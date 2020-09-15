"""Copyright (c) 2019 AIT Lab, ETH Zurich, Manuel Kaufmann, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import zipfile
from constants import Constants as C
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest


def get_activation_fn(activation=C.RELU):
    """
    Return tensorflow activation function given string name.

    Args:
        activation: The requested activation function.

    Returns: The tf op corresponding to the requested activation function.
    """
    # Check if the activation is already callable.
    if callable(activation):
        return activation

    if activation is None:
        return None
    elif activation == C.RELU:
        return tf.nn.relu
    else:
        raise Exception("Activation function is not implemented.")


def export_code(file_list, output_file):
    """
    Adds the given file paths to a zip file.
    Args:
        file_list: List of paths to files
        output_file: Name and path of the zip archive to be created
    """
    zipf = zipfile.ZipFile(output_file, mode="w", compression=zipfile.ZIP_DEFLATED)
    for f in file_list:
        zipf.write(f)
    zipf.close()


def export_results(eval_result, output_file):
    """
    Write predictions into a csv file that can be uploaded to the submission system.
    Args:
        eval_result: A dictionary {sample_id => (prediction, seed)}. This is exactly what is returned
          by `evaluate_test.evaluate_model`.
        output_file: Where to store the file.
    """

    def to_csv(fname, poses, ids, split=None):
        n_samples, seq_length, dof = poses.shape
        data_r = np.reshape(poses, [n_samples, seq_length * dof])
        cols = ['dof{}'.format(i) for i in range(seq_length * dof)]

        # add split id very last
        if split is not None:
            data_r = np.concatenate([data_r, split[..., np.newaxis]], axis=-1)
            cols.append("split")

        data_frame = pd.DataFrame(data_r,
                                  index=ids,
                                  columns=cols)
        data_frame.index.name = 'Id'

        if not fname.endswith('.gz'):
            fname += '.gz'

        data_frame.to_csv(fname, float_format='%.8f', compression='gzip')

    sample_file_ids = []
    sample_poses = []
    for k in eval_result:
        sample_file_ids.append(k)
        sample_poses.append(eval_result[k][0])

    to_csv(output_file, np.stack(sample_poses), sample_file_ids)


class CustomMultiRNNCell(tf.nn.rnn_cell.MultiRNNCell):
    def __init__(self, cells, state_is_tuple=True, intermediate_outputs=True):
        """
        Extends tensorflow MultiRNNCell such that outputs of the intermediate cells can be accessed.
        """
        super(CustomMultiRNNCell, self).__init__(cells, state_is_tuple)
        self._intermediate_outputs = intermediate_outputs

    @property
    def output_size(self):
        if self._intermediate_outputs:
            if self._state_is_tuple:
                return tuple(cell.output_size for cell in self._cells)
            else:
                return sum([cell.output_size for cell in self._cells])
        else:
            return self._cells[-1].output_size

    def call(self, inputs, state):
        """Run this multi-layer cell on inputs, starting from state."""
        cur_state_pos = 0
        cur_inp = inputs
        new_states = []
        new_outputs = []
        for i, cell in enumerate(self._cells):
            with vs.variable_scope("cell_%d" % i):
                if self._state_is_tuple:
                    if not nest.is_sequence(state):
                        raise ValueError("Expected state to be a tuple of length %d, but received: %s" % (len(self.state_size), state))
                    cur_state = state[i]
                else:
                    cur_state = array_ops.slice(state, [0, cur_state_pos], [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                cur_inp, new_state = cell(cur_inp, cur_state)
                new_states.append(new_state)
                new_outputs.append(cur_inp)

        new_states = (tuple(new_states) if self._state_is_tuple else array_ops.concat(new_states, 1))
        if self._intermediate_outputs:
            new_outputs = (tuple(new_outputs) if self._state_is_tuple else array_ops.concat(new_outputs, 1))
            return new_outputs, new_states
        else:
            return cur_inp, new_states


def get_rnn_cell(**kwargs):
    """
    Creates an rnn cell object.
    Args:
        **kwargs: must contain `cell_type`, `size` and `num_layers` key-value pairs. `dropout_keep_prob` is optional.
            `dropout_keep_prob` can be a list of ratios where each cell has different dropout ratio in a stacked
            architecture. If it is a scalar value, then the whole architecture (either a single cell or stacked cell)
            has one DropoutWrapper.
    Returns:
    """
    cell_type = kwargs['cell_type']
    size = kwargs['size']
    num_layers = kwargs['num_layers']
    dropout_keep_prob = kwargs.get('dropout_keep_prob', 1.0)
    intermediate_outputs = kwargs.get('intermediate_outputs', False)

    separate_dropout = False
    if isinstance(dropout_keep_prob, list) and len(dropout_keep_prob) == num_layers:
        separate_dropout = True

    if cell_type == C.LSTM:
        rnn_cell_constructor = tf.contrib.rnn.LSTMCell
    elif cell_type == C.BLSTM:
        rnn_cell_constructor = tf.contrib.rnn.LSTMBlockCell
    elif cell_type.lower() == C.GRU:
        rnn_cell_constructor = tf.contrib.rnn.GRUCell
    elif cell_type.lower() == C.LayerNormLSTM.lower():
        rnn_cell_constructor = tf.contrib.rnn.LayerNormBasicLSTMCell
    else:
        raise Exception("Unsupported RNN Cell.")

    rnn_cells = []
    for i in range(num_layers):
        cell = rnn_cell_constructor(size)
        if separate_dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 input_keep_prob=dropout_keep_prob[i],
                                                 output_keep_prob=dropout_keep_prob,
                                                 state_keep_prob=1,
                                                 dtype=tf.float32,
                                                 seed=1)
        rnn_cells.append(cell)

    if num_layers > 1:
        # cell = tf.nn.rnn_cell.MultiRNNCell(cells=rnn_cells, state_is_tuple=True)
        cell = CustomMultiRNNCell(cells=rnn_cells, state_is_tuple=True, intermediate_outputs=intermediate_outputs)
    else:
        cell = rnn_cells[0]

    if separate_dropout and dropout_keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                             input_keep_prob=dropout_keep_prob,
                                             output_keep_prob=dropout_keep_prob,
                                             state_keep_prob=1,
                                             dtype=tf.float32,
                                             seed=1)
    return cell
