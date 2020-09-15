"""
SPL: training and evaluation of neural networks with a structured prediction layer.
Copyright (C) 2019 ETH Zurich, Emre Aksan, Manuel Kaufmann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


Structured prediction layer (SPL).

Given a context (i.e., rnn hidden state or something similar), implements pose prediction hierarchically by following
the skeletal structure. `build` method implements the operations in TF graph.

Skeletal structure for full-body human pose is defined below. Similarly, you can introduce a new skeleton configuration.
"""

import tensorflow as tf
import constants as C

# TODO (emre) this skeleton representation is a uncommon and tedious as well. Something more straightforward?
# [(Parent ID, Joint ID, Joint Name), (...)] where each entry in a list corresponds to the joints at the same
# level in the joint tree.
SMPL_SKELETON = [
    [(-1, 0, "l_hip"), (-1, 1, "r_hip"), (-1, 2, "spine1")],
    [(0, 3, "l_knee"), (1, 4, "r_knee"), (2, 5, "spine2")],
    [(5, 6, "spine3")],
    [(6, 7, "neck"), (6, 8, "l_collar"), (6, 9, "r_collar")],
    [(7, 10, "head"), (8, 11, "l_shoulder"), (9, 12, "r_shoulder")],
    [(11, 13, "l_elbow"), (12, 14, "r_elbow")]
]


class SPL(object):
    def __init__(self, hidden_layers, hidden_units, joint_size, reuse, config, is_training, sparse=False):

        self.config = config
        self.per_joint_layers = hidden_layers
        self.per_joint_units = hidden_units
        self.reuse = reuse
        self.is_training = is_training
        self.sparse_spl = sparse

        self.skeleton = SMPL_SKELETON
        self.num_joints = 15

        self.joint_size = joint_size
        self.human_size = self.num_joints * self.joint_size

        kinematic_tree = dict()
        for joint_list in self.skeleton:
            for joint_entry in joint_list:
                parent_list_ = [joint_entry[0]] if joint_entry[0] > -1 else []
                kinematic_tree[joint_entry[1]] = [parent_list_, joint_entry[1], joint_entry[2]]

        def get_all_parents(parent_list, parent_id, tree):
            if parent_id not in parent_list:
                parent_list.append(parent_id)
                for parent in tree[parent_id][0]:
                    get_all_parents(parent_list, parent, tree)

        self.prediction_order = list()
        self.indexed_skeleton = dict()

        # Reorder the structure so that we can access joint information by using its index.
        self.prediction_order = list(range(len(kinematic_tree)))
        for joint_id in self.prediction_order:
            joint_entry = kinematic_tree[joint_id]
            if self.sparse_spl:
                new_entry = joint_entry
            else:
                parent_list_ = list()
                if len(joint_entry[0]) > 0:
                    get_all_parents(parent_list_, joint_entry[0][0], kinematic_tree)
                new_entry = [parent_list_, joint_entry[1], joint_entry[2]]
            self.indexed_skeleton[joint_id] = new_entry

    def build(self, context):
        """
        Builds layers to make predictions.
        """
        joint_predictions = dict()

        for joint_key in self.prediction_order:
            parent_joint_ids, joint_id, joint_name = self.indexed_skeleton[joint_key]

            joint_inputs = [context]
            for parent_joint_id in parent_joint_ids:
                joint_inputs.append(joint_predictions[parent_joint_id])

            joint_predictions[joint_id] = self._predict_joint(tf.concat(joint_inputs, axis=-1),
                                                              self.joint_size,
                                                              joint_name)
        # Concatenate joints.
        pose_prediction = tf.concat(list(joint_predictions.values()), axis=-1)
        assert pose_prediction.get_shape()[-1] == self.human_size, "Prediction not matching with the skeleton."
        return pose_prediction

    def _predict_joint(self, inputs, output_size, name):
        """
        Builds dense output layers given the inputs. First, creates a number of hidden layers and then makes the
        prediction without applying an activation function.
        Args:
            inputs (tf.Tensor):
            output_size (int):
            name (str):
        Returns:
            (tf.Tensor) prediction.
        """
        current_layer = inputs
        for layer_idx in range(self.per_joint_layers):
            with tf.variable_scope('out_dense_' + name + "_" + str(layer_idx), reuse=tf.AUTO_REUSE):
                current_layer = tf.layers.dense(inputs=current_layer, units=self.per_joint_units, activation=tf.nn.relu)

                # apply dropout before passing the current predictions further
                if self.config["spl_dropout"] and self.config["spl_dropout_rate"] > 0:
                    current_layer = tf.layers.dropout(current_layer, rate=self.config["spl_dropout_rate"],
                                                      training=self.is_training)

        with tf.variable_scope('out_dense_' + name + "_" + str(self.per_joint_layers), reuse=tf.AUTO_REUSE):
            return tf.layers.dense(inputs=current_layer, units=output_size, activation=None)
