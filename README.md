# Human Motion Prediction
## General Information
Authors: Pratyush Singh, Rohit Kaushik, Melvin Ott; Group: BruteForce


## Abstract
Modelling human motion is a challenging task in computer vision and graphics since human body movement can change drastically from one environment to another. It’s an important task for developing and deploying autonomous agents. In order to tackle this issue, we take inspiration from the Structured Prediction Layer (SPL) which decomposes the pose into individual joints and can be augmented with different neural network architectures. We introduce the SPL dropout layer and describe its effect on the prediction scores. Using the per joint loss instead ofthe standard mean squared error as well as a residual connection for modelling velocities help us stay afloat on the top of leaderboard in the Machine Perception course project at ETH Zurich.

## Recreating our best submission
You can use the following command:

```bash
bsub -n 6 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 --save_dir ./experiments \
 --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.0 --input_hidden_size 256 --input_hidden_layers 1 --output_hidden_layers 1 --output_hidden_size 128 --input_dropout_rate 0.04 --num_epochs 700 --experiment_name rnnspl \ 
 --learning_rate_decay_rate 0.91 --residual_velocity --cell_size 2048 --cell_type blstm --early_stopping_tolerance 60
```

and evaluate with:

```bash
bsub -n 6 -W 4:00 -R "rusage[mem=1024, ngpus_excl_p=1]"  -R "select[gpu_model0==GeForceGTX1080Ti]" -o outvalidate.txt python evaluate_test.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 --save_dir ./experiments --model_id <model_id> --export
```

## Notes

Make sure to **not** use a GTX 1080 on the cluster as this lead to problems in our experience

Proper licensing and permission required to use the codes.
