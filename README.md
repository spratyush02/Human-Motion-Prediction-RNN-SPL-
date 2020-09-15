# Human Motion Prediction
## General Information
Authors: Rohit Kaushik, Pratyush Singh, Melvin Ott; Group: BruteForce

## Setup
We didn't use any additional libraries, so you can install the depenencies as described on the project website

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