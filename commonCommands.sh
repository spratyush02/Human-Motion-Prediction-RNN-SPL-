bsub -n 6 -W 4:00 -R "rusage[mem=1024, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 --save_dir ./experiments --experiment_name sample

#1597236108 (online:3.1245266 ) #adagrad Optimizer

bsub -n 6 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 --save_dir ./experiments  
--model_type rnn_spl --optimizer adagrad --spl_dropout --spl_dropout_rate 0.0 --input_hidden_size 256 --input_hidden_layers 1 --output_hidden_layers 1 
--output_hidden_size 128 --input_dropout_rate 0.05 --num_epochs 1000 --experiment_name rnnspl --learning_rate_decay_rate 0.93 
--residual_velocity --cell_size 1024 --cell_type blstm --early_stopping_tolerance 200



#1597137370 (valid loss: 2.74) and 1597136472 (valid loss: 2.76)
bsub -n 6 -W 24:00 -R "rusage[mem=1024, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 \
--save_dir ./experiments  --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.1 --input_hidden_size 256 --input_hidden_layers 1 \
--output_hidden_layers 1 --output_hidden_size 64 --input_dropout_rate 0.1 --num_epochs 500 --experiment_name rnnspldef1024 --learning_rate_decay_rate 0.98 --residual_velocity \
--cell_size 512 --cell_type lstm

#1597137999 (valid loss: 2.7355; online: 2.252)
bsub -n 6 -W 24:00 -R "rusage[mem=1024, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 \
--save_dir ./experiments  --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.1 --input_hidden_size 256 --input_hidden_layers 1 \
--output_hidden_layers 1 --output_hidden_size 64 --input_dropout_rate 0.1 --num_epochs 500 --experiment_name rnnspl1024blstm --learning_rate_decay_rate 0.98 --residual_velocity \
--cell_size 1024 --cell_type blstm

#1597146040 (valid loss: 2.740; online: 2.2502)
bsub -n 6 -W 24:00 -R "rusage[mem=1024, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 \
--save_dir ./experiments  --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.0 --input_hidden_size 256 --input_hidden_layers 1 \
--output_hidden_layers 1 --output_hidden_size 64 --input_dropout_rate 0.1 --num_epochs 500 --experiment_name rnnspl1024blstm00drop --learning_rate_decay_rate 0.98 --residual_velocity \
--cell_size 1024 --cell_type blstm



#1597157709 (valid loss: 2.6855; online: 2.1872)
bsub -n 6 -W 24:00 -R "rusage[mem=1024, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 \
--save_dir ./experiments  --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.0 --input_hidden_size 256 --input_hidden_layers 1 \
--output_hidden_layers 1 --output_hidden_size 64 --input_dropout_rate 0.05 --num_epochs 1000 --experiment_name rnnspl1024blstm00dropmordecaymoredrop --learning_rate_decay_rate 0.95 --residual_velocity \
--cell_size 1024 --cell_type blstm

#1597164595 (valid loss: 2.674; online: 2.2006)
bsub -n 6 -W 24:00 -R "rusage[mem=1024, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 \
--save_dir ./experiments  --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.0 --input_hidden_size 256 --input_hidden_layers 1 \
--output_hidden_layers 1 --output_hidden_size 128 --input_dropout_rate 0.05 --num_epochs 1000 --experiment_name rnnspl1024blstm00dropmordecaymoredrop_2128 --learning_rate_decay_rate 0.95 --residual_velocity \
--cell_size 1024 --cell_type blstm --early_stopping_tolerance 50


#1597165376 (valid loss: 2.671; online: 2.1900)
bsub -n 6 -W 24:00 -R "rusage[mem=1024, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 \
--save_dir ./experiments  --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.0 --input_hidden_size 256 --input_hidden_layers 1 \
--output_hidden_layers 1 --output_hidden_size 128 --input_dropout_rate 0.1 --num_epochs 1000 --experiment_name rnnspl1024blstm00dropmordecaymoredrop_2128ot --learning_rate_decay_rate 0.95 --residual_velocity \
--cell_size 1024 --cell_type blstm --early_stopping_tolerance 50


#1597222727 (valid loss: 2.639; online: 2.138)
bsub -n 6 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 \
--save_dir ./experiments  --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.0 --input_hidden_size 256 --input_hidden_layers 1 \
--output_hidden_layers 1 --output_hidden_size 128 --input_dropout_rate 0.05 --num_epochs 1000 --experiment_name rnnspl1024blstm00dropmordecaymoredrop_212893 --learning_rate_decay_rate 0.93 --residual_velocity \
--cell_size 1024 --cell_type blstm --early_stopping_tolerance 60



#1597230977 (valid loss: 2.614; online: 2.128)
bsub -n 6 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 \
--save_dir ./experiments  --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.0 --input_hidden_size 256 --input_hidden_layers 1 \
--output_hidden_layers 1 --output_hidden_size 128 --input_dropout_rate 0.05 --num_epochs 1000 --experiment_name rnnspl1024blstm00dropmordecaymoredrop_212891 --learning_rate_decay_rate 0.92 --residual_velocity \
--cell_size 1024 --cell_type blstm --early_stopping_tolerance 60

# 1597264645 (valid loss: 2.619; online: 2.118)
bsub -n 6 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 \
--save_dir ./experiments  --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.0 --input_hidden_size 256 --input_hidden_layers 1 \
--output_hidden_layers 1 --output_hidden_size 128 --input_dropout_rate 0.05 --num_epochs 500 --experiment_name rnnspl1024blstm00dropmordecaymoredrop_2128905 --learning_rate_decay_rate 0.905 --residual_velocity \
--cell_size 1024 --cell_type blstm --early_stopping_tolerance 60


######### current best online score
#1597348180 (online: 2.072)
 bsub -n 6 -W 24:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" python train.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 --save_dir ./experiments  
 --model_type rnn_spl --spl_dropout --spl_dropout_rate 0.0 --input_hidden_size 256 --input_hidden_layers 1 --output_hidden_layers 1 --output_hidden_size 128 --input_dropout_rate 0.04 --num_epochs 700 --experiment_name rnnspl 
 --learning_rate_decay_rate 0.91 --residual_velocity --cell_size 2048 --cell_type blstm --early_stopping_tolerance 60





bsub -n 6 -W 1:00 -R "rusage[mem=1024, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" -o outvalidate.txt python evaluate_test.py --data_dir /cluster/project/infk/hilliges/lectures/mp20/project4 --save_dir ./experiments --model_id 1597650923 --export --visualize
