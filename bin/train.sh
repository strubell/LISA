#!/usr/bin/env bash

config_file=$1

source ${config_file}

params=${@:2}

transition_stats=$data_dir/transition_probs.tsv

python3 src/train.py \
--train_files $train_files \
--dev_files $dev_files \
--transition_stats $transition_stats \
--data_config $data_config \
--model_configs $model_configs \
--task_configs $task_configs \
--layer_configs $layer_configs \
--attention_configs $attention_configs \
--best_eval_key $best_eval_key \
$params

