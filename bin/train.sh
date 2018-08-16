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
--model_config $model_config \
$params

