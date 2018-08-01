#!/usr/bin/env bash


data_dir=$DATA_DIR/conll05st-release-new
train_file=$data_dir/train-set.gz.parse.sdeps.combined.bio
dev_file=$data_dir/dev-set.gz.parse.sdeps.combined.bio
transition_stats=$data_dir/transition_probs.tsv

params=${@:1}

python3 train.py \
--train_file $train_file \
--dev_file $dev_file \
--transition_stats $transition_stats \
$params

