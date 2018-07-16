#!/usr/bin/env bash

save_dir=$1

train_file=$DATA_DIR/conll05st-release-new/train-set.gz.parse.sdeps.combined.bio
dev_file=$DATA_DIR/conll05st-release-new/dev-set.gz.parse.sdeps.combined.bio
embeddings_file=embeddings/glove.6B.100d.txt

srun --gres=gpu --partition=gpu python3 train.py \
--train_file $train_file \
--dev_file $dev_file \
--save_dir $save_dir \
--word_embedding_file $embeddings_file