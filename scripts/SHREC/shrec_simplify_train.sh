#!/usr/bin/env bash

## run the training
python train_cls.py \
--dataset_name shrec11_split16 \
--experiment_name base_shrec16 \
--num_classes 30 \
--split_size 16 \
--k_eig_list 249 128 64 \
--lr 3e-4 \
--batch_size 3 \
--weight_decay 0.05 \
--smoothing 0.1 \
--augment_data \
