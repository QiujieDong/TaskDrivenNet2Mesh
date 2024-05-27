#!/usr/bin/env bash

## run the training
python train_seg.py \
--dataset_name coseg_chairs \
--experiment_name chairs \
--num_classes 3 \
--k_eig_list 485 64 16 \
--lr 3e-4 \
--batch_size 3 \
--weight_decay 0.3 \
--smoothing 0.1 \
--augment_data \
