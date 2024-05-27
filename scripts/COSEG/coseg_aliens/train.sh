#!/usr/bin/env bash

## run the training
python train_seg.py \
--dataset_name coseg_aliens \
--experiment_name alines \
--mode train \
--num_classes 4 \
--k_eig_list 751 64 16 \
--lr 3e-4 \
--batch_size 3 \
--weight_decay 0.3 \
--smoothing 0.1 \
--augment_data \
--random_rotate_axis y \
--warm_up_epochs 20 \
--epochs 3000 \
--iter_num 20 \
