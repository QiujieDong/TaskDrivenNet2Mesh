#!/usr/bin/env bash

## run the training
python train_seg.py \
--dataset_name human_trainDense_testSimply \
--experiment_name human \
--num_classes 8 \
--k_eig_list 749 64 16 \
--lr 3e-3 \
--batch_size 3 \
--weight_decay 0.3 \
--smoothing 0.1 \
--augment_data \

