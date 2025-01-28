#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python train_plot.py \
--name plot \
--num_slots 8 \
--img_aug \
--loss_names 'mlm+metric+id+local+recon' \
--dataset_name 'ICFG-PEDES' \
--root_dir '/data root dir' \
--num_epoch 60 \
--lr 5e-6 \
