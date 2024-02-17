#!/bin/bash

DATA_PATH="F:\Code\datasets\hidisc_data_small"

model_backbone=${1:-"resnet50"}
NUM_GPUS=${2:-1}
BATCH_SIZE=${3:-10}
attack=${4:-"pgd_2"}
attack_eps=${5:-8}
dynamic_aug=${6:-"False"}
wandb_use=${7:-"False"}



echo "Running Adv Training for $model_backbone with $NUM_GPUS GPUs and batch size $BATCH_SIZE"

torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py  data.db_root=$DATA_PATH data.dynamic_aug=$dynamic_aug model.backbone=resnet50 \
training.attack.name=$attack training.attack.eps=$attack_eps training.batch_size=$BATCH_SIZE  out_dir="Results/Adv/${model_backbone}_attack_${attack}_eps_{$attack_eps}_dynaug_${dynamic_aug}" \
wandb.exp_name="Adv_backbone_${model_backbone}_attack_${attack}_eps_{$attack_eps}_dynamic_aug_${dynamic_aug}" wandb.use=$wandb_use



#python main.py data.db_root=$DATA_PATH data.dynamic_aug=$dynamic_aug model.backbone=resnet50 \
#training.attack.name=$attack training.attack.eps=$attack_eps training.batch_size=$BATCH_SIZE  out_dir="Results/Adv/${model_backbone}_attack_${attack}_eps_{$attack_eps}_dynaug_${dynamic_aug}" \
#wandb.exp_name="Adv_backbone${model_backbone}_attack_${attack}_eps_{$attack_eps}_dynamic_aug_${dynamic_aug}" wandb.use=$wandb_use
#
#


