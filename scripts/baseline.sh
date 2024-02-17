#!/bin/bash

DATA_PATH="F:\Code\datasets\hidisc_data_small"

model_backbone=${1:-"resnet50"}
NUM_GPUS=${2:-1}
BATCH_SIZE=${3:-10}
attack=${4:-"none"}
dynamic_aug=${5:-"False"}
wandb_use=${6:-"True"}



echo "Running Baseline for $model_backbone with $NUM_GPUS GPUs and batch size $BATCH_SIZE"

torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.pydata.db_root=$DATA_PATH data.dynamic_aug=$dynamic_aug model.backbone=resnet50 \
training.attack.name=$attack training.batch_size=$BATCH_SIZE  out_dir="Results/Baseline/${model_backbone}" \
wandb.exp_name="Baseline_backbone_${model_backbone}__dynamic_aug_${dynamic_aug}" wandb.use=$wandb_use



#
#python main.py data.db_root=$DATA_PATH data.dynamic_aug=$dynamic_aug model.backbone=resnet50 \
#training.attack.name=$attack training.batch_size=$BATCH_SIZE  out_dir="Results/Baseline/${model_backbone}" \
#wandb.exp_name="Baseline_backbone${model_backbone}_attack_${attack}_dynamic_aug_${dynamic_aug}" wandb.use=$wandb_use
#





