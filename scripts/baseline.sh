#!/bin/bash

DATA_PATH=${1:-"F:\Code\datasets\hidisc_data_small"}
model_backbone=${2:-"resnet50"}
NUM_GPUS=${3:-1}
BATCH_SIZE=${4:-10}
attack=${5:-"none"}
dynamic_aug=${6:-"False"}
wandb_use=${7:-"True"}



echo "Running Baseline for $model_backbone with $NUM_GPUS GPUs and batch size $BATCH_SIZE"

wandb_exp_name="Baseline_backbone_${model_backbone}__dynamic_aug_${dynamic_aug}"
out_dir="Results/Baseline/${model_backbone}"

torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=$DATA_PATH data.dynamic_aug=$dynamic_aug model.backbone=resnet50 \
training.attack.name=$attack training.batch_size=$BATCH_SIZE  out_dir=$out_dir \
wandb.exp_name=$wandb_exp_name wandb.use=$wandb_use



#
#python main.py data.db_root=$DATA_PATH data.dynamic_aug=$dynamic_aug model.backbone=resnet50 \
#training.attack.name=$attack training.batch_size=$BATCH_SIZE  out_dir="Results/Baseline/${model_backbone}" \
#wandb.exp_name="Baseline_backbone${model_backbone}_attack_${attack}_dynamic_aug_${dynamic_aug}" wandb.use=$wandb_use
#





