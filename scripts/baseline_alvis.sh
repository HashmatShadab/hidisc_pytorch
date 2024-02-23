#!/bin/bash

NUM_GPUS=$1
BATCH_SIZE=$2
model_name=$3



# change the out_dir and wandb.exp_name to reflect the model name and dynamic_aug

dir="Results/Baseline/${model_name}_dynamic_aug_False_sanity_check_after_model_train_before_loss"
exp_name="Baseline_backbone_${model_name}_dynamic_aug_False_sanity_check_after_model_train_before_loss"



torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.dynamic_aug=False model.backbone=$model_name \
training.attack.name=none training.batch_size=$BATCH_SIZE  out_dir=$dir \
wandb.exp_name=$exp_name wandb.use=True


#torchrun --nproc_per_node=4 --master_port="$RANDOM" main.py data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.dynamic_aug=True model.backbone=resnet50 \
#training.attack.name=none training.batch_size=10  out_dir=Results/Baseline/resnet50_dynamic_aug_True \
#wandb.exp_name=Baseline_backbone_resnet50__dynamic_aug_True wandb.use=True
