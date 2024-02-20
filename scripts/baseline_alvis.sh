#!/bin/bash

NUM_GPUS=$1
BATCH_SIZE=$2






torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.dynamic_aug=False model.backbone=resnet50 \
training.attack.name=none training.batch_size=$BATCH_SIZE  out_dir=Results/Baseline/resnet50_dynamic_aug_False \
wandb.exp_name=Baseline_backbone_resnet50__dynamic_aug_False wandb.use=True


#torchrun --nproc_per_node=4 --master_port="$RANDOM" main.py data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.dynamic_aug=True model.backbone=resnet50 \
#training.attack.name=none training.batch_size=10  out_dir=Results/Baseline/resnet50_dynamic_aug_True \
#wandb.exp_name=Baseline_backbone_resnet50__dynamic_aug_True wandb.use=True
