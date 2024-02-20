#!/bin/bash


NUM_GPUS=$1
BATCH_SIZE=$2
exp_num=$3
model_name=$4

# if exp_num == 1, run the firstscript


if [ $exp_num -eq 1 ]
then
  echo "Running Adv Training for resnet50 with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=True"
    torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.dynamic_aug=True model.backbone=$model_name \
    training.attack.name=pgd  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/${model_name}_attack_pgd_eps_8_dynaug_True \
    wandb.exp_name=Adv_backbone_${model_name}_attack_pgd_eps_8_dynamic_aug_True wandb.use=True
fi

if [ $exp_num -eq 2 ]
then
    echo "Running Adv Training for resnet50 with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=False"
    torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.dynamic_aug=False model.backbone=resnet50 \
    training.attack.name=pgd  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/resnet50_attack_pgd_eps_8_dynaug_False\
    wandb.exp_name=Adv_backbone_resnet50_attack_pgd_eps_8_dynamic_aug_False wandb.use=True
fi

if [ $exp_num -eq 3 ]
then
    echo "Running Adv Training for resnet50_multi_bn with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=True"
    torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.dynamic_aug=True model.backbone=resnet50_multi_bn \
    training.attack.name=pgd  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/resnet50_multi_bn_attack_pgd_eps_8_dynaug_True \
    wandb.exp_name=Adv_backbone_resnet50_multi_bn_attack_pgd_eps_8_dynamic_aug_True wandb.use=True
fi

if [ $exp_num -eq 4 ]
then
    echo "Running Adv Training for resnet50_multi_bn with $NUM_GPUS GPUs and batch size $BATCH_SIZE and dynamic_aug=False"
    torchrun --nproc_per_node=$NUM_GPUS --master_port="$RANDOM" main.py data.db_root=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH data.dynamic_aug=False model.backbone=resnet50_multi_bn \
    training.attack.name=pgd  training.attack.eps=8 training.batch_size=$BATCH_SIZE  out_dir=Results/Adv/resnet50_multi_bn_attack_pgd_eps_8_dynaug_False\
    wandb.exp_name=Adv_backbone_resnet50_multi_bn_attack_pgd_eps_8_dynamic_aug_False wandb.use=True
fi

