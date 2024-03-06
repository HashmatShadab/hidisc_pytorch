#!/bin/bash

model_name=$1
exp_num=$2
BATCH_SIZE=${3:-10}
ssl_ckpt=${4:-"F:\Code\Projects\hidisc_pytorch\Results\Baseline\resnet50_timm_pretrained_single_gpu_exp21\checkpoint.pth"}

# get the parent name of ssl_ckpt
parentname="$(basename "$(dirname "$ssl_ckpt")")"
echo "$parentname"

if [ $exp_num -eq 1 ]
then
    wandb_exp_name="FT_backbone_${model_name}_ft_linear_class_balanced_false_attack_pgd_7_eps_8_exp_${exp_num}"
    out_dir="FT/pretrain_${parentname}/backbone_${model_name}_ft_linear_class_balanced_false_attack_pgd_7_eps_8_exp_${exp_num}"

    echo "Saving Results in  $out_dir"

    python finetuning.py distributed.single_gpu=True data.balance_study_per_class=True \
    model.backbone=$model_name model.start_from_ssl_ckpt=$ssl_ckpt model.finetuning=linear model.num_classes=7 \
    training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=pgd training.attack_eps=8.0 training.attack_steps=7 \
    out_dir=$out_dir \
    wandb.exp_name=$wandb_exp_name wandb.use=False
fi

if [ $exp_num -eq 2 ]
then
    wandb_exp_name="FT_backbone_${model_name}_ft_linear_class_balanced_false_attack_pgd_7_eps_8_exp_${exp_num}"
    out_dir="FT/pretrain_${parentname}/backbone_${model_name}_ft_linear_class_balanced_false_attack_pgd_7_eps_8_exp_${exp_num}"

    echo "Saving Results in  $out_dir"

    python finetuning.py distributed.single_gpu=True data.balance_study_per_class=True \
    model.backbone=$model_name model.start_from_ssl_ckpt=false model.finetuning=linear model.num_classes=7 \
    training.num_epochs=20 training.batch_size=$BATCH_SIZE training.train_attack=pgd training.attack_eps=8.0 training.attack_steps=7 \
    out_dir=$out_dir \
    wandb.exp_name=$wandb_exp_name wandb.use=False

fi
