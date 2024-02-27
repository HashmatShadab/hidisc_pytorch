#!/bin/bash

DATA_PATH=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH

model_name=$1
ckpt_path=$2

echo "Model Name: $model_name"
echo "Checkpoint Path: $ckpt_path"

# print current directory
echo "Current Directory: $PWD"

# get parent directory starting from the current directory
ckpt_dir=$(dirname $PWD/$ckpt_path)
# get experiment name as the folder name of ckpt_path
exp_name=$(basename $ckpt_dir)

echo "Experiment Name: $exp_name"

python eval_knn.py data.db_root=$DATA_PATH  model.backbone=$model_name \
eval.ckpt_path=$ckpt_path  out_dir=eval_knn_results/$exp_name


