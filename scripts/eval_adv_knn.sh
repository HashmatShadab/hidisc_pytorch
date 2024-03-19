#!/bin/bash

DATA_PATH=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH

model_name=$1
ckpt_path=$2

ckpt_dir=$(dirname $PWD/$ckpt_path)
echo "ckpt_dir: $ckpt_dir"



echo "saving results to: $ckpt_dir/eval_knn_results"
#
python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name \
--eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
