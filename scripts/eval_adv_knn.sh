#!/bin/bash

DATA_PATH=/mimer/NOBACKUP/groups/alvis_cvl/Fahad/OpenSRH

model_name=$1




#ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp24_with_embedding256"
#
#echo "ckpt_dir: $ckpt_dir"
#echo "saving results to: $ckpt_dir/eval_knn_results"
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in $ckpt_dir/*.pth; do
#
#    echo $ckpt_path
#    python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#    python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#
#done
#
#
#
#ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp25_with_embedding512"
#
#echo "ckpt_dir: $ckpt_dir"
#echo "saving results to: $ckpt_dir/eval_knn_results"
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in $ckpt_dir/*.pth; do
#
#    echo $ckpt_path
#    python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#    python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#
#done
#
#
#
#ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp26_with_embedding768"
#
#echo "ckpt_dir: $ckpt_dir"
#echo "saving results to: $ckpt_dir/eval_knn_results"
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in $ckpt_dir/*.pth; do
#
#    echo $ckpt_path
#    python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 768  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#    python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 768  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#
#done
#
#
#ckpt_dir="Results/Adv/resnet50_timm_pretrained_dynamicaug_true_epsilon_warmup_5000_only_adv_exp27_with_embedding1024"
#
#echo "ckpt_dir: $ckpt_dir"
#echo "saving results to: $ckpt_dir/eval_knn_results"
#
## loop over all the checkpoints in the directory ending with .pth
#for ckpt_path in $ckpt_dir/*.pth; do
#
#    echo $ckpt_path
#    python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 1024  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#    python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 1024  \
#    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
#
#done
#



######################################################################################################################

ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp19"

echo "ckpt_dir: $ckpt_dir"
echo "saving results to: $ckpt_dir/eval_knn_results"

# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $ckpt_dir/*.pth; do

    echo $ckpt_path
    python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
    python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name   \
    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results

done





ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp24_with_embedding256"

echo "ckpt_dir: $ckpt_dir"
echo "saving results to: $ckpt_dir/eval_knn_results"

# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $ckpt_dir/*.pth; do

    echo $ckpt_path
    python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256  \
    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
    python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 256  \
    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results

done



ckpt_dir="Results/Adv/resnet50_at_dynamicaug_true_epsilon_warmup_5000_only_adv_exp25_with_embedding512/"

echo "ckpt_dir: $ckpt_dir"
echo "saving results to: $ckpt_dir/eval_knn_results"

# loop over all the checkpoints in the directory ending with .pth
for ckpt_path in $ckpt_dir/*.pth; do

    echo $ckpt_path
    python adv_eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512  \
    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results
    python eval_knn.py --data_db_root $DATA_PATH  --model_backbone $model_name --model_num_embedding_out 512  \
    --eval_ckpt_path $ckpt_path  --save_results_path $ckpt_dir/eval_knn_results

done
