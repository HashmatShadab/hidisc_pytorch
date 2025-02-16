#!/bin/bash


# Default values for arguments
EXP_NUM=${1:-18}
EPSILON=${2:-8}


# Run evaluation scripts
bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true $EPSILON 7 64 "ffgsm_knn"

bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true $EPSILON 7 64 "mifgsm_knn"
bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true $EPSILON 10 64 "mifgsm_knn"
bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true $EPSILON 20 64 "mifgsm_knn"

bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true $EPSILON 7 64 "bim_knn"
bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true $EPSILON 10 64 "bim_knn"
bash scripts/eval_adv_knn_40k_any_attack.sh $EXP_NUM true $EPSILON 20 64 "bim_knn"


echo "All evaluation scripts executed successfully!"
