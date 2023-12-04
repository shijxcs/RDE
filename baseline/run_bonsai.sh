#!/bin/bash

dataset=$1
data_dir="data/$dataset"
results_dir="data/$dataset/bonsai"
model_dir="data/$dataset/bonsai/model"

trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
trn_ft_lbl_file="${data_dir}/trn_X_XY.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"
trn_score_file="${results_dir}/trn_SC.txt"
tst_score_file="${results_dir}/tst_SC.txt"

mkdir -p $results_dir
mkdir -p $model_dir

# training
# Reads training features (in $trn_ft_file), training labels (in $trn_lbl_file), and writes FastXML model to $model_dir

# NOTE: The usage of Bonsai for other datasets requires setting parameter `-m` to 2 for smaller datasets like EUR-Lex, Wikipedia-31K 
#       and to 3 for larger datasets like Delicious-200K, WikiLSHTC-325K, Amazon-670K, Wikipedia-500K, Amazon-3M.


depth=2
if [ $dataset = "amazon670k" ]; then
   depth=3
fi

baseline/bonsai/bonsai_train $trn_ft_file $trn_lbl_file $trn_ft_lbl_file $model_dir \
    -T 3 \
    -s 0 \
    -t 3 \
    -w 100 \
    -b 1.0 \
    -c 1.0 \
    -m $depth \
    -f 0.1 \
    -fcent 0 \
    -k 0.0001 \
    -siter 20 \
    -q 0 \
    -ptype 0 \
    -ctype 0

# testing
# Reads test features (in $tst_ft_file), FastXML model (in $model_dir), and writes test label scores to $score_file
baseline/bonsai/bonsai_predict $trn_ft_file $trn_score_file $model_dir
baseline/bonsai/bonsai_predict $tst_ft_file $tst_score_file $model_dir

rm -rf $model_dir
