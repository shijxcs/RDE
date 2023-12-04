#!/bin/bash

dataset=$1
data_dir="data/$dataset"
results_dir="data/$dataset/fastxml"
model_dir="data/$dataset/fastxml/model"

trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"
trn_score_file="${results_dir}/trn_SC.txt"
tst_score_file="${results_dir}/tst_SC.txt"

mkdir -p $results_dir
mkdir -p $model_dir

baseline/fastxml/fastXML_train $trn_ft_file $trn_lbl_file $model_dir -T 5 -s 0 -t 50 -b 1.0 -c 1.0 -m 10 -l 10

baseline/fastxml/fastXML_predict $trn_ft_file $trn_score_file $model_dir
baseline/fastxml/fastXML_predict $tst_ft_file $tst_score_file $model_dir

rm -rf $model_dir
