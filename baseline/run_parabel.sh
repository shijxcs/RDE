#!/bin/bash

dataset=$1
data_dir="data/$dataset"
results_dir="data/$dataset/parabel"
model_dir="data/$dataset/parabel/model"

trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"
trn_score_file="${results_dir}/trn_SC.txt"
tst_score_file="${results_dir}/tst_SC.txt"

mkdir -p $results_dir
mkdir -p $model_dir

baseline/parabel/parabel_train $trn_ft_file $trn_lbl_file $model_dir -T 5 -s 0 -t 50 -b 1.0 -c 1.0 -m 100 -tcl 0.1 -tce 0 -e 0.0001 -n 20 -k 0 -q 0

baseline/parabel/parabel_predict $trn_ft_file $model_dir $trn_score_file -t 50
baseline/parabel/parabel_predict $tst_ft_file $model_dir $tst_score_file -t 50

rm -rf $model_dir
