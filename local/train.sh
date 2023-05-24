#!/bin/bash
stage=0
stop_stage=0
train_output_path=$1
port=$2
root_dir=$3
config_path=$4
model=$5

# 需要先单卡跑一遍往 vctk-16 里面生成 *.spec.pt 文件
# check https://github.com/OlaWod/FreeVC/issues/22#issuecomment-1369973615
# train freevc
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 train.py \
           --output-dir=${root_dir}/${train_output_path} \
           --config=${config_path} \
           --model=${model} \
           --port=${port}
fi



