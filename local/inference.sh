#!/bin/bash
stage=0
stop_stage=0
train_output_path=$1
ptfile_name=$2
root_dir=$3
config_path=$4
model=$5

# inference with FreeVC
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 convert.py \
        --hpfile=${config_path} \
        --ptfile=${root_dir}/${train_output_path}/${model}/${ptfile_name} \
        --txtpath=convert.txt \
        --outdir=${root_dir}/${train_output_path}/${model}/outputs
fi
