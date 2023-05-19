#!/bin/bash
stage=0
stop_stage=0
train_output_path=$1
ptfile_name=$2
root_dir=$3

# inference with FreeVC
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 convert.py \
        --hpfile=configs/freevc.json \
        --ptfile=${root_dir}/${train_output_path}/freevc/${ptfile_name} \
        --txtpath=convert.txt \
        --outdir=${root_dir}/${train_output_path}/freevc//outputs
fi

# inference with FreeVC-s
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 convert.py \
        --hpfile=configs/freevc-s.json \
        --ptfile=${root_dir}/${train_output_path}/freevc-s/${ptfile_name} \
        --txtpath=convert.txt \
        --outdir=${root_dir}/${train_output_path}/freevc-s/outputs
fi