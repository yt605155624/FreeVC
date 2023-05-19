#!/bin/bash
stage=0
stop_stage=0
train_output_path=$1
port=$2
root_dir='/nfs-speech-tx/dev/yuantian04/Voice_Conversion/FreeVC/FreeVC_base/FreeVC'


# train freevc
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   python3 train.py \
           --output-dir=${root_dir}/${train_output_path} \
           --config=configs/freevc.json \
           --model=freevc \
           --port=${port}
fi

# train freevc-s
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 train.py -c configs/freevc-s.json -m freevc-s 
fi

