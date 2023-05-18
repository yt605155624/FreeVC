#!/bin/bash

set -e

gpus=0,1
stage=0
stop_stage=100
train_output_path=exp/default

# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ./local/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    CUDA_VISIBLE_DEVICES=${gpus} ./local/preprocess.sh || exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
   
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${train_output_path} || exit -1
fi
# 可以成功运行
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    
    CUDA_VISIBLE_DEVICES=${gpus} ./local/inference.sh || exit -1
fi

