#!/bin/bash
stage=0
stop_stage=0
train_output_path=$1


# train freevc
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   # 单卡
   CUDA_VISIBLE_DEVICES=0 python3 train.py \
            --output-dir=${train_output_path} \
            --config=configs/freevc.json \
            --model=freevc
   # 多卡
#    python3 -m torch.distributed.launch train.py -c configs/freevc.json -m freevc
fi

# train freevc-s
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 train.py -c configs/freevc-s.json -m freevc-s 
fi

