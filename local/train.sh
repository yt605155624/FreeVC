#!/bin/bash
stage=0
stop_stage=0

# train freevc
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
   python3 train.py -c configs/freevc.json -m freevc
fi

# train freevc-s
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 train.py -c configs/freevc-s.json -m freevc-s 
fi

