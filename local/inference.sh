#!/bin/bash
stage=0
stop_stage=1

# inference with FreeVC
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 convert.py \
        --hpfile=configs/freevc.json \
        --ptfile=pretrained_models/checkpoints/freevc.pth \
        --txtpath=convert.txt \
        --outdir=outputs/freevc
fi

# inference with FreeVC-s
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    python3 convert.py \
        --hpfile=configs/freevc-s.json \
        --ptfile=pretrained_models/checkpoints/freevc-s.pth \
        --txtpath=convert.txt \
        --outdir=outputs/freevc-s
fi