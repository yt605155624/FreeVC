#!/bin/bash
stage=0
stop_stage=100

sr_min=68
sr_max=92
root_dir='/nfs-speech-tx/dev/yuantian04/Voice_Conversion/FreeVC/FreeVC_base/FreeVC'

# for VCTK only
# 同时下采样到 16k 和 22.05k, 22.05k 用于 preprocess_sr, 因为 HiFiGAN 是 22.05k 的 
# 但是 preprocess_sr 里面 load 的时候用了 hps.sampling_rate，所以这个操作是否一定需要？
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 downsample.py \
        --in_dir=~/datasets/VCTK/wav48_silence_trimmed/ \
        --sr1=16000 \
        --out_dir1=./dataset/vctk-16k \
        --sr2=22050 \
        --out_dir2=./dataset/vctk-22k \
        --num-cpu=20
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ln -s dataset/vctk-16k DUMMY
fi

# shuffle *.txt in ./filelists, ./filelists 中仅包含 VCTK, 如果加了新数据一定要跑这个 stage
# 我改了数据预处理, 多了 s5, 所以也一定要跑这个
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python3 preprocess_flist.py \
        --source_dir=DUMMY \
        --list_dir=./filelists
fi

# preprocess_spk.py -> FreeVC, with pretrained speaker encoder

# preprocess_ssl.py -> FreeVC (w/o SR), uses pretrained speaker encoder, without SR-based data augmentation
# preprocess_sr.py -> with SR-based augmentation
# preprocess_sr.py 中包含提取 SSL 的过程
# preprocess_spk.py + preprocess_sr.py 就是默认的 FreeVC

# 结束之后有个段错误，之后看看
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python3 preprocess_spk.py \
        --in_dir=dataset/vctk-16k \
        --out_dir_root=dataset \
        --num_workers=12
fi


# 看下  stage 4 和 stage 5 的区别
# 在不同卡上同时执行多个

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 preprocess_sr.py \
        --sr=16000 \
        --config=hifigan/config.json \
        --in_dir=dataset/vctk-22k \
        --wav_dir=${root_dir}/dataset/sr/wav \
        --ssl_dir=${root_dir}/dataset/sr/wavlm \
        --min=68 \
        --max=72 \
        --num_workers=20 & CUDA_VISIBLE_DEVICES=0 python3 preprocess_sr.py \
            --sr=16000 \
            --config=hifigan/config.json \
            --in_dir=dataset/vctk-22k \
            --wav_dir=${root_dir}/dataset/sr/wav \
            --ssl_dir=${root_dir}/dataset/sr/wavlm \
            --min=73 \
            --max=76 \
            --num_workers=20 & CUDA_VISIBLE_DEVICES=1 python3 preprocess_sr.py \
                --sr=16000 \
                --config=hifigan/config.json \
                --in_dir=dataset/vctk-22k \
                --wav_dir=${root_dir}/dataset/sr/wav \
                --ssl_dir=${root_dir}/dataset/sr/wavlm \
                --min=77 \
                --max=80 \
                --num_workers=20 & CUDA_VISIBLE_DEVICES=1 python3 preprocess_sr.py \
                    --sr=16000 \
                    --config=hifigan/config.json \
                    --in_dir=dataset/vctk-22k \
                    --wav_dir=${root_dir}/dataset/sr/wav \
                    --ssl_dir=${root_dir}/dataset/sr/wavlm \
                    --min=81 \
                    --max=84 \
                    --num_workers=20 & CUDA_VISIBLE_DEVICES=2 python3 preprocess_sr.py \
                        --sr=16000 \
                        --config=hifigan/config.json \
                        --in_dir=dataset/vctk-22k \
                        --wav_dir=${root_dir}/dataset/sr/wav \
                        --ssl_dir=${root_dir}/dataset/sr/wavlm \
                        --min=85 \
                        --max=88 \
                        --num_workers=20 & CUDA_VISIBLE_DEVICES=2 python3 preprocess_sr.py \
                            --sr=16000 \
                            --config=hifigan/config.json \
                            --in_dir=dataset/vctk-22k \
                            --wav_dir=${root_dir}/dataset/sr/wav \
                            --ssl_dir=${root_dir}/dataset/sr/wavlm \
                            --min=89 \
                            --max=92 \
                            --num_workers=20
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
   ln -snf ${root_dir}/dataset/sr/ ./dataset/
fi
