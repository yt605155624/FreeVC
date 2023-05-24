#!/bin/bash
stage=5
stop_stage=5
root_dir=$1
# 增加新的数据集的时候，把 vctk 生成的 dataset mv 为 dataset_0、filelists_0
# 新数据集 mv 为 dataset_1、filelists_1 
# 生成 filelists_1 的时候需要注意下 DUMMY 路径，可以先把原始的 DUMMY rename 一下
# 运行 stage1 把 新数据集的 16k/wav 也软链接到 DUMMY
# concat filelists_0 和 filelists_1 到 filelists 或者重新执行 stage2 生成新的 DUMMY
## 后者 vctk 原始的训练训练集会变
# 新数据集也叫 vctk-16 只是用 dataset_num 区分
dataset_num='1'


# for VCTK only
# 同时下采样到 16k 和 22.05k, 22.05k 用于 preprocess_sr, 因为 HiFiGAN 是 22.05k 的 
# 但是 preprocess_sr 里面 load 的时候用了 hps.sampling_rate，所以这个操作是否一定需要？
# !!!! vctk-16k 在 nfs 上速度会比较慢
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python3 downsample_addspk.py \
        --in_dir=~/datasets/ \
        --spk_names 'deer_ai_newer' 'snowball_newer' \
        --sr1=16000 \
        --out_dir1=dataset_${dataset_num}/vctk-16k \
        --sr2=22050 \
        --out_dir2=${root_dir}/dataset_${dataset_num}/vctk-22k \
        --num-cpu=20
fi

# !!!! vctk-16k 在 nfs 上速度会比较慢
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p DUMMY
    ln -snf $(pwd)/dataset_${dataset_num}/vctk-16k/* DUMMY
fi

# shuffle *.txt in ./filelists, ./filelists 中仅包含 VCTK, 如果加了新数据一定要跑这个 stage
# 我改了数据预处理, 多了 s5, 所以也一定要跑这个
# 记得后续和预训练数据集合并 filelists 或者 mv 为 filelists
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python3 preprocess_flist.py \
        --source_dir=DUMMY \
        --list_dir=./filelists_${dataset_num} \
        --valnum_per_spk=10 \
        --testnum_per_spk=10
fi

# preprocess_spk.py -> FreeVC, with pretrained speaker encoder

# preprocess_ssl.py -> FreeVC (w/o SR), uses pretrained speaker encoder, without SR-based data augmentation
# preprocess_sr.py -> with SR-based augmentation
# preprocess_sr.py 中包含提取 SSL 的过程
# preprocess_spk.py + preprocess_sr.py 就是默认的 FreeVC

# 结束之后有个段错误，之后看看
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    python3 preprocess_spk.py \
        --in_dir=dataset_${dataset_num}/vctk-16k \
        --out_dir_root=dataset_${dataset_num} \
        --num_workers=12
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    CUDA_VISIBLE_DEVICES=0 python3 preprocess_sr.py \
        --sr=16000 \
        --config=hifigan/config.json \
        --in_dir=${root_dir}/dataset_${dataset_num}/vctk-22k \
        --wav_dir=${root_dir}/dataset_${dataset_num}/sr/wav \
        --ssl_dir=${root_dir}/dataset_${dataset_num}/sr/wavlm \
        --min=68 \
        --max=72 \
        --num_workers=20 & CUDA_VISIBLE_DEVICES=0 python3 preprocess_sr.py \
            --sr=16000 \
            --config=hifigan/config.json \
            --in_dir=${root_dir}/dataset_${dataset_num}/vctk-22k \
            --wav_dir=${root_dir}/dataset_${dataset_num}/sr/wav \
            --ssl_dir=${root_dir}/dataset_${dataset_num}/sr/wavlm \
            --min=73 \
            --max=76 \
            --num_workers=20 & CUDA_VISIBLE_DEVICES=1 python3 preprocess_sr.py \
                --sr=16000 \
                --config=hifigan/config.json \
                --in_dir=${root_dir}/dataset_${dataset_num}/vctk-22k \
                --wav_dir=${root_dir}/dataset_${dataset_num}/sr/wav \
                --ssl_dir=${root_dir}/dataset_${dataset_num}/sr/wavlm \
                --min=77 \
                --max=80 \
                --num_workers=20 & CUDA_VISIBLE_DEVICES=1 python3 preprocess_sr.py \
                    --sr=16000 \
                    --config=hifigan/config.json \
                    --in_dir=${root_dir}/dataset_${dataset_num}/vctk-22k \
                    --wav_dir=${root_dir}/dataset_${dataset_num}/sr/wav \
                    --ssl_dir=${root_dir}/dataset_${dataset_num}/sr/wavlm \
                    --min=81 \
                    --max=84 \
                    --num_workers=20 & CUDA_VISIBLE_DEVICES=2 python3 preprocess_sr.py \
                        --sr=16000 \
                        --config=hifigan/config.json \
                        --in_dir=${root_dir}/dataset_${dataset_num}/vctk-22k \
                        --wav_dir=${root_dir}/dataset_${dataset_num}/sr/wav \
                        --ssl_dir=${root_dir}/dataset_${dataset_num}/sr/wavlm \
                        --min=85 \
                        --max=88 \
                        --num_workers=20 & CUDA_VISIBLE_DEVICES=2 python3 preprocess_sr.py \
                            --sr=16000 \
                            --config=hifigan/config.json \
                            --in_dir=${root_dir}/dataset_${dataset_num}/vctk-22k \
                            --wav_dir=${root_dir}/dataset_${dataset_num}/sr/wav \
                            --ssl_dir=${root_dir}/dataset_${dataset_num}/sr/wavlm \
                            --min=89 \
                            --max=92 \
                            --num_workers=20
fi

# ${root_dir}/dataset/ 下的所有东西全部软链接到当前路径
# data_utils.py 里面是替换 lifelists 里 DUMMY 为 dataset/*
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # 以 speaker 目录为单位操作
    # 真正训练时候只需要这两个
    # !!!! spk_emb 在 nfs 上速度会比较慢
    mkdir -p dataset/spk_emb
    ln -snf $(pwd)/dataset_${dataset_num}/spk_emb/* dataset/spk_emb
    mkdir -p dataset/sr/wavlm
    ln -snf ${root_dir}/dataset_${dataset_num}/sr/wavlm/* dataset/sr/wavlm
    # ln -snf $(pwd)/filelists_1 $(pwd)/filelists
    # mkdir -p dataset/sr/wav
    # ln -snf ${root_dir}/dataset_${dataset_num}/sr/wav/* dataset/sr/wav
    # mkdir -p dataset/vctk-16k
    # ln -snf ${root_dir}/dataset_${dataset_num}/vctk-16k/* dataset/vctk-16k
    # mkdir -p dataset/vctk-22k
    # ln -snf ${root_dir}/dataset_${dataset_num}/vctk-22k/* dataset/vctk-22k

fi
