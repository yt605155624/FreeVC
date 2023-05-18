## 复现 VCTK 数据集上的实验
实验路径 `/home/yuantian04/Voice_Conversion/FreeVC/FreeVC_base/FreeVC`
1. 按照 README.md 下载 WavLM-Large 和 HiFiGAN VCTK_V1 预训练模型并放到指定位置 (可以整理到一个 pretrained_models 里再分别软链接到指定位置), see [Which WavLM and which hifi-gan should doenload?](https://github.com/OlaWod/FreeVC/issues/19)
```bash
tree pretrained_models/
```
```text
pretrained_models/
├── checkpoints
│   ├── 24kHz
│   │   ├── D-freevc-24.pth
│   │   └── freevc-24.pth
│   ├── D-freevc.pth
│   ├── D-freevc-s.pth
│   ├── freevc.pth
│   └── freevc-s.pth
├── VCTK_V1
│   ├── config.json
│   └── generator_v1
├── VCTK_V1-20230517T031805Z-001.zip
└── WavLM-Large.pt
```

```bash
unzip pretrained_models.zip
cd pretrained_models
unzip VCTK_V1-20230517T031805Z-001.zip
cd ../
# 软链接 WavLM
cd wavlm
ln -snf ../pretrained_models/WavLM-Large.pt
cd ../
# 软链接 HiFiGAN
cd hifigan
ln -snf ../pretrained_models/VCTK_V1/config.json .
ln -snf ../pretrained_models/VCTK_V1/generator_v1 .
cd ../
```


2. 下载 VCTK-Corpus-0.92.zip 到 `~/datasets` 并解压缩
```bash
unzip -d VCTK VCTK-Corpus-0.92.zip
```
3. 安装 [pyTorch](https://pytorch.org/get-started/previous-versions/)
依赖要求 torch=1.10.0 但是 torch 1.10.0 不支持 cuda 11.7, 所以选择 1.13.1
```bash
# CUDA 11.7
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# torch 检查安装
torch.__version__
torch.cuda.is_available() # 查看CUDA是否可用
torch.cuda.device_count() # 查看可用的CUDA数量
torch.version.cuda # 查看CUDA的版本号
```
4. 安装依赖
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
5. Inference
- 需要先执行数据预处理获得 DUMMY 文件中的 source audio 和 target audio, 输入音频采样率无要求，代码里会按照模型的采样率下采样
- 是 any-to-any 的，没有 spk_id 输入

5. 数据预处理

6. 训练
