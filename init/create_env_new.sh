#!/usr/bin/bash

# ===== 0. 让脚本里能用 conda（很重要）=====
# 按你的实际安装路径改，比如 ~/miniconda3 或 ~/anaconda3
source /home/wy/anaconda3/etc/profile.d/conda.sh

# ===== 1. 安装 rust 编译器（保持不变，只是提醒会有点慢）=====
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

# ===== 2. 创建并激活 conda 环境 =====
conda create -n lidar_diffusion python=3.10.11 -y
conda activate lidar_diffusion

# ===== 3. 只在这个脚本里使用清华 pip 源 =====
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# ===== 4. 安装依赖（已切到清华）=====
pip install --upgrade pip

pip install \
    torchmetrics==0.5.0 \
    pytorch-lightning==1.4.2 \
    omegaconf==2.1.1 \
    einops==0.3.0 \
    transformers==4.36.2 \
    imageio==2.9.0 \
    imageio-ffmpeg==0.4.2 \
    opencv-python \
    kornia==0.7.0 \
    wandb \
    more_itertools

pip install gdown

# ===== 5. GitHub 相关：改成先 clone，再本地安装（后面重跑会快很多）=====

mkdir -p deps
cd deps

if [ ! -d taming-transformers ]; then
    git clone https://github.com/CompVis/taming-transformers.git
fi
pip install -e taming-transformers

if [ ! -d CLIP ]; then
    git clone https://github.com/openai/CLIP.git
fi
pip install -e CLIP

cd ..

# ===== 6. 可选：torchsparse（一样会走 GitHub，按需打开）=====
# apt-get install -y libsparsehash-dev
# pip install git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

mkdir -p dataset/
