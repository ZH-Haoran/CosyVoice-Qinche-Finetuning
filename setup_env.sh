#!/bin/bash
# CosyVoice 环境配置脚本
# 使用方法: source setup_env.sh

# 1. 激活 conda 环境
source /data_hss/miniconda3/etc/profile.d/conda.sh
conda activate cosyvoice

# 2. 安装 cuda-toolkit (包含 nvcc，用于 deepspeed/finetuning)
# 检查是否已安装
if ! command -v nvcc &> /dev/null; then
    echo "正在安装 cuda-toolkit-12.1..."
    conda install -y cuda-toolkit -c nvidia/label/cuda-12.1.0
fi

# 3. 设置 CUDA 环境变量 (使用 conda 安装的 cuda-toolkit)
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "============================================"
echo "环境已配置完成!"
echo "Python: $(python --version 2>&1)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA: $(nvcc --version 2>&1 | tail -1)"
echo "============================================"
