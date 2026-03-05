#!/bin/bash
# CosyVoice2 秦彻音色推理脚本
# 用法: bash inference.sh
#
# 功能:
#   - 加载预训练模型和微调后的checkpoint
#   - 使用提示音频进行音色复刻
#   - 支持单条文本或多条文本文件推理
#   - 支持流式推理
#   - 支持上传结果到 WandB

set -e  # 遇到错误立即退出

cd "$(dirname "$0")"

# ========================================
# 环境设置
# ========================================
export PYTHONPATH=$(cd ../../../ && pwd):$(cd ../../../third_party/Matcha-TTS && pwd):$PYTHONPATH
export CUDA_HOME=${CONDA_PREFIX:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 设置GPU
export CUDA_VISIBLE_DEVICES="0"

# ========================================
# 模型配置
# ========================================
# 预训练模型目录
MODEL_DIR=../../../pretrained_models/CosyVoice2-0.5B

# 微调后的模型路径 (留空则不加载，使用预训练模型)
FLOW_CHECKPOINT=exp/cosyvoice2/flow/epoch_17_whole.pt
# FLOW_CHECKPOINT=""  # 不加载Flow微调权重

LLM_CHECKPOINT=""  # LLM微调权重（可选）
# LLM_CHECKPOINT="exp/cosyvoice2/llm/epoch_10_whole.pt"

# ========================================
# 推理配置
# ========================================
# 提示音频（用于提取音色）
PROMPT_AUDIO=data/qinche/wavs/qinche_0001.wav

# 提示音频对应文本（留空则自动查找）
PROMPT_TEXT=""

# 输出目录
OUTPUT_DIR=outputs_flow

# GPU ID
GPU_ID=0

# ========================================
# 文本配置
# ========================================
# 方式1: 单条文本推理（优先）
TEXT=""

# 方式2: 从文本文件读取多条
TEXT_FILE=inference_texts.txt

# ========================================
# 其他选项
# ========================================
# 是否使用流式推理 (true/false)
STREAM=false

# 是否上传到 WandB (true/false)
USE_WANDB=false
WANDB_PROJECT="cosyvoice-qinche"
WANDB_RUN_NAME=""

# ========================================
# 显示配置信息
# ========================================
echo "========================================"
echo "CosyVoice2 秦彻音色推理"
echo "========================================"
echo "预训练模型: $MODEL_DIR"
echo "Flow checkpoint: ${FLOW_CHECKPOINT:-未设置（使用预训练）}"
echo "LLM checkpoint: ${LLM_CHECKPOINT:-未设置（使用预训练）}"
echo "提示音频: $PROMPT_AUDIO"
echo "提示文本: ${PROMPT_TEXT:-自动查找}"
echo "输出目录: $OUTPUT_DIR"
echo "GPU ID: $GPU_ID"
echo "文本: ${TEXT:-使用文本文件 $TEXT_FILE}"
echo "流式推理: $STREAM"
echo "WandB: ${USE_WANDB:-禁用}"
echo "========================================"
echo ""

# ========================================
# 执行推理
# ========================================
ARGS=()

# 基础参数
ARGS+=("--model_dir" "$MODEL_DIR")
ARGS+=("--prompt_audio" "$PROMPT_AUDIO")
ARGS+=("--output_dir" "$OUTPUT_DIR")
ARGS+=("--text_file" "$TEXT_FILE")

# Flow checkpoint
if [ -n "$FLOW_CHECKPOINT" ] && [ -f "$FLOW_CHECKPOINT" ]; then
    ARGS+=("--flow_checkpoint" "$FLOW_CHECKPOINT")
elif [ -n "$FLOW_CHECKPOINT" ]; then
    echo "[警告] Flow checkpoint 文件不存在: $FLOW_CHECKPOINT"
    echo "将使用预训练模型进行推理"
fi

# LLM checkpoint
if [ -n "$LLM_CHECKPOINT" ] && [ -f "$LLM_CHECKPOINT" ]; then
    ARGS+=("--llm_checkpoint" "$LLM_CHECKPOINT")
elif [ -n "$LLM_CHECKPOINT" ]; then
    echo "[警告] LLM checkpoint 文件不存在: $LLM_CHECKPOINT"
fi

# Prompt text
if [ -n "$PROMPT_TEXT" ]; then
    ARGS+=("--prompt_text" "$PROMPT_TEXT")
fi

# 单条文本（优先）
if [ -n "$TEXT" ]; then
    ARGS+=("--text" "$TEXT")
fi

# 流式推理
if [ "$STREAM" = "true" ]; then
    ARGS+=("--stream")
fi

# WandB
if [ "$USE_WANDB" = "true" ]; then
    ARGS+=("--use_wandb")
    ARGS+=("--wandb_project" "$WANDB_PROJECT")
    if [ -n "$WANDB_RUN_NAME" ]; then
        ARGS+=("--wandb_run_name" "$WANDB_RUN_NAME")
    fi
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 执行推理
python inference.py "${ARGS[@]}"
