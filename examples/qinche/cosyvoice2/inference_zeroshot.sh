#!/bin/bash
# Zero-shot 推理脚本 - 使用预训练模型（无微调）

set -e

cd /data_hss/zhanghaoran/CosyVoice/examples/qinche/cosyvoice2

echo "Task Starts"

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
# WandB 配置 (可选)
# ========================================
export WANDB_PROJECT="cosyvoice-qinche-formal"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
export WANDB_RUN_NAME="zeroshot-${TIMESTAMP}"
export WANDB_RUN_ID="zeroshot-${TIMESTAMP}"
export WANDB_API_KEY="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"

# ========================================
# 推理配置
# ========================================
MODEL_DIR=../../../pretrained_models/CosyVoice2-0.5B
OUTPUT_DIR=outputs_zeroshot
PROMPT_AUDIO=../../../data/qinche/inference_ref/prompt_2.wav
PROMPT_TEXT="还记得我把你捡回暗点的那个地方吗，那里有一台管风琴。"
TEXT_FILE=inference_texts.txt

mkdir -p $OUTPUT_DIR

echo "========================================"
echo "Zero-shot 推理 - 使用预训练模型"
echo "========================================"
echo "模型目录: $MODEL_DIR"
echo "提示音频: $PROMPT_AUDIO"
echo "提示文本: $PROMPT_TEXT"
echo "输入文本: $TEXT_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

# 运行推理（不指定 checkpoint，使用预训练模型）
python inference.py \
    --model_dir="$MODEL_DIR" \
    --prompt_audio="$PROMPT_AUDIO" \
    --prompt_text="$PROMPT_TEXT" \
    --text_file="$TEXT_FILE" \
    --output_dir="$OUTPUT_DIR" \
    --use_wandb \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_id="$WANDB_RUN_ID" \
    --wandb_run_name="$WANDB_RUN_NAME"

echo ""
echo "========================================"
echo "推理完成"
echo "输出目录: $OUTPUT_DIR"
echo "音频已上传到 WandB: $WANDB_RUN_NAME"
echo "========================================"
