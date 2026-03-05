#!/bin/bash
# CosyVoice2 秦彻音色微调训练脚本 - LLM模块
# 目标: 全量微调 LLM 模块
# 用法: bash run_llm.sh

set -e  # 遇到错误立即退出
# 训练后推理不中断脚本
set +e  # 推理阶段允许失败

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
# WandB 配置 (可选 - 需要先 wandb login)
# ========================================
export WANDB_PROJECT="cosyvoice-qinche-formal"
# 生成唯一的时间戳，确保 WANDB_RUN_NAME 和 WANDB_RUN_ID 使用相同的值
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
export WANDB_RUN_NAME="llm-finetune-lr${LR}-batch${BATCH_SIZE}-${TIMESTAMP}"
export WANDB_RUN_ID="llm-ft-${TIMESTAMP}"
export WANDB_API_KEY="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"

# ========================================
# 模型配置
# ========================================
MODEL=llm
CHECKPOINT=../../../pretrained_models/CosyVoice2-0.5B/llm.pt
CONFIG=conf/cosyvoice2_llm.yaml
QWEN_PRETRAIN=../../../pretrained_models/CosyVoice2-0.5B/CosyVoice-BlankEN
ONNX_PATH=../../../pretrained_models/CosyVoice2-0.5B

# 数据路径
TRAIN_DATA=train.data.list
DEV_DATA=dev.data.list

# ========================================
# 从 yaml 文件读取超参数
# ========================================
# 如果这里设置了值，会覆盖 yaml 中的；留空则使用 yaml 中的值
LR=""  # 留空则从 yaml 读取
BATCH_SIZE=""  # 留空则从 yaml 读取

# 读取 yaml 文件中的学习率
if [ -z "$LR" ]; then
    LR=$(grep -A 3 "optim_conf:" $CONFIG | grep "lr:" | awk '{print $2}' | tr -d ' \r')
fi

# 读取 yaml 文件中的 max_epoch
MAX_EPOCH=$(grep "max_epoch:" $CONFIG | awk '{print $2}' | tr -d ' \r')

# 读取 yaml 文件中的 max_frames_in_batch 作为 batch_size
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=$(grep "max_frames_in_batch:" $CONFIG | awk '{print $2}' | tr -d ' \r')
fi

# 输出路径（根据 lr 和 batchsize 区分）
MODEL_DIR=exp/cosyvoice2/llm_lr${LR}_batch${BATCH_SIZE}
TENSORBOARD_DIR=tensorboard/cosyvoice2/llm_lr${LR}_batch${BATCH_SIZE}

mkdir -p $MODEL_DIR
mkdir -p $TENSORBOARD_DIR

echo "========================================"
echo "CosyVoice2 LLM 模块微调 - 秦彻音色复刻"
echo "========================================"
echo "模型: $MODEL"
echo "预训练权重: $CHECKPOINT"
echo "训练数据: $TRAIN_DATA"
echo "验证数据: $DEV_DATA"
echo "模型输出: $MODEL_DIR"
echo "TensorBoard: $TENSORBOARD_DIR"
echo "WandB Project: $WANDB_PROJECT"
echo "WandB Run: $WANDB_RUN_NAME"
echo "========================================"
echo "训练策略: 全量微调 LLM 模块"
echo "学习率: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "最大epoch: $MAX_EPOCH"
echo "========================================"

# 启动训练
torchrun --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=1002 --rdzv_backend="c10d" --rdzv_endpoint="localhost:1235" \
    ../../../cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --config $CONFIG \
    --train_data $TRAIN_DATA \
    --cv_data $DEV_DATA \
    --qwen_pretrain_path $QWEN_PRETRAIN \
    --onnx_path $ONNX_PATH \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --model_dir $MODEL_DIR \
    --tensorboard_dir $TENSORBOARD_DIR \
    --ddp.dist_backend nccl \
    --num_workers 2 \
    --prefetch 100 \
    --pin_memory \
    --use_amp

# ========================================
# 训练后推理
# ========================================
set -e  # 推理时遇到错误退出

echo ""
echo "========================================"
echo "训练完成，开始推理"
echo "========================================"

# 统计 checkpoint 数量
CKPT_COUNT=$(ls -1 $MODEL_DIR/epoch_*_whole.pt 2>/dev/null | wc -l)
echo "找到 $CKPT_COUNT 个 checkpoint"

if [ "$CKPT_COUNT" -eq 0 ]; then
    echo "错误: 没有找到任何 checkpoint 文件"
    exit 1
fi

# 推理函数
run_inference() {
    local ckpt=$1
    local epoch=$2
    local tag=$3  # 标识：best 或 last
    local output_dir="${MODEL_DIR}/inference_epoch_${epoch}_${tag}"

    echo ""
    echo "========================================"
    echo "推理 Epoch: $epoch ($tag)"
    echo "Checkpoint: $ckpt"
    echo "输出目录: $output_dir"
    echo "========================================"

    # 只使用 prompt_2 进行推理，使用同一个 wandb run id
    # 注意: prompt_text 必须与 prompt_audio 中说的内容完全一致！
    python inference.py \
        --model_dir="../../../pretrained_models/CosyVoice2-0.5B" \
        --llm_checkpoint="$ckpt" \
        --prompt_audio="../../../data/qinche/inference_ref/prompt_2.wav" \
        --prompt_text="还记得我把你捡回暗点的那个地方吗，那里有一台管风琴。" \
        --text_file="inference_texts.txt" \
        --output_dir="$output_dir" \
        --audio_prefix="$tag" \
        --use_wandb \
        --wandb_project="$WANDB_PROJECT" \
        --wandb_run_id="$WANDB_RUN_ID" \
        --wandb_run_name="${WANDB_RUN_NAME}_epoch${epoch}_${tag}_prompt2"
}

# ========================================
# 找到 CV loss 最低的 checkpoint
# ========================================
echo ""
echo "查找 CV loss 最低的 checkpoint..."

BEST_CKPT=""
BEST_EPOCH=""
BEST_LOSS=999999

for yaml_file in $(ls -v $MODEL_DIR/epoch_*_whole.yaml 2>/dev/null); do
    # 从 yaml 文件读取 CV loss
    # loss_dict 格式为:
    # loss_dict:
    #   acc: 0.123
    #   loss: 3.456
    cv_loss=$(grep -A 5 "loss_dict:" "$yaml_file" | grep "  loss:" | awk '{print $2}')
    epoch=$(basename "$yaml_file" | grep -oP 'epoch_\K\d+')

    if [ -n "$cv_loss" ]; then
        echo "  Epoch $epoch: CV loss = $cv_loss"
        if (( $(awk "BEGIN {print ($cv_loss < $BEST_LOSS)}") )); then
            BEST_LOSS=$cv_loss
            BEST_EPOCH=$epoch
            BEST_CKPT="${MODEL_DIR}/epoch_${epoch}_whole.pt"
        fi
    fi
done

if [ -z "$BEST_CKPT" ]; then
    echo "警告: 无法找到 CV loss 信息，将使用最后一个 checkpoint"
    BEST_CKPT=$(ls -v $MODEL_DIR/epoch_*_whole.pt | tail -1)
    BEST_EPOCH=$(basename $BEST_CKPT | grep -oP 'epoch_\K\d+')
fi

echo ""
echo "最低 CV loss checkpoint: Epoch $BEST_EPOCH (loss: $BEST_LOSS)"

# ========================================
# 找到最后一个 checkpoint
# ========================================
LAST_CKPT=$(ls -v $MODEL_DIR/epoch_*_whole.pt | tail -1)
LAST_EPOCH=$(basename $LAST_CKPT | grep -oP 'epoch_\K\d+')

echo "最后一个 checkpoint: Epoch $LAST_EPOCH"

# ========================================
# 运行推理（只对 best 和 last 两个 checkpoint）
# ========================================
echo ""
echo "========================================"
echo "开始推理（仅 CV loss 最低 + 最后一个 checkpoint）"
echo "========================================"

# 推理 best checkpoint
if [ -n "$BEST_CKPT" ] && [ -f "$BEST_CKPT" ]; then
    run_inference "$BEST_CKPT" "$BEST_EPOCH" "best"
fi

# 推理 last checkpoint（如果与 best 不同）
if [ "$BEST_EPOCH" != "$LAST_EPOCH" ]; then
    run_inference "$LAST_CKPT" "$LAST_EPOCH" "last"
else
    echo ""
    echo "Best 和 Last 是同一个 checkpoint (Epoch $BEST_EPOCH)，跳过重复推理"
fi

echo ""
echo "========================================"
echo "推理完成"
echo "  - Best checkpoint: Epoch $BEST_EPOCH (CV loss: $BEST_LOSS)"
echo "  - Last checkpoint: Epoch $LAST_EPOCH"
echo "  - 仅使用 prompt_2"
echo "  - 音频已上传到 WandB"
echo "========================================"
