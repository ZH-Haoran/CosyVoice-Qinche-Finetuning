#!/usr/bin/env python3
"""
CosyVoice2 秦彻音色推理测试脚本
使用微调后的模型进行音色复刻
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import argparse
import onnxruntime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'third_party/Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# WandB support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SpeakerSimilarityCalculator:
    """计算说话人相似度"""

    def __init__(self, campplus_model_path: str):
        """初始化 campplus ONNX 模型"""
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            campplus_model_path,
            sess_options=option,
            providers=["CPUExecutionProvider"]
        )

    def extract_embedding(self, wav_path: str) -> torch.Tensor:
        """提取说话人嵌入"""
        # 加载音频并重采样到 16kHz
        speech = load_wav(wav_path, 16000)
        # 提取 fbank 特征
        feat = kaldi.fbank(
            speech,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000
        )
        # 均值归一化
        feat = feat - feat.mean(dim=0, keepdim=True)
        # ONNX 推理
        embedding = self.session.run(
            None,
            {self.session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
        )[0].flatten()
        return torch.tensor(embedding)

    def compute_similarity(self, wav1_path: str, wav2_path: str) -> float:
        """计算两个音频的说话人相似度（余弦相似度）"""
        emb1 = self.extract_embedding(wav1_path)
        emb2 = self.extract_embedding(wav2_path)
        # 余弦相似度
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return similarity


def main():
    parser = argparse.ArgumentParser(description='秦彻音色复刻推理测试')
    parser.add_argument('--model_dir', type=str,
                        default='../../../pretrained_models/CosyVoice2-0.5B',
                        help='预训练模型目录')
    parser.add_argument('--flow_checkpoint', type=str,
                        default=None,
                        help='微调后的Flow模型路径（可选）')
    parser.add_argument('--llm_checkpoint', type=str,
                        default=None,
                        help='微调后的LLM模型路径（可选）')
    parser.add_argument('--prompt_audio', type=str,
                        default='../../../data/qinche/inference_ref/prompt_1.wav',
                        help='提示音频（用于提取音色）')
    parser.add_argument('--prompt_text', type=str,
                        default=None,
                        help='提示音频对应文本（如不提供，将使用默认值）')
    parser.add_argument('--output_dir', type=str,
                        default='outputs',
                        help='输出目录')
    parser.add_argument('--text', type=str,
                        default=None,
                        help='要合成的文本（单条）')
    parser.add_argument('--text_file', type=str,
                        default='inference_texts.txt',
                        help='要合成的文本文件（多条，每行一条，#开头为注释）')
    parser.add_argument('--stream', action='store_true',
                        help='是否使用流式推理')
    parser.add_argument('--use_wandb', action='store_true',
                        help='是否上传音频到 WandB')
    parser.add_argument('--wandb_project', type=str,
                        default='cosyvoice-qinche',
                        help='WandB 项目名称')
    parser.add_argument('--wandb_run_name', type=str,
                        default=None,
                        help='WandB 运行名称')
    parser.add_argument('--wandb_run_id', type=str,
                        default=None,
                        help='WandB run ID（用于 resume 已有的 run）')
    parser.add_argument('--no_wandb_init', action='store_true',
                        help='不初始化 WandB，使用外部传入的 run')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 准备测试文本
    test_texts = []
    if args.text:
        test_texts = [args.text]
    elif args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, 'r', encoding='utf-8') as f:
            test_texts = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    else:
        # 默认测试文本
        test_texts = [
            "你好，我是秦彻。很高兴认识你。",
            "今天天气真不错，适合出去走走。",
            "在这个充满挑战的时代，我们需要更多的勇气和智慧。",
            "时光荏苒，岁月如梭，转眼间我们已经走过了那么长的路。",
            "每一个梦想都值得被认真对待，每一份努力都不会被辜负。",
        ]

    # WandB run（外部传入或本地初始化）
    wandb_run = None
    if args.use_wandb and WANDB_AVAILABLE:
        if args.no_wandb_init:
            # 不初始化，尝试使用已有的 run
            if wandb.run is not None:
                wandb_run = wandb.run
                print(f"\n[WandB] 使用外部 run: {wandb_run.name}")
            else:
                print(f"\n[警告] --no_wandb_init 但没有外部 WandB run")
        else:
            # 正常初始化
            ckpt_name = "pretrained"
            if args.flow_checkpoint:
                ckpt_name = os.path.basename(args.flow_checkpoint)
            elif args.llm_checkpoint:
                ckpt_name = os.path.basename(args.llm_checkpoint)
            run_name = args.wandb_run_name or f"inference-{ckpt_name}"

            # 如果提供了 run_id，使用 resume 模式
            if args.wandb_run_id:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    id=args.wandb_run_id,
                    resume="allow",
                    config={
                        "model_dir": args.model_dir,
                        "flow_checkpoint": args.flow_checkpoint,
                        "llm_checkpoint": args.llm_checkpoint,
                        "prompt_audio": args.prompt_audio,
                        "num_test_texts": len(test_texts),
                        "stream": args.stream,
                    }
                )
                print(f"\n[WandB] 已恢复 run: {wandb.run.url}")
            else:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config={
                        "model_dir": args.model_dir,
                        "flow_checkpoint": args.flow_checkpoint,
                        "llm_checkpoint": args.llm_checkpoint,
                        "prompt_audio": args.prompt_audio,
                        "num_test_texts": len(test_texts),
                        "stream": args.stream,
                    }
                )
                print(f"\n[WandB] 已初始化: {wandb.run.url}")

    # 加载模型
    print(f"\n[模型] 加载 CosyVoice2 from {args.model_dir}")
    cosyvoice = CosyVoice2(args.model_dir)

    # 加载微调后的 checkpoint
    if args.flow_checkpoint and os.path.exists(args.flow_checkpoint):
        print(f"[模型] 加载 Flow checkpoint: {args.flow_checkpoint}")
        state_dict = torch.load(args.flow_checkpoint, map_location=cosyvoice.model.device, weights_only=True)
        # 过滤掉 epoch 和 step 键，只保留模型权重
        state_dict = {k: v for k, v in state_dict.items() if k not in ['epoch', 'step']}
        cosyvoice.model.flow.load_state_dict(state_dict, strict=True)

    if args.llm_checkpoint and os.path.exists(args.llm_checkpoint):
        print(f"[模型] 加载 LLM checkpoint: {args.llm_checkpoint}")
        state_dict = torch.load(args.llm_checkpoint, map_location=cosyvoice.model.device, weights_only=True)
        # 过滤掉 epoch 和 step 键，只保留模型权重
        state_dict = {k: v for k, v in state_dict.items() if k not in ['epoch', 'step']}
        cosyvoice.model.llm.load_state_dict(state_dict, strict=True)

    # 初始化 Speaker Similarity 计算器
    campplus_path = os.path.join(args.model_dir, 'campplus.onnx')
    spk_sim_calculator = SpeakerSimilarityCalculator(campplus_path)
    print(f"[Speaker Similarity] 已加载 campplus 模型: {campplus_path}")

    # 读取 prompt_text
    prompt_text = args.prompt_text
    if prompt_text is None:
        prompt_text = "希望你以后能够做的比我还好呦。"
        print(f"[提示] 使用默认 prompt_text: {prompt_text}")

    print(f"[提示] Prompt audio: {args.prompt_audio}")
    print(f"[提示] Prompt text: {prompt_text}")
    print(f"[输出] 输出目录: {args.output_dir}")
    print(f"[文本] 共 {len(test_texts)} 条测试文本\n")

    # 推理
    similarities = []
    for idx, text in enumerate(test_texts):
        print(f"[{idx+1}/{len(test_texts)}] 合成: {text[:50]}...")

        output_wav = os.path.join(args.output_dir, f"output_{idx:03d}.wav")

        try:
            # 使用 zero_shot 推理
            for result in cosyvoice.inference_zero_shot(
                tts_text=text,
                prompt_text=prompt_text,
                prompt_wav=args.prompt_audio,
                stream=args.stream
            ):
                tts_speech = result['tts_speech']
                torchaudio.save(output_wav, tts_speech, cosyvoice.sample_rate)

                # 计算音频时长
                duration = tts_speech.shape[1] / cosyvoice.sample_rate

                # 计算 Speaker Similarity
                spk_sim = spk_sim_calculator.compute_similarity(args.prompt_audio, output_wav)
                similarities.append(spk_sim)

                print(f"    -> 保存: {output_wav} (时长: {duration:.2f}s, SPK_SIM: {spk_sim:.4f})")

                # 上传到 WandB
                if wandb_run is not None:
                    wandb_run.log({
                        f"audio_{idx}": wandb.Audio(output_wav, sample_rate=cosyvoice.sample_rate, caption=text[:50]),
                        f"text_{idx}": text,
                        f"duration_{idx}": duration,
                        f"speaker_similarity_{idx}": spk_sim,
                    })

        except Exception as e:
            print(f"    [错误] 合成失败: {e}")
            continue

    # 计算并打印平均 Speaker Similarity
    if similarities:
        avg_spk_sim = sum(similarities) / len(similarities)
        print(f"\n[Speaker Similarity] 平均值: {avg_spk_sim:.4f}")

        # 上传平均相似度到 WandB
        if wandb_run is not None:
            wandb_run.log({
                "avg_speaker_similarity": avg_spk_sim,
            })

    print(f"\n[完成] 所有音频已保存到: {args.output_dir}")

    # WandB finish
    if wandb_run is not None and not args.no_wandb_init:
        wandb.finish()


if __name__ == '__main__':
    main()
