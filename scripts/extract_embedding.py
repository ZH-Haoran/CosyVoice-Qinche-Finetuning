#!/usr/bin/env python3
"""
提取 Speaker Embedding
修正版: 使用正确的路径
"""
import os
import sys
import json
import torch
import torchaudio
from tqdm import tqdm

# 设置工作目录
os.chdir('/data_hss/zhanghaoran/CosyVoice')


def load_segments(segments_file):
    """加载ASR分段结果"""
    with open(segments_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    print(f"加载 {len(segments)} 个分段")
    return segments


def extract_and_save_embeddings(segments, output_dir, onnx_path):
    """提取并保存embedding"""
    os.makedirs(output_dir, exist_ok=True)

    # 加载ONNX模型
    try:
        import onnxruntime
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess = onnxruntime.InferenceSession(onnx_path, sess_options)
        print("ONNX模型加载成功")
    except Exception as e:
        print(f"ONNX加载失败: {e}")
        raise

    utt2embedding = {}
    spk2embedding = {}

    for seg in tqdm(segments, desc="提取embedding"):
        # 获取音频信息
        source_file = seg['source_file']
        start_ms = seg['start']
        end_ms = seg['end']
        text = seg['text']

        # 构建音频路径
        audio_path = f"data/qinche/raw_audios/{source_file}"
        if not os.path.exists(audio_path):
            print(f"音频文件不存在: {audio_path}")
            continue

        # 加载音频
        waveform, sr = torchaudio.load(audio_path)

        # 提取片段
        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        audio_segment = waveform[:, start_sample:end_sample]

        # 保存临时文件
        temp_wav = "/tmp/temp_audio.wav"
        torchaudio.save(temp_wav, audio_segment, sr)

        # 提取embedding
        inputs = {sess.get_inputs()[0].name: temp_wav}
        outputs = sess.run(None, inputs)
        embedding = outputs[0]

        # 记录
        utt_id = f"qinche_{seg['idx']:04d}"
        utt2embedding[utt_id] = embedding

        # 说话人embedding (平均)
        spk_id = "qinche"
        if spk_id not in spk2embedding:
            spk2embedding[spk_id] = []
        spk2embedding[spk_id].append(embedding)

    # 保存
    torch.save(utt2embedding, os.path.join(output_dir, 'utt2embedding.pt'))

    # 计算说话人平均embedding
    for spk_id, embs in spk2embedding.items():
        avg_embedding = torch.mean(torch.stack(embs), dim=0)
        spk2embedding[spk_id] = avg_embedding
    torch.save(spk2embedding, os.path.join(output_dir, 'spk2embedding.pt'))

    print(f"保存embedding到 {output_dir}/")
    print(f"utt2embedding: {len(utt2embedding)} 条")
    print(f"spk2embedding: {len(spk2embedding)} 个说话人")


def main():
    # 设置CUDA环境
    conda_prefix = sys.prefix
    if conda_prefix:
        os.environ['CUDA_HOME'] = conda_prefix
        os.environ['PATH'] = os.path.join(conda_prefix, 'bin') + ':' + os.environ.get('PATH', '')
        os.environ['LD_LIBRARY_PATH'] = os.path.join(conda_prefix, 'lib64') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

    segments_file = "data/qinche/processed/all_segments.json"
    output_dir = "data/qinche/processed"
    onnx_path = "pretrained_models/CosyVoice2-0.5B/campplus.onnx"

    segments = load_segments(segments_file)
    extract_and_save_embeddings(segments, output_dir, onnx_path)


if __name__ == "__main__":
    main()
