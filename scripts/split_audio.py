#!/usr/bin/env python3
"""
根据ASR分段结果分割音频
增强版：添加信噪比筛选和音量归一化
"""
import os
import json
import torchaudio
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime


def calculate_snr(waveform, sample_rate, noise_frames=10):
    """
    计算信噪比（SNR）
    使用简单的基于能量的SNR估计方法

    参数:
        waveform: 音频波形 (1, T)
        sample_rate: 采样率
        noise_frames: 用于估计噪声的帧数（从开头和结尾各取多少帧）

    返回:
        snr_db: SNR值（分贝）
    """
    waveform_np = waveform.squeeze().numpy()
    num_samples = len(waveform_np)
    noise_samples = int(noise_frames * 0.01 * sample_rate)  # 每帧大约10ms

    if num_samples < noise_samples * 4:
        # 音频太短，无法估计噪声
        return 100.0  # 返回一个高SNR值

    # 从开头和结尾提取噪声估计
    noise_start = waveform_np[:noise_samples]
    noise_end = waveform_np[-noise_samples:]
    noise = np.concatenate([noise_start, noise_end])

    # 估计信号部分（去掉噪声部分）
    signal = waveform_np[noise_samples:-noise_samples]

    # 计算能量
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        # 噪声能量极低
        return 100.0

    # 计算SNR (dB)
    snr_db = 10 * np.log10(signal_power / noise_power)

    return snr_db


def normalize_volume(waveform, target_dBFS=-20.0):
    """
    音量归一化

    参数:
        waveform: 音频波形 (1, T)
        target_dBFS: 目标响度值（默认-20 dBFS）

    返回:
        normalized_waveform: 归一化后的波形
        scale_factor: 缩放因子
    """
    # 计算当前峰值
    peak = torch.max(torch.abs(waveform))

    # 如果峰值太小（静音），不处理
    if peak < 1e-6:
        return waveform, 1.0

    # Peak normalization: 将峰值归一化到目标值
    # target_dBFS 转换为线性值: 10^(dBFS/20)
    target_linear = 10 ** (target_dBFS / 20.0)

    # 计算缩放因子
    scale_factor = target_linear / (peak + 1e-8)

    # 应用归一化
    normalized_waveform = waveform * scale_factor

    # 限制到 [-1, 1] 范围（防止削波）
    normalized_waveform = torch.clamp(normalized_waveform, -1.0, 1.0)

    return normalized_waveform, scale_factor


def main():
    # 配置
    INPUT_DIR = "data/qinche/raw_audios"
    SEGMENTS_FILE = "data/qinche/processed/all_segments.json"
    OUTPUT_DIR = "data/qinche/wavs"
    TRAIN_DIR = "data/qinche/train"
    TEST_DIR = "data/qinche/test"

    # 过滤条件
    MIN_DURATION = 2.0   # 最小时长(秒)
    MAX_DURATION = 15.0  # 最大时长(秒)
    MIN_TEXT_LEN = 3     # 最少字符数
    TEST_RATIO = 0.1     # 测试集比例 (10%)

    # 新增：音频质量过滤条件
    MIN_SNR_DB = -100.0   # 最小信噪比（分贝），设置为 -100 dB 相当于不过滤
    MIN_RMS = 0.001       # 最小RMS值（防止静音音频），降低阈值
    TARGET_DBCFS = -20.0  # 目标响度（dBFS）用于归一化

    print(f"[{datetime.now()}] 开始音频分割...")
    print(f"音频质量过滤: SNR >= {MIN_SNR_DB} dB, RMS >= {MIN_RMS}")
    print(f"音量归一化: 目标 {TARGET_DBCFS} dBFS")

    # 创建目录
    for d in [OUTPUT_DIR, TRAIN_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    spk_id = "qinche"

    # 加载分段信息
    with open(SEGMENTS_FILE, 'r', encoding='utf-8') as f:
        all_segments = json.load(f)

    print(f"共加载 {len(all_segments)} 个分段")

    # 过滤符合条件的分段
    valid_segments = []
    quality_filtered_count = 0
    snr_below_threshold = 0
    rms_below_threshold = 0
    snr_distribution = []

    for seg in all_segments:
        duration = (seg['end'] - seg['start']) / 1000  # ms -> s
        text_len = len(seg.get('text', ''))

        if not (MIN_DURATION <= duration <= MAX_DURATION and text_len >= MIN_TEXT_LEN):
            continue

        source_file = os.path.join(INPUT_DIR, seg['source_file'])
        if not os.path.exists(source_file):
            continue

        # 加载音频片段进行质量检查
        start_s = seg['start'] / 1000
        end_s = seg['end'] / 1000

        try:
            waveform, sr = torchaudio.load(source_file)
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            audio_segment = waveform[:, start_sample:end_sample]

            # 检查采样率
            if sr != 24000:
                print(f"警告: {seg['source_file']} 采样率为 {sr} Hz，不是 24000 Hz")

            # 检查通道数
            if waveform.shape[0] != 1:
                print(f"警告: {seg['source_file']} 不是单声道，有 {waveform.shape[0]} 个通道")

            # 计算RMS（检查是否静音）
            rms = torch.sqrt(torch.mean(audio_segment ** 2)).item()
            if rms < MIN_RMS:
                rms_below_threshold += 1
                quality_filtered_count += 1
                continue

            # 计算SNR
            snr = calculate_snr(audio_segment, sr)
            snr_distribution.append(snr)

            if snr < MIN_SNR_DB:
                snr_below_threshold += 1
                quality_filtered_count += 1
                continue

            seg['audio_path'] = source_file
            seg['duration'] = duration
            seg['rms'] = rms
            seg['snr'] = snr
            valid_segments.append(seg)

        except Exception as e:
            print(f"处理分段 {seg.get('idx', 'unknown')} 时出错: {e}")
            continue

    print(f"符合时长和文本条件: {len(valid_segments) + quality_filtered_count} 个分段")
    print(f"质量过滤掉: {quality_filtered_count} 个")
    print(f"  - SNR < {MIN_SNR_DB} dB: {snr_below_threshold} 个")
    print(f"  - RMS < {MIN_RMS}: {rms_below_threshold} 个")

    if snr_distribution:
        print(f"剩余分段SNR统计: min={min(snr_distribution):.1f} dB, "
              f"max={max(snr_distribution):.1f} dB, "
              f"mean={np.mean(snr_distribution):.1f} dB, "
              f"median={np.median(snr_distribution):.1f} dB")

    # 随机打乱并划分训练集/测试集
    import random
    random.seed(42)
    random.shuffle(valid_segments)

    test_count = max(1, int(len(valid_segments) * TEST_RATIO))
    test_segments = valid_segments[:test_count]
    train_segments = valid_segments[test_count:]

    print(f"训练集: {len(train_segments)} 个分段")
    print(f"测试集: {len(test_segments)} 个分段")

    # 处理训练集
    print("\n处理训练集...")
    train_stats = process_segments(train_segments, OUTPUT_DIR, TRAIN_DIR, spk_id,
                                  TARGET_DBCFS)

    # 处理测试集
    print("\n处理测试集...")
    test_stats = process_segments(test_segments, OUTPUT_DIR, TEST_DIR, spk_id,
                                 TARGET_DBCFS)

    # 统计信息
    print("\n" + "="*50)
    print("数据处理完成!")
    print("="*50)
    print(f"训练集: {train_stats['count']} 条, {train_stats['duration']/60:.2f} 分钟")
    print(f"测试集: {test_stats['count']} 条, {test_stats['duration']/60:.2f} 分钟")
    print(f"总计: {train_stats['count'] + test_stats['count']} 条")
    print(f"总时长: {(train_stats['duration'] + test_stats['duration'])/60:.2f} 分钟")
    print(f"说话人: {spk_id}")
    print(f"音量归一化: 目标 {TARGET_DBCFS} dBFS")
    print("="*50)


def process_segments(segments, output_dir, data_dir, spk_id, target_dbcfs):
    """处理分段列表"""
    utt2wav = {}
    utt2text = {}
    utt2spk = {}
    spk2utt = {spk_id: []}

    total_duration = 0
    scale_factors = []

    for seg in tqdm(segments, desc=f"分割音频到 {data_dir}"):
        utt_id = f"qinche_{seg['idx']:04d}"
        audio_path = seg['audio_path']
        text = seg['text'].strip()

        # 时间戳 (ms -> s)
        start_s = seg['start'] / 1000
        end_s = seg['end'] / 1000

        # 加载音频
        waveform, sr = torchaudio.load(audio_path)

        # 提取音频片段
        start_sample = int(start_s * sr)
        end_sample = int(end_s * sr)
        audio_segment = waveform[:, start_sample:end_sample]

        # 音量归一化
        normalized_segment, scale_factor = normalize_volume(audio_segment, target_dbcfs)
        scale_factors.append(scale_factor)

        # 确保是单声道
        if normalized_segment.shape[0] > 1:
            # 转为单声道（取平均值）
            normalized_segment = torch.mean(normalized_segment, dim=0, keepdim=True)

        # 确保采样率是 24kHz
        if sr != 24000:
            # 重采样
            resampler = torchaudio.transforms.Resample(sr, 24000)
            normalized_segment = resampler(normalized_segment)
            sr = 24000

        # 保存为 16-bit PCM WAV
        output_path = os.path.join(output_dir, f"{utt_id}.wav")
        torchaudio.save(output_path, normalized_segment, sr,
                       encoding="PCM_S", bits_per_sample=16)

        # 记录
        utt2wav[utt_id] = output_path
        utt2text[utt_id] = text
        utt2spk[utt_id] = spk_id
        spk2utt[spk_id].append(utt_id)

        total_duration += (end_s - start_s)

    if scale_factors:
        print(f"  音量归一化统计: mean={np.mean(scale_factors):.3f}, "
              f"min={min(scale_factors):.3f}, max={max(scale_factors):.3f}")

    # 保存索引文件
    save_index(data_dir, utt2wav, utt2text, utt2spk, spk2utt)

    return {
        'count': len(utt2wav),
        'duration': total_duration
    }


def save_index(data_dir, utt2wav, utt2text, utt2spk, spk2utt):
    """保存训练数据索引文件"""
    with open(f"{data_dir}/wav.scp", 'w') as f:
        for k, v in sorted(utt2wav.items()):
            f.write(f"{k} {v}\n")

    with open(f"{data_dir}/text", 'w') as f:
        for k, v in sorted(utt2text.items()):
            f.write(f"{k} {v}\n")

    with open(f"{data_dir}/utt2spk", 'w') as f:
        for k, v in sorted(utt2spk.items()):
            f.write(f"{k} {v}\n")

    with open(f"{data_dir}/spk2utt", 'w') as f:
        for k, v in sorted(spk2utt.items()):
            f.write(f"{k} {' '.join(v)}\n")

    print(f"  已保存到 {data_dir}/")
    print(f"    wav.scp: {len(utt2wav)} 条")
    print(f"    text: {len(utt2text)} 条")


if __name__ == "__main__":
    main()
