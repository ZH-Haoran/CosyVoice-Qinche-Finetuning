#!/usr/bin/env python3
"""
音频分割与ASR标注脚本 - 使用FunASR进行语音识别
改进版：更细粒度的分段（支持逗号分段）
"""
import os
import sys
import json
import argparse
import re
from datetime import datetime

# 设置 CUDA 环境变量
conda_prefix = sys.prefix
if conda_prefix:
    os.environ['CUDA_HOME'] = conda_prefix
    os.environ['PATH'] = os.path.join(conda_prefix, 'bin') + ':' + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = os.path.join(conda_prefix, 'lib64') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

from funasr import AutoModel


# 句子结束标点（强分段点）
STRONG_END_PATTERN = re.compile(r'[。！？…;.!?]')
# 弱分段点（逗号等）
WEAK_END_PATTERN = re.compile(r'[，,、]')

# 其他标点符号（需要跳过）
OTHER_PUNCS = set('：:"\u201c\u201d\u2018\u2019\'\"()（）')

# 分段配置
MIN_SEGMENT_CHARS = 3    # 最小字符数
MAX_SEGMENT_CHARS = 25   # 最大字符数（超过则强制在逗号处分段）
MIN_SEGMENT_DURATION = 1.0   # 最小时长（秒）
MAX_SEGMENT_DURATION = 15.0  # 最大时长（秒）


def split_text_to_segments(text, timestamps):
    """
    将文本分割成适合TTS训练的片段
    规则：
    1. 遇到句号、问号、感叹号等强标点，强制分段
    2. 遇到逗号，始终分段（更激进的分段策略）
    3. 使用时长过滤，只保留合适时长的片段
    """
    chars_with_punc = list(text)
    non_punc_chars = [c for c in chars_with_punc
                      if not STRONG_END_PATTERN.match(c)
                      and not WEAK_END_PATTERN.match(c)
                      and c not in OTHER_PUNCS]

    # 验证数量匹配
    if len(non_punc_chars) != len(timestamps):
        print(f"  警告: 字符数({len(non_punc_chars)}) != 时间戳数({len(timestamps)})")
        min_len = min(len(non_punc_chars), len(timestamps))
        # 需要同步截断
        new_chars = []
        new_timestamps = timestamps[:min_len]
        char_idx = 0
        for c in chars_with_punc:
            if char_idx >= min_len:
                break
            if not STRONG_END_PATTERN.match(c) and not WEAK_END_PATTERN.match(c) and c not in OTHER_PUNCS:
                new_chars.append(c)
                char_idx += 1
        non_punc_chars = new_chars
        timestamps = new_timestamps

    segments = []
    current_text = []
    current_ts_start_idx = 0
    char_ts_idx = 0  # 当前字符对应的timestamp索引

    for char in chars_with_punc:
        is_strong_end = STRONG_END_PATTERN.match(char)
        is_weak_end = WEAK_END_PATTERN.match(char)
        is_other_punc = char in OTHER_PUNCS

        if is_strong_end:
            # 强分段点，保存当前句子
            if current_text and len(current_text) >= MIN_SEGMENT_CHARS:
                end_ts_idx = char_ts_idx - 1
                if current_ts_start_idx <= end_ts_idx < len(timestamps):
                    start_ms = timestamps[current_ts_start_idx][0]
                    end_ms = timestamps[end_ts_idx][1]
                    duration = (end_ms - start_ms) / 1000
                    if duration >= MIN_SEGMENT_DURATION:
                        seg_text = ''.join(current_text).strip()
                        if seg_text:
                            segments.append({
                                'text': seg_text,
                                'start': start_ms,
                                'end': end_ms
                            })
            current_text = []
            current_ts_start_idx = char_ts_idx

        elif is_weak_end:
            # 弱分段点（逗号），始终分段
            if current_text and len(current_text) >= MIN_SEGMENT_CHARS:
                end_ts_idx = char_ts_idx - 1
                if current_ts_start_idx <= end_ts_idx < len(timestamps):
                    start_ms = timestamps[current_ts_start_idx][0]
                    end_ms = timestamps[end_ts_idx][1]
                    # 检查时长是否合适
                    duration = (end_ms - start_ms) / 1000
                    if duration >= MIN_SEGMENT_DURATION:
                        seg_text = ''.join(current_text).strip()
                        if seg_text:
                            segments.append({
                                'text': seg_text,
                                'start': start_ms,
                                'end': end_ms
                            })
            current_text = []
            current_ts_start_idx = char_ts_idx

        elif is_other_punc:
            # 其他标点，跳过
            continue

        else:
            # 普通字符
            current_text.append(char)
            char_ts_idx += 1

    # 处理最后剩余的内容
    if current_text and len(current_text) >= MIN_SEGMENT_CHARS:
        end_ts_idx = char_ts_idx - 1
        if current_ts_start_idx <= end_ts_idx < len(timestamps):
            start_ms = timestamps[current_ts_start_idx][0]
            end_ms = timestamps[end_ts_idx][1]
            duration = (end_ms - start_ms) / 1000
            if duration >= MIN_SEGMENT_DURATION:
                seg_text = ''.join(current_text).strip()
                if seg_text:
                    segments.append({
                        'text': seg_text,
                        'start': start_ms,
                        'end': end_ms
                    })

    return segments


def process_audio_file(model, audio_path):
    """处理单个音频文件"""
    print(f"  处理: {os.path.basename(audio_path)}")

    result = model.generate(
        input=audio_path,
        batch_size_s=300,
        return_raw_text=True,
    )

    all_segments = []
    for res in result:
        text = res.get('text', '')
        timestamps = res.get('timestamp', [])

        if not text or not timestamps:
            continue

        segments = split_text_to_segments(text, timestamps)

        for seg in segments:
            seg['source_file'] = os.path.basename(audio_path)
            all_segments.append(seg)

    print(f"    提取到 {len(all_segments)} 个分段")
    return all_segments


def main():
    parser = argparse.ArgumentParser(description='处理秦彻音频数据')
    parser.add_argument('--input_dir', type=str, default='data/qinche/raw_audios',
                        help='输入音频目录')
    parser.add_argument('--output_dir', type=str, default='data/qinche/processed',
                        help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{datetime.now()}] 开始ASR处理...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")

    print("\n正在加载FunASR模型...")
    model = AutoModel(
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        device="cuda",
        disable_update=True,
    )
    print("模型加载完成!")

    all_results = []
    audio_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.wav')])
    print(f"\n找到 {len(audio_files)} 个音频文件")

    for audio_file in audio_files:
        audio_path = os.path.join(args.input_dir, audio_file)
        segments = process_audio_file(model, audio_path)
        all_results.extend(segments)

    for i, seg in enumerate(all_results):
        seg['idx'] = i

    output_file = os.path.join(args.output_dir, 'all_segments.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 统计
    total_duration = sum(seg['end'] - seg['start'] for seg in all_results) / 1000
    durations = [(seg['end'] - seg['start']) / 1000 for seg in all_results]

    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"{'='*50}")
    print(f"总分段数: {len(all_results)}")
    print(f"总时长: {total_duration/60:.2f} 分钟 ({total_duration:.1f} 秒)")
    print(f"平均片段时长: {sum(durations)/len(durations):.2f} 秒")
    print(f"最短片段: {min(durations):.2f} 秒")
    print(f"最长片段: {max(durations):.2f} 秒")
    print(f"结果已保存到: {output_file}")

    # 时长分布统计
    short_count = sum(1 for d in durations if d < 2)
    medium_count = sum(1 for d in durations if 2 <= d < 15)
    long_count = sum(1 for d in durations if d >= 15)
    print(f"\n时长分布:")
    print(f"  < 2秒: {short_count} 条")
    print(f"  2-15秒: {medium_count} 条 (适合训练)")
    print(f"  >= 15秒: {long_count} 条")

    print(f"\n前15个分段示例:")
    for seg in all_results[:15]:
        duration = (seg['end'] - seg['start']) / 1000
        print(f"  [{seg['start']/1000:.2f}s - {seg['end']/1000:.2f}s] ({duration:.2f}s) {seg['text'][:40]}...")

    return all_results


if __name__ == "__main__":
    main()
