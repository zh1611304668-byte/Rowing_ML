#!/usr/bin/env python3
"""
步骤7: 使用训练好的ML模型重新检测划桨事件
通过模型预测"核心期"来精确定位事件，实现数据反哺
"""

import argparse
import os
import numpy as np
import pandas as pd
import pickle  # Changed from lightgbm
from scipy import stats as scipy_stats
from scipy.fft import fft


def extract_window_features(window: np.ndarray, sample_rate: float = 100.0):
    """提取单个窗口的特征（与5_feature_extraction.py保持一致）"""
    features = {}
    
    # 时域特征
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        axis_data = window[:, axis_idx]
        features[f'{axis_name}_mean'] = np.mean(axis_data)
        features[f'{axis_name}_std'] = np.std(axis_data)
        features[f'{axis_name}_min'] = np.min(axis_data)
        features[f'{axis_name}_max'] = np.max(axis_data)
        features[f'{axis_name}_ptp'] = np.ptp(axis_data)
        features[f'{axis_name}_rms'] = np.sqrt(np.mean(axis_data ** 2))
        features[f'{axis_name}_skew'] = scipy_stats.skew(axis_data)
        features[f'{axis_name}_kurtosis'] = scipy_stats.kurtosis(axis_data)
        zero_crossings = np.sum(np.diff(np.sign(axis_data)) != 0)
        features[f'{axis_name}_zcr'] = zero_crossings / len(axis_data)
    
    # 加速度幅值
    acc_mag = np.linalg.norm(window, axis=1)
    features['mag_mean'] = np.mean(acc_mag)
    features['mag_std'] = np.std(acc_mag)
    features['mag_max'] = np.max(acc_mag)
    features['mag_min'] = np.min(acc_mag)
    
    # 频域特征
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        axis_data = window[:, axis_idx]
        fft_vals = np.abs(fft(axis_data))
        freqs = np.fft.fftfreq(len(axis_data), 1/sample_rate)
        
        pos_mask = freqs > 0
        fft_vals = fft_vals[pos_mask]
        freqs = freqs[pos_mask]
        
        if len(fft_vals) > 0:
            dominant_freq_idx = np.argmax(fft_vals)
            features[f'{axis_name}_dominant_freq'] = freqs[dominant_freq_idx]
            features[f'{axis_name}_dominant_power'] = fft_vals[dominant_freq_idx]
        else:
            features[f'{axis_name}_dominant_freq'] = 0.0
            features[f'{axis_name}_dominant_power'] = 0.0
        
        bands = [(0, 2), (2, 5), (5, 10), (10, 50)]
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs < high)
            band_energy = np.sum(fft_vals[band_mask] ** 2)
            features[f'{axis_name}_energy_{low}_{high}Hz'] = band_energy
    
    # 自定义特征
    std_x, std_y, std_z = np.std(window[:, 0]), np.std(window[:, 1]), np.std(window[:, 2])
    max_std = max(std_x, std_y, std_z)
    
    if max_std > 0:
        features['std_ratio_x'] = std_x / max_std
        features['std_ratio_y'] = std_y / max_std
        features['std_ratio_z'] = std_z / max_std
    else:
        features['std_ratio_x'] = features['std_ratio_y'] = features['std_ratio_z'] = 0.33
    
    if len(window) > 1:
        gradient = np.diff(window, axis=0)
        features['gradient_mean'] = np.mean(np.linalg.norm(gradient, axis=1))
        features['gradient_max'] = np.max(np.linalg.norm(gradient, axis=1))
    else:
        features['gradient_mean'] = features['gradient_max'] = 0.0
    
    features['dynamic_range'] = np.max(acc_mag) - np.min(acc_mag)
    
    return features


def detect_events_with_ml(csv_path: str, model_path: str, 
                          window_size: int = 40, stride: int = 5,
                          core_prob_threshold: float = 0.5):
    """使用ML模型检测划桨事件"""
    
    print(f"[INFO] 加载数据: {csv_path}")
    df = pd.read_csv(csv_path)
    
    acc_cols = ['acc_dyn_x', 'acc_dyn_y', 'acc_dyn_z']
    for col in acc_cols + ['time']:
        if col not in df.columns:
            raise ValueError(f"CSV缺少必需列: {col}")
    
    time_data = df['time'].values
    acc_data = df[acc_cols].values
    
    print(f"[INFO] 加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"[INFO] 模型类型: {type(model).__name__}")
    
    print(f"[INFO] 开始滑动窗口检测...")
    print(f"  窗口大小: {window_size} 样本 (400ms)")
    print(f"  步长: {stride} 样本 ({stride*10}ms)")
    print(f"  核心期阈值: {core_prob_threshold}")
    
    # 滑动窗口提取特征
    features_list = []
    window_times = []
    
    total_windows = (len(acc_data) - window_size) // stride + 1
    
    # 预加载特征名以便构建 DataFrame
    dummy_feat = extract_window_features(acc_data[0:window_size])
    feature_cols = list(dummy_feat.keys())

    for i in range(0, len(acc_data) - window_size + 1, stride):
        window = acc_data[i:i+window_size]
        center_time = time_data[i + window_size // 2]
        
        # 提取基础特征
        features = extract_window_features(window)
        features_list.append(features)
        window_times.append(center_time)
        
        if len(features_list) % 5000 == 0:
            print(f"  已提取 {len(features_list)}/{total_windows} 个窗口特征...")
            
    # 转换为 DataFrame以批量处理
    feature_df = pd.DataFrame(features_list)
    
    # ========== 添加时间序列特征 (与 Script 5 保持一致) ==========
    print("\n[INFO] 计算时间序列特征...")
    
    # 关键特征列表
    key_cols = ['y_mean', 'y_std', 'y_rms', 'y_ptp', 
                'mag_mean', 'mag_max', 'mag_std',
                'gradient_max', 'dynamic_range']
    
    # 过滤存在的列
    key_cols = [col for col in key_cols if col in feature_df.columns]
    
    # 1. 差分特征
    for col in key_cols:
        feature_df[f'{col}_diff'] = feature_df[col].diff().fillna(0)
    
    # 2. 滚动统计
    for col in key_cols:
        feature_df[f'{col}_roll3_mean'] = feature_df[col].rolling(window=3, min_periods=1).mean()
        feature_df[f'{col}_roll3_std'] = feature_df[col].rolling(window=3, min_periods=1).std().fillna(0)
    
    # 3. 加速度特征
    for col in ['y_rms', 'mag_max']:
        if col in feature_df.columns:
            feature_df[f'{col}_accel'] = feature_df[col].diff().diff().fillna(0)
    
    # 4. 动量特征
    for col in ['y_rms', 'mag_mean']:
        if col in feature_df.columns:
            feature_df[f'{col}_momentum'] = feature_df[col].diff().rolling(window=5, min_periods=1).sum().fillna(0)
            
    print(f"[INFO] 最终特征数量: {len(feature_df.columns)}")
    
    # 批量预测
    print(f"[INFO] 开始批量预测...")
    # 5类: [背景, 准备, 核心, 恢复, 过渡]
    pred_probs = model.predict_proba(feature_df)
    core_probs = pred_probs[:, 2]  # 核心期概率 (Index 2)
    
    print(f"[INFO] 预测完成，共 {len(core_probs)} 个窗口")
    
    # 找到核心期的峰值作为事件
    print(f"[INFO] 识别事件（核心期概率 > {core_prob_threshold}）...")
    
    core_probs = np.array(core_probs)
    window_times = np.array(window_times)
    
    # 找到概率超过阈值的连续区域
    events = []
    in_region = False
    region_start = 0
    
    for i in range(len(core_probs)):
        if core_probs[i] > core_prob_threshold and not in_region:
            region_start = i
            in_region = True
        elif core_probs[i] <= core_prob_threshold and in_region:
            # 找到这个区域内的峰值
            region_probs = core_probs[region_start:i]
            if len(region_probs) > 0:
                peak_idx = region_start + np.argmax(region_probs)
                events.append({
                    'time': window_times[peak_idx],
                    'prob': core_probs[peak_idx]
                })
            in_region = False
    
    # 处理最后一个区域
    if in_region:
        region_probs = core_probs[region_start:]
        if len(region_probs) > 0:
            peak_idx = region_start + np.argmax(region_probs)
            events.append({
                'time': window_times[peak_idx],
                'prob': core_probs[peak_idx]
            })
    
    # 过滤过近的事件（最小间隔0.8秒）
    min_interval = 0.8
    filtered_events = []
    last_time = -999
    
    for event in events:
        if event['time'] - last_time >= min_interval:
            filtered_events.append(event)
            last_time = event['time']
    
    event_times = [e['time'] for e in filtered_events]
    
    print(f"[INFO] 检测到 {len(event_times)} 个事件")
    
    # 统计
    if len(event_times) > 1:
        intervals = np.diff(event_times)
        stroke_rate = 60.0 / np.median(intervals)
        print(f"\n=== 检测统计 ===")
        print(f"事件总数: {len(event_times)}")
        print(f"间隔中位数: {np.median(intervals):.2f}秒")
        print(f"预估划桨率: {stroke_rate:.1f}次/分钟")
        print(f"平均置信度: {np.mean([e['prob'] for e in filtered_events]):.3f}")
    
    return np.array(event_times)


def main():
    parser = argparse.ArgumentParser(description='步骤7: 用ML模型重新检测事件')
    
    default_csv = (r"D:\Desktop\python\rowing_ML\clean_report"
                   r"\Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                   r"F92041BC-2503-4150-8196-2B45C0258ED8_clean.csv")
    default_model = r"D:\Desktop\python\rowing_ML\models\lightgbm_model.txt"
    
    parser.add_argument('--csv_path', type=str, default=default_csv,
                       help='清洗后的CSV路径')
    parser.add_argument('--model_path', type=str, default=default_model,
                       help='训练好的模型路径')
    parser.add_argument('--window_size', type=int, default=40,
                       help='窗口大小(样本数)')
    parser.add_argument('--stride', type=int, default=5,
                       help='步长(样本数)')
    parser.add_argument('--core_prob_threshold', type=float, default=0.5,
                       help='核心期概率阈值(0-1)')
    parser.add_argument('--out_path', type=str, default=None,
                       help='输出事件文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {args.csv_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    # 检测事件
    events = detect_events_with_ml(
        args.csv_path,
        args.model_path,
        args.window_size,
        args.stride,
        args.core_prob_threshold
    )
    
    # 保存结果
    if args.out_path is None:
        base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        args.out_path = f"{base_name}_events_ml.txt"
    
    np.savetxt(args.out_path, events, fmt='%.6f')
    print(f"\n[完成] ML检测的事件已保存: {args.out_path}")
    print(f"\n下一步: 使用这些事件重新生成标签")
    print(f"  python scripts/4_generate_labels.py --event_path {args.out_path}")


if __name__ == '__main__':
    main()
