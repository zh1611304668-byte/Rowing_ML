#!/usr/bin/env python3
"""
ML推理脚本 - 使用训练好的模型对新数据进行预测
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.fft import fft

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def extract_time_features(window: np.ndarray):
    """提取时域特征 (3轴) - 与训练脚本保持一致"""
    features = {}
    
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        axis_data = window[:, axis_idx]
        
        # 基本统计
        features[f'{axis_name}_mean'] = np.mean(axis_data)
        features[f'{axis_name}_std'] = np.std(axis_data)
        features[f'{axis_name}_min'] = np.min(axis_data)
        features[f'{axis_name}_max'] = np.max(axis_data)
        features[f'{axis_name}_ptp'] = np.ptp(axis_data)
        
        # RMS
        features[f'{axis_name}_rms'] = np.sqrt(np.mean(axis_data ** 2))
        
        # 偏度和峰度
        features[f'{axis_name}_skew'] = scipy_stats.skew(axis_data)
        features[f'{axis_name}_kurtosis'] = scipy_stats.kurtosis(axis_data)
        
        # 过零率
        zero_crossings = np.sum(np.diff(np.sign(axis_data)) != 0)
        features[f'{axis_name}_zcr'] = zero_crossings / len(axis_data)
    
    # 加速度幅值
    acc_mag = np.linalg.norm(window, axis=1)
    features['mag_mean'] = np.mean(acc_mag)
    features['mag_std'] = np.std(acc_mag)
    features['mag_max'] = np.max(acc_mag)
    features['mag_min'] = np.min(acc_mag)
    
    return features


def extract_freq_features(window: np.ndarray, sample_rate: float = 100.0):
    """提取频域特征 - 与训练脚本保持一致"""
    features = {}
    
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        axis_data = window[:, axis_idx]
        
        # FFT
        fft_vals = np.abs(fft(axis_data))
        freqs = np.fft.fftfreq(len(axis_data), 1/sample_rate)
        
        # 只取正频率部分
        pos_mask = freqs > 0
        fft_vals = fft_vals[pos_mask]
        freqs = freqs[pos_mask]
        
        # 主频率
        if len(fft_vals) > 0:
            dominant_freq_idx = np.argmax(fft_vals)
            features[f'{axis_name}_dominant_freq'] = freqs[dominant_freq_idx]
            features[f'{axis_name}_dominant_power'] = fft_vals[dominant_freq_idx]
        else:
            features[f'{axis_name}_dominant_freq'] = 0.0
            features[f'{axis_name}_dominant_power'] = 0.0
        
        # 频段能量分布
        bands = [(0, 2), (2, 5), (5, 10), (10, 50)]
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs < high)
            band_energy = np.sum(fft_vals[band_mask] ** 2)
            features[f'{axis_name}_energy_{low}_{high}Hz'] = band_energy
    
    return features


def extract_custom_features(window: np.ndarray):
    """提取自定义特征 - 与训练脚本保持一致"""
    features = {}
    
    # 三轴标准差比值
    std_x = np.std(window[:, 0])
    std_y = np.std(window[:, 1])
    std_z = np.std(window[:, 2])
    
    max_std = max(std_x, std_y, std_z)
    if max_std > 0:
        features['std_ratio_x'] = std_x / max_std
        features['std_ratio_y'] = std_y / max_std
        features['std_ratio_z'] = std_z / max_std
    else:
        features['std_ratio_x'] = 0.33
        features['std_ratio_y'] = 0.33
        features['std_ratio_z'] = 0.33
    
    # 加速度梯度
    if len(window) > 1:
        gradient = np.diff(window, axis=0)
        features['gradient_mean'] = np.mean(np.linalg.norm(gradient, axis=1))
        features['gradient_max'] = np.max(np.linalg.norm(gradient, axis=1))
    else:
        features['gradient_mean'] = 0.0
        features['gradient_max'] = 0.0
    
    # 动态范围
    acc_mag = np.linalg.norm(window, axis=1)
    features['dynamic_range'] = np.max(acc_mag) - np.min(acc_mag)
    
    return features


def extract_features_from_df(
    df: pd.DataFrame,
    window_size: int = 40,
    stride: int = 1,
    sample_rate: float = 100.0
) -> pd.DataFrame:
    """从DataFrame中提取滑动窗口特征"""
    
    acc_cols = ['acc_dyn_x', 'acc_dyn_y', 'acc_dyn_z']
    acc_data = df[acc_cols].values
    time_data = df['time'].values
    
    features_list = []
    window_times = []
    
    print(f"[INFO] 开始特征提取...")
    print(f"[INFO] 窗口大小: {window_size} 样本 ({window_size/sample_rate*1000:.0f}ms)")
    print(f"[INFO] 步长: {stride} 样本 ({stride/sample_rate*1000:.0f}ms)")
    
    total_windows = (len(acc_data) - window_size) // stride + 1
    
    for i in range(0, len(acc_data) - window_size + 1, stride):
        # 提取窗口
        window = acc_data[i:i+window_size]
        
        # 窗口的时间 (取中心点)
        center_idx = i + window_size // 2
        window_time = time_data[center_idx]
        
        # 提取特征
        features = {}
        features.update(extract_time_features(window))
        features.update(extract_freq_features(window, sample_rate))
        features.update(extract_custom_features(window))
        
        features_list.append(features)
        window_times.append(window_time)
        
        if (len(features_list) % 5000 == 0):
            print(f"[INFO] 已处理 {len(features_list)}/{total_windows} 窗口...")
    
    print(f"[INFO] 特征提取完成! 总窗口数: {len(features_list)}")
    
    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['time'] = window_times
    
    # 添加时间序列特征 (与训练保持一致)
    print("[INFO] 添加时间序列特征...")
    
    key_cols = ['y_mean', 'y_std', 'y_rms', 'y_ptp', 
                'mag_mean', 'mag_max', 'mag_std',
                'gradient_max', 'dynamic_range']
    
    key_cols = [col for col in key_cols if col in features_df.columns]
    
    # 1. 差分特征
    for col in key_cols:
        features_df[f'{col}_diff'] = features_df[col].diff().fillna(0)
    
    # 2. 滚动统计
    for col in key_cols:
        features_df[f'{col}_roll3_mean'] = features_df[col].rolling(window=3, min_periods=1).mean()
        features_df[f'{col}_roll3_std'] = features_df[col].rolling(window=3, min_periods=1).std().fillna(0)
    
    # 3. 加速度特征
    for col in ['y_rms', 'mag_max']:
        if col in features_df.columns:
            features_df[f'{col}_accel'] = features_df[col].diff().diff().fillna(0)
    
    # 4. 动量特征
    for col in ['y_rms', 'mag_mean']:
        if col in features_df.columns:
            features_df[f'{col}_momentum'] = features_df[col].diff().rolling(window=5, min_periods=1).sum().fillna(0)
    
    # ⚠️ 添加 stroke_id 列以匹配训练特征（推理时设为0）
    features_df['stroke_id'] = 0
    
    print(f"[INFO] 特征总数: {len(features_df.columns) - 1}  (不含time)")
    
    return features_df


def visualize_predictions(df_orig: pd.DataFrame, predictions: np.ndarray, 
                         probabilities: np.ndarray, out_dir: str, 
                         window_size: int = 40):
    """可视化预测结果"""
    print("\n[INFO] 生成预测可视化...")
    
    # 标签配置
    label_names = {0: '背景', 1: '准备', 2: '核心', 3: '恢复', 4: '过渡'}
    colors = {
        0: '#BBDEFB',  # 浅蓝 - 背景
        1: '#FFF59D',  # 黄色 - 准备
        2: '#FF5252',  # 红色 - 核心
        3: '#90CAF9',  # 蓝色 - 恢复
        4: '#CE93D8'   # 紫色 - 过渡
    }
    
    # 选择一段数据进行可视化 (前10秒)
    sample_rate = 100.0
    viz_duration = 10.0  # 秒
    viz_samples = int(viz_duration * sample_rate)
    
    # 创建多个可视化样本
    n_samples = min(3, len(df_orig) // viz_samples)
    
    for sample_idx in range(n_samples):
        start_idx = sample_idx * viz_samples
        end_idx = start_idx + viz_samples
        
        if end_idx > len(df_orig):
            break
        
        # 提取数据段
        df_segment = df_orig.iloc[start_idx:end_idx]
        
        # 对应的预测结果 (考虑窗口中心偏移)
        pred_start = start_idx
        pred_end = min(end_idx, len(predictions))
        
        if pred_start >= len(predictions):
            continue
        
        preds_segment = predictions[pred_start:pred_end]
        probs_segment = probabilities[pred_start:pred_end]
        
        # 创建图表
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        
        time_segment = df_segment['time'].values
        pred_time = time_segment[:len(preds_segment)]
        
        # 1. 原始加速度数据
        ax = axes[0]
        ax.plot(time_segment, df_segment['acc_dyn_x'], 'r-', alpha=0.6, linewidth=0.8, label='X轴')
        ax.plot(time_segment, df_segment['acc_dyn_y'], 'g-', alpha=0.6, linewidth=0.8, label='Y轴')
        ax.plot(time_segment, df_segment['acc_dyn_z'], 'b-', alpha=0.6, linewidth=0.8, label='Z轴')
        ax.set_ylabel('加速度 (m/s²)', fontsize=11)
        ax.set_title(f'样本 {sample_idx+1}: 原始加速度数据', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 2. 预测标签
        ax = axes[1]
        for label_id in sorted(label_names.keys()):
            mask = preds_segment == label_id
            if np.any(mask):
                ax.scatter(pred_time[mask], preds_segment[mask], 
                          c=colors[label_id], label=label_names[label_id],
                          s=10, alpha=0.7)
        ax.set_ylabel('预测标签', fontsize=11)
        ax.set_ylim([-0.5, 4.5])
        ax.set_yticks(range(5))
        ax.set_yticklabels([label_names[i] for i in range(5)])
        ax.set_title('ML模型预测结果', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', ncol=5)
        ax.grid(True, alpha=0.3)
        
        # 3. 预测概率分布
        ax = axes[2]
        for label_id in range(5):
            ax.plot(pred_time, probs_segment[:, label_id], 
                   color=colors[label_id], label=label_names[label_id],
                   linewidth=1.5, alpha=0.7)
        ax.set_ylabel('概率', fontsize=11)
        ax.set_ylim([0, 1])
        ax.set_title('各阶段预测概率', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', ncol=5)
        ax.grid(True, alpha=0.3)
        
        # 4. 预测置信度 (最大概率)
        ax = axes[3]
        max_probs = np.max(probs_segment, axis=1)
        ax.plot(pred_time, max_probs, 'k-', linewidth=1.5, label='置信度')
        ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.5, label='0.5阈值')
        ax.fill_between(pred_time, 0, max_probs, alpha=0.3, color='gray')
        ax.set_ylabel('置信度', fontsize=11)
        ax.set_xlabel('时间 (s)', fontsize=11)
        ax.set_ylim([0, 1])
        ax.set_title('预测置信度', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'prediction_sample_{sample_idx+1}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存可视化: prediction_sample_{sample_idx+1}.png")
    
    # 生成统计图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 预测标签分布
    ax = axes[0]
    label_counts = pd.Series(predictions).value_counts().sort_index()
    bars = ax.bar([label_names[i] for i in label_counts.index], 
                   label_counts.values,
                   color=[colors[i] for i in label_counts.index],
                   alpha=0.7)
    ax.set_ylabel('计数', fontsize=11)
    ax.set_title('预测标签分布统计', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加百分比标签
    total = len(predictions)
    for bar, count in zip(bars, label_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}\n({count/total*100:.1f}%)',
               ha='center', va='bottom', fontsize=9)
    
    # 平均置信度
    ax = axes[1]
    mean_confidence = []
    for label_id in range(5):
        mask = predictions == label_id
        if np.any(mask):
            mean_conf = np.mean(np.max(probabilities[mask], axis=1))
            mean_confidence.append(mean_conf)
        else:
            mean_confidence.append(0)
    
    bars = ax.bar([label_names[i] for i in range(5)], mean_confidence,
                   color=[colors[i] for i in range(5)], alpha=0.7)
    ax.set_ylabel('平均置信度', fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_title('各阶段平均预测置信度', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, conf in zip(bars, mean_confidence):
        if conf > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{conf:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    stats_path = os.path.join(out_dir, 'prediction_statistics.png')
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 保存统计图: prediction_statistics.png")


def main():
    parser = argparse.ArgumentParser(description='ML推理脚本')
    
    parser.add_argument('--data', type=str, required=True,
                       help='新数据CSV文件路径')
    parser.add_argument('--model', type=str, default='models/rf_rigorous_model.pkl',
                       help='训练好的模型路径')
    parser.add_argument('--window_size', type=int, default=40,
                       help='窗口大小(样本数)')
    parser.add_argument('--stride', type=int, default=1,
                       help='滑动步长(样本数)')
    parser.add_argument('--sample_rate', type=float, default=100.0,
                       help='采样率(Hz)')
    parser.add_argument('--out_dir', type=str, default='detection_comparison',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 1. 检查文件
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据文件不存在: {args.data}")
    
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    
    # 2. 加载模型
    print(f"\n[INFO] 加载训练好的模型: {args.model}")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    print(f"[INFO] 模型加载成功!")
    print(f"  - 模型类型: {type(model).__name__}")
    print(f"  - 特征数量: {model.n_features_in_}")
    
    # 3. 加载新数据
    print(f"\n[INFO] 加载新数据: {args.data}")
    df = pd.read_csv(args.data)
    print(f"[INFO] 数据行数: {len(df)}")
    print(f"[INFO] 数据列: {list(df.columns)}")
    
    # 检查必要的列
    required_cols = ['time', 'acc_dyn_x', 'acc_dyn_y', 'acc_dyn_z']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必要的列: {missing_cols}")
    
    # 4. 提取特征
    print(f"\n[INFO] 提取特征...")
    features_df = extract_features_from_df(
        df,
        window_size=args.window_size,
        stride=args.stride,
        sample_rate=args.sample_rate
    )
    
    # 5. 准备特征矩阵 (排除time列)
    feature_cols = [col for col in features_df.columns if col != 'time']
    X = features_df[feature_cols].values
    
    print(f"\n[INFO] 特征矩阵形状: {X.shape}")
    print(f"[INFO] 模型期望特征数: {model.n_features_in_}")
    
    if X.shape[1] != model.n_features_in_:
        raise ValueError(f"特征数量不匹配! 提取了{X.shape[1]}个特征，但模型需要{model.n_features_in_}个")
    
    # 6. 进行预测
    print(f"\n[INFO] 开始预测...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # 7. 保存结果
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 保存预测结果
    results_df = features_df.copy()
    results_df['predicted_label'] = predictions
    for i in range(5):
        results_df[f'prob_label_{i}'] = probabilities[:, i]
    results_df['max_probability'] = np.max(probabilities, axis=1)
    
    base_name = os.path.splitext(os.path.basename(args.data))[0]
    results_path = os.path.join(args.out_dir, f'{base_name}_predictions.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n[INFO] 预测结果已保存: {results_path}")
    
    # 8. 统计分析
    print(f"\n{'='*60}")
    print("预测结果统计")
    print('='*60)
    
    label_names = {0: '背景', 1: '准备', 2: '核心', 3: '恢复', 4: '过渡'}
    
    print("\n标签分布:")
    label_counts = pd.Series(predictions).value_counts().sort_index()
    total = len(predictions)
    for label, count in label_counts.items():
        pct = count / total * 100
        print(f"  {label_names.get(label, str(label))} ({label}): {count:6d} ({pct:5.2f}%)")
    
    print("\n各阶段平均置信度:")
    for label_id in range(5):
        mask = predictions == label_id
        if np.any(mask):
            mean_conf = np.mean(np.max(probabilities[mask], axis=1))
            print(f"  {label_names[label_id]}: {mean_conf:.4f}")
    
    overall_conf = np.mean(np.max(probabilities, axis=1))
    print(f"\n整体平均置信度: {overall_conf:.4f}")
    
    # 9. 可视化
    print(f"\n[INFO] 生成可视化...")
    visualize_predictions(df, predictions, probabilities, args.out_dir, args.window_size)
    
    print(f"\n{'='*60}")
    print("推理完成!")
    print('='*60)
    print(f"\n📁 输出文件:")
    print(f"  预测结果: {results_path}")
    print(f"  可视化目录: {args.out_dir}")
    print(f"\n✅ 下一步建议:")
    print("  1. 查看可视化图片，验证预测效果")
    print("  2. 如果效果不理想，可以考虑:")
    print("     - 检查数据质量")
    print("     - 调整窗口大小和步长")
    print("     - 重新训练模型")


if __name__ == '__main__':
    main()
