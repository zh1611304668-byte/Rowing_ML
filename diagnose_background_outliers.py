#!/usr/bin/env python3
"""
诊断背景阶段离群值
分析为什么背景阶段会有大量离群值
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def analyze_background_outliers(features_path: str, labeled_path: str, out_dir: str = "outlier_analysis"):
    """分析背景阶段的离群值"""
    
    print("[INFO] 加载数据...")
    features_df = pd.read_csv(features_path)
    labeled_df = pd.read_csv(labeled_path)
    
    # 筛选背景标签
    bg_features = features_df[features_df['label'] == 0].copy()
    print(f"[INFO] 背景样本数: {len(bg_features)}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 分析关键特征的离群值
    key_features = ['y_std', 'y_ptp', 'mag_std', 'gradient_max', 'y_rms']
    
    print("\n=== 背景阶段离群值分析 ===\n")
    
    outlier_times = []  # 记录离群值对应的时间点
    
    for feature in key_features:
        if feature not in bg_features.columns:
            continue
        
        values = bg_features[feature].values
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_count = np.sum(outlier_mask)
        outlier_pct = outlier_count / len(values) * 100
        
        print(f"{feature}:")
        print(f"  Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}")
        print(f"  离群值范围: < {lower_bound:.4f} 或 > {upper_bound:.4f}")
        print(f"  离群值数量: {outlier_count} ({outlier_pct:.2f}%)")
        print(f"  离群值统计: min={values[outlier_mask].min():.4f}, max={values[outlier_mask].max():.4f}")
        print()
        
        # 记录离群值的时间
        if outlier_count > 0:
            outlier_indices = bg_features[outlier_mask].index
            outlier_times.extend(features_df.loc[outlier_indices, 'time'].values)
    
    # 去重并排序时间点
    outlier_times = sorted(set(outlier_times))
    print(f"[INFO] 检测到 {len(outlier_times)} 个唯一时间点存在离群特征\n")
    
    # 可视化：查看离群值在时间轴上的分布
    if len(outlier_times) > 0 and 'time' in bg_features.columns:
        _visualize_outlier_distribution(
            labeled_df, bg_features, outlier_times, out_dir
        )
    
    # 深入分析：离群值窗口对应的原始信号
    if len(outlier_times) > 0:
        _analyze_outlier_windows(
            labeled_df, features_df, outlier_times[:10], out_dir  # 只看前10个
        )
    
    print(f"\n[完成] 分析结果已保存到: {out_dir}")


def _visualize_outlier_distribution(labeled_df, bg_features, outlier_times, out_dir):
    """可视化离群值在时间轴上的分布"""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # 上图：背景阶段在整个时间轴上的分布
    all_time = labeled_df['time'].values
    all_labels = labeled_df['label'].values
    
    axes[0].scatter(all_time[all_labels == 0], 
                   np.zeros(np.sum(all_labels == 0)),
                   c='lightgray', s=0.5, alpha=0.3, label='背景样本')
    axes[0].scatter(outlier_times, 
                   np.zeros(len(outlier_times)),
                   c='red', s=10, marker='x', label='离群值窗口')
    
    # 标记划桨事件（核心阶段）
    core_mask = all_labels == 2
    core_times = all_time[core_mask]
    axes[0].scatter(core_times, 
                   np.ones(len(core_times)) * 0.5,
                   c='orange', s=5, alpha=0.6, marker='v', label='划桨核心期')
    
    axes[0].set_xlabel('时间 (s)', fontsize=12)
    axes[0].set_yticks([0, 0.5])
    axes[0].set_yticklabels(['背景', '核心'])
    axes[0].set_title('背景阶段离群值在时间轴上的分布', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='x', alpha=0.3)
    
    # 下图：离群值与划桨事件的距离分析
    if len(core_times) > 0 and len(outlier_times) > 0:
        distances_to_event = []
        for ot in outlier_times:
            min_dist = np.min(np.abs(core_times - ot))
            distances_to_event.append(min_dist)
        
        axes[1].hist(distances_to_event, bins=50, color='red', alpha=0.6, edgecolor='black')
        axes[1].set_xlabel('离最近划桨事件的距离 (s)', fontsize=12)
        axes[1].set_ylabel('离群值数量', fontsize=12)
        axes[1].set_title('离群值离划桨事件的距离分布', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        median_dist = np.median(distances_to_event)
        axes[1].axvline(median_dist, color='blue', linestyle='--', 
                       linewidth=2, label=f'中位距离: {median_dist:.2f}s')
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'outlier_distribution.png'), dpi=150)
    plt.close()
    print("[INFO] 已生成: outlier_distribution.png")


def _analyze_outlier_windows(labeled_df, features_df, sample_times, out_dir):
    """深入分析离群值窗口的原始信号"""
    
    print("\n[INFO] 分析离群值窗口的原始信号...")
    
    for i, center_time in enumerate(sample_times, 1):
        # 提取窗口前后2秒的数据
        window_sec = 2.0
        start_time = center_time - window_sec
        end_time = center_time + window_sec
        
        mask = (labeled_df['time'] >= start_time) & (labeled_df['time'] <= end_time)
        window_df = labeled_df[mask].copy()
        
        if len(window_df) == 0:
            continue
        
        # 绘制原始信号
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # 三轴加速度
        colors = ['#1976D2', '#388E3C', '#D32F2F']
        for ax_idx, ax_name in enumerate(['X', 'Y', 'Z']):
            col_name = f'acc_dyn_{ax_name.lower()}'
            axes[ax_idx].plot(window_df['time'], window_df[col_name], 
                            color=colors[ax_idx], linewidth=1.5, label=f'{ax_name}轴')
            
            # 标记标签区域
            label_colors = {0: 'lightgray', 1: 'yellow', 2: 'red', 3: 'orange'}
            for label_val in [0, 1, 2, 3]:
                label_mask = window_df['label'] == label_val
                if np.any(label_mask):
                    axes[ax_idx].fill_between(
                        window_df['time'], 
                        window_df[col_name].min() - 0.02,
                        window_df[col_name].max() + 0.02,
                        where=label_mask,
                        color=label_colors[label_val],
                        alpha=0.2
                    )
            
            # 标记离群值中心点
            axes[ax_idx].axvline(center_time, color='red', linestyle='--', 
                               linewidth=2, label='离群值中心')
            
            axes[ax_idx].set_ylabel(f'{ax_name}轴加速度 (g)', fontsize=11)
            axes[ax_idx].legend(loc='upper right', fontsize=9)
            axes[ax_idx].grid(True, alpha=0.3)
        
        axes[0].set_title(
            f'离群值窗口 {i}: 中心时间 {center_time:.2f}s\n' +
            f'[灰=背景, 黄=准备, 红=核心, 橙=恢复]',
            fontsize=13, fontweight='bold'
        )
        axes[2].set_xlabel('时间 (s)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'outlier_window_{i:02d}.png'), dpi=150)
        plt.close()
    
    print(f"[INFO] 已生成 {len(sample_times)} 个离群值窗口可视化")


def main():
    parser = argparse.ArgumentParser(description='诊断背景阶段离群值')
    
    default_features = (r"D:\Desktop\python\rowing_ML\datasets"
                       r"\Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                       r"F92041BC-2503-4150-8196-2B45C0258ED8_clean_labeled_features.csv")
    default_labeled = (r"D:\Desktop\python\rowing_ML\datasets"
                      r"\Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                      r"F92041BC-2503-4150-8196-2B45C0258ED8_clean_labeled.csv")
    
    parser.add_argument('--features_csv', type=str, default=default_features,
                       help='特征CSV路径')
    parser.add_argument('--labeled_csv', type=str, default=default_labeled,
                       help='标注CSV路径')
    parser.add_argument('--out_dir', type=str, default='outlier_analysis',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.features_csv):
        raise FileNotFoundError(f"特征文件不存在: {args.features_csv}")
    if not os.path.exists(args.labeled_csv):
        raise FileNotFoundError(f"标注文件不存在: {args.labeled_csv}")
    
    analyze_background_outliers(args.features_csv, args.labeled_csv, args.out_dir)


if __name__ == '__main__':
    main()
