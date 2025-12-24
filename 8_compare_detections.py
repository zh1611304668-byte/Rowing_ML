#!/usr/bin/env python3
"""
检测方法对比分析
对比传统峰值检测方法和ML检测方法的差异
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_events(event_path: str) -> np.ndarray:
    """加载事件时间戳"""
    events = np.loadtxt(event_path, ndmin=1).astype(float)
    
    # 自动检测单位
    median_val = np.median(events) if len(events) > 0 else 0
    if median_val > 1e5:  # 毫秒
        events = events / 1000.0
        print(f"  检测到单位为毫秒，已转换为秒")
    
    return np.sort(events)


def match_events(events_a: np.ndarray, events_b: np.ndarray, 
                 tolerance: float = 0.3) -> Tuple[List, List, List]:
    """
    匹配两组事件
    
    Args:
        events_a: 第一组事件（参考）
        events_b: 第二组事件（待匹配）
        tolerance: 匹配容差（秒）
    
    Returns:
        matched: [(a_idx, b_idx, time_diff), ...]
        only_in_a: 只在A中的事件索引
        only_in_b: 只在B中的事件索引
    """
    matched = []
    used_b = set()
    only_in_a = []
    
    for i, time_a in enumerate(events_a):
        # 找最近的B事件
        diffs = np.abs(events_b - time_a)
        min_idx = np.argmin(diffs)
        min_diff = diffs[min_idx]
        
        if min_diff <= tolerance and min_idx not in used_b:
            matched.append((i, min_idx, min_diff))
            used_b.add(min_idx)
        else:
            only_in_a.append(i)
    
    only_in_b = [j for j in range(len(events_b)) if j not in used_b]
    
    return matched, only_in_a, only_in_b


def analyze_detection_quality(matched: List, events_a: np.ndarray, 
                              events_b: np.ndarray, only_in_a: List, 
                              only_in_b: List) -> dict:
    """分析检测质量指标"""
    
    n_matched = len(matched)
    n_only_a = len(only_in_a)
    n_only_b = len(only_in_b)
    n_total_a = len(events_a)
    n_total_b = len(events_b)
    
    # 召回率：传统方法中有多少被ML找到
    recall = n_matched / n_total_a if n_total_a > 0 else 0
    
    # 精确率：ML检测中有多少是真的（假设传统方法是ground truth）
    precision = n_matched / n_total_b if n_total_b > 0 else 0
    
    # F1分数
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 时间误差统计
    time_errors = [diff for _, _, diff in matched]
    mean_error = np.mean(time_errors) if time_errors else 0
    std_error = np.std(time_errors) if time_errors else 0
    max_error = np.max(time_errors) if time_errors else 0
    
    stats = {
        'n_traditional': n_total_a,
        'n_ml': n_total_b,
        'n_matched': n_matched,
        'n_missing': n_only_a,  # ML漏检
        'n_extra': n_only_b,    # ML新发现（或误检）
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'mean_time_error': mean_error,
        'std_time_error': std_error,
        'max_time_error': max_error
    }
    
    return stats


def visualize_event_comparison(events_traditional: np.ndarray, 
                               events_ml: np.ndarray,
                               matched: List, only_traditional: List, 
                               only_ml: List, csv_data: pd.DataFrame,
                               out_dir: str):
    """可视化事件对比"""
    
    print("[INFO] 生成对比可视化...")
    
    # 图1：事件时间轴对比
    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
    
    # 上图：两种方法的事件分布
    axes[0].scatter(events_traditional, np.ones(len(events_traditional)), 
                   c='blue', s=50, alpha=0.6, marker='|', linewidths=2,
                   label=f'传统检测 ({len(events_traditional)}个)')
    axes[0].scatter(events_ml, np.zeros(len(events_ml)), 
                   c='red', s=50, alpha=0.6, marker='|', linewidths=2,
                   label=f'ML检测 ({len(events_ml)}个)')
    
    # 标记匹配的事件
    for a_idx, b_idx, _ in matched:
        axes[0].plot([events_traditional[a_idx], events_ml[b_idx]], 
                    [1, 0], 'g-', alpha=0.3, linewidth=0.5)
    
    axes[0].set_ylabel('检测方法', fontsize=11)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['ML', '传统'])
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].set_title('事件检测对比 - 时间轴视图', fontsize=14, fontweight='bold')
    
    # 中图：差异事件
    if only_traditional:
        axes[1].scatter(events_traditional[only_traditional], 
                       np.ones(len(only_traditional)),
                       c='blue', s=100, marker='x', linewidths=2,
                       label=f'ML漏检 ({len(only_traditional)}个)')
    
    if only_ml:
        axes[1].scatter(events_ml[only_ml], 
                       np.zeros(len(only_ml)),
                       c='red', s=100, marker='x', linewidths=2,
                       label=f'ML新发现 ({len(only_ml)}个)')
    
    axes[1].set_ylabel('差异类型', fontsize=11)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['ML独有', '传统独有'])
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_title('差异事件分布', fontsize=13, fontweight='bold')
    
    # 下图：原始信号（Y轴加速度）
    if 'time' in csv_data.columns and 'acc_dyn_y' in csv_data.columns:
        axes[2].plot(csv_data['time'], csv_data['acc_dyn_y'], 
                    'gray', alpha=0.5, linewidth=0.5, label='Y轴加速度')
        
        # 标记传统检测事件
        for event_time in events_traditional:
            axes[2].axvline(event_time, color='blue', alpha=0.3, linewidth=1)
        
        # 标记ML独有事件（可能是误检或新发现）
        for idx in only_ml:
            axes[2].axvline(events_ml[idx], color='red', alpha=0.5, 
                          linewidth=2, linestyle='--')
        
        axes[2].set_xlabel('时间 (s)', fontsize=12)
        axes[2].set_ylabel('Y轴加速度 (g)', fontsize=11)
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title('原始信号与检测事件', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'detection_comparison_timeline.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 时间轴对比图已保存")
    
    # 图2：时间误差分布
    if matched:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        time_errors = [diff * 1000 for _, _, diff in matched]  # 转换为毫秒
        
        # 直方图
        axes[0].hist(time_errors, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(time_errors), color='red', linestyle='--', 
                       linewidth=2, label=f'均值: {np.mean(time_errors):.1f}ms')
        axes[0].set_xlabel('时间误差 (ms)', fontsize=11)
        axes[0].set_ylabel('频数', fontsize=11)
        axes[0].set_title('检测时间误差分布', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 累积分布
        sorted_errors = np.sort(time_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        axes[1].plot(sorted_errors, cumulative, 'b-', linewidth=2)
        axes[1].axvline(100, color='red', linestyle='--', alpha=0.5, label='100ms')
        axes[1].axvline(200, color='orange', linestyle='--', alpha=0.5, label='200ms')
        axes[1].set_xlabel('时间误差 (ms)', fontsize=11)
        axes[1].set_ylabel('累积百分比 (%)', fontsize=11)
        axes[1].set_title('时间误差累积分布', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'detection_time_error.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 时间误差分布图已保存")


def visualize_sample_differences(events_traditional: np.ndarray, 
                                 events_ml: np.ndarray,
                                 only_traditional: List, only_ml: List,
                                 csv_data: pd.DataFrame, out_dir: str,
                                 num_samples: int = 5):
    """可视化差异事件的样本"""
    
    print("[INFO] 生成差异事件样本可视化...")
    
    # 采样ML独有事件（可能的新发现或误检）
    if only_ml:
        sample_indices = np.random.choice(only_ml, 
                                         min(num_samples, len(only_ml)), 
                                         replace=False)
        
        for i, idx in enumerate(sample_indices, 1):
            event_time = events_ml[idx]
            
            # 提取事件前后3秒的数据
            window_sec = 3.0
            start_time = event_time - window_sec
            end_time = event_time + window_sec
            
            mask = (csv_data['time'] >= start_time) & (csv_data['time'] <= end_time)
            window_df = csv_data[mask].copy()
            
            if len(window_df) == 0:
                continue
            
            # 绘制三轴加速度
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            
            colors = ['#1976D2', '#388E3C', '#D32F2F']
            for ax_idx, (ax_name, col_name) in enumerate([('X', 'acc_dyn_x'), 
                                                           ('Y', 'acc_dyn_y'), 
                                                           ('Z', 'acc_dyn_z')]):
                if col_name not in window_df.columns:
                    continue
                
                axes[ax_idx].plot(window_df['time'], window_df[col_name],
                                color=colors[ax_idx], linewidth=1.5, 
                                label=f'{ax_name}轴')
                
                # 标记ML检测点
                axes[ax_idx].axvline(event_time, color='red', linestyle='--',
                                   linewidth=2, label='ML检测点')
                
                # 标记附近的传统检测点
                nearby_traditional = events_traditional[
                    (events_traditional >= start_time) & 
                    (events_traditional <= end_time)
                ]
                for trad_time in nearby_traditional:
                    axes[ax_idx].axvline(trad_time, color='blue', 
                                       linestyle=':', linewidth=1.5, 
                                       alpha=0.5)
                
                axes[ax_idx].set_ylabel(f'{ax_name}轴 (g)', fontsize=11)
                axes[ax_idx].legend(loc='upper right', fontsize=9)
                axes[ax_idx].grid(True, alpha=0.3)
            
            axes[0].set_title(
                f'ML独有事件 #{i}: 时间={event_time:.2f}s\n' +
                f'[红虚线=ML检测, 蓝点线=传统检测]',
                fontsize=13, fontweight='bold'
            )
            axes[2].set_xlabel('时间 (s)', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'ml_unique_event_{i:02d}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ 已生成 {len(sample_indices)} 个ML独有事件样本")


def generate_report(stats: dict, out_dir: str):
    """生成详细的文本报告"""
    
    report_path = os.path.join(out_dir, 'detection_comparison_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("检测方法对比分析报告\n")
        f.write("传统峰值检测 vs 机器学习检测\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("【1. 事件数量统计】\n")
        f.write(f"  传统方法检测数量: {stats['n_traditional']:4d}\n")
        f.write(f"  ML方法检测数量:   {stats['n_ml']:4d}\n")
        f.write(f"  差值:             {stats['n_ml'] - stats['n_traditional']:+4d} " +
                f"({(stats['n_ml'] - stats['n_traditional']) / stats['n_traditional'] * 100:+.1f}%)\n\n")
        
        f.write("【2. 匹配情况】\n")
        f.write(f"  成功匹配:         {stats['n_matched']:4d}\n")
        f.write(f"  ML漏检:          {stats['n_missing']:4d} " +
                f"(传统检测到但ML未检测)\n")
        f.write(f"  ML独有:          {stats['n_extra']:4d} " +
                f"(ML检测到但传统未检测)\n\n")
        
        f.write("【3. 性能指标】\n")
        f.write(f"  召回率 (Recall):   {stats['recall']*100:5.2f}% " +
                f"(ML找到了传统方法的多少)\n")
        f.write(f"  精确率 (Precision): {stats['precision']*100:5.2f}% " +
                f"(ML检测中有多少是真的)\n")
        f.write(f"  F1分数:            {stats['f1_score']:5.3f}\n\n")
        
        f.write("【4. 时间精度】\n")
        f.write(f"  平均时间误差:      {stats['mean_time_error']*1000:6.2f} ms\n")
        f.write(f"  时间误差标准差:    {stats['std_time_error']*1000:6.2f} ms\n")
        f.write(f"  最大时间误差:      {stats['max_time_error']*1000:6.2f} ms\n\n")
        
        f.write("【5. 结论与建议】\n")
        
        # 智能分析
        if stats['recall'] >= 0.95 and stats['precision'] >= 0.90:
            conclusion = "✅ 优秀：ML方法接近或超越传统方法"
            suggestion = "建议: 可以使用ML方法替代传统方法"
        elif stats['recall'] >= 0.85 and stats['precision'] >= 0.80:
            conclusion = "✓ 良好：ML方法表现不错，但仍有改进空间"
            suggestion = "建议: 继续优化特征工程或模型参数"
        elif stats['n_extra'] > stats['n_missing'] * 1.5:
            conclusion = "⚠ 警告：ML方法可能存在较多误检"
            suggestion = "建议: 检查ML独有事件，调整分类阈值"
        elif stats['n_missing'] > stats['n_matched'] * 0.2:
            conclusion = "⚠ 警告：ML方法漏检较多"
            suggestion = "建议: 检查特征提取或模型训练数据"
        else:
            conclusion = "→ 一般：两种方法各有优劣"
            suggestion = "建议: 深入分析差异事件，考虑集成方法"
        
        f.write(f"  {conclusion}\n")
        f.write(f"  {suggestion}\n\n")
        
        if stats['n_extra'] > 0:
            f.write("【6. ML独有事件分析】\n")
            f.write(f"  ML检测到 {stats['n_extra']} 个传统方法未检测的事件\n")
            f.write("  可能原因:\n")
            f.write("    - ML发现了真实但微弱的划桨动作（好）\n")
            f.write("    - ML误检了干扰信号为划桨（坏）\n")
            f.write("  验证方法:\n")
            f.write("    - 查看生成的 ml_unique_event_*.png 图片\n")
            f.write("    - 检查是否有明显的划桨特征\n\n")
        
        f.write("=" * 70 + "\n")
    
    print(f"[INFO] 详细报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='检测方法对比分析')
    
    default_traditional = (r"Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                          r"F92041BC-2503-4150-8196-2B45C0258ED8_clean_events.txt")
    default_ml = (r"Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                 r"F92041BC-2503-4150-8196-2B45C0258ED8_clean_events_ml.txt")
    default_csv = (r"clean_report\Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                  r"F92041BC-2503-4150-8196-2B45C0258ED8_clean.csv")
    
    parser.add_argument('--traditional', type=str, default=default_traditional,
                       help='传统检测的事件文件')
    parser.add_argument('--ml', type=str, default=default_ml,
                       help='ML检测的事件文件')
    parser.add_argument('--csv', type=str, default=default_csv,
                       help='原始CSV数据文件')
    parser.add_argument('--tolerance', type=float, default=0.3,
                       help='事件匹配容差(秒)')
    parser.add_argument('--out_dir', type=str, default='detection_comparison',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.traditional):
        raise FileNotFoundError(f"传统检测文件不存在: {args.traditional}")
    if not os.path.exists(args.ml):
        raise FileNotFoundError(f"ML检测文件不存在: {args.ml}")
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV文件不存在: {args.csv}")
    
    # 加载数据
    print("[INFO] 加载传统检测事件...")
    events_traditional = load_events(args.traditional)
    print(f"  传统检测事件数: {len(events_traditional)}")
    
    print("[INFO] 加载ML检测事件...")
    events_ml = load_events(args.ml)
    print(f"  ML检测事件数: {len(events_ml)}")
    
    print("[INFO] 加载原始数据...")
    csv_data = pd.read_csv(args.csv)
    print(f"  数据行数: {len(csv_data)}")
    
    # 匹配事件
    print(f"\n[INFO] 匹配事件 (容差={args.tolerance}s)...")
    matched, only_traditional, only_ml = match_events(
        events_traditional, events_ml, args.tolerance
    )
    
    print(f"  匹配成功: {len(matched)}")
    print(f"  ML漏检: {len(only_traditional)}")
    print(f"  ML独有: {len(only_ml)}")
    
    # 分析
    print("\n[INFO] 分析检测质量...")
    stats = analyze_detection_quality(matched, events_traditional, 
                                      events_ml, only_traditional, only_ml)
    
    # 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 可视化
    visualize_event_comparison(events_traditional, events_ml, matched,
                               only_traditional, only_ml, csv_data, args.out_dir)
    
    visualize_sample_differences(events_traditional, events_ml,
                                only_traditional, only_ml, csv_data, 
                                args.out_dir, num_samples=5)
    
    # 生成报告
    generate_report(stats, args.out_dir)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("对比分析摘要")
    print("=" * 70)
    print(f"传统检测: {stats['n_traditional']:4d} 个事件")
    print(f"ML检测:   {stats['n_ml']:4d} 个事件")
    print(f"匹配:     {stats['n_matched']:4d} 个")
    print(f"ML漏检:  {stats['n_missing']:4d} 个")
    print(f"ML独有:  {stats['n_extra']:4d} 个")
    print(f"\n召回率:   {stats['recall']*100:5.2f}%")
    print(f"精确率:   {stats['precision']*100:5.2f}%")
    print(f"F1分数:   {stats['f1_score']:5.3f}")
    print(f"平均误差: {stats['mean_time_error']*1000:6.2f} ms")
    print("=" * 70)
    
    print(f"\n[完成] 所有分析结果已保存到: {args.out_dir}")


if __name__ == '__main__':
    main()
