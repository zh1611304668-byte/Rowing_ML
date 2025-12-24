#!/usr/bin/env python3
"""
自动标注生成器
根据划桨事件时间戳为IMU数据生成训练标签

标签定义:
- 0: 背景/静止
- 1: 划桨准备期 (事件前200ms)
- 2: 划桨核心期 (事件前后±100ms)
- 3: 划桨恢复期 (事件后300ms)
"""

import argparse
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_events(event_path: str, event_unit: str = "auto") -> np.ndarray:
    """加载划桨事件时间戳"""
    events = np.loadtxt(event_path, ndmin=1).astype(float)
    
    # 自动检测单位
    if event_unit == "auto":
        median_val = np.median(events) if len(events) > 0 else 0
        if median_val > 1e5:  # 大概率是毫秒
            events = events / 1000.0
            print(f"[INFO] 自动检测到事件单位为毫秒，已转换为秒")
        else:
            print(f"[INFO] 自动检测到事件单位为秒")
    elif event_unit == "ms":
        events = events / 1000.0
    
    return events


def generate_labels(
    time_s: np.ndarray,
    events: np.ndarray,
    acc_raw: Optional[np.ndarray],  # 改为原始加速度（带正负）
    prepare_window_ms: float = 200.0,
    core_window_ms: float = 100.0,
    recover_window_ms: float = 800.0,
    recover_min_still_ms: float = 200.0,
    recover_smooth_sec: float = 0.10,
    recover_hard_cap_ms: float = 1600.0,
    recover_left_window_ms: float = 250.0,
    recover_drop_ratio: float = 1.5,
    recover_abs_threshold: float = 0.018,
    recover_min_duration_ms: float = 350.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成多类别标签
    
    新逻辑：恢复期从核心期结束开始，到加速度从负值回升至背景平均值时结束
    
    Returns:
        labels: 0=背景, 1=准备, 2=核心, 3=恢复
        stroke_ids: 每个样本所属的划桨编号 (-1表示背景)
    """
    n = len(time_s)
    labels = np.zeros(n, dtype=int)
    stroke_ids = np.full(n, -1, dtype=int)

    # 采样率估计
    dt = np.median(np.diff(time_s)) if len(time_s) > 1 else 0.01
    fs = 1.0 / max(dt, 1e-6)
    min_still_samples = max(1, int(round(recover_min_still_ms / 1000.0 * fs)))
    left_window_samples = max(1, int(round(recover_left_window_ms / 1000.0 * fs)))
    min_duration_samples = max(1, int(round(recover_min_duration_ms / 1000.0 * fs)))

    # 平滑原始加速度信号（全局平滑，用于所有事件）
    acc_smooth = None
    if acc_raw is not None:
        smooth_win = max(1, int(round(recover_smooth_sec * fs))) if recover_smooth_sec > 0 else 1
        if smooth_win > 1:
            kernel = np.ones(smooth_win) / float(smooth_win)
            acc_smooth = np.convolve(acc_raw, kernel, mode="same")
        else:
            acc_smooth = acc_raw.copy()

    events_sorted = np.sort(events)
    total_events = len(events_sorted)
    
    for stroke_id, event_time in enumerate(events_sorted):
        prev_event = events_sorted[stroke_id - 1] if stroke_id > 0 else None
        next_event = events_sorted[stroke_id + 1] if stroke_id + 1 < total_events else None

        # 各阶段时间范围
        prepare_start = event_time - prepare_window_ms / 1000.0
        core_start = event_time - core_window_ms / 1000.0
        core_end = event_time + core_window_ms / 1000.0
        # 恢复期：默认窗口
        default_recover_end = event_time + recover_window_ms / 1000.0
        if next_event is not None:
            next_prepare_start = next_event - prepare_window_ms / 1000.0
            default_recover_end = min(default_recover_end, next_prepare_start)
        # 硬上限，避免过长拖尾
        recover_hard_cap = event_time + recover_hard_cap_ms / 1000.0
        recover_end = min(default_recover_end, recover_hard_cap)
        recover_end = max(recover_end, core_end)  # 至少覆盖核心结束

        # 对齐到时间索引
        prepare_start_idx = np.searchsorted(time_s, prepare_start)
        core_start_idx = np.searchsorted(time_s, core_start)
        core_end_idx = np.searchsorted(time_s, core_end)
        recover_end_idx_default = np.searchsorted(time_s, recover_end)

        # 用于动态阈值的峰值估计（核心+恢复默认窗口内的绝对值峰值）
        if acc_smooth is not None:
            peak_window_end = min(len(acc_smooth) - 1, np.searchsorted(time_s, core_end + recover_window_ms / 1000.0))
            if peak_window_end > core_start_idx:
                local_peak_abs = float(np.max(np.abs(acc_smooth[core_start_idx:peak_window_end])))
            else:
                local_peak_abs = 0.0
        else:
            local_peak_abs = 0.0
        # 动态右侧阈值：随本次振幅变化，限制在 [abs_threshold, 0.05]
        dyn_abs_threshold = min(max(recover_abs_threshold, local_peak_abs * 0.06), 0.05)

        recover_end_idx = recover_end_idx_default
        
        # 简化的恢复期自适应逻辑：
        # 从核心期结束开始，寻找“右侧绝对值均值”的最小窗口，且满足左大右小。
        if acc_smooth is not None:
            acc_abs = np.abs(acc_smooth)
            if next_event is not None:
                search_stop = np.searchsorted(time_s, next_prepare_start)
            else:
                search_stop = len(time_s) - 1
            search_stop = min(search_stop, len(time_s) - 1)
            search_stop = max(search_stop, core_end_idx + min_still_samples)

            best_right_mean = float("inf")
            for idx in range(core_end_idx + min_duration_samples, search_stop - min_still_samples + 1):
                right_slice = acc_abs[idx:idx + min_still_samples]
                right_mean = float(np.mean(right_slice))

                left_start = max(0, idx - left_window_samples)
                left_slice = acc_abs[left_start:idx] if idx > left_start else right_slice
                left_mean = float(np.mean(left_slice))

                # 仅在右侧达到目前为止的最小均值时考虑截断，避免过早切断
                best_right_mean = min(best_right_mean, right_mean)

                # 动态阈值基于本次局部峰值，避免高振幅时阈值过紧
                if (left_mean > right_mean * recover_drop_ratio) and (right_mean < dyn_abs_threshold):
                    recover_end_idx = idx + min_still_samples
                    break

            recover_end_idx = min(recover_end_idx, search_stop)

        # 标注各阶段
        labels[prepare_start_idx:core_start_idx] = 1  # 准备期
        labels[core_start_idx:core_end_idx] = 2       # 核心期
        labels[core_end_idx:recover_end_idx] = 3      # 恢复期

        # 记录所属划桨编号
        stroke_ids[prepare_start_idx:recover_end_idx] = stroke_id
    
    return labels, stroke_ids


def visualize_labels(df: pd.DataFrame, acc_cols: Tuple[str, str, str], 
                     out_dir: str, num_samples: int = 10):
    """增强版可视化标注结果"""
    os.makedirs(out_dir, exist_ok=True)
    
    # 颜色映射 - 使用更鲜明的颜色（含背景填充）
    label_colors = {
        0: '#BBDEFB',  # 浅蓝 - 背景
        1: '#FFF59D',  # 黄色 - 准备
        2: '#FF5252',  # 红色 - 核心
        3: '#FFB74D'   # 橙色 - 恢复
    }
    label_names = {0: '背景', 1: '准备', 2: '核心', 3: '恢复'}
    
    # 随机采样几段数据可视化
    stroke_events = df[df['label'] == 2]['time'].values
    if len(stroke_events) == 0:
        print("[WARNING] 没有找到核心期标签，跳过可视化")
        return
        
    num_samples = min(num_samples, len(stroke_events))
    sampled_events = np.random.choice(stroke_events, num_samples, replace=False)
    
    print(f"[INFO] 找到 {len(stroke_events)} 个划桨事件，随机采样 {num_samples} 个")
    
    for i, event_time in enumerate(sampled_events, 1):
        # 提取10秒时间窗口
        sample_duration_sec = 10.0
        start_time = event_time - sample_duration_sec / 2
        end_time = event_time + sample_duration_sec / 2
        
        mask = (df['time'] >= start_time) & (df['time'] <= end_time)
        df_sample = df[mask].copy()
        
        if len(df_sample) == 0:
            continue
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # ===================
        # 上图：三轴加速度 + 背景区域着色
        # ===================
        
        # 先绘制标签背景区域
        for label_value in [1, 2, 3]:
            label_mask = df_sample['label'].values == label_value
            if not np.any(label_mask):
                continue
            
            time_vals = df_sample['time'].values
            
            # 找到连续区域并绘制
            in_region = False
            region_start = None
            for j in range(len(label_mask)):
                if label_mask[j] and not in_region:
                    region_start = time_vals[j]
                    in_region = True
                elif not label_mask[j] and in_region:
                    axes[0].axvspan(region_start, time_vals[j-1], 
                                  color=label_colors[label_value], 
                                  alpha=0.25, zorder=0)
                    in_region = False
            if in_region:  # 最后一个区域
                axes[0].axvspan(region_start, time_vals[-1], 
                              color=label_colors[label_value], 
                              alpha=0.25, zorder=0)
        
        # 绘制三轴加速度曲线
        colors_axis = ['#1976D2', '#388E3C', '#D32F2F']  # 蓝、绿、红
        for ax_idx, ax_name in enumerate(['X', 'Y', 'Z']):
            axes[0].plot(df_sample['time'], df_sample[acc_cols[ax_idx]], 
                        label=f'{ax_name}轴', linewidth=1.8, 
                        color=colors_axis[ax_idx], alpha=0.9, zorder=1)
        
        # 标记事件时刻
        axes[0].axvline(event_time, color='#FF1744', linestyle='--', 
                       linewidth=2.5, label='事件时刻', zorder=2)
        
        axes[0].set_ylabel('加速度 (g)', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=11, framealpha=0.9)
        axes[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        axes[0].set_title(
            f'样本 {i}/{num_samples}: 划桨事件 @ {event_time:.2f}s  ' + 
            f'[黄=准备, 红=核心, 橙=恢复]', 
            fontsize=15, fontweight='bold', pad=15
        )
        
        # ===================
        # 下图：加速度幅值 + 标签填充
        # ===================
        
        acc_mag = np.linalg.norm(df_sample[list(acc_cols)].values, axis=1)
        
        # 绘制幅值曲线
        axes[1].plot(df_sample['time'], acc_mag, color='#0D47A1', 
                    linewidth=2.5, label='加速度幅值', zorder=2)
        
        # 为不同标签区域填充颜色
        for label_value in [0, 1, 2, 3]:
            label_mask = df_sample['label'] == label_value
            if np.any(label_mask):
                axes[1].fill_between(
                    df_sample['time'], 0, acc_mag,
                    where=label_mask,
                    color=label_colors[label_value],
                    alpha=0.5,
                    label=label_names[label_value],
                    zorder=1
                )
        
        # 标记事件时刻
        axes[1].axvline(event_time, color='#FF1744', linestyle='--', 
                       linewidth=2.5, label='事件时刻', zorder=3)
        
        axes[1].set_xlabel('时间 (s)', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('加速度幅值 (g)', fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=11, ncol=2, framealpha=0.9)
        axes[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'label_vis_sample_{i:02d}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if i % 5 == 0 or i == num_samples:
            print(f"[{i}/{num_samples}] 已生成可视化图片")
    
    print(f"[INFO] 标注可视化已保存到: {out_dir} ({num_samples}个样本)")


def generate_statistics(df: pd.DataFrame, out_dir: str):
    """生成标注统计报告"""
    os.makedirs(out_dir, exist_ok=True)
    
    label_counts = df['label'].value_counts().sort_index()
    total = len(df)
    
    stats = {
        'total_samples': int(total),
        'label_distribution': {
            '背景 (0)': int(label_counts.get(0, 0)),
            '准备 (1)': int(label_counts.get(1, 0)),
            '核心 (2)': int(label_counts.get(2, 0)),
            '恢复 (3)': int(label_counts.get(3, 0)),
        },
        'label_percentage': {
            '背景 (0)': f"{label_counts.get(0, 0) / total * 100:.2f}%",
            '准备 (1)': f"{label_counts.get(1, 0) / total * 100:.2f}%",
            '核心 (2)': f"{label_counts.get(2, 0) / total * 100:.2f}%",
            '恢复 (3)': f"{label_counts.get(3, 0) / total * 100:.2f}%",
        },
        'unique_strokes': int(df['stroke_id'].max() + 1) if len(df) > 0 else 0,
    }
    
    # 保存统计报告
    import json
    with open(os.path.join(out_dir, 'label_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 绘制标签分布图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    labels_list = ['背景', '准备', '核心', '恢复']
    sizes = [label_counts.get(i, 0) for i in range(4)]
    colors = ['lightgray', 'yellow', 'red', 'orange']
    
    axes[0].pie(sizes, labels=labels_list, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('标签分布 (百分比)', fontsize=14)
    
    axes[1].bar(labels_list, sizes, color=colors, edgecolor='black')
    axes[1].set_ylabel('样本数', fontsize=12)
    axes[1].set_title('标签分布 (数量)', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'label_distribution.png'), dpi=150)
    plt.close()
    
    print(f"\n=== 标注统计 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"划桨总数: {stats['unique_strokes']}")
    print("\n标签分布:")
    for label, count in stats['label_distribution'].items():
        pct = stats['label_percentage'][label]
        print(f"  {label}: {count:7d} ({pct})")


def main():
    parser = argparse.ArgumentParser(description='生成训练标签')
    
    default_csv = (r"D:\Desktop\python\rowing_ML\clean_report"
                   r"\Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                   r"F92041BC-2503-4150-8196-2B45C0258ED8_clean.csv")
    default_events = (r"D:\Desktop\python\rowing_ML"
                      r"\Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                      r"F92041BC-2503-4150-8196-2B45C0258ED8_clean_events.txt")
    
    parser.add_argument('--csv_path', type=str, default=default_csv)
    parser.add_argument('--event_path', type=str, default=default_events)
    parser.add_argument('--event_unit', type=str, default='auto', choices=['auto', 's', 'ms'])
    parser.add_argument('--prepare_window_ms', type=float, default=300.0,
                       help='准备期窗口(ms)，在事件前')
    parser.add_argument('--core_window_ms', type=float, default=125.0,
                       help='核心期窗口(ms)，事件前100ms+后150ms=核心期250ms总长')
    parser.add_argument('--recover_window_ms', type=float, default=600.0,
                       help='恢复期窗口(ms)，在事件后；若下一个事件更近，则恢复延续到下一个准备期前')
    parser.add_argument('--recover_min_still_ms', type=float, default=200.0,
                       help='回落阈值需持续的最短时间(ms)')
    parser.add_argument('--recover_smooth_sec', type=float, default=0.10,
                       help='恢复期检测的平滑窗口(s)')
    parser.add_argument('--recover_hard_cap_ms', type=float, default=1200.0,
                       help='恢复期最长持续时间(ms)，防止跨到下一划准备期')
    parser.add_argument('--recover_left_window_ms', type=float, default=250.0,
                       help='左侧窗口时长(ms)，用于比较恢复前后的均值')
    parser.add_argument('--recover_drop_ratio', type=float, default=1.3,
                       help='左侧均值需大于右侧均值的倍数')
    parser.add_argument('--recover_abs_threshold', type=float, default=0.02,
                       help='右侧均值需低于的绝对值阈值(g)')
    parser.add_argument('--recover_min_duration_ms', type=float, default=400.0,
                       help='恢复期搜索最早开始时间(ms)，避免刚结束核心就被截断')
    parser.add_argument('--out_dir', type=str, default='datasets')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='生成标签可视化图片（默认开启）')
    parser.add_argument('--num_vis_samples', type=int, default=10,
                       help='可视化样本数量')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {args.csv_path}")
    if not os.path.exists(args.event_path):
        raise FileNotFoundError(f"事件文件不存在: {args.event_path}")
    
    print("[INFO] 加载数据...")
    df = pd.read_csv(args.csv_path)
    events = load_events(args.event_path, args.event_unit)
    
    # 峰值检测已对齐，无需再做全局时间前移
    peak_offset_sec = 0.0
    if peak_offset_sec != 0.0:
        events = events - peak_offset_sec
        print(f"[INFO] 应用峰值对齐修正：前移 {peak_offset_sec*1000:.0f}ms")
    
    print(f"[INFO] CSV行数: {len(df)}")
    print(f"[INFO] 事件数: {len(events)}")
    
    required_cols = ['time', 'acc_dyn_x', 'acc_dyn_y', 'acc_dyn_z']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV缺少必需列: {missing_cols}")
    
    print("[INFO] 生成标签...")
    time_s = df['time'].values
    
    # 使用原始加速度信号（选择Y轴，通常在划桨中有最明显的前后运动特征）
    acc_raw = df['acc_dyn_y'].values  # 改为使用原始Y轴加速度
    
    labels, stroke_ids = generate_labels(
        time_s, events, acc_raw,
        args.prepare_window_ms,
        args.core_window_ms,
        args.recover_window_ms,
        args.recover_min_still_ms,
        args.recover_smooth_sec,
        args.recover_hard_cap_ms,
        args.recover_left_window_ms,
        args.recover_drop_ratio,
        args.recover_abs_threshold,
        args.recover_min_duration_ms
    )
    
    df['label'] = labels
    df['stroke_id'] = stroke_ids
    
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    out_path = os.path.join(args.out_dir, f"{base_name}_labeled.csv")
    df.to_csv(out_path, index=False)
    print(f"[INFO] 标注数据已保存: {out_path}")
    
    generate_statistics(df, args.out_dir)
    
    if args.visualize:
        print("[INFO] 生成可视化...")
        acc_cols = ('acc_dyn_x', 'acc_dyn_y', 'acc_dyn_z')
        visualize_labels(df, acc_cols, args.out_dir, num_samples=args.num_vis_samples)
    
    print("\n[完成] 标注生成完毕!")


if __name__ == '__main__':
    main()
