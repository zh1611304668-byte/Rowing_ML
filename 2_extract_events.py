#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


TIME_COL_PRIORITY = [
    "time",
    "accelerometer_timestamp_since_reboot",
    "gyro_timestamp_since_reboot",
    "motion_timestamp_since_reboot",
    "log_time",
    "location_timestamp_since_1970",
    "timestamp",
]

ACC_CANDIDATES = [
    ("acc_dyn_x", "acc_dyn_y", "acc_dyn_z"),
    ("motion_user_acceleration_x", "motion_user_acceleration_y",
     "motion_user_acceleration_z"),
    ("accelerometer_acceleration_x", "accelerometer_acceleration_y",
     "accelerometer_acceleration_z"),
    ("accel_x", "accel_y", "accel_z"),
    ("ax", "ay", "az"),
]

SPEED_CANDIDATES = [
    "location_speed",
    "speed",
]


@dataclass
class AlgoParams:
    min_rms_mag: float = 0.05  # 优化: 提高RMS阈值过滤噪声
    min_speed_mps: float = 0.5  # 优化: 提高速度阈值
    activity_gate: str = "and"  # 优化: 速度和加速度都需满足
    rms_window_sec: float = 0.5
    segment_min_sec: float = 0.0
    segment_gap_sec: float = 1.0


def pick_columns(df: pd.DataFrame, candidates: List[Tuple[str, str, str]]) -> Optional[Tuple[str, str, str]]:
    for trio in candidates:
        if all(c in df.columns for c in trio):
            return trio
    return None


def pick_time_col(df: pd.DataFrame, override: Optional[str]) -> Optional[str]:
    if override:
        return override if override in df.columns else None
    for c in TIME_COL_PRIORITY:
        if c in df.columns:
            return c
    return None


def estimate_sample_rate(time_s: np.ndarray) -> float:
    dt = np.diff(time_s)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return 100.0
    return float(1.0 / np.median(dt))


def compute_rms(acc: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return np.linalg.norm(acc, axis=1)
    rms = np.zeros(len(acc))
    sq = np.sum(acc * acc, axis=1)
    csum = np.cumsum(sq)
    for i in range(len(acc)):
        start = max(0, i - win + 1)
        total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        rms[i] = np.sqrt(total / (i - start + 1))
    return rms


def compute_active_axis(time_s: np.ndarray,
                        acc: np.ndarray,
                        interval_ms: float,
                        window_sec: float) -> np.ndarray:
    n = len(time_s)
    if n == 0:
        return np.zeros(0, dtype=int)

    if interval_ms <= 0:
        axis = int(np.argmax(np.std(acc, axis=0)))
        return np.full(n, axis, dtype=int)

    interval_sec = interval_ms / 1000.0
    window_sec = max(window_sec, 0.01)
    csum = np.vstack([np.zeros(3), np.cumsum(acc, axis=0)])
    csum_sq = np.vstack([np.zeros(3), np.cumsum(acc * acc, axis=0)])

    active_axis = 1
    next_select_time = time_s[0]
    axis_idx = np.zeros(n, dtype=int)

    for i, t in enumerate(time_s):
        if t >= next_select_time:
            window_start = t - window_sec
            start_idx = int(np.searchsorted(time_s, window_start, side="left"))
            end_idx = i + 1
            count = end_idx - start_idx
            if count > 1:
                sums = csum[end_idx] - csum[start_idx]
                sumsqs = csum_sq[end_idx] - csum_sq[start_idx]
                mean = sums / count
                var = (sumsqs - (sums * sums) / count) / (count - 1)
                stds = np.sqrt(np.maximum(var, 0.0))
                active_axis = int(np.argmax(stds))
            next_select_time = t + interval_sec
        axis_idx[i] = active_axis

    return axis_idx


def build_detection_signal(time_s: np.ndarray,
                           acc: np.ndarray,
                           mode: str,
                           axis_interval_ms: float,
                           axis_window_sec: float,
                           polarity: str,
                           fixed_axis: Optional[int] = None,
                           invert_axis: bool = False):
    if mode == "active_axis":
        if fixed_axis is not None:
            axis_idx = np.full(len(acc), fixed_axis, dtype=int)
        else:
            axis_idx = compute_active_axis(time_s, acc, axis_interval_ms, axis_window_sec)
        signal = acc[np.arange(len(acc)), axis_idx]
        if invert_axis:
            signal = -signal
        if polarity == "positive":
            signal = np.where(signal > 0.0, signal, 0.0)
        elif polarity == "negative":
            signal = np.where(signal < 0.0, -signal, 0.0)
        else:
            signal = np.abs(signal)
        return signal, axis_idx

    return np.linalg.norm(acc, axis=1), None


def smooth_signal(signal: np.ndarray, fs: float, window_sec: float) -> np.ndarray:
    """Simple moving-average smoothing to stabilize peak detection."""
    if window_sec <= 0:
        return signal
    win_samples = max(1, int(round(window_sec * fs)))
    if win_samples <= 1:
        return signal
    kernel = np.ones(win_samples, dtype=float) / float(win_samples)
    return np.convolve(signal, kernel, mode="same")


def parse_fixed_axis(val: Optional[str]) -> Optional[int]:
    """Parse user-specified axis selector into 0/1/2 or None."""
    if val is None:
        return None
    mapping = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}
    if isinstance(val, str):
        lower_val = val.strip().lower()
        if lower_val in mapping:
            return mapping[lower_val]
        if lower_val.isdigit():
            return int(lower_val)
    try:
        return int(val)
    except Exception:
        return None


def build_active_segments(time_s: np.ndarray,
                          rms_mag: np.ndarray,
                          speed_vals: Optional[np.ndarray],
                          params: AlgoParams) -> list:
    segments = []
    start_t = None
    last_active_t = None

    for i, t in enumerate(time_s):
        rms_active = (rms_mag[i] >= params.min_rms_mag)
        if speed_vals is None:
            is_active = rms_active
        else:
            speed_active = (speed_vals[i] >= params.min_speed_mps)
            if params.activity_gate == "and":
                is_active = rms_active and speed_active
            else:
                is_active = rms_active or speed_active

        if is_active:
            if start_t is None:
                start_t = t
            last_active_t = t
        else:
            if start_t is not None and last_active_t is not None:
                if (t - last_active_t) > params.segment_gap_sec:
                    end_t = last_active_t
                    if (end_t - start_t) >= params.segment_min_sec:
                        segments.append((start_t, end_t))
                    start_t = None
                    last_active_t = None

    if start_t is not None and last_active_t is not None:
        if (last_active_t - start_t) >= params.segment_min_sec:
            segments.append((start_t, last_active_t))

    return segments


def extract_events_find_peaks(time_s: np.ndarray,
                              acc: np.ndarray,
                              params: AlgoParams,
                              speed_vals: Optional[np.ndarray] = None,
                              segments: Optional[list] = None,
                              prominence: Optional[float] = None,
                              height: Optional[float] = None,
                              distance_ms: Optional[float] = None,
                              width_ms: Optional[float] = None,
                              signal_mode: str = "mag",
                              axis_interval_ms: float = 1000.0,
                              axis_window_sec: float = 1.0,
                              polarity: str = "positive",
                              smooth_window_sec: float = 0.25,
                              use_adaptive_thresholds: bool = True,
                              fixed_axis: Optional[int] = None,
                              invert_axis: bool = False):
    try:
        from scipy.signal import find_peaks
    except Exception as exc:
        raise RuntimeError("scipy is required for --detector peaks") from exc

    fs_est = estimate_sample_rate(time_s)
    signal, axis_idx = build_detection_signal(
        time_s, acc, signal_mode, axis_interval_ms, axis_window_sec, polarity,
        fixed_axis=fixed_axis, invert_axis=invert_axis
    )
    # Smooth to reduce small fluctuations that hide true peaks.
    signal = smooth_signal(signal, fs_est, smooth_window_sec)

    distance = None
    if distance_ms is not None and distance_ms > 0:
        distance = max(1, int(round(distance_ms / 1000.0 * fs_est)))
    width = None
    if width_ms is not None and width_ms > 0:
        width = max(1, int(round(width_ms / 1000.0 * fs_est)))

    # Auto-set thresholds when users leave them at zero/None; improves recall on weak strokes.
    adaptive_height = None
    adaptive_prom = None
    if use_adaptive_thresholds and (not prominence or prominence <= 0 or not height or height <= 0):
        median_val = float(np.median(signal))
        mad = float(np.median(np.abs(signal - median_val)))
        noise = 1.4826 * mad  # robust std estimate
        adaptive_height = median_val + max(0.02, 1.05 * noise)
        adaptive_prom = max(0.015, 0.9 * noise)
        if not height or height <= 0:
            height = adaptive_height
        if not prominence or prominence <= 0:
            prominence = adaptive_prom

    peaks, _ = find_peaks(
        signal,
        prominence=prominence if prominence and prominence > 0 else None,
        height=height if height and height > 0 else None,
        distance=distance,
        width=width,
    )
    raw_peaks = int(len(peaks))
    if raw_peaks == 0:
        return np.array([]), 0, {"peaks_raw": 0, "peaks_filtered": 0}

    active_mask = None
    if params.min_rms_mag > 0 or speed_vals is not None:
        win = max(1, int(round(params.rms_window_sec * fs_est)))
        rms_series = compute_rms(acc, win)
        rms_active = rms_series >= params.min_rms_mag
        if speed_vals is None:
            active_mask = rms_active
        else:
            speed_active = speed_vals >= params.min_speed_mps
            if params.activity_gate == "and":
                active_mask = rms_active & speed_active
            else:
                active_mask = rms_active | speed_active

    if segments is not None and len(segments) > 0:
        seg_mask = np.zeros_like(time_s, dtype=bool)
        for start_t, end_t in segments:
            seg_mask |= (time_s >= start_t) & (time_s <= end_t)
        active_mask = seg_mask if active_mask is None else (active_mask & seg_mask)

    if active_mask is not None:
        peaks = np.asarray([p for p in peaks if active_mask[p]], dtype=int)

    events_ms = time_s[peaks] * 1000.0
    debug_info = {
        "peaks_raw": raw_peaks,
        "peaks_filtered": int(len(peaks)),
        "fs_est": float(fs_est),
        "signal_mode": signal_mode,
        "polarity": polarity,
        "smooth_window_sec": float(smooth_window_sec),
    }
    if axis_idx is not None:
        debug_info["active_axis_changes"] = int(np.sum(np.diff(axis_idx) != 0))
    if adaptive_height is not None or adaptive_prom is not None:
        debug_info["adaptive_height"] = adaptive_height
        debug_info["adaptive_prominence"] = adaptive_prom
    if fixed_axis is not None:
        debug_info["fixed_axis"] = int(fixed_axis)
        debug_info["invert_axis"] = bool(invert_axis)
    return events_ms, int(len(events_ms)), debug_info


def visualize_detected_events(time_s: np.ndarray, 
                              acc: np.ndarray,
                              acc_cols: Tuple[str, str, str],
                              events_s: np.ndarray,
                              out_dir: str = 'event_detection_vis',
                              num_samples: int = 10):
    """可视化检测到的划桨事件"""
    os.makedirs(out_dir, exist_ok=True)
    
    if len(events_s) == 0:
        print("[WARNING] 没有检测到事件，跳过可视化")
        return
    
    num_samples = min(num_samples, len(events_s))
    sampled_events = np.random.choice(events_s, num_samples, replace=False)
    
    print(f"[INFO] 检测到 {len(events_s)} 个事件，随机采样 {num_samples} 个进行可视化")
    
    for i, event_time in enumerate(sampled_events,1):
        # 提取10秒时间窗口
        sample_duration_sec = 10.0
        start_time = event_time - sample_duration_sec / 2
        end_time = event_time + sample_duration_sec / 2
        
        mask = (time_s >= start_time) & (time_s <= end_time)
        if not np.any(mask):
            continue
        
        time_sample = time_s[mask]
        acc_sample = acc[mask]
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # === 上图：三轴加速度 ===
        colors_axis = ['#1976D2', '#388E3C', '#D32F2F']  # 蓝、绿、红
        for ax_idx, ax_name in enumerate(['X', 'Y', 'Z']):
            axes[0].plot(time_sample, acc_sample[:, ax_idx], 
                        label=f'{ax_name}轴', linewidth=1.8, 
                        color=colors_axis[ax_idx], alpha=0.9)
        
        # 标记检测到的事件
        axes[0].axvline(event_time, color='#FF1744', linestyle='--', 
                       linewidth=2.5, label='检测事件', zorder=10)
        
        # 标记窗口内的其他事件
        other_events = events_s[(events_s >= start_time) & 
                                (events_s <= end_time) & 
                                (events_s != event_time)]
        for other_t in other_events:
            axes[0].axvline(other_t, color='#FF6F00', linestyle=':', 
                          linewidth=1.5, alpha=0.7)
        
        axes[0].set_ylabel('加速度 (g)', fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=11, framealpha=0.9)
        axes[0].grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        axes[0].set_title(
            f'事件检测可视化 {i}/{num_samples}: 事件 @ {event_time:.2f}s',
            fontsize=15, fontweight='bold', pad=15
        )
        
        # === 下图：加速度幅值 ===
        acc_mag = np.linalg.norm(acc_sample, axis=1)
        
        axes[1].plot(time_sample, acc_mag, color='#0D47A1', 
                    linewidth=2.5, label='加速度幅值', zorder=2)
        
        # 填充事件附近区域（±200ms）
        event_highlight_width = 0.2  # 200ms
        axes[1].axvspan(event_time - event_highlight_width, 
                       event_time + event_highlight_width,
                       color='#FF5252', alpha=0.2, zorder=1,
                       label='检测区域(±200ms)')
        
        # 标记检测到的事件
        axes[1].axvline(event_time, color='#FF1744', linestyle='--', 
                       linewidth=2.5, label='检测事件', zorder=10)
        
        # 标记其他事件
        for other_t in other_events:
            axes[1].axvline(other_t, color='#FF6F00', linestyle=':', 
                          linewidth=1.5, alpha=0.7, zorder=9)
        
        axes[1].set_xlabel('时间 (s)', fontsize=13, fontweight='bold')
        axes[1].set_ylabel('加速度幅值 (g)', fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=11, framealpha=0.9)
        axes[1].grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'event_detection_{i:02d}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if i % 5 == 0 or i == num_samples:
            print(f"[{i}/{num_samples}] 已生成可视化图片")
    
    print(f"[INFO] 事件检测可视化已保存到: {out_dir} ({num_samples}个样本)")


def main():
    default_csv_path = ("D:/Desktop/python/rowing_ML/clean_report/"
                        "Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                        "F92041BC-2503-4150-8196-2B45C0258ED8_clean.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default=default_csv_path)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--time_unit", type=str, default="s", choices=["s", "ms"])

    # 关键检测参数设在代码里，开箱即用无需额外命令行
    parser.add_argument("--min_rms_mag", type=float, default=0.06)
    parser.add_argument("--min_speed_mps", type=float, default=0.6)
    parser.add_argument("--activity_gate", type=str, default="and", choices=["or", "and"])
    parser.add_argument("--rms_window_sec", type=float, default=0.5)
    parser.add_argument("--segment_min_sec", type=float, default=0.0)
    parser.add_argument("--segment_gap_sec", type=float, default=1.0)
    parser.add_argument("--peak_prominence", type=float, default=0.07)
    parser.add_argument("--peak_height", type=float, default=0.15)
    parser.add_argument("--peak_distance_ms", type=float, default=1000.0)
    parser.add_argument("--peak_width_ms", type=float, default=60.0)
    parser.add_argument("--peak_signal", type=str, default="active_axis",
                        choices=["mag", "active_axis"])
    parser.add_argument("--peak_polarity", type=str, default="positive",
                        choices=["positive", "negative", "both"])
    parser.add_argument("--active_axis_interval_ms", type=float, default=1000.0)
    parser.add_argument("--active_axis_window_sec", type=float, default=1.0)
    parser.add_argument("--fixed_axis", type=str, default="y",
                        help="Force axis for detection: x/y/z/0/1/2. Useful to lock to drive-axis.")
    parser.add_argument("--invert_axis", action="store_true",
                        help="Flip sign of fixed axis if your sensor frame is reversed.")
    parser.add_argument("--smooth_window_sec", type=float, default=0.25,
                        help="Moving-average window (sec) applied before peak finding.")
    parser.add_argument("--disable_adaptive_thresholds", action="store_true",
                        help="Disable auto height/prominence based on signal noise.")
    parser.add_argument("--print_stats", action="store_true", default=True)
    parser.add_argument("--debug_counts", action="store_true", default=True)
    parser.add_argument("--visualize", action="store_true", default=True,
                       help='生成事件检测可视化（默认开启）')
    parser.add_argument("--num_vis_samples", type=int, default=5,
                       help='可视化样本数量')

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"csv_path not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    time_col = pick_time_col(df, args.time_col)
    if time_col is None:
        raise ValueError("No valid time column found.")

    acc_cols = pick_columns(df, ACC_CANDIDATES)
    if acc_cols is None:
        raise ValueError("No acceleration columns found.")

    speed_col = None
    for c in SPEED_CANDIDATES:
        if c in df.columns:
            speed_col = c
            break

    df = df.sort_values(time_col).reset_index(drop=True)
    time_s = df[time_col].astype(float).values
    if time_col != "time":
        time_s = time_s - time_s[0]

    acc = df[list(acc_cols)].astype(float).values
    speed_vals = None
    if speed_col is not None:
        speed_series = pd.to_numeric(df[speed_col], errors="coerce")
        speed_series = speed_series.interpolate(limit_direction="both").fillna(0.0)
        speed_vals = speed_series.values.astype(float)

    # 使用命令行参数或优化后的默认值
    params = AlgoParams(
        min_rms_mag=args.min_rms_mag,
        min_speed_mps=args.min_speed_mps,
        activity_gate=args.activity_gate,
        rms_window_sec=args.rms_window_sec,
        segment_min_sec=args.segment_min_sec,
        segment_gap_sec=args.segment_gap_sec,
    )

    if args.print_stats:
        acc_mag = np.linalg.norm(acc, axis=1)
        print("acc_mag stats:",
              f"median={np.median(acc_mag):.6f},",
              f"p95={np.percentile(acc_mag, 95):.6f},",
              f"max={np.max(acc_mag):.6f}")
        print("axis std:", np.std(acc[:, 0]), np.std(acc[:, 1]), np.std(acc[:, 2]))
        if args.fixed_axis is not None:
            print("Using fixed axis:",
                  {"0": "X", "1": "Y", "2": "Z", "x": "X", "y": "Y", "z": "Z"}.get(args.fixed_axis, args.fixed_axis),
                  "invert" if args.invert_axis else "")

    segments = None
    if params.segment_min_sec > 0.0:
        dt = np.diff(time_s)
        fs = 1.0 / np.median(dt[dt > 0]) if np.any(dt > 0) else 100.0
        win = max(1, int(round(params.rms_window_sec * fs)))
        rms_series = compute_rms(acc, win)
        segments = build_active_segments(time_s, rms_series, speed_vals, params)
        if args.print_stats:
            total_active = sum((e - s) for s, e in segments)
            print(f"segments: {len(segments)} total_active_sec={total_active:.1f}")

    events_ms, count, debug_info = extract_events_find_peaks(
        time_s, acc, params, speed_vals=speed_vals, segments=segments,
        prominence=args.peak_prominence,
        height=args.peak_height,
        distance_ms=args.peak_distance_ms,
        width_ms=args.peak_width_ms,
        signal_mode=args.peak_signal,
        axis_interval_ms=args.active_axis_interval_ms,
        axis_window_sec=args.active_axis_window_sec,
        polarity=args.peak_polarity,
        smooth_window_sec=args.smooth_window_sec,
        use_adaptive_thresholds=not args.disable_adaptive_thresholds,
        fixed_axis=parse_fixed_axis(args.fixed_axis),
        invert_axis=args.invert_axis,
    )
    if args.time_unit == "s":
        events_out = events_ms / 1000.0
    else:
        events_out = events_ms

    if args.out_path is None:
        base = os.path.splitext(os.path.basename(args.csv_path))[0]
        out_path = f"{base}_events.txt"
    else:
        out_path = args.out_path

    np.savetxt(out_path, events_out, fmt="%.6f")
    print("Events saved:", out_path)
    print("Event count:", int(count))
    if debug_info is not None and args.debug_counts:
        if segments is not None:
            debug_info["segment_count"] = len(segments)
            debug_info["segment_total_sec"] = float(sum((e - s) for s, e in segments))
        print("Debug:", debug_info)
    if count == 0:
        print("Warning: event count is 0.")
        print("  Try lowering --peak_height or --peak_prominence.")
    
    # 生成可视化
    if args.visualize and count > 0:
        visualize_detected_events(time_s, acc, acc_cols, events_out, 
                                 'event_detection_vis', args.num_vis_samples)


if __name__ == "__main__":
    main()
