#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ACC_CANDIDATES = [
    ("acc_dyn_x", "acc_dyn_y", "acc_dyn_z"),
    ("motion_user_acceleration_x", "motion_user_acceleration_y", "motion_user_acceleration_z"),
    ("accelerometer_acceleration_x", "accelerometer_acceleration_y", "accelerometer_acceleration_z"),
]

SPEED_CANDIDATES = ["location_speed", "speed"]


def pick_columns(df: pd.DataFrame, candidates) -> Optional[Tuple[str, str, str]]:
    for trio in candidates:
        if all(c in df.columns for c in trio):
            return trio
    return None


def load_events(path: str, unit: str) -> np.ndarray:
    ev = np.loadtxt(path, ndmin=1).astype(float)
    if unit == "s":
        return ev
    if unit == "ms":
        return ev / 1000.0
    # auto
    if np.median(ev) > 1e5:
        return ev / 1000.0
    return ev


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


def main():
    parser = argparse.ArgumentParser()
    default_csv_path = ("D:/Desktop/python/rowing_ML/clean_report/"
                        "Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                        "F92041BC-2503-4150-8196-2B45C0258ED8_clean.csv")
    default_event_path = ("D:/Desktop/python/rowing_ML/"
                          "Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                          "F92041BC-2503-4150-8196-2B45C0258ED8_clean_events.txt")
    parser.add_argument("--csv_path", type=str, default=default_csv_path)
    parser.add_argument("--event_path", type=str, default=default_event_path)
    parser.add_argument("--event_unit", type=str, default="auto", choices=["auto", "s", "ms"])
    parser.add_argument("--out_dir", type=str, default="event_check")
    parser.add_argument("--rms_window_sec", type=float, default=0.5)
    parser.add_argument("--rms_threshold", type=float, default=0.05)
    parser.add_argument("--page_sec", type=float, default=12.0)
    parser.add_argument("--num_pages", type=int, default=-1,
                        help="单个事件截图数量，<=0 表示所有事件都画出来")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"csv_path not found: {args.csv_path}")
    if not os.path.exists(args.event_path):
        raise FileNotFoundError(f"event_path not found: {args.event_path}")

    df = pd.read_csv(args.csv_path)
    if "time" not in df.columns:
        raise ValueError("CSV must have a 'time' column.")

    acc_cols = pick_columns(df, ACC_CANDIDATES)
    if acc_cols is None:
        raise ValueError("No acceleration columns found.")

    speed_col = None
    for c in SPEED_CANDIDATES:
        if c in df.columns:
            speed_col = c
            break

    time_s = df["time"].values.astype(float)
    acc = df[list(acc_cols)].values.astype(float)
    speed = df[speed_col].values.astype(float) if speed_col else None

    events = load_events(args.event_path, args.event_unit)
    events = events[(events >= time_s[0]) & (events <= time_s[-1])]

    # interval histogram
    if len(events) >= 2:
        intervals = np.diff(np.sort(events))
        plt.figure(figsize=(8, 4))
        plt.hist(intervals, bins=80, color="steelblue")
        plt.title("Event interval histogram")
        plt.xlabel("interval (s)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "interval_hist.png"))
        plt.close()

    # overlay plot
    acc_mag = np.linalg.norm(acc, axis=1)
    plt.figure(figsize=(12, 4))
    plt.plot(time_s, acc_mag, linewidth=1, label="|acc_dyn|")
    if speed is not None:
        plt.plot(time_s, speed / max(np.max(speed), 1e-6), linewidth=1, label="speed (norm)")
    for t in events:
        plt.axvline(t, color="r", alpha=0.2)
    plt.title("Events overlay on acc_mag (and speed)")
    plt.xlabel("time (s)")
    plt.ylabel("magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "events_overlay.png"))
    plt.close()

    # active vs inactive event density
    fs = 1.0 / np.median(np.diff(time_s))
    win = max(1, int(round(args.rms_window_sec * fs)))
    rms = compute_rms(acc, win)
    active_mask = rms >= args.rms_threshold

    # count events in active vs inactive time
    active_time = np.sum(active_mask) / fs
    inactive_time = (len(active_mask) - np.sum(active_mask)) / fs

    active_events = 0
    inactive_events = 0
    for t in events:
        idx = np.searchsorted(time_s, t)
        idx = min(max(idx, 0), len(active_mask) - 1)
        if active_mask[idx]:
            active_events += 1
        else:
            inactive_events += 1

    report = {
        "event_count": int(len(events)),
        "active_time_sec": float(active_time),
        "inactive_time_sec": float(inactive_time),
        "active_events": int(active_events),
        "inactive_events": int(inactive_events),
        "active_event_rate_per_min": float(60.0 * active_events / max(active_time, 1e-6)),
        "inactive_event_rate_per_min": float(60.0 * inactive_events / max(inactive_time, 1e-6)),
    }

    with open(os.path.join(args.out_dir, "event_quality.json"), "w", encoding="utf-8") as f:
        f.write(pd.Series(report).to_json())

    # event pages (one image per event)
    half = args.page_sec / 2
    page_limit = len(events) if args.num_pages is None or args.num_pages <= 0 else args.num_pages
    for i, t in enumerate(events[:page_limit]):
        mask = (time_s >= t - half) & (time_s <= t + half)
        if not np.any(mask):
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(time_s[mask], acc_mag[mask], label="|acc_dyn|", linewidth=1)
        plt.axvline(t, color="r", linestyle="--", label="event")
        plt.title(f"Event page {i+1} @ {t:.2f}s")
        plt.xlabel("time (s)")
        plt.ylabel("acc magnitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"event_page_{i+1:02d}.png"))
        plt.close()
    print(f"Saved per-event pages: {min(page_limit, len(events))} (page_sec={args.page_sec}s)")

    # === NEW: Stroke Cycle Visualization ===
    # Calculate average stroke rate and identify anomalies
    if len(events) >= 2:
        intervals = np.diff(np.sort(events))
        median_interval = np.median(intervals)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Identify anomalies (intervals > 1.5x median or < 0.5x median)
        anomaly_threshold_high = median_interval * 1.5
        anomaly_threshold_low = median_interval * 0.5
        
        print(f"\n=== Stroke Cycle Statistics ===")
        print(f"Total events: {len(events)}")
        print(f"Median interval: {median_interval:.3f}s ({60/median_interval:.1f} strokes/min)")
        print(f"Mean interval: {mean_interval:.3f}s (±{std_interval:.3f}s)")
        print(f"Anomaly thresholds: {anomaly_threshold_low:.3f}s - {anomaly_threshold_high:.3f}s")
        
        # Create cycle overview plots showing multiple strokes
        # Show 20-30 seconds windows to capture ~10-15 strokes
        cycle_window_sec = 30.0
        num_cycle_plots = min(5, int((time_s[-1] - time_s[0]) / cycle_window_sec))
        
        sorted_events = np.sort(events)
        
        for plot_idx in range(num_cycle_plots):
            # Find a good starting point with events
            start_time = time_s[0] + plot_idx * cycle_window_sec
            end_time = start_time + cycle_window_sec
            
            # Get events in this window
            window_events = sorted_events[(sorted_events >= start_time) & (sorted_events <= end_time)]
            
            if len(window_events) < 2:
                continue
            
            # Create figure with 2 subplots: 3-axis and magnitude
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
            
            # Mask for this time window
            mask = (time_s >= start_time) & (time_s <= end_time)
            t_window = time_s[mask]
            acc_window = acc[mask]
            
            # Upper plot: 3-axis acceleration
            ax1.plot(t_window, acc_window[:, 0], label=f'{acc_cols[0]} (X)', linewidth=1.2, alpha=0.8)
            ax1.plot(t_window, acc_window[:, 1], label=f'{acc_cols[1]} (Y)', linewidth=1.2, alpha=0.8)
            ax1.plot(t_window, acc_window[:, 2], label=f'{acc_cols[2]} (Z)', linewidth=1.2, alpha=0.8)
            ax1.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
            ax1.set_ylabel('Acceleration (g)', fontsize=11, fontweight='bold')
            ax1.set_title(f'划桨周期详细分析 #{plot_idx+1} | Time: {start_time:.1f}s - {end_time:.1f}s', 
                         fontsize=13, fontweight='bold', pad=15)
            ax1.legend(loc='upper right', fontsize=9)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Lower plot: magnitude with event markers
            acc_mag_window = np.linalg.norm(acc_window, axis=1)
            ax2.plot(t_window, acc_mag_window, color='navy', linewidth=1.5, label='|Acceleration|')
            
            # Mark events and annotate intervals
            for i, evt_time in enumerate(window_events):
                # Draw vertical line for event
                ax1.axvline(evt_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
                ax2.axvline(evt_time, color='red', linestyle='--', linewidth=2, alpha=0.7)
                
                # Annotate interval to next event
                if i < len(window_events) - 1:
                    next_evt = window_events[i + 1]
                    interval = next_evt - evt_time
                    mid_time = (evt_time + next_evt) / 2
                    
                    # Determine if interval is anomalous
                    is_anomaly = (interval > anomaly_threshold_high) or (interval < anomaly_threshold_low)
                    color = 'red' if is_anomaly else 'darkgreen'
                    weight = 'bold' if is_anomaly else 'normal'
                    
                    # Add interval annotation
                    spm = 60 / interval  # strokes per minute
                    label = f'{interval:.2f}s\n{spm:.1f}spm'
                    
                    # Find y position for annotation (slightly above max in this interval)
                    interval_mask = (t_window >= evt_time) & (t_window <= next_evt)
                    if np.any(interval_mask):
                        y_pos = np.max(acc_mag_window[interval_mask]) * 1.1
                    else:
                        y_pos = np.max(acc_mag_window) * 0.9
                    
                    ax2.annotate(label, xy=(mid_time, y_pos), 
                               ha='center', va='bottom',
                               fontsize=10, fontweight=weight, color=color,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow' if is_anomaly else 'lightgreen', 
                                        alpha=0.7, edgecolor=color, linewidth=1.5))
            
            ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('|Acceleration| (g)', fontsize=11, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=9)
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            # Add text box with cycle statistics
            stats_text = f'Events in window: {len(window_events)}\n'
            stats_text += f'Median interval: {median_interval:.3f}s\n'
            stats_text += f'Expected rate: {60/median_interval:.1f} spm'
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"cycle_analysis_{plot_idx+1:02d}.png"), dpi=150)
            plt.close()
            
            print(f"Saved cycle analysis plot {plot_idx+1} with {len(window_events)} events")
        
        # Create a summary plot showing all intervals
        plt.figure(figsize=(14, 6))
        interval_times = sorted_events[:-1]  # x-axis: time of each stroke
        plt.plot(interval_times, intervals, 'o-', linewidth=2, markersize=6, 
                color='steelblue', label='Stroke Interval')
        plt.axhline(median_interval, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_interval:.3f}s')
        plt.axhline(anomaly_threshold_high, color='red', linestyle=':', linewidth=1.5, 
                   label=f'High threshold: {anomaly_threshold_high:.3f}s')
        plt.axhline(anomaly_threshold_low, color='red', linestyle=':', linewidth=1.5,
                   label=f'Low threshold: {anomaly_threshold_low:.3f}s')
        
        # Highlight anomalies
        anomaly_mask = (intervals > anomaly_threshold_high) | (intervals < anomaly_threshold_low)
        if np.any(anomaly_mask):
            plt.scatter(interval_times[anomaly_mask], intervals[anomaly_mask], 
                       color='red', s=100, marker='X', zorder=5, label='Anomalies')
        
        plt.title('划桨间隔时间序列 - 漏桨检测', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=11, fontweight='bold')
        plt.ylabel('Interval (s)', fontsize=11, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "interval_timeline.png"), dpi=150)
        plt.close()
        
        print(f"Saved interval timeline plot")
        print(f"Number of anomalous intervals: {np.sum(anomaly_mask)}")

    print(f"\n✓ All visualizations saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
