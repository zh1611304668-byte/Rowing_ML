#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TIME_COL_PRIORITY = [
    "accelerometer_timestamp_since_reboot",
    "gyro_timestamp_since_reboot",
    "motion_timestamp_since_reboot",
    "log_time",
    "location_timestamp_since_1970",
    "timestamp",
    "time",
]

ACC_CANDIDATES = [
    ("accelerometer_acceleration_x", "accelerometer_acceleration_y",
     "accelerometer_acceleration_z"),
    ("accel_x", "accel_y", "accel_z"),
    ("ax", "ay", "az"),
]

USER_ACC_CANDIDATES = [
    ("motion_user_acceleration_x", "motion_user_acceleration_y",
     "motion_user_acceleration_z"),
    ("user_acc_x", "user_acc_y", "user_acc_z"),
]

GYRO_CANDIDATES = [
    ("gyro_rotation_x", "gyro_rotation_y", "gyro_rotation_z"),
    ("gyroscope_rotation_x", "gyroscope_rotation_y", "gyroscope_rotation_z"),
    ("gyro_x", "gyro_y", "gyro_z"),
]

GRAVITY_CANDIDATES = [
    ("motion_gravity_x", "motion_gravity_y", "motion_gravity_z"),
    ("gravity_x", "gravity_y", "gravity_z"),
]

SPEED_CANDIDATES = [
    "location_speed",
    "speed",
]


@dataclass
class CleanConfig:
    sample_rate: int = 100
    max_gap_sec: float = 0.5
    min_segment_sec: float = 10.0
    acc_unit: str = "auto"  # auto, g, mps2
    max_acc_g: float = 8.0
    remove_gravity: str = "none"  # none, gravity, lpf
    gravity_tau_sec: float = 1.0
    resample: bool = True
    trim_inactive_edges: bool = True
    trim_rms_threshold: float = 0.06
    trim_window_sec: float = 1.0
    trim_min_active_sec: float = 20.0
    trim_pad_sec: float = 2.0


def pick_columns(df: pd.DataFrame, candidates: List[Tuple[str, str, str]]) -> Optional[Tuple[str, str, str]]:
    for trio in candidates:
        if all(c in df.columns for c in trio):
            return trio
    return None


def pick_time_col(df: pd.DataFrame, override: Optional[str]) -> Optional[str]:
    if override:
        return override if override in df.columns else None

    candidates = [c for c in TIME_COL_PRIORITY if c in df.columns]
    if not candidates:
        return None

    best = None
    best_score = None
    for c in candidates:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 2:
            continue
        s = s.sort_values()
        dt = np.diff(s.values)
        dt = dt[dt > 0]
        if len(dt) == 0:
            continue
        median_dt = float(np.median(dt))
        uniq = int(s.nunique())
        score = (median_dt, -uniq)
        if best_score is None or score < best_score:
            best = c
            best_score = score

    return best if best is not None else candidates[0]


def lpf_gravity(acc: np.ndarray, fs: int, tau: float) -> np.ndarray:
    dt = 1.0 / fs
    alpha = dt / (tau + dt)
    g = np.zeros_like(acc)
    g[0] = acc[0]
    for i in range(1, len(acc)):
        g[i] = (1.0 - alpha) * g[i - 1] + alpha * acc[i]
    return g


def compute_dt_stats(dt: np.ndarray) -> dict:
    if len(dt) == 0:
        return {}
    return {
        "count": int(len(dt)),
        "mean": float(np.mean(dt)),
        "median": float(np.median(dt)),
        "p01": float(np.percentile(dt, 1)),
        "p05": float(np.percentile(dt, 5)),
        "p95": float(np.percentile(dt, 95)),
        "p99": float(np.percentile(dt, 99)),
        "max": float(np.max(dt)),
    }


def compute_rms_vector(acc: np.ndarray, win: int) -> np.ndarray:
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


def trim_inactive_edges(df: pd.DataFrame,
                        acc_cols: Tuple[str, str, str],
                        cfg: CleanConfig) -> Tuple[pd.DataFrame, dict]:
    t = df["time"].values.astype(float)
    if len(t) == 0:
        return df, {"trim_applied": False}
    dt = np.diff(t)
    dt = dt[dt > 0]
    fs_est = float(1.0 / np.median(dt)) if len(dt) > 0 else float(cfg.sample_rate)
    win = max(1, int(round(cfg.trim_window_sec * fs_est)))
    acc = df[list(acc_cols)].values.astype(float)
    rms = compute_rms_vector(acc, win)
    active_mask = rms >= cfg.trim_rms_threshold

    segments = []
    start = None
    for i, active in enumerate(active_mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(active_mask) - 1))

    kept = []
    for s, e in segments:
        dur = float(t[e] - t[s])
        if dur >= cfg.trim_min_active_sec:
            kept.append((s, e))

    if not kept:
        return df, {
            "trim_applied": False,
            "trim_reason": "no_active_segment",
        }

    trim_start = max(t[0], float(t[kept[0][0]] - cfg.trim_pad_sec))
    trim_end = min(t[-1], float(t[kept[-1][1]] + cfg.trim_pad_sec))
    mask = (t >= trim_start) & (t <= trim_end)
    df_trim = df[mask].copy()
    if len(df_trim) > 0:
        df_trim["time"] = df_trim["time"] - float(df_trim["time"].iloc[0])

    info = {
        "trim_applied": True,
        "trim_start_sec": float(trim_start),
        "trim_end_sec": float(trim_end),
        "trim_active_segments": int(len(kept)),
        "trim_rms_threshold": float(cfg.trim_rms_threshold),
        "trim_window_sec": float(cfg.trim_window_sec),
        "trim_min_active_sec": float(cfg.trim_min_active_sec),
        "trim_pad_sec": float(cfg.trim_pad_sec),
    }
    return df_trim, info


def segment_by_gaps(t: np.ndarray, max_gap_sec: float) -> np.ndarray:
    if len(t) == 0:
        return np.array([], dtype=int)
    dt = np.diff(t, prepend=t[0])
    seg_id = np.zeros(len(t), dtype=int)
    cur = 0
    for i in range(1, len(t)):
        if dt[i] <= 0 or dt[i] > max_gap_sec:
            cur += 1
        seg_id[i] = cur
    return seg_id


def resample_segment(df: pd.DataFrame, fs: int, cols: List[str]) -> pd.DataFrame:
    t = df["time"].values.astype(float)
    t0, t1 = t[0], t[-1]
    dt = 1.0 / fs
    t_new = np.arange(t0, t1, dt)

    out = {"time": t_new}
    for c in cols:
        out[c] = np.interp(t_new, t, df[c].values.astype(float))
    return pd.DataFrame(out)


def plot_dt_hist(dt: np.ndarray, out_path: str) -> None:
    if len(dt) == 0:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(dt, bins=100, color="steelblue")
    plt.title("Time delta histogram")
    plt.xlabel("dt (s)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_acc_mag(df: pd.DataFrame, acc_cols: Tuple[str, str, str], out_path: str, label: str) -> None:
    if len(df) == 0:
        return
    acc = df[list(acc_cols)].values
    acc_mag = np.linalg.norm(acc, axis=1)
    t = df["time"].values

    plt.figure(figsize=(10, 4))
    plt.plot(t, acc_mag, linewidth=1)
    plt.title(label)
    plt.xlabel("time (s)")
    plt.ylabel("acc magnitude")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_missing_rate(df: pd.DataFrame, cols: List[str], out_path: str) -> None:
    rates = []
    for c in cols:
        if c in df.columns:
            rates.append(1.0 - float(df[c].notna().mean()))
        else:
            rates.append(1.0)
    plt.figure(figsize=(10, 4))
    plt.bar(cols, rates)
    plt.xticks(rotation=45, ha="right")
    plt.title("Missing rate")
    plt.ylabel("missing fraction")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    default_csv_path = (r"D:\Desktop\python\rowing_ML\row_data\club-level\iPhone"
                        r"\Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_"
                        r"F92041BC-2503-4150-8196-2B45C0258ED8.csv")
    parser.add_argument("--csv_path", type=str, default=default_csv_path)
    parser.add_argument("--out_path", type=str, default=None)
    parser.add_argument("--report_dir", type=str, default="clean_report")
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--session_id", type=str, default=None)

    parser.add_argument("--sample_rate", type=int, default=100)
    parser.add_argument("--max_gap_sec", type=float, default=0.5)
    parser.add_argument("--min_segment_sec", type=float, default=10.0)
    parser.add_argument("--acc_unit", type=str, default="auto", choices=["auto", "g", "mps2"])
    parser.add_argument("--max_acc_g", type=float, default=8.0)
    parser.add_argument("--acc_source", type=str, default="auto", choices=["auto", "user", "raw"])
    parser.add_argument("--remove_gravity", type=str, default="gravity", choices=["none", "gravity", "lpf"])
    parser.add_argument("--gravity_tau_sec", type=float, default=1.0)
    parser.add_argument("--no_resample", action="store_true")
    parser.add_argument("--keep_speed", action="store_true")
    parser.add_argument("--trim_inactive_edges", dest="trim_inactive_edges",
                        action="store_true")
    parser.add_argument("--no_trim_inactive_edges", dest="trim_inactive_edges",
                        action="store_false")
    parser.set_defaults(trim_inactive_edges=True)
    parser.add_argument("--trim_rms_threshold", type=float, default=0.06)
    parser.add_argument("--trim_window_sec", type=float, default=1.0)
    parser.add_argument("--trim_min_active_sec", type=float, default=20.0)
    parser.add_argument("--trim_pad_sec", type=float, default=2.0)

    args = parser.parse_args()

    cfg = CleanConfig(
        sample_rate=args.sample_rate,
        max_gap_sec=args.max_gap_sec,
        min_segment_sec=args.min_segment_sec,
        acc_unit=args.acc_unit,
        max_acc_g=args.max_acc_g,
        remove_gravity=args.remove_gravity,
        gravity_tau_sec=args.gravity_tau_sec,
        resample=not args.no_resample,
        trim_inactive_edges=args.trim_inactive_edges,
        trim_rms_threshold=args.trim_rms_threshold,
        trim_window_sec=args.trim_window_sec,
        trim_min_active_sec=args.trim_min_active_sec,
        trim_pad_sec=args.trim_pad_sec,
    )

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"csv_path not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)

    if args.session_id is not None:
        for col in ["session_id", "session", "Session", "file_id"]:
            if col in df.columns:
                df = df[df[col].astype(str) == str(args.session_id)]
                break

    time_col = pick_time_col(df, args.time_col)
    if time_col is None:
        raise ValueError("No valid time column found.")

    user_acc_cols = pick_columns(df, USER_ACC_CANDIDATES)
    raw_acc_cols = pick_columns(df, ACC_CANDIDATES)
    if args.acc_source == "user":
        if user_acc_cols is None:
            raise ValueError("No user acceleration columns found.")
        acc_cols = user_acc_cols
        acc_source = "user"
    elif args.acc_source == "raw":
        if raw_acc_cols is None:
            raise ValueError("No raw accelerometer columns found.")
        acc_cols = raw_acc_cols
        acc_source = "raw"
    else:
        if user_acc_cols is not None:
            acc_cols = user_acc_cols
            acc_source = "user"
        elif raw_acc_cols is not None:
            acc_cols = raw_acc_cols
            acc_source = "raw"
        else:
            raise ValueError("No accelerometer columns found.")

    gyro_cols = pick_columns(df, GYRO_CANDIDATES)
    gravity_cols = pick_columns(df, GRAVITY_CANDIDATES)
    speed_col = None
    if args.keep_speed:
        for c in SPEED_CANDIDATES:
            if c in df.columns:
                speed_col = c
                break

    report = {
        "input_rows": int(len(df)),
        "time_col": time_col,
        "acc_source": acc_source,
        "acc_cols": acc_cols,
        "gyro_cols": gyro_cols,
        "gravity_cols": gravity_cols,
        "speed_col": speed_col,
        "trim_inactive_edges": cfg.trim_inactive_edges,
        "trim_rms_threshold": cfg.trim_rms_threshold,
        "trim_window_sec": cfg.trim_window_sec,
        "trim_min_active_sec": cfg.trim_min_active_sec,
        "trim_pad_sec": cfg.trim_pad_sec,
    }

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[time_col])
    report["drop_time_nan"] = int(before - len(df))

    df = df.sort_values(time_col)
    df = df.reset_index(drop=True)
    df["time"] = df[time_col].values - float(df[time_col].iloc[0])

    before = len(df)
    df = df.drop_duplicates(subset=["time"])
    report["drop_time_duplicate"] = int(before - len(df))

    dt = np.diff(df["time"].values)
    report["dt_stats"] = compute_dt_stats(dt)

    seg_id = segment_by_gaps(df["time"].values, cfg.max_gap_sec)
    df["segment_id"] = seg_id

    seg_stats = []
    for sid in np.unique(seg_id):
        seg = df[df["segment_id"] == sid]
        dur = float(seg["time"].iloc[-1] - seg["time"].iloc[0])
        seg_stats.append((sid, dur, len(seg)))

    report["segment_count_raw"] = int(len(seg_stats))

    # Drop short segments
    keep_ids = [sid for sid, dur, _ in seg_stats if dur >= cfg.min_segment_sec]
    before = len(df)
    df = df[df["segment_id"].isin(keep_ids)].copy()
    report["drop_short_segments_rows"] = int(before - len(df))
    report["segment_count_kept"] = int(len(keep_ids))

    # Drop missing acc
    before = len(df)
    df = df.dropna(subset=list(acc_cols))
    report["drop_missing_acc_rows"] = int(before - len(df))

    # Unit conversion if needed
    acc = df[list(acc_cols)].values
    acc_mag = np.linalg.norm(acc, axis=1)
    if cfg.acc_unit == "auto":
        if np.median(acc_mag) > 3.0:
            df[list(acc_cols)] = df[list(acc_cols)] / 9.80665
            report["acc_unit_used"] = "mps2_to_g"
        else:
            report["acc_unit_used"] = "g"
    elif cfg.acc_unit == "mps2":
        df[list(acc_cols)] = df[list(acc_cols)] / 9.80665
        report["acc_unit_used"] = "mps2_to_g"
    else:
        report["acc_unit_used"] = "g"

    # Remove gravity (only for raw acc)
    if acc_source == "raw":
        if cfg.remove_gravity == "gravity":
            if gravity_cols is not None:
                gx, gy, gz = gravity_cols
                df[acc_cols[0]] = df[acc_cols[0]] - df[gx]
                df[acc_cols[1]] = df[acc_cols[1]] - df[gy]
                df[acc_cols[2]] = df[acc_cols[2]] - df[gz]
                report["gravity_removal"] = "gravity_cols"
            else:
                acc = df[list(acc_cols)].values
                g = lpf_gravity(acc, cfg.sample_rate, cfg.gravity_tau_sec)
                df[acc_cols[0]] = acc[:, 0] - g[:, 0]
                df[acc_cols[1]] = acc[:, 1] - g[:, 1]
                df[acc_cols[2]] = acc[:, 2] - g[:, 2]
                report["gravity_removal"] = "lpf_fallback"
        elif cfg.remove_gravity == "lpf":
            acc = df[list(acc_cols)].values
            g = lpf_gravity(acc, cfg.sample_rate, cfg.gravity_tau_sec)
            df[acc_cols[0]] = acc[:, 0] - g[:, 0]
            df[acc_cols[1]] = acc[:, 1] - g[:, 1]
            df[acc_cols[2]] = acc[:, 2] - g[:, 2]
            report["gravity_removal"] = "lpf"
        else:
            report["gravity_removal"] = "none"
    else:
        report["gravity_removal"] = "user_acc"

    # Standardize to dynamic acceleration column names
    acc_dyn_cols = ("acc_dyn_x", "acc_dyn_y", "acc_dyn_z")
    df = df.rename(columns={
        acc_cols[0]: acc_dyn_cols[0],
        acc_cols[1]: acc_dyn_cols[1],
        acc_cols[2]: acc_dyn_cols[2],
    })
    acc_cols = acc_dyn_cols
    report["acc_dyn_cols"] = acc_dyn_cols

    # Outlier filter by acc magnitude
    acc = df[list(acc_cols)].values
    acc_mag = np.linalg.norm(acc, axis=1)
    before = len(df)
    df = df[acc_mag <= cfg.max_acc_g].copy()
    report["drop_outlier_rows"] = int(before - len(df))

    # Trim inactive edges
    if cfg.trim_inactive_edges:
        before = len(df)
        df, trim_info = trim_inactive_edges(df, acc_cols, cfg)
        report.update(trim_info)
        report["trim_drop_rows"] = int(before - len(df))

    # Resample per segment
    cols = ["time"] + list(acc_cols)
    if gyro_cols is not None:
        cols += list(gyro_cols)
    if speed_col is not None:
        df[speed_col] = pd.to_numeric(df[speed_col], errors="coerce")
        df[speed_col] = df[speed_col].interpolate(limit_direction="both")
        cols += [speed_col]

    if cfg.resample:
        resampled = []
        for sid in sorted(df["segment_id"].unique()):
            seg = df[df["segment_id"] == sid].copy()
            if len(seg) < 2:
                continue
            seg = resample_segment(seg, cfg.sample_rate, cols[1:])
            seg["segment_id"] = sid
            resampled.append(seg)
        if resampled:
            df = pd.concat(resampled, ignore_index=True)
        report["resampled"] = True
    else:
        report["resampled"] = False

    # Keep only cleaned columns in output
    keep_cols = cols + ["segment_id"]
    df = df[keep_cols].copy()
    report["output_cols"] = keep_cols

    report["output_rows"] = int(len(df))

    # Output paths
    report_dir = args.report_dir
    os.makedirs(report_dir, exist_ok=True)
    if args.out_path is None:
        base = os.path.splitext(os.path.basename(args.csv_path))[0]
        out_path = os.path.join(report_dir, f"{base}_clean.csv")
    else:
        out_path = args.out_path

    df.to_csv(out_path, index=False)
    report["out_path"] = out_path

    if len(df) == 0:
        with open(os.path.join(report_dir, "clean_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Cleaned CSV:", out_path)
        print("Report dir:", report_dir)
        print("Warning: output is empty. Check time column choice and segment thresholds.")
        print(json.dumps(report, indent=2))
        return

    # Plots
    dt = np.diff(df["time"].values)
    plot_dt_hist(dt, os.path.join(report_dir, "time_delta_hist.png"))
    plot_acc_mag(df, acc_cols, os.path.join(report_dir, "acc_mag_overview.png"),
                 "acc magnitude overview")

    key_cols = list(acc_cols)
    if gyro_cols is not None:
        key_cols += list(gyro_cols)
    plot_missing_rate(df, key_cols, os.path.join(report_dir, "missing_rate.png"))

    with open(os.path.join(report_dir, "clean_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Cleaned CSV:", out_path)
    print("Report dir:", report_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
