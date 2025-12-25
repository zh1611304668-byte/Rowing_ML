#!/usr/bin/env python3
"""
特征提取脚本 (带3D可视化)
从标注数据中提取滑动窗口特征用于模型训练

提取以下特征:
1. 时域特征: 均值、标准差、峰峰值、RMS、过零率
2. 频域特征: FFT能量分布、主频率
3. 自定义特征: 轴间标准差比、加速度梯度

新增功能:
4. 3D PCA可视化: 多角度3D图、交互式HTML、主成分对比
"""

import argparse
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats as scipy_stats
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def extract_time_features(window: np.ndarray) -> Dict[str, float]:
    """提取时域特征 (3轴)"""
    features = {}
    
    for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
        axis_data = window[:, axis_idx]
        
        # 基本统计
        features[f'{axis_name}_mean'] = np.mean(axis_data)
        features[f'{axis_name}_std'] = np.std(axis_data)
        features[f'{axis_name}_min'] = np.min(axis_data)
        features[f'{axis_name}_max'] = np.max(axis_data)
        features[f'{axis_name}_ptp'] = np.ptp(axis_data)  # peak-to-peak
        
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


def extract_freq_features(window: np.ndarray, sample_rate: float = 100.0) -> Dict[str, float]:
    """提取频域特征"""
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
        
        # 频段能量分布 (0-2Hz, 2-5Hz, 5-10Hz, >10Hz)
        bands = [(0, 2), (2, 5), (5, 10), (10, 50)]
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs < high)
            band_energy = np.sum(fft_vals[band_mask] ** 2)
            features[f'{axis_name}_energy_{low}_{high}Hz'] = band_energy
    
    return features


def extract_custom_features(window: np.ndarray) -> Dict[str, float]:
    """提取自定义特征"""
    features = {}
    
    # 三轴标准差比值 (用于判断活跃轴)
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
    
    # 加速度梯度 (变化速率)
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
    window_size: int = 40,  # 400ms @ 100Hz
    stride: int = 1,
    sample_rate: float = 100.0
) -> pd.DataFrame:
    """从DataFrame中提取滑动窗口特征"""
    
    acc_cols = ['acc_dyn_x', 'acc_dyn_y', 'acc_dyn_z']
    acc_data = df[acc_cols].values
    labels = df['label'].values
    time_data = df['time'].values
    
    features_list = []
    window_labels = []
    window_times = []
    
    print(f"[INFO] 开始特征提取...")
    print(f"[INFO] 窗口大小: {window_size} 样本 ({window_size/sample_rate*1000:.0f}ms)")
    print(f"[INFO] 步长: {stride} 样本 ({stride/sample_rate*1000:.0f}ms)")
    
    total_windows = (len(acc_data) - window_size) // stride + 1
    
    for i in range(0, len(acc_data) - window_size + 1, stride):
        # 提取窗口
        window = acc_data[i:i+window_size]
        
        # 窗口的标签 (取中心点的标签)
        center_idx = i + window_size // 2
        window_label = labels[center_idx]
        window_time = time_data[center_idx]
        
        # 提取特征
        features = {}
        features.update(extract_time_features(window))
        features.update(extract_freq_features(window, sample_rate))
        features.update(extract_custom_features(window))
        
        features_list.append(features)
        window_labels.append(window_label)
        window_times.append(window_time)
        
        if (len(features_list) % 5000 == 0):
            print(f"[INFO] 已处理 {len(features_list)}/{total_windows} 窗口...")
    
    print(f"[INFO] 特征提取完成! 总窗口数: {len(features_list)}")
    
    # 转换为DataFrame
    features_df = pd.DataFrame(features_list)
    features_df['label'] = window_labels
    features_df['time'] = window_times
    
    # Add stroke_id if available in input df
    if 'stroke_id' in df.columns:
        stroke_ids = df['stroke_id'].values
        window_stroke_ids = [stroke_ids[i + window_size // 2] for i in range(0, len(acc_data) - window_size + 1, stride)]
        features_df['stroke_id'] = window_stroke_ids
    
    # ========== 新增：时间序列特征 ==========
    print("\n[INFO] 添加时间序列特征...")
    
    # 关键特征列表（用于构建时间特征）
    key_cols = ['y_mean', 'y_std', 'y_rms', 'y_ptp', 
                'mag_mean', 'mag_max', 'mag_std',
                'gradient_max', 'dynamic_range']
    
    # 过滤存在的列
    key_cols = [col for col in key_cols if col in features_df.columns]
    
    # 1. 差分特征（窗口间变化，区分上升/下降趋势）
    for col in key_cols:
        features_df[f'{col}_diff'] = features_df[col].diff().fillna(0)
    
    # 2. 滚动统计（短期趋势，3窗口）
    for col in key_cols:
        features_df[f'{col}_roll3_mean'] = features_df[col].rolling(window=3, min_periods=1).mean()
        features_df[f'{col}_roll3_std'] = features_df[col].rolling(window=3, min_periods=1).std().fillna(0)
    
    # 3. 加速度特征（二阶差分）
    for col in ['y_rms', 'mag_max']:  # 核心指标
        if col in features_df.columns:
            features_df[f'{col}_accel'] = features_df[col].diff().diff().fillna(0)
    
    # 4. 动量特征（累积变化）
    for col in ['y_rms', 'mag_mean']:
        if col in features_df.columns:
            features_df[f'{col}_momentum'] = features_df[col].diff().rolling(window=5, min_periods=1).sum().fillna(0)
    
    print(f"[INFO] 添加了 {len(features_df.columns) - len(features_list[0]) - 2} 个时间特征")
    print(f"[INFO] 新特征总数: {len(features_df.columns) - 2}  (不含label和time)")
    
    return features_df


def visualize_pca_3d(features_df: pd.DataFrame, out_dir: str) -> Optional[Tuple]:
    """生成3D PCA可视化（多角度静态图）"""
    print("\n[INFO] 生成3D PCA可视化...")
    
    # 准备数据
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['label', 'time', 'stroke_id']]
    
    if len(feature_cols) < 3:
        print("[WARNING] 特征数不足3个，无法进行3D PCA")
        return None
    
    X = features_df[feature_cols].values
    y = features_df['label'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA降维到3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"  PCA解释方差: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%, "
          f"PC3={pca.explained_variance_ratio_[2]*100:.1f}%, "
          f"总计={sum(pca.explained_variance_ratio_)*100:.1f}%")
    
    # 标签配置 (与 generate_labels.py 保持一致)
    label_names = {0: '背景', 1: '准备', 2: '核心', 3: '恢复', 4: '过渡'}
    colors = {
        0: '#BBDEFB',  # 浅蓝 - 背景
        1: '#FFF59D',  # 黄色 - 准备
        2: '#FF5252',  # 红色 - 核心
        3: '#90CAF9',  # 蓝色 - 恢复
        4: '#CE93D8'   # 紫色 - 过渡
    }
    
    # 创建多角度视图
    fig = plt.figure(figsize=(20, 15))
    
    # 定义4个不同的视角
    views = [
        (30, 45, "视角1: 标准视角"),
        (10, 120, "视角2: 侧视"),
        (60, 200, "视角3: 俯视"),
        (80, 300, "视角4: 仰视")
    ]
    
    for idx, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # 绘制每个类别
        for label in sorted(np.unique(y)):
            mask = y == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                      c=colors.get(label, 'gray'),
                      label=label_names.get(label, str(label)),
                      alpha=0.6, s=15, edgecolors='none')
        
        # 设置标签和标题
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=10)
        ax.set_title(title, fontsize=12, pad=10)
        
        # 设置视角
        ax.view_init(elev=elev, azim=azim)
        
        # 只在第一个子图显示图例
        if idx == 1:
            ax.legend(fontsize=10, loc='upper left')
        
        # 网格
        ax.grid(True, alpha=0.3)
    
    # 总标题
    fig.suptitle(f'PCA 3D降维可视化 (总解释方差: {sum(pca.explained_variance_ratio_)*100:.1f}%)',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_3d_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 静态3D PCA可视化已保存")
    
    return X_pca, y, pca


def visualize_pca_interactive(X_pca: np.ndarray, y: np.ndarray, pca: PCA, out_dir: str):
    """生成交互式3D PCA可视化（使用plotly）"""
    try:
        import plotly.express as px
        
        print("  生成交互式3D PCA可视化...")
        
        # 准备数据
        label_names = {0: '背景', 1: '准备', 2: '核心', 3: '恢复', 4: '过渡'}
        df_plot = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'PC3': X_pca[:, 2],
            'label': y,
            'label_name': [label_names.get(int(l), str(l)) for l in y]
        })
        
        # 创建3D散点图
        fig = px.scatter_3d(
            df_plot, 
            x='PC1', y='PC2', z='PC3',
            color='label_name',
            color_discrete_map={
                '背景': '#BBDEFB',
                '准备': '#FFF59D',
                '核心': '#FF5252',
                '恢复': '#90CAF9',
                '过渡': '#CE93D8'
            },
            opacity=0.6,
            title=f'PCA 3D交互式可视化 (解释方差: {sum(pca.explained_variance_ratio_)*100:.1f}%)',
            labels={
                'PC1': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                'PC3': f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
                'label_name': '阶段'
            }
        )
        
        # 调整布局
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)',
            ),
            width=1200,
            height=900,
            font=dict(size=12)
        )
        
        # 保存HTML
        html_path = os.path.join(out_dir, 'pca_3d_interactive.html')
        fig.write_html(html_path)
        print("  ✓ 交互式3D PCA可视化已保存 (可在浏览器中打开)")
        
    except ImportError:
        print("  [INFO] 未安装plotly，跳过交互式可视化")
        print("        提示: 运行 'pip install plotly' 以启用交互式3D图")


def visualize_pca_components(X_pca: np.ndarray, y: np.ndarray, pca: PCA, out_dir: str):
    """生成PCA主成分两两对比图"""
    print("  生成PCA主成分对比图...")
    
    label_names = {0: '背景', 1: '准备', 2: '核心', 3: '恢复', 4: '过渡'}
    colors = {0: '#BBDEFB', 1: '#FFF59D', 2: '#FF5252', 3: '#90CAF9', 4: '#CE93D8'}
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    pairs = [(0, 1, 'PC1 vs PC2'), (0, 2, 'PC1 vs PC3'), (1, 2, 'PC2 vs PC3')]
    
    for idx, (pc1, pc2, title) in enumerate(pairs):
        ax = axes[idx]
        
        for label in sorted(np.unique(y)):
            mask = y == label
            ax.scatter(X_pca[mask, pc1], X_pca[mask, pc2],
                      c=colors.get(label, 'gray'),
                      label=label_names.get(label, str(label)),
                      alpha=0.6, s=20, edgecolors='none')
        
        ax.set_xlabel(f'PC{pc1+1} ({pca.explained_variance_ratio_[pc1]*100:.1f}%)', fontsize=11)
        ax.set_ylabel(f'PC{pc2+1} ({pca.explained_variance_ratio_[pc2]*100:.1f}%)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_components_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ PCA主成分对比图已保存")


def visualize_feature_boxplots(features_df: pd.DataFrame, out_dir: str):
    """生成关键特征的箱线图对比"""
    print("  生成特征箱线图...")
    
    # 选择关键特征（包含您之前关注的特征）
    key_features = [
        'y_std', 'y_ptp', 'y_rms',  # Y轴特征（主要发力方向）
        'mag_std', 'mag_mean', 'mag_max',  # 幅值特征
        'gradient_max', 'dynamic_range',  # 动态特征
        'x_std', 'z_std'  # 辅助轴特征
    ]
    
    # 过滤存在的特征
    key_features = [f for f in key_features if f in features_df.columns]
    
    if len(key_features) == 0:
        print("  [WARNING] 未找到关键特征，跳过箱线图")
        return
    
    # 动态计算子图布局
    n_features = min(len(key_features), 12)  # 最多显示12个
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    label_names = {0: '背景', 1: '准备', 2: '核心', 3: '恢复', 4: '过渡'}
    colors = ['#BBDEFB', '#FFF59D', '#FF5252', '#90CAF9', '#CE93D8']
    
    for idx, feature in enumerate(key_features[:n_features]):
        ax = axes[idx]
        
        # 准备数据
        data_by_label = []
        labels = []
        for label in sorted(features_df['label'].unique()):
            data_by_label.append(features_df[features_df['label'] == label][feature].values)
            labels.append(label_names.get(label, str(label)))
        
        # 绘制箱线图
        bp = ax.boxplot(data_by_label, labels=labels, patch_artist=True)
        
        # 设置颜色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(feature, fontsize=10)
        ax.set_title(f'{feature} 按标签分布', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 特征箱线图已保存")


def main():
    parser = argparse.ArgumentParser(description='特征提取（支持增强数据）')
    
    # 自动查找数据文件（优先使用增强数据）
    import glob
    
    # 查找增强数据
    augmented_files = glob.glob('datasets_augmented/*_augmented.csv')
    labeled_files = glob.glob('datasets/*_labeled.csv')
    
    # 默认值：优先使用最新的增强数据
    default_file = None
    if augmented_files:
        default_file = max(augmented_files, key=os.path.getmtime)
        print(f"[INFO] 检测到增强数据: {default_file}")
    elif labeled_files:
        default_file = max(labeled_files, key=os.path.getmtime)
        print(f"[INFO] 检测到标注数据: {default_file}") 
    
    parser.add_argument('--labeled_csv', type=str, default=default_file,
                       help='标注后的CSV路径（支持增强数据）')
    parser.add_argument('--window_size', type=int, default=40,
                       help='窗口大小(样本数)')
    parser.add_argument('--stride', type=int, default=10,
                       help='滑动步长(样本数)')
    parser.add_argument('--sample_rate', type=float, default=100.0,
                       help='采样率(Hz)')
    parser.add_argument('--out_dir', type=str, default='datasets',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.labeled_csv is None:
        print("[ERROR] 未找到标注或增强数据文件！")
        print("  请确保运行了步骤4 (生成标签) 或步骤5 (数据增强)")
        return
    
    if not os.path.exists(args.labeled_csv):
        raise FileNotFoundError(f"数据文件不存在: {args.labeled_csv}")
    
    # 检测是否为增强数据
    is_augmented = '_augmented.csv' in args.labeled_csv
    data_status = "增强数据" if is_augmented else "标注数据"
    
    print(f"\n{'='*60}")
    print(f"特征提取 - 使用{data_status}")
    print('='*60)
    
    if not os.path.exists(args.labeled_csv):
        raise FileNotFoundError(f"标注文件不存在: {args.labeled_csv}")
    
    print(f"[INFO] 加载数据: {os.path.basename(args.labeled_csv)}")
    df = pd.read_csv(args.labeled_csv)
    print(f"[INFO] 数据行数: {len(df):,}")
    
    if is_augmented:
        print(f"[INFO] ✨ 正在使用增强数据进行特征提取")
    
    # 提取特征
    features_df = extract_features_from_df(
        df,
        window_size=args.window_size,
        stride=args.stride,
        sample_rate=args.sample_rate
    )
    
    # 保存特征数据
    os.makedirs(args.out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.labeled_csv))[0]
    # Remove _labeled or _augmented suffix, then add _features
    base_name = base_name.replace('_labeled', '').replace('_augmented', '')
    suffix = '_augmented_features' if is_augmented else '_labeled_features'
    out_path = os.path.join(args.out_dir, f"{base_name}{suffix}.csv")
    features_df.to_csv(out_path, index=False)
    print(f"\n[INFO] 特征数据已保存: {out_path}")
    
    # 打印特征统计
    print(f"\n=== 特征统计 ===")
    print(f"总特征数: {len(features_df.columns) - 2}  (不含label和time)")
    print(f"特征窗口数: {len(features_df)}")
    print(f"\n标签分布:")
    label_counts = features_df['label'].value_counts().sort_index()
    label_map = {0: '背景', 1: '准备', 2: '核心', 3: '恢复', 4: '过渡'}
    for label, count in label_counts.items():
        pct = count / len(features_df) * 100
        label_name = label_map.get(label, str(label))
        print(f"  {label_name} ({label}): {count:6d} ({pct:5.2f}%)")
    
    # 生成可视化
    vis_dir = os.path.join(args.out_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 3D PCA可视化
    result = visualize_pca_3d(features_df, vis_dir)
    if result is not None:
        X_pca, y, pca = result
        visualize_pca_interactive(X_pca, y, pca, vis_dir)
        visualize_pca_components(X_pca, y, pca, vis_dir)
    
    # 特征箱线图
    visualize_feature_boxplots(features_df, vis_dir)
    
    print(f"\n[INFO] 可视化已保存到: {vis_dir}")
    print("  生成的文件：")
    print("    - pca_3d_visualization.png (3D PCA多视角)")
    print("    - pca_3d_interactive.html (交互式3D PCA)")
    print("    - pca_components_comparison.png (主成分对比)")
    print("    - feature_boxplots.png (特征箱线图)")
    
    print("\n[完成] 特征提取与可视化完毕!")


if __name__ == '__main__':
    main()
