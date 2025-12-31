#!/usr/bin/env python3
"""
0. 数据合并脚本
自动合并所有已标注的数据文件，生成统一的训练数据集

使用方法：直接运行即可
"""

import os
import glob
import pandas as pd


def find_labeled_files(search_dir='datasets'):
    """查找所有已标注的CSV文件"""
    pattern = os.path.join(search_dir, '*_labeled.csv')
    files = glob.glob(pattern)
    # 排除combined文件
    files = [f for f in files if 'combined' not in f.lower()]
    return sorted(files)


def main():
    print("="*60)
    print("数据合并脚本")
    print("="*60)
    
    # 查找所有已标注文件
    labeled_files = find_labeled_files('datasets')
    
    if not labeled_files:
        print("[ERROR] 未找到已标注的数据文件！")
        print("  请先运行以下脚本生成标注数据：")
        print("  1. python scripts/2_extract_events.py")
        print("  2. python scripts/4_generate_labels.py")
        return
    
    print(f"\n[INFO] 找到 {len(labeled_files)} 个已标注文件：")
    for i, f in enumerate(labeled_files, 1):
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {i}. {os.path.basename(f)} ({size_mb:.1f} MB)")
    
    # 读取并合并
    print("\n[INFO] 开始合并...")
    dfs = []
    max_stroke_id = 0  # 累计stroke_id偏移
    
    for i, f in enumerate(labeled_files, 1):
        df = pd.read_csv(f)
        
        # 添加数据源标识
        df['source_file'] = os.path.basename(f)
        df['source_id'] = i
        
        # 调整stroke_id使其在合并后不重复（使用累计偏移）
        if 'stroke_id' in df.columns:
            df['stroke_id'] = df['stroke_id'] + max_stroke_id
            max_stroke_id = df['stroke_id'].max() + 1  # 更新偏移量
        
        print(f"  [{i}/{len(labeled_files)}] {os.path.basename(f)}: {len(df):,} 行")
        dfs.append(df)
    
    # 合并
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 保存
    out_path = 'datasets/combined_labeled.csv'
    combined_df.to_csv(out_path, index=False)
    
    print(f"\n[INFO] 合并完成！")
    print(f"  总行数: {len(combined_df):,}")
    print(f"  保存到: {out_path}")
    
    # 统计标签分布
    print("\n[INFO] 合并后标签分布:")
    label_names = {0: '背景', 1: '准备', 2: '核心', 3: '恢复'}
    label_counts = combined_df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / len(combined_df) * 100
        name = label_names.get(label, str(label))
        print(f"  {name} ({label}): {count:8,} ({pct:5.2f}%)")
    
    # 统计每个数据源的划桨数
    print("\n[INFO] 每个数据源的划桨数:")
    for source_id in combined_df['source_id'].unique():
        source_df = combined_df[combined_df['source_id'] == source_id]
        strokes = source_df['stroke_id'].nunique()
        source_name = source_df['source_file'].iloc[0]
        print(f"  数据源 {source_id}: {strokes} 个划桨 ({source_name[:50]}...)")
    
    print("\n" + "="*60)
    print("✅ 下一步：运行特征提取")
    print("  python scripts/5_feature_extraction_with_vis.py --labeled_csv datasets/combined_labeled.csv --stride 20")
    print("="*60)


if __name__ == '__main__':
    main()
