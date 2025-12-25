import pandas as pd

print("=" * 60)
print("数据处理流程追踪")
print("=" * 60)

# 1. 清洗后的原始数据
clean_file = r'd:\Desktop\python\rowing_ML\clean_report\Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_F92041BC-2503-4150-8196-2B45C0258ED8_clean.csv'
df_clean = pd.read_csv(clean_file)
print(f"\n1. 清洗后数据: {len(df_clean):,} 行")

# 2. 标注后的数据  
labeled_file = r'd:\Desktop\python\rowing_ML\datasets\Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_F92041BC-2503-4150-8196-2B45C0258ED8_clean_labeled.csv'
df_labeled = pd.read_csv(labeled_file)
print(f"2. 标注后数据: {len(df_labeled):,} 行")

# 标签分布
print("\n标签分布:")
for label in range(5):
    count = (df_labeled['label'] == label).sum()
    pct = count / len(df_labeled) * 100
    print(f"  Label {label}: {count:6d} ({pct:5.2f}%)")

# 3. 检查特征文件
import glob
import os
feature_files = glob.glob(r'd:\Desktop\python\rowing_ML\datasets\*features*.csv')
print(f"\n3. 找到 {len(feature_files)} 个特征文件:")
for f in feature_files:
    df_feat = pd.read_csv(f)
    basename = os.path.basename(f)
    print(f"   {basename}: {len(df_feat):,} 行")

print("\n" + "=" * 60)
print("窗口数量变化分析")
print("=" * 60)

# 计算理论窗口数
window_size = 40
stride = 10
theoretical_windows = (len(df_labeled) - window_size) // stride + 1
print(f"\n理论窗口数 (window_size={window_size}, stride={stride}):")
print(f"  ({len(df_labeled)} - {window_size}) // {stride} + 1 = {theoretical_windows:,}")

if len(feature_files) > 0:
    df_feat = pd.read_csv(feature_files[0])
    actual_windows = len(df_feat)
    print(f"\n实际窗口数: {actual_windows:,}")
    print(f"差异: {theoretical_windows - actual_windows:,} ({(theoretical_windows - actual_windows)/theoretical_windows*100:.2f}%)")
