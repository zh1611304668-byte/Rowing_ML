import pandas as pd
import numpy as np

# 读取预测结果
df = pd.read_csv(r'd:\Desktop\python\rowing_ML\detection_comparison\Boat-20180422T103229_1641_rpc364_data_1CLX_1_B_2CDF0487-83FC-45CC-B590-FF42D74E0D6D_clean_predictions.csv')

print('=' * 60)
print('标签分布统计')
print('=' * 60)
label_counts = df['predicted_label'].value_counts().sort_index()
total = len(df)
label_names = {0: '背景', 1: '准备', 2: '核心', 3: '恢复', 4: '过渡'}

for label in range(5):
    count = label_counts.get(label, 0)
    pct = count / total * 100
    print(f'{label_names[label]:4s} (Label {label}): {count:8d} ({pct:6.2f}%)')

print('\n' + '=' * 60)
print('各标签平均置信度')
print('=' * 60)

for label in range(5):
    mask = df['predicted_label'] == label
    if mask.any():
        mean_conf = df.loc[mask, 'max_probability'].mean()
        print(f'{label_names[label]:4s} (Label {label}): {mean_conf:.6f}')

print(f'\n整体平均置信度: {df["max_probability"].mean():.6f}')

print('\n' + '=' * 60)
print('置信度分布统计')
print('=' * 60)
print(df['max_probability'].describe())
