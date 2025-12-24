#!/usr/bin/env python3
"""
严谨版 Random Forest 训练脚本
核心改进：
1. 时间序列交叉验证（TimeSeriesSplit）- 避免数据泄漏
2. 全面评估指标（macro F1 + confusion matrix + per-class metrics）
3. Permutation Importance - 修正树模型偏差
4. 输出每帧概率分布（为 HMM/Viterbi 准备）
"""

import argparse
import os
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_features_file(search_dir='datasets'):
    """自动查找最新的特征CSV文件"""
    pattern = os.path.join(search_dir, '*features*.csv')
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        return None
    
    # 按修改时间排序，返回最新的
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


def plot_confusion_matrix_heatmap(y_true, y_pred, out_path, labels):
    """绘制带百分比和绝对数量的混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制归一化热力图
    sns.heatmap(cm_norm, annot=False, fmt='.2%', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': '比例'}, ax=ax)
    
    # 手动添加标注（百分比 + 绝对数量）
    for i in range(len(labels)):
        for j in range(len(labels)):
            percentage = cm_norm[i, j] * 100
            count = cm[i, j]
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{percentage:.1f}%\n({count})',
                   ha='center', va='center', fontsize=10, color=text_color)
    
    ax.set_title('混淆矩阵（归一化）', fontsize=14, pad=15)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_xlabel('预测标签', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 混淆矩阵热力图已保存: {out_path}")


def plot_per_class_metrics(report_dict, out_path, labels):
    """绘制每类的 Precision、Recall、F1-Score"""
    metrics = ['precision', 'recall', 'f1-score']
    
    # 提取每个类别的指标（使用标签名作为键）
    data = {metric: [] for metric in metrics}
    for label in labels:
        # classification_report 使用 target_names 作为键
        class_report = report_dict.get(label, {})
        for metric in metrics:
            value = class_report.get(metric, 0.0)
            data[metric].append(value)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, data[metric], width, 
                     label=metric.capitalize(), color=colors[i], alpha=0.8)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Per-class 性能指标已保存: {out_path}")


def plot_permutation_importance(perm_imp, feature_names, out_path, top_n=20):
    """绘制 Permutation Importance"""
    mean_importance = perm_imp.importances_mean
    std_importance = perm_imp.importances_std
    
    # 排序并取前 N 个
    indices = np.argsort(mean_importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制条形图
    y_pos = np.arange(top_n)
    ax.barh(y_pos, mean_importance[indices], 
           xerr=std_importance[indices], color='teal', alpha=0.7,
           error_kw={'elinewidth': 1, 'capsize': 3})
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Permutation Importance (mean ± std)', fontsize=12)
    ax.set_title(f'Top {top_n} Features - Permutation Importance', fontsize=14)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Permutation Importance 已保存: {out_path}")


def plot_importance_comparison(rf_model, perm_imp, feature_names, out_path, top_n=15):
    """对比 RF 内置 importance vs Permutation Importance"""
    rf_importance = rf_model.feature_importances_
    perm_importance = perm_imp.importances_mean
    
    # 找出两者的 topN 并集
    rf_top = set(np.argsort(rf_importance)[::-1][:top_n])
    perm_top = set(np.argsort(perm_importance)[::-1][:top_n])
    top_features = sorted(rf_top | perm_top, 
                         key=lambda x: perm_importance[x], reverse=True)[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(top_features))
    width = 0.35
    
    # 归一化到相同范围便于比较
    rf_norm = rf_importance / rf_importance.max() if rf_importance.max() > 0 else rf_importance
    perm_norm = perm_importance / perm_importance.max() if perm_importance.max() > 0 else perm_importance
    
    ax.barh(x - width/2, [rf_norm[i] for i in top_features], width, 
           label='RF Built-in', color='steelblue', alpha=0.7)
    ax.barh(x + width/2, [perm_norm[i] for i in top_features], width,
           label='Permutation', color='darkorange', alpha=0.7)
    
    ax.set_yticks(x)
    ax.set_yticklabels([feature_names[i] for i in top_features])
    ax.invert_yaxis()
    ax.set_xlabel('Normalized Importance', fontsize=12)
    ax.set_title('Feature Importance Comparison (Normalized)', fontsize=14)
    ax.legend()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Importance 对比图已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='严谨版 Random Forest 训练脚本')
    
    parser.add_argument('--data', type=str, default=None,
                       help='特征CSV文件路径（留空自动查找最新）')
    parser.add_argument('--cv_splits', type=int, default=5,
                       help='TimeSeriesSplit 折数')
    parser.add_argument('--n_estimators', type=int, default=300,
                       help='随机森林树的数量')
    parser.add_argument('--class_weights', type=str, default='balanced_subsample',
                       help='类别权重（balanced_subsample 或逗号分隔的权重如 1,10,15,10）')
    parser.add_argument('--out_dir', type=str, default='models',
                       help='输出目录')
    parser.add_argument('--vis_dir', type=str, default='datasets/visualizations',
                       help='可视化输出目录')
    
    args = parser.parse_args()
    
    # 1. 查找数据文件
    if args.data is None:
        print("[INFO] 未指定数据文件，自动搜索最新文件...")
        data_file = find_latest_features_file()
        if data_file is None:
            print("[ERROR] 未找到特征文件！")
            print("[INFO] 请先运行特征提取脚本")
            return
        print(f"[INFO] 找到最新文件: {data_file}")
    else:
        data_file = args.data
        if not os.path.exists(data_file):
            print(f"[ERROR] 文件不存在: {data_file}")
            return
    
    # 2. 加载数据
    print("\n[INFO] 加载数据...")
    df = pd.read_csv(data_file)
    print(f"[INFO] 数据行数: {len(df)}")
    print(f"[INFO] 列数: {len(df.columns)}")
    
    # 3. 分离特征和标签
    exclude_cols = ['label', 'time']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"\n[INFO] 特征维度: {X.shape}")
    print(f"[INFO] 特征数量: {len(feature_cols)}")
    
    # 4. 标签分布统计
    labels = ['背景', '准备', '核心', '恢复']
    print(f"\n[INFO] 标签分布:")
    for label_idx in range(4):
        count = np.sum(y == label_idx)
        pct = count / len(y) * 100
        print(f"  {labels[label_idx]} ({label_idx}): {count:6d} ({pct:5.2f}%)")
    
    # 5. 处理类别权重
    if args.class_weights == 'balanced_subsample':
        class_weight = 'balanced_subsample'
        print(f"\n[INFO] 使用自动平衡权重: balanced_subsample")
    else:
        try:
            weights = [float(w) for w in args.class_weights.split(',')]
            if len(weights) != 4:
                raise ValueError("权重数量必须为4")
            class_weight = {i: weights[i] for i in range(4)}
            print(f"\n[INFO] 使用手动权重: {class_weight}")
        except Exception as e:
            print(f"[ERROR] 权重格式错误: {e}")
            print("[INFO] 使用默认: balanced_subsample")
            class_weight = 'balanced_subsample'
    
    # 6. 构建 Random Forest 模型
    print(f"\n[INFO] 构建 Random Forest 模型...")
    print(f"  - n_estimators: {args.n_estimators}")
    print(f"  - max_depth: None (不限制)")
    print(f"  - min_samples_leaf: 5")
    print(f"  - class_weight: {class_weight}")
    
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None,
        min_samples_leaf=5,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=42
    )
    
    # 7. 时间序列交叉验证（手动实现）
    print(f"\n[INFO] 使用 TimeSeriesSplit (n_splits={args.cv_splits}) 交叉验证...")
    tscv = TimeSeriesSplit(n_splits=args.cv_splits)
    
    # 手动执行交叉验证
    print("[INFO] 执行交叉验证预测...")
    y_pred_cv = np.zeros(len(y), dtype=int)
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"  Fold {fold_idx}/{args.cv_splits} - Train: {len(train_idx)}, Test: {len(test_idx)}")
        
        # 训练模型
        rf_fold = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=None,
            min_samples_leaf=5,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42
        )
        rf_fold.fit(X[train_idx], y[train_idx])
        
        # 预测测试集
        y_pred_cv[test_idx] = rf_fold.predict(X[test_idx])
    
    # 8. 评估指标
    print("\n" + "="*60)
    print("交叉验证结果")
    print("="*60)
    
    # Classification Report
    report = classification_report(y, y_pred_cv, 
                                   target_names=labels, 
                                   output_dict=True,
                                   zero_division=0)
    
    print("\n分类报告:")
    print(classification_report(y, y_pred_cv, target_names=labels, zero_division=0))
    
    # Macro F1
    macro_f1 = report['macro avg']['f1-score']
    print(f"⭐ Macro F1-Score: {macro_f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_cv)
    print(f"\n混淆矩阵:")
    print(cm)
    
    # 9. 训练最终模型（用于特征重要性和预测概率）
    print(f"\n[INFO] 训练最终模型（全量数据）...")
    rf.fit(X, y)
    
    # 10. Permutation Importance
    print(f"\n[INFO] 计算 Permutation Importance (n_repeats=10)...")
    perm_imp = permutation_importance(rf, X, y, 
                                      n_repeats=10, 
                                      random_state=42, 
                                      n_jobs=-1)
    
    # 11. 输出概率分布（为 HMM 准备）
    print(f"\n[INFO] 生成每帧概率分布...")
    y_proba = rf.predict_proba(X)
    y_pred_final = rf.predict(X)
    
    # 创建概率DataFrame
    proba_df = pd.DataFrame(y_proba, columns=[f'prob_{label}' for label in labels])
    proba_df['true_label'] = y
    proba_df['pred_label'] = y_pred_final
    proba_df['is_correct'] = (y == y_pred_final).astype(int)
    
    if 'time' in df.columns:
        proba_df.insert(0, 'time', df['time'].values)
    
    # 12. 保存结果
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(args.out_dir, 'rf_rigorous_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"\n[INFO] 模型已保存: {model_path}")
    
    # 保存概率分布
    proba_path = os.path.join(args.out_dir, 'rf_frame_proba.csv')
    proba_df.to_csv(proba_path, index=False)
    print(f"[INFO] 每帧概率已保存: {proba_path}")
    print(f"  - 包含列: {list(proba_df.columns)}")
    
    # 保存训练报告
    report_data = {
        'data_file': data_file,
        'n_samples': int(len(X)),
        'n_features': int(len(feature_cols)),
        'cv_strategy': f'TimeSeriesSplit(n_splits={args.cv_splits})',
        'model_config': {
            'n_estimators': args.n_estimators,
            'max_depth': None,
            'min_samples_leaf': 5,
            'class_weight': str(class_weight)
        },
        'metrics': {
            'accuracy': float(report['accuracy']),
            'macro_f1': float(macro_f1),
            'macro_precision': float(report['macro avg']['precision']),
            'macro_recall': float(report['macro avg']['recall']),
        },
        'per_class_metrics': {
            label: {
                'precision': float(report.get(label, {}).get('precision', 0.0)),
                'recall': float(report.get(label, {}).get('recall', 0.0)),
                'f1-score': float(report.get(label, {}).get('f1-score', 0.0)),
                'support': int(report.get(label, {}).get('support', 0))
            } for label in labels
        },
        'confusion_matrix': cm.tolist(),
        'top_10_features_permutation': [
            {
                'feature': feature_cols[i],
                'importance': float(perm_imp.importances_mean[i]),
                'std': float(perm_imp.importances_std[i])
            }
            for i in np.argsort(perm_imp.importances_mean)[::-1][:10]
        ]
    }
    
    report_path = os.path.join(args.out_dir, 'rf_training_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 训练报告已保存: {report_path}")
    
    # 13. 生成可视化
    print(f"\n[INFO] 生成可视化...")
    
    # 混淆矩阵热力图
    cm_heatmap_path = os.path.join(args.vis_dir, 'rf_confusion_matrix_heatmap.png')
    plot_confusion_matrix_heatmap(y, y_pred_cv, cm_heatmap_path, labels)
    
    # Per-class 性能指标
    per_class_path = os.path.join(args.vis_dir, 'rf_per_class_metrics.png')
    plot_per_class_metrics(report, per_class_path, labels)
    
    # Permutation Importance
    perm_imp_path = os.path.join(args.vis_dir, 'rf_permutation_importance.png')
    plot_permutation_importance(perm_imp, feature_cols, perm_imp_path, top_n=20)
    
    # Importance 对比
    comparison_path = os.path.join(args.vis_dir, 'rf_importance_comparison.png')
    plot_importance_comparison(rf, perm_imp, feature_cols, comparison_path, top_n=15)
    
    # 14. 总结
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"\n📁 输出文件:")
    print(f"  模型:     {model_path}")
    print(f"  概率:     {proba_path}")
    print(f"  报告:     {report_path}")
    print(f"\n📊 可视化:")
    print(f"  混淆矩阵: {cm_heatmap_path}")
    print(f"  Per-class: {per_class_path}")
    print(f"  Perm Imp:  {perm_imp_path}")
    print(f"  对比图:    {comparison_path}")
    
    print(f"\n🎯 关键指标:")
    print(f"  Accuracy:  {report['accuracy']:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    
    # 准备 vs 恢复混淆分析
    prep_to_rec = cm[1, 3]
    rec_to_prep = cm[3, 1]
    prep_total = cm[1].sum()
    rec_total = cm[3].sum()
    
    print(f"\n⚠️  准备 vs 恢复 混淆分析:")
    print(f"  准备→恢复: {prep_to_rec}/{prep_total} ({prep_to_rec/prep_total*100:.1f}%)")
    print(f"  恢复→准备: {rec_to_prep}/{rec_total} ({rec_to_prep/rec_total*100:.1f}%)")
    
    print("\n✅ 下一步建议:")
    print("  1. 查看混淆矩阵热力图，分析准备/恢复混淆情况")
    print("  2. 基于 rf_frame_proba.csv 实现 Viterbi 序列平滑")
    print("  3. 添加时间结构特征（窗口差分、滑动统计）")


if __name__ == '__main__':
    main()
