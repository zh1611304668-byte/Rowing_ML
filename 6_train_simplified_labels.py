#!/usr/bin/env python3
"""
简化标签训练脚本
将5类标签简化为4类：合并 背景(0) 和 过渡(4) 为 "背景/过渡(0)"

新标签映射：
- 0: 背景/过渡 (原0和4)
- 1: 准备 (原1)
- 2: 核心 (原2)
- 3: 恢复 (原3)
"""

import argparse
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免线程错误
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def find_latest_features_file(search_dir='datasets'):
    pattern = os.path.join(search_dir, '*features*.csv')
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def simplify_labels(y):
    """将5类标签简化为4类"""
    # 原标签: 0=背景, 1=准备, 2=核心, 3=恢复, 4=过渡
    # 新标签: 0=背景/过渡, 1=准备, 2=核心, 3=恢复
    y_new = y.copy()
    y_new[y == 4] = 0  # 过渡 → 背景/过渡
    return y_new


def create_stroke_groups(df):
    """
    创建分组标识，优先级：session_id > stroke_id > time-based
    
    用于 Leave-One-Session-Out (LOSO) 交叉验证
    """
    # 优先级1: 使用 session_id (最佳实践，用于LOSO)
    if 'session_id' in df.columns:
        groups = df['session_id'].values
        n_groups = len(np.unique(groups))
        print(f"[INFO] ✓ 使用 session_id 分组，共 {n_groups} 个独立训练会话")
        print(f"[INFO]   → 将执行 Leave-One-Session-Out (LOSO) 验证")
        return groups, 'session_id'
    
    # 优先级2: 回退到 stroke_id
    if 'stroke_id' in df.columns:
        groups = df['stroke_id'].values
        n_groups = len(np.unique(groups))
        print(f"[WARNING] session_id 未找到，使用 stroke_id 分组，共 {n_groups} 组")
        print(f"[WARNING]   → 将执行 GroupShuffleSplit（警告：可能存在伪泛化）")
        return groups, 'stroke_id'
    
    # 优先级3: 回退到时间分组（最差方案）
    if 'time' in df.columns:
        groups = (df['time'].values).astype(int)
    else:
        groups = np.arange(len(df)) // 100
    
    n_groups = len(np.unique(groups))
    print(f"[WARNING] 未找到 session_id 或 stroke_id，使用伪分组 {n_groups} 个")
    print(f"[WARNING]   → 交叉验证结果可能不可靠！")
    return groups, 'pseudo'


def plot_confusion_matrix(y_true, y_pred, out_path, labels):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=False, cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            pct = cm_norm[i, j] * 100
            count = cm[i, j]
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{pct:.1f}%\n({count})',
                   ha='center', va='center', fontsize=11, color=color)
    
    ax.set_title('混淆矩阵 - 简化4类标签', fontsize=14)
    ax.set_ylabel('真实标签')
    ax.set_xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] 混淆矩阵已保存: {out_path}")


def plot_feature_importance(model, feature_cols, out_path, top_n=20):
    """绘制特征重要性图"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(indices)))
    
    ax.barh(range(len(indices)), importances[indices][::-1], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i][:25] for i in indices[::-1]], fontsize=10)
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title(f'Random Forest 特征重要性 (Top {top_n})', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] 特征重要性图已保存: {out_path}")


def plot_classification_metrics(report, labels, out_path):
    """绘制分类指标对比图（Precision/Recall/F1）"""
    metrics = ['precision', 'recall', 'f1-score']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    
    for ax_idx, metric in enumerate(metrics):
        values = [report.get(label, {}).get(metric, 0) for label in labels]
        bars = axes[ax_idx].bar(labels, values, color=colors[:len(labels)], alpha=0.8)
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        axes[ax_idx].set_ylim(0, 1.15)
        axes[ax_idx].set_ylabel(metric.capitalize(), fontsize=12)
        axes[ax_idx].set_title(f'{metric.capitalize()} per Class', fontsize=13)
        axes[ax_idx].grid(axis='y', alpha=0.3)
        axes[ax_idx].tick_params(axis='x', rotation=15)
    
    plt.suptitle('分类指标对比', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] 分类指标对比图已保存: {out_path}")


def plot_cv_fold_results(fold_f1s, out_path):
    """绘制交叉验证各折结果"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    folds = range(1, len(fold_f1s) + 1)
    colors = ['#4CAF50' if f1 > np.mean(fold_f1s) else '#FF5722' for f1 in fold_f1s]
    bars = ax.bar(folds, fold_f1s, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加均值线
    mean_f1 = np.mean(fold_f1s)
    ax.axhline(mean_f1, color='#2196F3', linestyle='--', linewidth=2, 
               label=f'平均 F1: {mean_f1:.4f}')
    
    # 添加数值标签
    for bar, val in zip(bars, fold_f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Macro F1', fontsize=12)
    ax.set_title('交叉验证各折 Macro F1 分数', fontsize=14)
    ax.set_xticks(folds)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] 交叉验证结果图已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='简化标签训练')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--cv_splits', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=300)
    parser.add_argument('--out_dir', type=str, default='models')
    parser.add_argument('--vis_dir', type=str, default='datasets/visualizations')
    args = parser.parse_args()
    
    # 加载数据
    data_file = args.data or find_latest_features_file()
    if not data_file:
        print("[ERROR] 未找到特征文件!")
        return
    
    print(f"\n[INFO] 加载: {data_file}")
    df = pd.read_csv(data_file)
    print(f"[INFO] 样本数: {len(df)}")
    
    # 分离特征和标签
    metadata_cols = ['label', 'time', 'stroke_id', 'session_id', 'date', 
                     'rower_level', 'boat_type', 'device_id']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    X = df[feature_cols].values
    y_original = df['label'].values
    
    # 简化标签
    y = simplify_labels(y_original)
    labels = ['背景/过渡', '准备', '核心', '恢复']
    
    print("\n[INFO] 标签简化映射:")
    print("  原5类 → 新4类")
    print("  背景(0) + 过渡(4) → 背景/过渡(0)")
    print("  准备(1) → 准备(1)")
    print("  核心(2) → 核心(2)")
    print("  恢复(3) → 恢复(3)")
    
    # 创建分组
    groups, group_type = create_stroke_groups(df)
    n_unique_groups = len(np.unique(groups))
    
    # 标签分布
    print("\n[INFO] 新标签分布:")
    for i, label in enumerate(labels):
        count = np.sum(y == i)
        pct = count / len(y) * 100
        print(f"  {label} ({i}): {count:6d} ({pct:5.2f}%)")
    
    # ========== 选择交叉验证策略 ==========
    # 根据分组类型选择最佳CV策略
    if group_type == 'session_id' and n_unique_groups >= 3:
        # 使用 Leave-One-Session-Out (最严格的泛化测试)
        cv_strategy = LeaveOneGroupOut()
        cv_name = "Leave-One-Session-Out (LOSO)"
        n_splits = n_unique_groups
        print(f"\n[INFO] 🎯 执行 {cv_name} 验证...")
        print(f"[INFO]   每次留出1个完整训练会话作为测试集")
        print(f"[INFO]   总共 {n_splits} 次交叉验证")
    else:
        # 回退到 GroupShuffleSplit
        cv_strategy = GroupShuffleSplit(n_splits=args.cv_splits, test_size=0.2, random_state=42)
        cv_name = f"GroupShuffleSplit ({args.cv_splits}次)"
        n_splits = args.cv_splits
        print(f"\n[INFO] 执行 {cv_name} 验证...")
        if group_type != 'session_id':
            print(f"[WARNING] ⚠️  未使用session_id，验证结果可能存在伪泛化！")
    
    y_pred_cv = np.full(len(y), -1, dtype=int)  # -1表示未预测
    fold_f1s = []
    fold_results = []  # 存储每个fold的详细结果
    
    for fold, (train_idx, test_idx) in enumerate(cv_strategy.split(X, y, groups), 1):
        # 获取当前fold测试的session信息
        if group_type == 'session_id':
            test_sessions = np.unique(groups[test_idx])
            session_info = f", Test Session: {test_sessions[0]}"
        else:
            session_info = ""
        
        print(f"  Fold {fold}/{n_splits}: Train={len(train_idx)}, Test={len(test_idx)}{session_info}")
        
        rf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=None,
            min_samples_leaf=5,
            class_weight=None,
            n_jobs=-1,
            random_state=42 + fold
        )
        rf.fit(X[train_idx], y[train_idx])
        pred = rf.predict(X[test_idx])
        y_pred_cv[test_idx] = pred
        
        # 计算每折的F1
        from sklearn.metrics import f1_score
        fold_f1 = f1_score(y[test_idx], pred, average='macro', zero_division=0)
        fold_f1s.append(fold_f1)
        
        # 存储详细结果
        fold_result = {
            'fold': fold,
            'f1_score': fold_f1,
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        }
        if group_type == 'session_id':
            fold_result['test_session'] = str(test_sessions[0])
        fold_results.append(fold_result)
        
        print(f"    Fold {fold} Macro F1: {fold_f1:.4f}")
    
    # 对未被预测的样本用最后一个模型预测
    not_predicted = y_pred_cv == -1
    if np.any(not_predicted):
        y_pred_cv[not_predicted] = rf.predict(X[not_predicted])
    
    print(f"\n  {'='*50}")
    print(f"  {cv_name} 结果汇总")
    print(f"  {'='*50}")
    print(f"  平均 Macro F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"  最佳 Fold F1:  {np.max(fold_f1s):.4f}")
    print(f"  最差 Fold F1:  {np.min(fold_f1s):.4f}")
    print(f"  {'='*50}")
    
    # 评估
    print("\n" + "="*60)
    print(f"{cv_name} - 简化标签结果")
    print("="*60)
    
    report = classification_report(y, y_pred_cv, target_names=labels, 
                                   output_dict=True, zero_division=0)
    print("\n分类报告:")
    print(classification_report(y, y_pred_cv, target_names=labels, zero_division=0))
    
    macro_f1 = report['macro avg']['f1-score']
    accuracy = report['accuracy']
    print(f"\n⭐ Macro F1: {macro_f1:.4f}")
    print(f"⭐ Accuracy: {accuracy:.4f}")
    
    # 保存可视化
    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    plot_confusion_matrix(y, y_pred_cv,
                         os.path.join(args.vis_dir, 'rf_simplified_confusion_matrix.png'),
                         labels)
    
    # 新增：分类指标对比图
    plot_classification_metrics(report, labels,
                               os.path.join(args.vis_dir, 'rf_classification_metrics.png'))
    
    # 新增：交叉验证各折结果图
    plot_cv_fold_results(fold_f1s,
                        os.path.join(args.vis_dir, 'rf_cv_fold_results.png'))
    
    # 训练最终模型
    print("\n[INFO] 训练最终模型...")
    rf_final = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight=None,
        n_jobs=-1,
        random_state=42
    )
    rf_final.fit(X, y)
    
    # 新增：特征重要性图
    plot_feature_importance(rf_final, feature_cols,
                           os.path.join(args.vis_dir, 'rf_feature_importance.png'),
                           top_n=25)
    
    # 保存
    model_path = os.path.join(args.out_dir, 'rf_simplified_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': rf_final,
            'feature_cols': feature_cols,
            'labels': labels,
            'label_mapping': '0=背景/过渡, 1=准备, 2=核心, 3=恢复'
        }, f)
    print(f"[INFO] 模型已保存: {model_path}")
    
    # 保存报告
    report_data = {
        'n_samples': len(df),
        'n_classes': 4,
        'cv_strategy': cv_name,
        'n_unique_groups': int(n_unique_groups),
        'group_type': group_type,
        'macro_f1': float(macro_f1),
        'macro_f1_std': float(np.std(fold_f1s)),
        'accuracy': float(accuracy),
        'fold_results': fold_results,
        'per_class_f1': {label: float(report.get(label, {}).get('f1-score', 0)) 
                        for label in labels}
    }
    report_path = os.path.join(args.out_dir, 'rf_simplified_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    print(f"⭐ 简化4类 Macro F1: {macro_f1:.4f}")
    print(f"⭐ 对比原5类 Macro F1: ~0.54")
    print(f"  改善: {(macro_f1 - 0.54) * 100:+.1f}%")


if __name__ == '__main__':
    main()
