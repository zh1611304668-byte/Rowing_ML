"""
元数据提取工具

从文件路径中提取关键元数据信息，用于后续的分组交叉验证。
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional


def extract_metadata_from_filename(filepath: str) -> Dict[str, str]:
    """
    从文件路径提取元数据
    
    参数:
        filepath: 文件完整路径
        
    返回:
        包含元数据的字典:
        - rower_level: 'club-level' 或 'elite'
        - boat_type: 'Boat2x', 'Boat', 'Pelvis2x', 'Waist' 等
        - date: 'YYYYMMDD' 格式
        - time: 'HHMMSS' 格式
        - session_id: 训练会话ID
        - device_id: 设备标识符 (如 'rpc364')
        - filename: 原始文件名
        
    示例:
        输入: 'd:/row_data/club-level/iPhone/Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_*.csv'
        输出: {
            'rower_level': 'club-level',
            'boat_type': 'Boat2x',
            'date': '20180420',
            'time': '085713',
            'session_id': '1633',
            'device_id': 'rpc364',
            'filename': 'Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_*.csv'
        }
    """
    filepath = str(filepath)
    path = Path(filepath)
    filename = path.name
    
    metadata = {
        'filename': filename,
        'rower_level': None,
        'boat_type': None,
        'date': None,
        'time': None,
        'session_id': None,
        'device_id': None
    }
    
    # 1. 从路径中提取 rower_level
    if 'club-level' in filepath or 'club_level' in filepath:
        metadata['rower_level'] = 'club-level'
    elif 'elite' in filepath:
        metadata['rower_level'] = 'elite'
    
    # 2. 从文件名提取信息
    # 文件名模式: Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_*.csv
    # 或: Boat-20180422T103229_1641_rpc364_data_1CLX_1_B_*.csv
    
    # 提取 boat_type (文件名开头到第一个'-'之间)
    boat_match = re.match(r'^([^-]+)', filename)
    if boat_match:
        metadata['boat_type'] = boat_match.group(1)
    
    # 提取 date 和 time (格式: YYYYMMDDTHHMMSS)
    datetime_match = re.search(r'(\d{8})T(\d{6})', filename)
    if datetime_match:
        metadata['date'] = datetime_match.group(1)
        metadata['time'] = datetime_match.group(2)
    
    # 提取 session_id (日期时间后的数字)
    session_match = re.search(r'T\d{6}_(\d+)_', filename)
    if session_match:
        metadata['session_id'] = session_match.group(1)
    
    # 提取 device_id (session_id 后的标识符)
    device_match = re.search(r'_\d+_([a-zA-Z0-9]+)_data', filename)
    if device_match:
        metadata['device_id'] = device_match.group(1)
    
    return metadata


def extract_metadata_from_clean_filename(filepath: str) -> Dict[str, str]:
    """
    从清洗后的文件名提取元数据
    
    清洗后的文件名通常是: *_clean.csv
    需要从原始文件名模式中提取
    
    参数:
        filepath: 清洗后的文件完整路径
        
    返回:
        元数据字典
    """
    filepath = str(filepath)
    
    # 如果是 _clean.csv 结尾，去掉后缀
    if filepath.endswith('_clean.csv'):
        original_pattern = filepath.replace('_clean.csv', '.csv')
    else:
        original_pattern = filepath
    
    return extract_metadata_from_filename(original_pattern)


def add_metadata_to_dataframe(df, filepath: str):
    """
    将元数据作为新列添加到 DataFrame
    
    参数:
        df: pandas DataFrame
        filepath: 数据文件路径
        
    返回:
        添加了元数据列的 DataFrame
    """
    import pandas as pd
    
    metadata = extract_metadata_from_filename(filepath)
    
    # 添加元数据列（所有行使用相同的值）
    for key, value in metadata.items():
        if key != 'filename':  # filename 不需要添加到数据中
            df[key] = value
    
    return df


def get_unique_sessions(directory: str) -> list:
    """
    扫描目录，获取所有唯一的 session_id
    
    参数:
        directory: 数据目录路径
        
    返回:
        session_id 列表
    """
    sessions = set()
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                filepath = os.path.join(root, file)
                metadata = extract_metadata_from_filename(filepath)
                if metadata['session_id']:
                    sessions.add(metadata['session_id'])
    
    return sorted(list(sessions))


def print_metadata_summary(directory: str):
    """
    打印目录中所有文件的元数据摘要
    
    参数:
        directory: 数据目录路径
    """
    import pandas as pd
    
    all_metadata = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and not file.startswith('.'):
                filepath = os.path.join(root, file)
                metadata = extract_metadata_from_filename(filepath)
                all_metadata.append(metadata)
    
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        print("\n" + "="*80)
        print("元数据摘要")
        print("="*80)
        print(f"\n总文件数: {len(df)}")
        print(f"\n运动员级别分布:")
        print(df['rower_level'].value_counts())
        print(f"\n船只类型分布:")
        print(df['boat_type'].value_counts())
        print(f"\n唯一 session 数量: {df['session_id'].nunique()}")
        print(f"Session IDs: {sorted(df['session_id'].unique())}")
        print("="*80 + "\n")
    else:
        print("[WARNING] 未找到任何 CSV 文件")


if __name__ == '__main__':
    # 测试代码
    print("元数据提取工具测试\n")
    
    # 测试样例
    test_files = [
        'd:/Desktop/python/rowing_ML/row_data/club-level/iPhone/Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_F92041BC-2503-4150-8196-2B45C0258ED8.csv',
        'd:/Desktop/python/rowing_ML/row_data/elite/iPhone/Boat-20180422T103229_1641_rpc364_data_1CLX_1_B_2CDF0487-83FC-45CC-B590-FF42D74E0D6D.csv',
        'd:/Desktop/python/rowing_ML/clean_report/Boat2x-20180420T085713_1633_rpc364_data_1CLX_1_B_F92041BC-2503-4150-8196-2B45C0258ED8_clean.csv'
    ]
    
    for filepath in test_files:
        print(f"文件: {Path(filepath).name}")
        metadata = extract_metadata_from_filename(filepath)
        for key, value in metadata.items():
            if key != 'filename':
                print(f"  {key:15s}: {value}")
        print()
