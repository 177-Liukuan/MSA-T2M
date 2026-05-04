"""
基础数据集探索脚本 - 查看数据集的基本信息和统计
"""
import numpy as np
import os
from pathlib import Path


def explore_basic_info():
    """探索数据集基本信息"""
    data_root = Path('../')

    print("=" * 60)
    print("HumanML3D-272 数据集基本信息")
    print("=" * 60)

    # 1. 统计motion数据文件数量
    motion_dir = data_root / 'motion_data'
    motion_files = sorted(motion_dir.glob('*.npy'))
    print(f"\n📁 Motion数据文件总数: {len(motion_files)}")

    # 2. 加载并查看第一个motion样本
    print(f"\n📊 加载第一个motion样本: {motion_files[0].name}")
    first_motion = np.load(motion_files[0])
    print(f"   - Shape: {first_motion.shape}")
    print(f"   - 数据类型: {first_motion.dtype}")
    print(f"   - 数值范围: [{first_motion.min():.4f}, {first_motion.max():.4f}]")
    print(f"   - 平均值: {first_motion.mean():.4f}")
    print(f"   - 标准差: {first_motion.std():.4f}")

    # 3. 统计所有motion的shape分布
    print(f"\n📏 统计所有motion的长度分布...")
    lengths = []
    for motion_file in motion_files[:100]:  # 只统计前100个以加快速度
        motion = np.load(motion_file)
        lengths.append(motion.shape[0])

    lengths = np.array(lengths)
    print(f"   - 最短序列长度: {lengths.min()}")
    print(f"   - 最长序列长度: {lengths.max()}")
    print(f"   - 平均序列长度: {lengths.mean():.2f}")
    print(f"   - 中位数序列长度: {np.median(lengths):.2f}")

    # 4. 加载归一化参数
    print(f"\n🔢 归一化参数:")
    mean = np.load(data_root / 'mean_std' / 'Mean.npy')
    std = np.load(data_root / 'mean_std' / 'Std.npy')
    print(f"   - Mean shape: {mean.shape}")
    print(f"   - Std shape: {std.shape}")
    print(f"   - Mean 前5个值: {mean[:5]}")
    print(f"   - Std 前5个值: {std[:5]}")

    # 5. 查看文本数据
    text_dir = data_root / 'texts'
    if text_dir.exists():
        text_files = sorted(text_dir.glob('*.txt'))
        print(f"\n📝 文本描述文件总数: {len(text_files)}")
        if len(text_files) > 0:
            print(f"   第一个文本文件: {text_files[0].name}")
            with open(text_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"   内容预览: {content}...")

    # 6. 查看数据集划分
    split_dir = data_root / 'split'
    if split_dir.exists():
        print(f"\n🔀 数据集划分:")
        for split_file in split_dir.glob('*.txt'):
            with open(split_file, 'r') as f:
                lines = f.readlines()
                print(f"   - {split_file.name}: {len(lines)} 样本")


if __name__ == '__main__':
    explore_basic_info()
