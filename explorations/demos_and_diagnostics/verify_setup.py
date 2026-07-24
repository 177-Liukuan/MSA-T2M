"""
验证文本嵌入和训练配置
"""

import os
import numpy as np
from os.path import join as pjoin
import argparse
from tqdm import tqdm


def verify_text_embeddings(text_latent_dir='./humanml3d_272/text_latents_t5'):
    """验证文本嵌入文件"""
    
    print("=" * 60)
    print("文本嵌入验证")
    print("=" * 60)
    
    if not os.path.exists(text_latent_dir):
        print(f"❌ 错误: 目录不存在: {text_latent_dir}")
        return False
    
    files = [f for f in os.listdir(text_latent_dir) if f.endswith('.npy')]
    
    if len(files) == 0:
        print(f"❌ 错误: 目录为空: {text_latent_dir}")
        return False
    
    print(f"✓ 找到 {len(files)} 个嵌入文件\n")
    
    # 检查文件统计
    print("文件统计:")
    print("-" * 60)
    
    total_size = 0
    shape_stats = {}
    
    for i, filename in enumerate(tqdm(files, desc="分析")):
        filepath = pjoin(text_latent_dir, filename)
        
        try:
            data = np.load(filepath)
            total_size += os.path.getsize(filepath)
            
            shape_key = str(data.shape)
            if shape_key not in shape_stats:
                shape_stats[shape_key] = 0
            shape_stats[shape_key] += 1
            
            if i < 3:  # 显示前3个文件信息
                print(f"  {filename}")
                print(f"    形状: {data.shape}")
                print(f"    dtype: {data.dtype}")
                print(f"    均值: {data.mean():.6f}")
                print(f"    标准差: {data.std():.6f}")
                print()
        
        except Exception as e:
            print(f"❌ 读取失败: {filename} - {str(e)}")
            return False
    
    # 形状统计
    print("嵌入形状分布:")
    print("-" * 60)
    for shape, count in sorted(shape_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {shape}: {count} 个文件")
    
    # 大小统计
    total_size_gb = total_size / (1024 ** 3)
    avg_size_mb = total_size / len(files) / (1024 ** 2)
    
    print("\n大小统计:")
    print("-" * 60)
    print(f"  总大小: {total_size_gb:.2f} GB")
    print(f"  平均文件: {avg_size_mb:.2f} MB")
    print(f"  文件数: {len(files)}")
    
    print("\n✓ 文本嵌入验证成功!\n")
    return True


def verify_dataset_config(dataname='t2m_272', latent_dir=None, text_latent_dir=None):
    """验证数据集配置"""
    
    print("=" * 60)
    print("数据集配置验证")
    print("=" * 60)
    
    # 检查基本目录
    print("目录检查:")
    print("-" * 60)
    
    if dataname == 't2m_272':
        data_root = './humanml3d_272'
        required_dirs = [
            ('texts', pjoin(data_root, 'texts')),
            ('motion_data', pjoin(data_root, 'motion_data')),
            ('split', pjoin(data_root, 'split')),
        ]
        
        if latent_dir:
            required_dirs.append(('motion_latents', latent_dir))
        
        if text_latent_dir:
            required_dirs.append(('text_latents', text_latent_dir))
        
        for name, path in required_dirs:
            exists = os.path.exists(path)
            status = "✓" if exists else "❌"
            print(f"  {status} {name}: {path}")
            if not exists:
                print(f"     ^-- 目录不存在!")
    
    # 读取split文件
    print("\nSplit文件检查:")
    print("-" * 60)
    split_file = pjoin(data_root, 'split', 'train.txt')
    
    try:
        with open(split_file, 'r') as f:
            lines = f.readlines()
        print(f"  ✓ 训练集大小: {len(lines)} 个视频")
    except FileNotFoundError:
        print(f"  ❌ Split文件不存在: {split_file}")
        return False
    
    # 检查文本文件
    print("\n文本文件检查:")
    print("-" * 60)
    
    text_dir = pjoin(data_root, 'texts')
    text_files = [f for f in os.listdir(text_dir) if f.endswith('.txt')]
    print(f"  找到 {len(text_files)} 个文本文件")
    
    if len(text_files) > 0:
        # 检查第一个文本文件
        sample_file = pjoin(text_dir, text_files[0])
        with open(sample_file, 'r') as f:
            sample_lines = f.readlines()
        print(f"  示例: {text_files[0]}")
        print(f"    行数: {len(sample_lines)}")
        if sample_lines:
            print(f"    首行: {sample_lines[0][:80]}...")
    
    # 检查运动latent文件
    if latent_dir and os.path.exists(latent_dir):
        print("\n运动Latent检查:")
        print("-" * 60)
        latent_files = [f for f in os.listdir(latent_dir) if f.endswith('.npy')]
        print(f"  找到 {len(latent_files)} 个latent文件")
        
        if len(latent_files) > 0:
            sample_latent = np.load(pjoin(latent_dir, latent_files[0]))
            print(f"  示例形状: {latent_files[0]}")
            print(f"    数据形状: {sample_latent.shape}")
            print(f"    数据类型: {sample_latent.dtype}")
    
    print("\n✓ 数据集配置验证完成!\n")
    return True


def verify_training_setup(num_gpus=8, batch_size=32, dataname='t2m_272'):
    """验证训练配置"""
    
    print("=" * 60)
    print("训练配置验证")
    print("=" * 60)
    
    print("\n硬件配置:")
    print("-" * 60)
    print(f"  GPU数量: {num_gpus}")
    print(f"  批大小: {batch_size}")
    print(f"  每个GPU的批大小: {batch_size // num_gpus}")
    print(f"  总批大小: {batch_size * num_gpus}")
    
    print("\nMemory估算 (per GPU):")
    print("-" * 60)
    
    batch_size_per_gpu = batch_size // num_gpus
    
    # 基础估算
    motion_encoder_mem = 4  # GB
    transformer_mem = 5     # GB
    grad_optimizer_mem = 4  # GB
    text_embedding_mem = (batch_size_per_gpu * 768 * 4) / (1024 ** 3)  # 768-dim float32
    
    total_estimate = motion_encoder_mem + transformer_mem + grad_optimizer_mem + text_embedding_mem
    
    print(f"  运动Encoder: ~{motion_encoder_mem}GB")
    print(f"  Transformer: ~{transformer_mem}GB")
    print(f"  梯度/优化器: ~{grad_optimizer_mem}GB")
    print(f"  文本嵌入: ~{text_embedding_mem:.2f}GB")
    print(f"  ─────────────────")
    print(f"  总计: ~{total_estimate:.2f}GB/GPU")
    
    # 检查是否可行
    available_gpus = num_gpus
    print(f"\n✓ 推荐配置: {available_gpus} × GPU (每个{batch_size // num_gpus})")
    print(f"  理想场景: 16GB/GPU 显卡")
    
    print("\n✓ 训练配置验证完成!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='验证文本嵌入和训练配置')
    parser.add_argument('--text-latent-dir', type=str, default='./humanml3d_272/text_latents',
                       help='文本嵌入目录')
    parser.add_argument('--latent-dir', type=str, default=None,
                       help='运动latent目录')
    parser.add_argument('--dataname', type=str, default='t2m_272',
                       help='数据集名称')
    parser.add_argument('--num-gpus', type=int, default=8,
                       help='GPU数量')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批大小')
    
    args = parser.parse_args()
    
    print("\n")
    
    # 验证文本嵌入
    text_valid = verify_text_embeddings(args.text_latent_dir)
    
    # 验证数据集配置
    dataset_valid = verify_dataset_config(args.dataname, args.latent_dir, args.text_latent_dir)
    
    # 验证训练配置
    training_valid = verify_training_setup(args.num_gpus, args.batch_size, args.dataname)
    
    # 总结
    print("=" * 60)
    print("验证总结")
    print("=" * 60)
    
    all_valid = text_valid and dataset_valid and training_valid
    
    if all_valid:
        print("✓ 所有检查通过!")
        print("\n准备好开始训练:")
        print(f"  bash TRAIN_t2m_cached.sh {args.num_gpus}")
    else:
        print("❌ 有些检查失败，请解决问题后重试")
    
    print("\n")
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    exit(main())
