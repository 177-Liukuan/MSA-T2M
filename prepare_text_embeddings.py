"""
准备文本嵌入 - 使用T5模型预计算所有文本的嵌入并保存到磁盘
这样在训练时无需加载T5模型，节约GPU内存
"""

import os
import torch
import numpy as np
from os.path import join as pjoin
import codecs as cs
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


def prepare_text_embeddings(dataset_name='t2m_272', 
                            batch_size=64,
                            output_dir='./humanml3d_272/text_latents',
                            t5_model_path='sentencet5-xxl/'):
    """
    预处理文本嵌入
    
    Args:
        dataset_name: 数据集名称 ('t2m_272')
        batch_size: 处理批大小
        output_dir: 输出嵌入的目录
        t5_model_path: T5模型路径
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载T5模型
    print(f"加载T5模型: {t5_model_path}")
    t5_model = SentenceTransformer(t5_model_path)
    t5_model.eval()
    for p in t5_model.parameters():
        p.requires_grad = False
    
    # 使用CUDA如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t5_model.to(device)
    print(f"使用设备: {device}")
    
    # 设置数据集路径
    if dataset_name == 't2m_272':
        data_root = './humanml3d_272'
        text_dir = pjoin(data_root, 'texts')
        split_file = pjoin(data_root, 'split', 'train.txt')
        fps = 30
        unit_length = 4
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 读取split文件获取所有ID
    id_list = []
    with cs.open(split_file, 'r') as f:
        for line in f.readlines():
            id_list.append(line.strip())
    
    print(f"总共要处理的视频数: {len(id_list)}")
    
    # 处理每个视频的文本
    processed_count = 0
    skipped_count = 0
    
    for name in tqdm(id_list, desc="处理文本嵌入"):
        try:
            # 读取文本文件
            with cs.open(pjoin(text_dir, name + '.txt')) as f:
                lines = f.readlines()
            
            if not lines:
                print(f"警告: {name} 的文本文件为空，跳过")
                skipped_count += 1
                continue
            
            # 提取所有文本描述
            text_list = []
            for line in lines:
                line_split = line.strip().split('#')
                caption = line_split[0]
                # 直接使用原始文本，包括空字符串
                # （空字符串会被T5正确编码）
                text_list.append(caption)
            
            # 编码文本
            with torch.no_grad():
                feat_text = t5_model.encode(text_list)
                feat_text = torch.from_numpy(feat_text).float()
            
            # 保存嵌入
            save_path = pjoin(output_dir, name + '.npy')
            np.save(save_path, feat_text.cpu().numpy())
            processed_count += 1
            
        except Exception as e:
            print(f"错误处理 {name}: {str(e)}")
            skipped_count += 1
            continue
    
    # 生成并保存空文本嵌入（用于文本mask操作）
    print("\n生成空文本嵌入...")
    empty_text = ''
    with torch.no_grad():
        empty_feat_text = t5_model.encode([empty_text])
        empty_feat_text = torch.from_numpy(empty_feat_text).float()
    
    empty_text_path = pjoin(output_dir, 'empty_text_embedding.npy')
    np.save(empty_text_path, empty_feat_text.cpu().numpy())
    print(f"✓ 空文本嵌入已保存到: {empty_text_path}")
    print(f"  形状: {empty_feat_text.shape}")
    
    print(f"\n完成!")
    print(f"成功处理: {processed_count}")
    print(f"跳过: {skipped_count}")
    print(f"文本嵌入已保存到: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='准备文本嵌入')
    parser.add_argument('--dataset-name', type=str, default='t2m_272', 
                       help='数据集名称')
    parser.add_argument('--output-dir', type=str, default='./humanml3d_272/text_latents',
                       help='输出嵌入的目录')
    parser.add_argument('--t5-model-path', type=str, default='sentencet5-xxl/',
                       help='T5模型路径')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='处理批大小')
    
    args = parser.parse_args()
    
    prepare_text_embeddings(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        t5_model_path=args.t5_model_path
    )
