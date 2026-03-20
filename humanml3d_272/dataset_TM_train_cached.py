"""
文本-运动数据集加载模块（使用预计算的文本嵌入）

修改点：
1. __getitem__ 返回预计算的feat_text而不是caption
2. 添加text_latent_dir参数
3. 从磁盘加载预计算的文本嵌入
"""

import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)


'''对于使用预计算文本嵌入训练文本-运动生成模型'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, unit_length=4, latent_dir=None, text_latent_dir=None):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.unit_length = unit_length
        self.text_latent_dir = text_latent_dir

        if dataset_name == 't2m_272':
            self.data_root = './humanml3d_272'
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            fps = 30
            self.max_motion_length = 78   
            dim_pose = 272
            split_file = pjoin(self.data_root, 'split', 'train.txt')

        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
     
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                # 加载运动token
                m_token_list = np.load(pjoin(latent_dir, '%s.npy'%name))
            except:
                continue

            # 读取文本（用于确定文本段长度）
            with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                text_data = []
                flag = False
                lines = f.readlines()

                for line in lines:
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    t_tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])

                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = t_tokens
                    text_dict['f_tag'] = f_tag
                    text_dict['to_tag'] = to_tag

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length):
                            m_token_list_new = [m_token_list[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)]] 

                            if len(m_token_list_new) == 0:
                                continue

                            new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                            data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                    'text':[text_dict]}
                            new_name_list.append(new_name)
                    
            if flag:
                data_dict[name] = {'m_token_list': m_token_list,
                                    'text':text_data}
                new_name_list.append(name)

        self.data_dict = data_dict
        self.name_list = new_name_list

        # 验证文本嵌入是否存在
        if self.text_latent_dir is not None:
            print("验证文本嵌入文件...")
            missing_count = 0
            for name in self.name_list:
                # 根据名称获取基础ID（处理分割的情况）
                base_name = name.split('_')[0]
                text_latent_path = pjoin(self.text_latent_dir, base_name + '.npy')
                if not os.path.exists(text_latent_path):
                    missing_count += 1
            
            if missing_count > 0:
                print(f"警告: {missing_count} 个文本嵌入文件缺失!")
            else:
                print(f"✓ 所有 {len(set([n.split('_')[0] for n in self.name_list]))} 个文本嵌入文件已找到")
            
            # 检查空文本嵌入是否存在
            empty_text_path = pjoin(self.text_latent_dir, 'empty_text_embedding.npy')
            if os.path.exists(empty_text_path):
                print(f"✓ 空文本嵌入已找到: {empty_text_path}")
            else:
                print(f"⚠ 警告: 未找到空文本嵌入文件: {empty_text_path}")
                print(f"  请运行 prepare_text_embeddings.py 重新生成")

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = np.array(m_token_list)

        text_data = random.choice(text_list)
        caption = text_data['caption']

        if len(m_tokens.shape) == 3:
            m_tokens = m_tokens.squeeze(0)
        
        coin = np.random.choice([False, False, True])
        if coin:
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        
        m_tokens_len = m_tokens.shape[0]
        
        if m_tokens_len < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.zeros((self.max_motion_length-m_tokens_len, m_tokens.shape[1]), dtype=int)], axis=0)
        
        # 加载预计算的文本嵌入
        if self.text_latent_dir is not None:
            base_name = self.name_list[item].split('_')[0]
            text_latent_path = pjoin(self.text_latent_dir, base_name + '.npy')
            
            try:
                feat_text_all = np.load(text_latent_path)  # shape: (num_texts, text_dim)
                
                # 如果有多个文本描述，随机选择一个
                if len(feat_text_all.shape) > 1 and feat_text_all.shape[0] > 1:
                    text_idx = random.randint(0, feat_text_all.shape[0] - 1)
                    feat_text = feat_text_all[text_idx]  # 返回 [text_dim]，让collate处理批处理
                else:
                    if len(feat_text_all.shape) > 1:
                        feat_text = feat_text_all[0]  # 从 [1, text_dim] 提取 [text_dim]
                    else:
                        feat_text = feat_text_all  # 已经是 [text_dim]
                
                # 转换为torch张量
                feat_text = torch.from_numpy(feat_text).float()
                
            except FileNotFoundError:
                print(f"错误: 未找到文本嵌入文件: {text_latent_path}")
                # 返回零张量作为fallback - 返回 [text_dim] 而不是 [1, text_dim]
                feat_text = torch.zeros(768)  # 通常T5输出维度是768
            
            return feat_text, m_tokens, m_tokens_len
        else:
            # 如果没有预计算的嵌入，返回原始caption
            return caption, m_tokens, m_tokens_len


def DATALoader(dataset_name,
                batch_size, 
                latent_dir, 
                text_latent_dir=None,
                unit_length=4,
                num_workers=8):
    """
    创建数据加载器
    
    Args:
        dataset_name: 数据集名称
        batch_size: 批大小
        latent_dir: 运动latent目录
        text_latent_dir: 文本嵌入目录（如果为None，将返回原始文本）
        unit_length: 单元长度
        num_workers: 数据加载进程数
    """
    
    train_loader = torch.utils.data.DataLoader(
        Text2MotionDataset(
            dataset_name, 
            latent_dir=latent_dir, 
            text_latent_dir=text_latent_dir,
            unit_length=unit_length
        ),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
