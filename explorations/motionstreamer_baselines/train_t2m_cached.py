"""
适配的T2M训练脚本 - 使用预计算的文本嵌入

修改点：
1. 移除T5模型加载
2. 直接从数据加载器获取预计算的文本嵌入
3. 节约大量GPU内存
"""

import os
import math
import torch
import numpy as np
import random
import json
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import warnings

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from accelerate import Accelerator

from models.llama_model import LLaMAHF, LLaMAHFConfig
from humanml3d_272 import dataset_TM_train_cached
import options.option_transformer as option_trans
import utils.utils_model as utils_model

warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

# warm-up + cosine decay scheduler
class WarmupCosineDecayScheduler:
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        
        self.cosine_scheduler = CosineAnnealingLR(optimizer, 
                                                  T_max=total_iters - warmup_iters, 
                                                  eta_min=min_lr)
        
    def warmup_lambda(self, current_iter):
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        return 1.0

    def step(self, current_iter):
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step(current_iter - self.warmup_iters)

    def state_dict(self):
        return {
            'warmup_iters': self.warmup_iters,
            'total_iters': self.total_iters,
            'min_lr': self.min_lr,
        }

    def load_state_dict(self, state_dict):
        self.warmup_iters = state_dict['warmup_iters']
        self.total_iters = state_dict['total_iters']
        self.min_lr = state_dict['min_lr']


args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok=True)


##### ---- Accelerator Setup ---- #####
accelerator = Accelerator()
comp_device = accelerator.device

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


##### ---- Dataloader ---- #####
# 指定text_latent_dir来使用预计算的文本嵌入
text_latent_dir = os.path.join('./humanml3d_272', 'text_latents_t5')
train_loader = dataset_TM_train_cached.DATALoader(
    args.dataname, 
    args.batch_size, 
    args.latent_dir,
    text_latent_dir=text_latent_dir,
    unit_length=2**args.down_t
)

# 加载空文本嵌入（用于文本mask操作）
empty_text_embedding_path = os.path.join(text_latent_dir, 'empty_text_embedding.npy')
if os.path.exists(empty_text_embedding_path):
    empty_text_embedding = torch.from_numpy(np.load(empty_text_embedding_path)).float()
    logger.info(f"✓ 加载空文本嵌入: {empty_text_embedding_path}")
    logger.info(f"  形状: {empty_text_embedding.shape}")
else:
    logger.warning(f"❌ 未找到空文本嵌入文件: {empty_text_embedding_path}")
    logger.warning("使用零向量作为空文本嵌入（不推荐）")
    empty_text_embedding = None


##### ---- Network ---- #####
config = LLaMAHFConfig.from_name('Normal_size')
config.block_size = 78
trans_encoder = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)

if args.resume_trans is not None:
    print('从 {} 加载transformer检查点'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    new_ckpt_trans = {}
    for key in ckpt['trans'].keys():
        if key.split('.')[0]=='module':
            new_key = '.'.join(key.split('.')[1:])
        else:
            new_key = key
        new_ckpt_trans[new_key] = ckpt['trans'][key]
    trans_encoder.load_state_dict(new_ckpt_trans, strict=True)

trans_encoder.train()
trans_encoder.to(comp_device)


##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = WarmupCosineDecayScheduler(optimizer, args.total_iter//10, args.total_iter)


trans_encoder, optimizer, train_loader = accelerator.prepare(trans_encoder, optimizer, train_loader)
train_loader_iter = dataset_TM_train_cached.cycle(train_loader)


diffmlps_batch_mul = 4
def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


#--------------2-forward strategy------------------
def cosine_decay(step, total_steps, start_value=1.0, end_value=0.0):
    step = torch.tensor(step, dtype=torch.float32)  
    total_steps = torch.tensor(total_steps, dtype=torch.float32)  
    
    cosine_factor = 0.5 * (1 + torch.cos(torch.pi * step / total_steps))
    return start_value + (end_value - start_value) * cosine_factor


def replace_with_pred(latents, pred_xstart, step, total_steps):
    decay_factor = cosine_decay(step, total_steps).to(latents.device)
    
    b, l, d = latents.shape
    num_replace = int(l * decay_factor)  
    
    replace_indices = torch.randperm(l)[:num_replace]  

    replace_mask = torch.zeros(b, l, dtype=torch.bool).to(latents.device)
    replace_mask[:, replace_indices] = 1  

    updated_latents = latents.clone()  
    updated_latents[replace_mask] = pred_xstart[replace_mask]
    
    return updated_latents


def forward_loss_withmask_2_forward(latents, trans, m_lens, feat_text, step, total_steps):
    """z: condition; latents: gt"""
    #--------------First Forward:-------------------------
    conditions = trans(latents, feat_text)  
    conditions = conditions.contiguous()
    z = conditions[:,:-1,:]
    #-------------------------------------------------

    b, l, d = latents.shape     
    mask = lengths_to_mask(m_lens, l)       
    mask = mask.reshape(b * l).repeat(diffmlps_batch_mul)

    target = latents.clone().detach()       
    target = target.reshape(b * l, -1)    
    z = z.reshape(b * l, -1)            
    
    with torch.no_grad():
        loss, pred_xstart = trans.diff_loss(target=target, z=z)  

    pred_xstart = pred_xstart.clone().detach()
    pred_xstart = pred_xstart.reshape(b, l, -1)           

    #--------------Second Forward:-------------------------
    # Update latents
    updated_latents = replace_with_pred(latents, pred_xstart, step, total_steps)    
    updated_conditions = trans(updated_latents, feat_text)  
    updated_conditions = updated_conditions.contiguous()
    updated_z = updated_conditions[:,:-1,:]      

    updated_target = latents.clone().detach()       

    updated_target = updated_target.reshape(b * l, -1).repeat(diffmlps_batch_mul, 1)    
    updated_z = updated_z.reshape(b * l, -1).repeat(diffmlps_batch_mul, 1)            

    updated_target = updated_target[mask]                   
    updated_z = updated_z[mask]                            

    updated_loss, _ = trans.diff_loss(target=updated_target, z=updated_z)  

    return updated_loss
#-------------------


##### ---- Training Loop ---- #####
nb_iter, avg_loss = 0, 0.

logger.info("开始训练...")
logger.info(f"文本嵌入目录: {text_latent_dir}")

while nb_iter <= args.total_iter:
    batch = next(train_loader_iter)
    feat_text, m_tokens, m_tokens_len = batch
    
    # feat_text 已经是预计算的嵌入，无需再编码
    feat_text = feat_text.to(comp_device)
    m_tokens, m_tokens_len = m_tokens.to(comp_device), m_tokens_len.to(comp_device)

    bs = feat_text.shape[0]
    
    # 10% 的batch进行文本mask（使用真实的空文本嵌入）
    num_masked = int(bs * 0.1)
    if num_masked > 0:
        mask_indices = random.sample(range(bs), num_masked)
        
        if empty_text_embedding is not None:
            # 使用预生成的空文本嵌入（与原始train_t2m.py一致）
            empty_embedding_device = empty_text_embedding.to(comp_device)
            # 如果原始feat_text形状是(bs, 1, 768)，需要匹配
            if feat_text.dim() == 3:
                feat_text[mask_indices] = empty_embedding_device.expand(len(mask_indices), -1, -1)
            else:
                feat_text[mask_indices] = empty_embedding_device.expand(len(mask_indices), -1)
        else:
            # fallback: 使用零向量（不推荐）
            feat_text[mask_indices] = 0.0

    # -------gt-------- 
    input_latent = m_tokens[:,:-1]    # continuous token
    loss = 0.0

    if args.num_gpus > 1:
        loss = forward_loss_withmask_2_forward(
            latents=input_latent, 
            trans=trans_encoder.module, 
            m_lens=m_tokens_len, 
            feat_text=feat_text, 
            step=nb_iter, 
            total_steps=args.total_iter
        )
    else:
        loss = forward_loss_withmask_2_forward(
            latents=input_latent, 
            trans=trans_encoder, 
            m_lens=m_tokens_len, 
            feat_text=feat_text, 
            step=nb_iter, 
            total_steps=args.total_iter
        )

    
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step(nb_iter)

    avg_loss = avg_loss + loss.item()

    nb_iter += 1
    args.print_iter = 100
    if nb_iter % args.print_iter == 0:
        if accelerator.is_main_process:
            avg_loss = avg_loss / args.print_iter
            writer.add_scalar('./Loss/train', avg_loss, nb_iter)
            writer.add_scalar('./LR/train', optimizer.param_groups[0]['lr'], nb_iter)
            msg = f"训练. 迭代 {nb_iter} : 损失. {avg_loss:.5f}"
            logger.info(msg)
        avg_loss = 0.

    args.save_iter = 10000
    if nb_iter % args.save_iter == 0:
        # save 
        if accelerator.is_main_process:
            torch.save({
                'trans': trans_encoder.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(args.out_dir, f'latest.pth'))
            logger.info(f"检查点已保存到: {os.path.join(args.out_dir, f'latest.pth')}")

                    
accelerator.wait_for_everyone()
logger.info("训练完成!")
