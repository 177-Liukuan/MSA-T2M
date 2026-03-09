import os
import sys
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import torch.nn as nn
import models.msa_vae as msa_vae
import options.option_msa_vae as option_msa_vae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from humanml3d_272 import dataset_eval_t2m
import warnings
warnings.filterwarnings('ignore')

os.chdir('Evaluator_272')
sys.path.insert(0, os.getcwd())

comp_device = torch.device('cuda')

##### ---- Args ---- #####
args = option_msa_vae.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Data ---- #####
val_loader = dataset_eval_t2m.DATALoader(args.dataname, True, 32,
                                          unit_length=2 ** args.down_t)

##### ---- Network ---- #####
clip_range = [-30, 20]
net = msa_vae.MSA_HumanVAE(
    hidden_size=args.hidden_size,
    down_t=args.down_t,
    stride_t=args.stride_t,
    depth=args.depth,
    dilation_growth_rate=args.dilation_growth_rate,
    activation='relu',
    latent_dim=args.latent_dim,
    clip_range=clip_range,
    trans_d_model=args.trans_d_model,
    trans_nhead=args.trans_nhead,
    trans_enc_layers=args.trans_enc_layers,
    trans_dec_layers=args.trans_dec_layers,
    trans_ff_size=args.trans_ff_size,
    trans_dropout=args.trans_dropout,
    clip_dim=args.clip_dim,
)

print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
state = ckpt['net'] if (isinstance(ckpt, dict) and 'net' in ckpt) else ckpt
net.load_state_dict(state, strict=True)
net.eval()
net.to(comp_device)


class EvalCompat(nn.Module):
    """Thin wrapper so net(x) returns (x_recon, mu, logvar) tuple."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out['x_recon'], out['mu'], out['logvar']


net_eval = EvalCompat(net)

##### ---- Evaluator ---- #####
from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

modelpath = './deps/distilbert-base-uncased'
textencoder  = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)

ckpt_eval = torch.load('./experiments/temos/EXP1/checkpoints/epoch=99.ckpt')

textencoder_ckpt = {k.replace('textencoder.', ''): v
                    for k, v in ckpt_eval['state_dict'].items()
                    if k.startswith('textencoder.')}
textencoder.load_state_dict(textencoder_ckpt, strict=True)
textencoder.eval()
textencoder.to(comp_device)

motionencoder_ckpt = {k.replace('motionencoder.', ''): v
                      for k, v in ckpt_eval['state_dict'].items()
                      if k.startswith('motionencoder.')}
motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
motionencoder.eval()
motionencoder.to(comp_device)

evaluator = [textencoder, motionencoder]

##### ---- Evaluate (repeat 3 times for stability) ---- #####
fid_list, mpjpe_list, div_list = [], [], []
top1_list, top2_list, top3_list, mm_list = [], [], [], []

num_repeats = 3
for rep in range(num_repeats):
    fid, mpjpe, div, top1, top2, top3, mm_dist, writer, logger = \
        eval_trans.evaluation_msa_vae_single(
            args.out_dir, val_loader, net_eval, logger, writer,
            evaluator=evaluator, device=comp_device,
        )
    fid_list.append(fid);   mpjpe_list.append(mpjpe);  div_list.append(div)
    top1_list.append(top1); top2_list.append(top2);    top3_list.append(top3)
    mm_list.append(mm_dist)

logger.info('\nFinal result (mean over {} repeats):'.format(num_repeats))
logger.info(f'FID:     {np.mean(fid_list):.4f} +/- {np.std(fid_list):.4f}')
logger.info(f'MPJPE:   {np.mean(mpjpe_list):.3f} mm')
logger.info(f'R@1:     {np.mean(top1_list):.4f}')
logger.info(f'R@2:     {np.mean(top2_list):.4f}')
logger.info(f'R@3:     {np.mean(top3_list):.4f}')
logger.info(f'MM-dist: {np.mean(mm_list):.4f}')
logger.info(f'Div:     {np.mean(div_list):.4f}')
