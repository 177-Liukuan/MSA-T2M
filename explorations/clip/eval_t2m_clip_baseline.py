import os
import json
import sys
import torch
import numpy as np
import warnings
from torch.utils.tensorboard import SummaryWriter

from models.llama_model import LLaMAHF, LLaMAHFConfig
import options.option_transformer as option_trans
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from humanml3d_272 import dataset_eval_t2m
import models.tae as tae

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TextProjector(torch.nn.Module):
    """Project CLIP embeddings (512d) to model dimension (768d)."""

    def __init__(self, input_dim=512, output_dim=768):
        super().__init__()
        self.proj = torch.nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class CLIPProjectedTokenizer:
    """Tokenizer-like adapter exposing .encode() for eval_trans compatibility.

    encode(text) output shape follows SentenceTransformer behavior used in original code:
    - str -> (768,)
    - list[str] -> (N, 768)
    """

    def __init__(self, clip_model, text_projector, device):
        self.clip_model = clip_model
        self.text_projector = text_projector
        self.device = device

    @torch.no_grad()
    def encode(self, text):
        import clip

        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        tokens = clip.tokenize(text, truncate=True).to(self.device)
        clip_feat = self.clip_model.encode_text(tokens).float()      # [N, 512]
        proj_feat = self.text_projector(clip_feat).float()           # [N, 768]

        proj_np = proj_feat.cpu().numpy()
        if is_single:
            return proj_np[0]
        return proj_np


os.chdir('Evaluator_272')
sys.path.insert(0, os.getcwd())

comp_device = torch.device('cuda')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok=True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
val_loader = dataset_eval_t2m.DATALoader(args.dataname, True, 32)

##### ---- Network ---- #####
import clip

clip_model, _ = clip.load('ViT-B/32', device=comp_device, jit=False)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

text_projector = TextProjector(input_dim=512, output_dim=768).to(comp_device)
text_projector.eval()

# Causal TAE
clip_range = [-30, 20]
net = tae.Causal_HumanTAE(
    hidden_size=args.hidden_size,
    down_t=args.down_t,
    stride_t=args.stride_t,
    depth=args.depth,
    dilation_growth_rate=args.dilation_growth_rate,
    activation='relu',
    latent_dim=args.latent_dim,
    clip_range=clip_range
)

config = LLaMAHFConfig.from_name('Normal_size')
config.block_size = 78
trans_encoder = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)

print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.to(comp_device)

if args.resume_trans is not None:
    print('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')

    # load trans encoder
    new_ckpt_trans = {}
    for key in ckpt['trans'].keys():
        if key.split('.')[0] == 'module':
            new_key = '.'.join(key.split('.')[1:])
        else:
            new_key = key
        new_ckpt_trans[new_key] = ckpt['trans'][key]
    trans_encoder.load_state_dict(new_ckpt_trans, strict=True)

    # load text projector from CLIP-baseline checkpoint
    if 'text_projector' not in ckpt:
        raise KeyError('Checkpoint does not contain text_projector. Please use CLIP-baseline checkpoint.')

    new_ckpt_proj = {}
    for key in ckpt['text_projector'].keys():
        if key.split('.')[0] == 'module':
            new_key = '.'.join(key.split('.')[1:])
        else:
            new_key = key
        new_ckpt_proj[new_key] = ckpt['text_projector'][key]
    text_projector.load_state_dict(new_ckpt_proj, strict=True)

trans_encoder.eval()
trans_encoder.to(comp_device)
text_projector.eval()
text_projector.to(comp_device)

# adapter with encode() API expected by eval_trans + llama sampling
tokenize_model = CLIPProjectedTokenizer(clip_model, text_projector, comp_device)

# load evaluator:
from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

modelpath = './deps/distilbert-base-uncased'
textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)

ckpt_path = '../Evaluator_272/experiments/temos/EXP1/checkpoints/epoch=99.ckpt'
print(f'Loading evaluator checkpoint from {ckpt_path}')
ckpt = torch.load(ckpt_path)

textencoder_ckpt = {}
for k, v in ckpt['state_dict'].items():
    if k.split('.')[0] == 'textencoder':
        name = k.replace('textencoder.', '')
        textencoder_ckpt[name] = v
textencoder.load_state_dict(textencoder_ckpt, strict=True)
textencoder.eval()
textencoder.to(comp_device)

motionencoder_ckpt = {}
for k, v in ckpt['state_dict'].items():
    if k.split('.')[0] == 'motionencoder':
        name = k.replace('motionencoder.', '')
        motionencoder_ckpt[name] = v
motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
motionencoder.eval()
motionencoder.to(comp_device)

#--------------------------------
evaluator = [textencoder, motionencoder]

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []

best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger = eval_trans.evaluation_transformer_272_single(
    val_loader,
    net,
    trans_encoder,
    tokenize_model,
    logger,
    evaluator,
    4.0,
)

fid.append(best_fid)
div.append(best_div)
top1.append(best_top1)
top2.append(best_top2)
top3.append(best_top3)
matching.append(best_matching)

logger.info('final result:')
logger.info(f'fid: {fid}')
logger.info(f'div: {div}')
logger.info(f'top1: {top1}')
logger.info(f'top2: {top2}')
logger.info(f'top3: {top3}')
logger.info(f'MM-dist (matching score) : {matching}')
