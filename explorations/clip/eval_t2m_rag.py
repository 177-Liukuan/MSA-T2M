import os
import sys
import json
import argparse
import glob
import warnings
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
import models.msa_vae as msa_vae
import options.option_transformer as option_trans
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from humanml3d_272 import dataset_eval_t2m

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class RAGRetriever:
    """Build in-memory h_cls retrieval library and provide top-k retrieval."""

    def __init__(self, hcls_dir, topk=3, device=torch.device('cuda')):
        self.hcls_dir = hcls_dir
        self.topk = topk
        self.device = device

        hcls_files = sorted(glob.glob(os.path.join(hcls_dir, '*.npy')))
        if len(hcls_files) == 0:
            raise RuntimeError(f'No h_cls npy files found in: {hcls_dir}')

        vectors = []
        for path in hcls_files:
            h = np.load(path).astype(np.float32)
            if h.ndim == 2:
                h = h.mean(axis=0)
            else:
                h = h.reshape(-1)
            if h.shape[0] != 512:
                continue
            vectors.append(h)

        if len(vectors) == 0:
            raise RuntimeError(f'No valid 512-d h_cls vectors found in: {hcls_dir}')

        lib = torch.from_numpy(np.stack(vectors)).float().to(device)
        self.library = lib
        self.library_norm = self._normalize(lib)

    @staticmethod
    def _normalize(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb):
        """text_emb: [B, 512] -> (top_hcls [B,K,512], top_scores [B,K])"""
        q = self._normalize(text_emb)
        sim = torch.matmul(q, self.library_norm.t())
        k = min(self.topk, sim.shape[1])
        top_scores, top_indices = torch.topk(sim, k=k, dim=1)
        top_hcls = self.library[top_indices]

        if k < self.topk:
            pad_h = torch.zeros(text_emb.shape[0], self.topk - k, 512, device=text_emb.device, dtype=top_hcls.dtype)
            pad_s = torch.full((text_emb.shape[0], self.topk - k), -1e6, device=text_emb.device, dtype=top_scores.dtype)
            top_hcls = torch.cat([top_hcls, pad_h], dim=1)
            top_scores = torch.cat([top_scores, pad_s], dim=1)

        return top_hcls, top_scores


class RAGEvalSampler:
    """Adapter exposing sample_for_eval_CFG for reuse with eval_trans."""

    def __init__(self, rag_model, clip_model, retriever, empty_text_emb, latent_dim=16, device=torch.device('cuda')):
        self.rag_model = rag_model
        self.clip_model = clip_model
        self.retriever = retriever
        self.empty_text_emb = empty_text_emb
        self.latent_dim = latent_dim
        self.device = device

    def eval(self):
        self.rag_model.eval()
        self.clip_model.eval()
        return self

    @torch.no_grad()
    def sample_for_eval_CFG(self, text, length=196, tokenize_model=None, device=torch.device('cuda'), unit_length=4, cfg=4.0):
        import clip

        _ = tokenize_model  # kept for eval_trans compatibility
        _ = device

        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = list(text)

        tokens = clip.tokenize(text_list, truncate=True).to(self.device)
        text_emb = self.clip_model.encode_text(tokens).float()  # [B, 512]

        top_hcls, top_scores = self.retriever.retrieve(text_emb)

        max_token_len = int(length) // unit_length
        xs = None
        bsz = text_emb.shape[0]

        for _k in range(max_token_len):
            if xs is None:
                prefix = torch.zeros((bsz, 0, self.latent_dim), device=self.device, dtype=torch.float32)
            else:
                prefix = xs

            next_token = self.rag_model.sample_next_with_cfg(
                motion_prefix=prefix,
                text_emb=text_emb,
                top3_h_cls=top_hcls,
                top3_sim_scores=top_scores,
                empty_text_emb=self.empty_text_emb,
                cfg_scale=cfg,
                temperature=1.0,
            )  # [B, latent_dim]

            next_token = next_token.unsqueeze(1)  # [B, 1, latent_dim]
            xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)

        return xs


def parse_args():
    # Parse RAG/MSA-specific args first, then delegate remaining args to
    # option_transformer parser to keep backward compatibility.
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--hcls_dir", type=str, default="./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv5_phase2_t2m_272_iter2000_α0")
    extra_parser.add_argument("--empty_text_path", type=str, default="./humanml3d_272/text_latents_clip/empty_cfg_text.npy")
    extra_parser.add_argument("--retrieval_topk", type=int, default=3)
    extra_parser.add_argument("--cfg_scale", type=float, default=4.0)

    # MSA-VAE architecture args
    extra_parser.add_argument("--trans_d_model", type=int, default=512)
    extra_parser.add_argument("--trans_nhead", type=int, default=8)
    extra_parser.add_argument("--trans_enc_layers", type=int, default=4)
    extra_parser.add_argument("--trans_dec_layers", type=int, default=4)
    extra_parser.add_argument("--trans_ff_size", type=int, default=1024)
    extra_parser.add_argument("--trans_dropout", type=float, default=0.1)
    extra_parser.add_argument("--clip_dim", type=int, default=512)

    custom_args, remaining = extra_parser.parse_known_args()

    argv_backup = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = option_trans.get_args_parser()
    finally:
        sys.argv = argv_backup

    # Inject parsed custom args into final namespace
    args.hcls_dir = custom_args.hcls_dir
    args.empty_text_path = custom_args.empty_text_path
    args.retrieval_topk = custom_args.retrieval_topk
    args.cfg_scale = custom_args.cfg_scale

    args.trans_d_model = custom_args.trans_d_model
    args.trans_nhead = custom_args.trans_nhead
    args.trans_enc_layers = custom_args.trans_enc_layers
    args.trans_dec_layers = custom_args.trans_dec_layers
    args.trans_ff_size = custom_args.trans_ff_size
    args.trans_dropout = custom_args.trans_dropout
    args.clip_dim = custom_args.clip_dim

    return args


def load_state_strip_module(state_dict):
    new_state = {}
    for key, value in state_dict.items():
        if key.split('.')[0] == 'module':
            new_key = '.'.join(key.split('.')[1:])
        else:
            new_key = key
        new_state[new_key] = value
    return new_state


def main():
    os.chdir('Evaluator_272')
    sys.path.insert(0, os.getcwd())

    comp_device = torch.device('cuda')

    args = parse_args()
    torch.manual_seed(args.seed)

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    val_loader = dataset_eval_t2m.DATALoader(args.dataname, True, 32)

    # MSA-VAE (Stage-1) decoder backbone
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

    print('loading MSA-VAE checkpoint from {}'.format(args.resume_pth))
    ckpt_vae = torch.load(args.resume_pth, map_location='cpu')
    vae_state = ckpt_vae['net'] if (isinstance(ckpt_vae, dict) and 'net' in ckpt_vae) else ckpt_vae
    net.load_state_dict(vae_state, strict=True)
    net.eval()
    net.to(comp_device)

    # RAG model
    config = LLaMAHFConfig.from_name('Normal_size')
    config.block_size = 78
    base_model = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)
    rag_model = LLaMARAGWrapper(base_model=base_model, text_dim=512, retrieval_dim=512, model_dim=config.n_embd)

    if args.resume_trans is None:
        raise ValueError('Please provide --resume-trans for RAG checkpoint.')

    print('loading RAG checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')

    if 'trans' in ckpt:
        rag_model.base_model.load_state_dict(load_state_strip_module(ckpt['trans']), strict=False)
    else:
        raise KeyError('RAG checkpoint missing key: trans')

    if 'rag' in ckpt:
        rag_model.load_state_dict(load_state_strip_module(ckpt['rag']), strict=False)
    else:
        raise KeyError('RAG checkpoint missing key: rag')

    rag_model.eval()
    rag_model.to(comp_device)

    # CLIP encoder for query text
    import clip
    clip_model, _ = clip.load('ViT-B/32', device=comp_device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Empty text embedding for CFG
    empty_text_path = args.empty_text_path
    if not os.path.exists(empty_text_path):
        fallback_path = './humanml3d_272/text_latents_clip/empty_cfg_text_clip.npy'
        if os.path.exists(fallback_path):
            empty_text_path = fallback_path
        else:
            raise FileNotFoundError(f'empty cfg text file not found: {args.empty_text_path} / {fallback_path}')

    empty_text_emb = torch.from_numpy(np.load(empty_text_path).astype(np.float32)).reshape(-1).to(comp_device)
    if empty_text_emb.shape[0] != 512:
        raise ValueError(f'empty text embedding must be 512-d, got {empty_text_emb.shape[0]}')

    # Retrieval library
    retriever = RAGRetriever(args.hcls_dir, topk=args.retrieval_topk, device=comp_device)

    # Adapter for eval_trans
    trans_for_eval = RAGEvalSampler(
        rag_model=rag_model,
        clip_model=clip_model,
        retriever=retriever,
        empty_text_emb=empty_text_emb,
        latent_dim=args.latent_dim,
        device=comp_device,
    )

    # Load evaluator (same metric pipeline as original eval_t2m.py)
    from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
    from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

    modelpath = './deps/distilbert-base-uncased'
    textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
    motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)

    ckpt_path = '../Evaluator_272/experiments/temos/EXP1/checkpoints/epoch=99.ckpt'
    print(f'Loading evaluator checkpoint from {ckpt_path}')
    ckpt_eval = torch.load(ckpt_path)

    textencoder_ckpt = {}
    for k, v in ckpt_eval['state_dict'].items():
        if k.split('.')[0] == 'textencoder':
            name = k.replace('textencoder.', '')
            textencoder_ckpt[name] = v
    textencoder.load_state_dict(textencoder_ckpt, strict=True)
    textencoder.eval()
    textencoder.to(comp_device)

    motionencoder_ckpt = {}
    for k, v in ckpt_eval['state_dict'].items():
        if k.split('.')[0] == 'motionencoder':
            name = k.replace('motionencoder.', '')
            motionencoder_ckpt[name] = v
    motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
    motionencoder.eval()
    motionencoder.to(comp_device)

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
        trans_for_eval,
        tokenize_model=None,
        logger=logger,
        evaluator=evaluator,
        cfg=args.cfg_scale,
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


if __name__ == '__main__':
    main()
