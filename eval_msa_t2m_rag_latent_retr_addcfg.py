# eval_msa_t2m_rag_latent_retr_addcfg.py
# Evaluation with 3-forward ADDITIVE velocity-space CFG for NEW model
# (independently trained text + retrieval dropout, retr_cfg_drop_prob=0.3).
#
# Additive CFG formula (applied inside diff_loss.sample_additive_cfg):
#   v_guided = v_nn + s_t*(v_tn - v_nn) + s_r*(v_tr - v_tn)
#
# All 3 z-vectors are in-distribution — no OOD hidden-space extrapolation.
#
# Usage: add --cfg_scale_t <s_t> --cfg_scale_r <s_r> to usual args.
#        --cfg_scale is still accepted (maps to s_t for backward compat).
import os
import sys
import json
import torch
import numpy as np

# Import everything from the stable base eval script
from eval_msa_t2m_rag_latent_retr import (
    RAGEvalSampler,
    OfflineTextEmbeddingLookup,
    RAGRetriever,
    RAGLatentRetriever,
    parse_args as _base_parse_args,
    load_state_strip_module,
    resolve_existing_path,
    resolve_data_root,
    resolve_reference_end_latent,
)

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model_latent_retr import (
    LLaMARAGLatentRetrWrapper,
    LLaMARAGLatentRetrGatedWrapper,
)
import models.msa_vae as msa_vae
import options.option_transformer as option_trans
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from humanml3d_272 import dataset_eval_t2m
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
#  Subclass: 3-forward additive CFG sampler
# ---------------------------------------------------------------------------

class RAGEvalSamplerAdditiveCFG(RAGEvalSampler):
    """RAGEvalSampler variant using 3-forward additive velocity-space CFG.

    Instead of 2-forward joint-dropout CFG, performs three independent
    forward passes per autoregressive step:
      Forward 1 (null_text, null_retr)  → z_nn
      Forward 2 (real_text, null_retr)  → z_tn
      Forward 3 (real_text, real_retr)  → z_tr
    Then calls diff_loss.sample_additive_cfg(z_nn, z_tn, z_tr, s_t, s_r).

    Parameters
    ----------
    cfg_scale_t : float
        Text guidance strength (maps to s_t). Default 6.0.
    cfg_scale_r : float
        Retrieval guidance strength (maps to s_r). Default 2.0.
    All other parameters are identical to RAGEvalSampler.
    """

    def __init__(self, *args, cfg_scale_t=6.0, cfg_scale_r=2.0, **kwargs):
        # cfg_scale_retr is kept in parent for informational purposes;
        # we use cfg_scale_t / cfg_scale_r directly.
        kwargs.setdefault('cfg_scale_retr', cfg_scale_r)
        super().__init__(*args, **kwargs)
        self.cfg_scale_t = float(cfg_scale_t)
        self.cfg_scale_r = float(cfg_scale_r)

    @torch.no_grad()
    def sample_for_eval_CFG(self, text, length=196, tokenize_model=None,
                             device=torch.device('cuda'), unit_length=4, cfg=None):
        """3-forward additive CFG sampling (cfg arg is ignored; uses self.cfg_scale_t)."""
        _ = tokenize_model
        _ = device
        # cfg arg from eval_trans is the outer cfg_scale; we use our own fields.
        s_t = self.cfg_scale_t
        s_r = self.cfg_scale_r

        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = list(text)

        # ── text embedding ───────────────────────────────────────────────────
        if self.text_source == 'online_t5':
            if self.text_encoder is None:
                raise RuntimeError('text_encoder required for online_t5 mode')
            text_np = np.asarray(self.text_encoder.encode(text_list), dtype=np.float32)
            text_emb = torch.from_numpy(text_np).float().to(self.device)
        else:
            if self.text_lookup is None:
                raise RuntimeError('text_lookup required for offline mode')
            text_emb = self.text_lookup.batch_lookup(text_list, self.device)

        if text_emb.shape[-1] != self.text_embed_dim:
            raise ValueError(
                f'text embedding dim mismatch: got {text_emb.shape[-1]}, '
                f'expected {self.text_embed_dim}'
            )

        # ── global h_cls retrieval ───────────────────────────────────────────
        if self.disable_rag:
            top_hcls = None
            top_scores = None
        else:
            top_hcls, top_scores = self.retriever.retrieve(text_emb)

        # ── local motion latent retrieval ────────────────────────────────────
        retr_latents_batch = None
        retr_latent_lens_batch = None
        if self.latent_retriever is not None and not self.disable_latent_retr:
            exclude_ids = [None] * len(text_list)
            retr_latents_batch, retr_latent_lens_batch = self.latent_retriever.retrieve(
                text_emb, exclude_sample_ids=exclude_ids
            )

        # ── autoregressive sampling with 3-forward additive CFG ─────────────
        max_token_len = max(1, int(length) // unit_length)
        xs = None
        bsz = text_emb.shape[0]
        finished = torch.zeros(bsz, dtype=torch.bool, device=self.device)

        all_false = torch.zeros(bsz, dtype=torch.bool, device=self.device)  # keep
        all_true  = torch.ones( bsz, dtype=torch.bool, device=self.device)  # drop

        for _k in range(max_token_len):
            prefix = (xs if xs is not None
                      else torch.zeros((bsz, 0, self.latent_dim),
                                       device=self.device, dtype=torch.float32))

            # Forward 1: null text, null retr → z_nn  (~1% training prob)
            z_nn = self.rag_model.forward(
                prefix, text_emb, top_hcls, top_scores,
                cfg_drop_mask=all_true,
                empty_text_emb=self.empty_text_emb,
                retr_latents=retr_latents_batch,
                retr_latent_lens=retr_latent_lens_batch,
                retr_cfg_drop_mask=all_true,
            )[:, -1, :]   # (bsz, D)

            # Forward 2: real text, null retr → z_tn  (~9% training prob)
            z_tn = self.rag_model.forward(
                prefix, text_emb, top_hcls, top_scores,
                cfg_drop_mask=all_false,
                empty_text_emb=self.empty_text_emb,
                retr_latents=retr_latents_batch,
                retr_latent_lens=retr_latent_lens_batch,
                retr_cfg_drop_mask=all_true,
            )[:, -1, :]   # (bsz, D)

            # Forward 3: real text, real retr → z_tr  (~81% training prob)
            z_tr = self.rag_model.forward(
                prefix, text_emb, top_hcls, top_scores,
                cfg_drop_mask=all_false,
                empty_text_emb=self.empty_text_emb,
                retr_latents=retr_latents_batch,
                retr_latent_lens=retr_latent_lens_batch,
                retr_cfg_drop_mask=all_false,
            )[:, -1, :]   # (bsz, D)

            # Additive CFG in velocity/noise space — all z are in-distribution
            next_token = self.rag_model.base_model.diff_loss.sample_additive_cfg(
                z_nn=z_nn, z_tn=z_tn, z_tr=z_tr,
                s_t=s_t, s_r=s_r, temperature=1.0,
            )   # (bsz, latent_dim)

            next_token = next_token.unsqueeze(1)   # (bsz, 1, latent_dim)
            xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)

            if self.enable_stopping and self.reference_end_latent is not None:
                cur = next_token.squeeze(1)
                dist_l2 = torch.linalg.norm(
                    cur - self.reference_end_latent.unsqueeze(0), dim=-1
                )
                finished = finished | (dist_l2 < self.stop_threshold)
                if torch.all(finished):
                    break

        if xs is None:
            xs = torch.zeros((bsz, 1, self.latent_dim),
                             device=self.device, dtype=torch.float32)
        return xs


# ---------------------------------------------------------------------------
#  Argument parsing (extends base with --cfg_scale_t / --cfg_scale_r)
# ---------------------------------------------------------------------------

def parse_args():
    import argparse

    extra_parser = argparse.ArgumentParser(add_help=False)
    # Paths
    extra_parser.add_argument('--hcls_dir', type=str,
                              default='./humanml3d_272/h_cls_latents_msa_vae/exp')
    extra_parser.add_argument('--text_latent_dir', type=str,
                              default='./humanml3d_272/text_latents_t5')
    extra_parser.add_argument('--empty_text_path', type=str,
                              default='./humanml3d_272/text_latents_t5/empty_text_embedding.npy')
    extra_parser.add_argument('--library_cache_dir', type=str, required=True)
    # Retrieval
    extra_parser.add_argument('--retrieval_topk', type=int, default=3)
    extra_parser.add_argument('--latent_retr_topk', type=int, default=3)
    extra_parser.add_argument('--latent_dim', type=int, default=16)
    # Additive CFG scales
    extra_parser.add_argument('--cfg_scale', type=float, default=6.0,
                              help='Text CFG scale (alias for --cfg_scale_t).')
    extra_parser.add_argument('--cfg_scale_t', type=float, default=None,
                              help='Text guidance scale s_t. Overrides --cfg_scale.')
    extra_parser.add_argument('--cfg_scale_r', type=float, default=2.0,
                              help='Retrieval guidance scale s_r.')
    extra_parser.add_argument('--cfg_scale_retr', type=float, default=None,
                              help='Alias for --cfg_scale_r (backward compat).')
    # Text / eval
    extra_parser.add_argument('--text_embed_dim', type=int, default=768)
    extra_parser.add_argument('--eval_split', type=str, default='test',
                              choices=['test', 'val'])
    extra_parser.add_argument('--text_source', type=str, default='online_t5',
                              choices=['offline', 'online_t5'])
    extra_parser.add_argument('--t5_model_path', type=str, default='sentencet5-xxl/')
    # Ablation
    extra_parser.add_argument('--disable_rag', action='store_true', default=False)
    extra_parser.add_argument('--disable_latent_retr', action='store_true', default=False)
    extra_parser.add_argument('--disable_ema', action='store_true', default=False)
    # CA arch
    extra_parser.add_argument('--ca_n_head', type=int, default=0)
    extra_parser.add_argument('--ca_every_n_layers', type=int, default=1)
    # MSA-VAE arch
    extra_parser.add_argument('--trans_d_model', type=int, default=768)
    extra_parser.add_argument('--trans_nhead', type=int, default=8)
    extra_parser.add_argument('--trans_enc_layers', type=int, default=6)
    extra_parser.add_argument('--trans_dec_layers', type=int, default=6)
    extra_parser.add_argument('--trans_ff_size', type=int, default=2048)
    extra_parser.add_argument('--trans_dropout', type=float, default=0.1)
    extra_parser.add_argument('--clip_dim', type=int, default=768)
    # Stopping
    extra_parser.add_argument('--reference_end_latent_path', type=str, default='')
    extra_parser.add_argument('--stop_threshold', type=float, default=3.0)
    extra_parser.add_argument('--enable_stopping', dest='enable_stopping',
                              action='store_true', default=True)
    extra_parser.add_argument('--disable_stopping', dest='enable_stopping',
                              action='store_false')

    custom_args, remaining = extra_parser.parse_known_args()

    argv_backup = sys.argv
    try:
        sys.argv = [sys.argv[0]] + remaining
        args = option_trans.get_args_parser()
    finally:
        sys.argv = argv_backup

    # Resolve cfg_scale_t (cfg_scale_t > cfg_scale > default 6.0)
    s_t = custom_args.cfg_scale_t if custom_args.cfg_scale_t is not None else custom_args.cfg_scale
    s_r = custom_args.cfg_scale_retr if custom_args.cfg_scale_retr is not None else custom_args.cfg_scale_r

    args.hcls_dir = custom_args.hcls_dir
    args.text_latent_dir = custom_args.text_latent_dir
    args.empty_text_path = custom_args.empty_text_path
    args.library_cache_dir = custom_args.library_cache_dir
    args.retrieval_topk = custom_args.retrieval_topk
    args.latent_retr_topk = custom_args.latent_retr_topk
    args.latent_dim_retr = custom_args.latent_dim
    args.cfg_scale = s_t        # used as outer cfg passed to eval_trans (ignored in subclass)
    args.cfg_scale_t = s_t
    args.cfg_scale_r = s_r
    args.text_embed_dim = custom_args.text_embed_dim
    args.eval_split = custom_args.eval_split
    args.text_source = custom_args.text_source
    args.t5_model_path = custom_args.t5_model_path
    args.disable_rag = custom_args.disable_rag
    args.disable_latent_retr = custom_args.disable_latent_retr
    args.use_ema = not custom_args.disable_ema
    args.ca_n_head = custom_args.ca_n_head
    args.ca_every_n_layers = custom_args.ca_every_n_layers
    args.trans_d_model = custom_args.trans_d_model
    args.trans_nhead = custom_args.trans_nhead
    args.trans_enc_layers = custom_args.trans_enc_layers
    args.trans_dec_layers = custom_args.trans_dec_layers
    args.trans_ff_size = custom_args.trans_ff_size
    args.trans_dropout = custom_args.trans_dropout
    args.clip_dim = custom_args.clip_dim
    args.reference_end_latent_path = custom_args.reference_end_latent_path
    args.stop_threshold = custom_args.stop_threshold
    args.enable_stopping = custom_args.enable_stopping
    return args


# ---------------------------------------------------------------------------
#  main  (identical model setup to base script; only sampler class differs)
# ---------------------------------------------------------------------------

def main():
    os.chdir('Evaluator_272')
    sys.path.insert(0, os.getcwd())

    comp_device = torch.device('cuda')
    args = parse_args()

    args.resume_pth = resolve_existing_path(args.resume_pth) if args.resume_pth is not None else None
    args.resume_trans = resolve_existing_path(args.resume_trans) if args.resume_trans is not None else None
    args.latent_dir = resolve_existing_path(args.latent_dir, must_be_dir=True)
    args.text_latent_dir = resolve_existing_path(args.text_latent_dir, must_be_dir=True)
    args.hcls_dir = resolve_existing_path(args.hcls_dir, must_be_dir=True)
    args.library_cache_dir = resolve_existing_path(args.library_cache_dir, must_be_dir=True)
    args.empty_text_path = resolve_existing_path(args.empty_text_path)
    args.t5_model_path = resolve_existing_path(args.t5_model_path, must_be_dir=True)
    data_root = resolve_data_root()

    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info(f'Additive CFG: s_t={args.cfg_scale_t}, s_r={args.cfg_scale_r}')

    val_loader = dataset_eval_t2m.DATALoader(args.dataname, args.eval_split == 'test', 32)

    # ── MSA-VAE decoder ──────────────────────────────────────────────────────
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
    print(f'Loading MSA-VAE checkpoint from {args.resume_pth}')
    ckpt_vae = torch.load(args.resume_pth, map_location='cpu')
    vae_state = ckpt_vae['net'] if (isinstance(ckpt_vae, dict) and 'net' in ckpt_vae) else ckpt_vae
    net.load_state_dict(vae_state, strict=True)
    net.eval()
    net.to(comp_device)

    # ── RAG model ────────────────────────────────────────────────────────────
    latent_dim_retr = getattr(args, 'latent_dim_retr', args.latent_dim)
    config = LLaMAHFConfig.from_name('Normal_size')
    config.block_size = 78
    base_model = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)
    ca_n_head = getattr(args, 'ca_n_head', 0)

    print(f'Loading RAG transformer checkpoint from {args.resume_trans}')
    ckpt = torch.load(args.resume_trans, map_location='cpu')

    # Auto-detect EMA vs non-EMA keys
    trans_key = 'trans_ema' if (args.use_ema and 'trans_ema' in ckpt) else 'trans'
    rag_key   = 'rag_ema'   if (args.use_ema and 'rag_ema'   in ckpt) else 'rag'

    # Auto-detect CA architecture from checkpoint
    _detected_ff_mult = 4
    if rag_key in ckpt:
        for k in load_state_strip_module(ckpt[rag_key]).keys():
            if 'ca_blocks' in k and 'ff.net.0.weight' in k:
                _w = load_state_strip_module(ckpt[rag_key])[k]
                _detected_ff_mult = _w.shape[0] // config.n_embd
                break

    _detected_ca_every = getattr(args, 'ca_every_n_layers', 1)
    if rag_key in ckpt:
        n_ca = sum(1 for k in load_state_strip_module(ckpt[rag_key]).keys()
                   if k.startswith('ca_blocks.') and k.endswith('.norm1.weight'))
        if n_ca > 0:
            for n in range(1, config.n_layer + 1):
                if config.n_layer // n == n_ca:
                    _detected_ca_every = n
                    break

    rag_model = LLaMARAGLatentRetrWrapper(
        base_model=base_model,
        model_dim=config.n_embd,
        disable_rag=args.disable_rag,
        latent_dim=latent_dim_retr,
        ca_every_n_layers=max(1, _detected_ca_every),
        ca_n_head=ca_n_head if ca_n_head > 0 else None,
        ff_mult=_detected_ff_mult,
        disable_latent_retr=args.disable_latent_retr,
    )

    if trans_key in ckpt:
        rag_model.base_model.load_state_dict(
            load_state_strip_module(ckpt[trans_key]), strict=False
        )
    else:
        raise KeyError('RAG checkpoint missing key: trans/trans_ema')

    if rag_key in ckpt:
        rag_model.load_state_dict(load_state_strip_module(ckpt[rag_key]), strict=False)
    elif not args.disable_rag and not args.disable_latent_retr:
        raise KeyError('RAG checkpoint missing key: rag/rag_ema')
    else:
        logger.info('Checkpoint has no rag key; running in ablation mode.')

    rag_model.eval()
    rag_model.to(comp_device)

    # ── Empty text embedding ─────────────────────────────────────────────────
    empty_text_path = args.empty_text_path
    for p in [
        empty_text_path,
        os.path.join(args.text_latent_dir, 'empty_text_embedding.npy'),
        os.path.join(args.text_latent_dir, 'empty_cfg_text_t5.npy'),
    ]:
        if p and os.path.exists(p):
            empty_text_path = p
            break
    if not os.path.exists(empty_text_path):
        raise FileNotFoundError(f'empty cfg text file not found: {args.empty_text_path}')

    empty_text_emb = (
        torch.from_numpy(np.load(empty_text_path).astype(np.float32))
        .reshape(-1).to(comp_device)
    )

    # ── Text source ──────────────────────────────────────────────────────────
    text_lookup = None
    text_encoder = None
    if args.text_source == 'online_t5':
        from sentence_transformers import SentenceTransformer
        text_encoder = SentenceTransformer(args.t5_model_path)
        text_encoder.eval()
        logger.info(f'Text source: online_t5, model={args.t5_model_path}')
    else:
        text_lookup = OfflineTextEmbeddingLookup(
            data_root=data_root,
            text_latent_dir=args.text_latent_dir,
            split=args.eval_split,
            text_embed_dim=args.text_embed_dim,
        )
        logger.info('Text source: offline precomputed latents')

    # ── Global h_cls retriever ───────────────────────────────────────────────
    retriever = None
    if not args.disable_rag:
        retriever = RAGRetriever(
            args.hcls_dir,
            topk=args.retrieval_topk,
            text_embed_dim=args.text_embed_dim,
            device=comp_device,
        )

    # ── Local latent retriever ───────────────────────────────────────────────
    latent_retriever = None
    if not args.disable_latent_retr:
        latent_retriever = RAGLatentRetriever(
            library_cache_dir=args.library_cache_dir,
            topk=args.latent_retr_topk,
            device=comp_device,
        )

    # ── Reference end latent ─────────────────────────────────────────────────
    reference_path, reference_end = resolve_reference_end_latent(args)
    reference_end_latent = None
    if args.enable_stopping:
        if reference_end is None:
            raise FileNotFoundError(
                'Cannot locate reference end latent. Provide --reference_end_latent_path '
                'or ensure it exists under --latent_dir.'
            )
        reference_end_latent = torch.from_numpy(reference_end).float().to(comp_device)
        logger.info(
            f'Continuous stopping: reference={reference_path}, threshold={args.stop_threshold}'
        )

    # ── Build additive-CFG sampler ───────────────────────────────────────────
    trans_for_eval = RAGEvalSamplerAdditiveCFG(
        rag_model=rag_model,
        retriever=retriever,
        latent_retriever=latent_retriever,
        empty_text_emb=empty_text_emb,
        latent_dim=args.latent_dim,
        device=comp_device,
        reference_end_latent=reference_end_latent,
        stop_threshold=args.stop_threshold,
        enable_stopping=args.enable_stopping,
        text_source=args.text_source,
        text_lookup=text_lookup,
        text_encoder=text_encoder,
        text_embed_dim=args.text_embed_dim,
        disable_rag=args.disable_rag,
        disable_latent_retr=args.disable_latent_retr,
        cfg_scale_t=args.cfg_scale_t,
        cfg_scale_r=args.cfg_scale_r,
    )

    # ── Load evaluator ───────────────────────────────────────────────────────
    from mld.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
    from mld.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

    modelpath = './deps/distilbert-base-uncased'
    textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)
    motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)

    ckpt_path = '../Evaluator_272/epoch=99.ckpt'
    print(f'Loading evaluator checkpoint from {ckpt_path}')
    ckpt_eval = torch.load(ckpt_path)

    textencoder.load_state_dict({
        k.replace('textencoder.', ''): v
        for k, v in ckpt_eval['state_dict'].items()
        if k.startswith('textencoder.')
    }, strict=True)
    textencoder.eval().to(comp_device)

    motionencoder.load_state_dict({
        k.replace('motionencoder.', ''): v
        for k, v in ckpt_eval['state_dict'].items()
        if k.startswith('motionencoder.')
    }, strict=True)
    motionencoder.eval().to(comp_device)

    evaluator = [textencoder, motionencoder]

    # ── Run evaluation ───────────────────────────────────────────────────────
    fid, div, top1, top2, top3, matching = [], [], [], [], [], []

    best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger = (
        eval_trans.evaluation_transformer_272_single(
            val_loader,
            net,
            trans_for_eval,
            tokenize_model=None,
            logger=logger,
            evaluator=evaluator,
            cfg=args.cfg_scale,   # ignored inside sample_for_eval_CFG (overridden)
        )
    )

    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)

    logger.info('=== Additive CFG Evaluation Results ===')
    logger.info(f'cfg_scale_t={args.cfg_scale_t}  cfg_scale_r={args.cfg_scale_r}')
    logger.info(f'fid: {fid}')
    logger.info(f'div: {div}')
    logger.info(f'top1: {top1}')
    logger.info(f'top2: {top2}')
    logger.info(f'top3: {top3}')
    logger.info(f'MM-dist (matching score): {matching}')


if __name__ == '__main__':
    main()
