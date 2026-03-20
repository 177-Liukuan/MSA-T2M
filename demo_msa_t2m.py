import os
import glob
import argparse
import warnings
import numpy as np
import torch

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
import models.msa_vae as msa_vae
from visualization.recover_visualize import recover_from_local_position
import visualization.plot_3d_global as plot_3d

warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


class RAGRetriever:
    """In-memory h_cls retrieval library for demo inference."""

    def __init__(self, hcls_dir, topk=3, device=torch.device('cuda')):
        self.topk = topk

        files = sorted(glob.glob(os.path.join(hcls_dir, '*.npy')))
        if len(files) == 0:
            raise RuntimeError(f'No h_cls npy files found in: {hcls_dir}')

        vectors = []
        for path in files:
            h = np.load(path).astype(np.float32)
            if h.ndim == 2:
                h = h.mean(axis=0)
            else:
                h = h.reshape(-1)
            if h.shape[0] == 512:
                vectors.append(h)

        if len(vectors) == 0:
            raise RuntimeError(f'No valid 512-d h_cls vectors found in: {hcls_dir}')

        lib = torch.from_numpy(np.stack(vectors)).float().to(device)
        self.lib = lib
        self.lib_norm = self._norm(lib)

    @staticmethod
    def _norm(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb):
        """text_emb: [B, 512] -> top_hcls: [B,K,512], top_scores: [B,K]"""
        q = self._norm(text_emb)
        sim = torch.matmul(q, self.lib_norm.t())

        k = min(self.topk, sim.shape[1])
        top_scores, top_idx = torch.topk(sim, k=k, dim=1)
        top_hcls = self.lib[top_idx]

        if k < self.topk:
            pad_h = torch.zeros(text_emb.shape[0], self.topk - k, 512, device=text_emb.device, dtype=top_hcls.dtype)
            pad_s = torch.full((text_emb.shape[0], self.topk - k), -1e6, device=text_emb.device, dtype=top_scores.dtype)
            top_hcls = torch.cat([top_hcls, pad_h], dim=1)
            top_scores = torch.cat([top_scores, pad_s], dim=1)

        return top_hcls, top_scores


def load_state_strip_module(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.split('.')[0] == 'module':
            out['.'.join(k.split('.')[1:])] = v
        else:
            out[k] = v
    return out


@torch.no_grad()
def sample_motion_latents_with_stop(
    rag_model,
    clip_model,
    retriever,
    text,
    empty_text_emb,
    reference_end_latent,
    threshold=0.1,
    length=196,
    unit_length=4,
    cfg_scale=4.0,
    latent_dim=16,
    device=torch.device('cuda'),
):
    import clip

    tokens = clip.tokenize([text], truncate=True).to(device)
    text_emb = clip_model.encode_text(tokens).float()  # [1, 512]
    top_hcls, top_scores = retriever.retrieve(text_emb)

    # Keep the same token budget style as original MotionStreamer.
    max_token_len = int(length) // unit_length

    # Normalize reference stop token shape to [1, latent_dim].
    reference_end_latent = reference_end_latent.reshape(-1)
    if reference_end_latent.numel() != latent_dim:
        raise ValueError(
            f'reference stop token dim mismatch: got {reference_end_latent.numel()}, expected {latent_dim}'
        )
    reference_end_latent = reference_end_latent.view(1, latent_dim)

    xs = None
    for _k in range(max_token_len):
        if xs is None:
            prefix = torch.zeros((1, 0, latent_dim), device=device, dtype=torch.float32)
        else:
            prefix = xs

        next_token = rag_model.sample_next_with_cfg(
            motion_prefix=prefix,
            text_emb=text_emb,
            top3_h_cls=top_hcls,
            top3_sim_scores=top_scores,
            empty_text_emb=empty_text_emb,
            cfg_scale=cfg_scale,
            temperature=1.0,
        )  # [1, latent_dim]

        # Print stop-token distance each step like original behavior.
        distance_l2 = torch.sqrt(torch.sum((next_token - reference_end_latent) ** 2))
        print(distance_l2)

        next_token = next_token.unsqueeze(1)  # [1,1,latent_dim]
        xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)

        if distance_l2 < threshold:
            break

    generated = 0 if xs is None else xs.shape[1]
    print(f'Generated tokens: {generated} (max {max_token_len})')
    return xs


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for MSA-T2M RAG model (GIF output).')

    # Keep input/output style aligned with original demo_t2m.py.
    parser.add_argument('--text', type=str, default='A man is jogging around.')
    parser.add_argument('--mode', type=str, default='pos', choices=['pos', 'rot'])

    # Accept original naming used by user command.
    parser.add_argument('--resume-pth', dest='resume_pth', type=str, default='Experiments/MSA_VAEv5_phase2_t2m_272_iter2000_α0/net_last.pth')
    parser.add_argument('--resume-trans', dest='resume_trans', type=str, default='Experiments/MotionStreamer_t2m_272_msa_rag/latest.pth')

    # Optional generation controls.
    parser.add_argument('--length', type=int, default=300)
    parser.add_argument('--unit_length', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=4.0)
    parser.add_argument('--threshold', type=float, default=0.1)

    # RAG retrieval controls.
    parser.add_argument('--hcls_dir', type=str, default='./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv5_phase2_t2m_272_iter2000_α0')
    parser.add_argument('--empty_text_path', type=str, default='./humanml3d_272/text_latents_clip/empty_cfg_text.npy')
    parser.add_argument('--retrieval_topk', type=int, default=3)

    # Stop-token reference (same idea as original MotionStreamer demo).
    parser.add_argument('--reference_end_latent', type=str, default='reference_end_latent_t2m_272.npy')

    # Visualization options.
    parser.add_argument('--fps', type=int, default=30)

    # Shared arch params.
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--down_t', type=int, default=2)
    parser.add_argument('--stride_t', type=int, default=2)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--dilation_growth_rate', type=int, default=3)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--num_diffusion_head_layers', type=int, default=9)

    # MSA-VAE arch params.
    parser.add_argument('--trans_d_model', type=int, default=512)
    parser.add_argument('--trans_nhead', type=int, default=8)
    parser.add_argument('--trans_enc_layers', type=int, default=6)
    parser.add_argument('--trans_dec_layers', type=int, default=6)
    parser.add_argument('--trans_ff_size', type=int, default=2048)
    parser.add_argument('--trans_dropout', type=float, default=0.1)
    parser.add_argument('--clip_dim', type=int, default=512)

    return parser.parse_args()


def main():
    args = parse_args()
    comp_device = torch.device('cuda')

    print(f'Input text: {args.text}')

    # 1) MSA-VAE decoder
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
    ).to(comp_device)

    print('loading checkpoint from {}'.format(args.resume_pth))
    ckpt_vae = torch.load(args.resume_pth, map_location='cpu')
    state_vae = ckpt_vae['net'] if (isinstance(ckpt_vae, dict) and 'net' in ckpt_vae) else ckpt_vae
    net.load_state_dict(state_vae, strict=True)
    net.eval()

    # 2) RAG diffusion-AR model
    config = LLaMAHFConfig.from_name('Normal_size')
    config.block_size = 78
    base_model = LLaMAHF(config, args.num_diffusion_head_layers, args.latent_dim, comp_device)
    rag_model = LLaMARAGWrapper(base_model=base_model, text_dim=512, retrieval_dim=512, model_dim=config.n_embd).to(comp_device)

    if args.resume_trans is not None:
        print('loading transformer checkpoint from {}'.format(args.resume_trans))
        ckpt_rag = torch.load(args.resume_trans, map_location='cpu')
        if 'trans' not in ckpt_rag or 'rag' not in ckpt_rag:
            raise KeyError('RAG checkpoint must contain both trans and rag keys.')
        rag_model.base_model.load_state_dict(load_state_strip_module(ckpt_rag['trans']), strict=False)
        rag_model.load_state_dict(load_state_strip_module(ckpt_rag['rag']), strict=False)
    rag_model.eval()

    # 3) CLIP + retrieval
    import clip

    clip_model, _ = clip.load('ViT-B/32', device=comp_device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    empty_text_path = args.empty_text_path
    if not os.path.exists(empty_text_path):
        fallback = './humanml3d_272/text_latents_clip/empty_cfg_text_clip.npy'
        if os.path.exists(fallback):
            empty_text_path = fallback
        else:
            raise FileNotFoundError(f'empty text embedding file not found: {args.empty_text_path} / {fallback}')
    empty_text_emb = torch.from_numpy(np.load(empty_text_path).astype(np.float32)).reshape(-1).to(comp_device)

    retriever = RAGRetriever(args.hcls_dir, topk=args.retrieval_topk, device=comp_device)

    reference_end_latent = np.load(args.reference_end_latent)
    reference_end_latent = torch.from_numpy(reference_end_latent).float().to(comp_device)

    # 4) sample latent -> decode motion
    motion_latents = sample_motion_latents_with_stop(
        rag_model=rag_model,
        clip_model=clip_model,
        retriever=retriever,
        text=args.text,
        empty_text_emb=empty_text_emb,
        reference_end_latent=reference_end_latent,
        threshold=args.threshold,
        length=args.length,
        unit_length=args.unit_length,
        cfg_scale=args.cfg_scale,
        latent_dim=args.latent_dim,
        device=comp_device,
    )

    # forward decode
    motion_seqs = net.forward_decoder(motion_latents)
    motion = motion_seqs.squeeze(0).detach().cpu().numpy()

    if not os.path.exists('demo_output'):
        os.makedirs('demo_output')

    if args.mode == 'pos':
        mean = np.load('humanml3d_272/mean_std/Mean.npy')
        std = np.load('humanml3d_272/mean_std/Std.npy')

        pred_xyz = recover_from_local_position(motion * std + mean, 22)
        xyz = pred_xyz.reshape(1, -1, 22, 3)

        out_path = f'demo_output/{args.text}.gif'
        plot_3d.draw_to_batch(xyz, [args.text], [out_path], fps=args.fps)
        print(f'Visualized result is saved in {out_path}')

    elif args.mode == 'rot':
        np.save('demo_output/global_rotation.npy', motion)
        print('You can further convert to BVH format and visualize in Blender following: https://github.com/Li-xingXiao/272-dim-Motion-Representation?tab=readme-ov-file#6-representation_272-to-bvh-conversion-optional (Step 6: Representation_272 to BVH conversion)')

    else:
        raise ValueError(f'Invalid mode: {args.mode}')


if __name__ == '__main__':
    main()
