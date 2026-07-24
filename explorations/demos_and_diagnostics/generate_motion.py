import argparse
import glob
import os
import re
import warnings
from datetime import datetime

import numpy as np
import torch

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
from models.msa_vae import MSA_HumanVAE
from visualization.recover_visualize import recover_from_local_rotation


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RAGRetriever:
    """In-memory retrieval library for h_cls vectors."""

    def __init__(self, hcls_dir, embed_dim, topk=3, device=torch.device("cuda")):
        self.hcls_dir = hcls_dir
        self.embed_dim = int(embed_dim)
        self.topk = int(topk)

        files = sorted(glob.glob(os.path.join(hcls_dir, "*.npy")))
        if len(files) == 0:
            raise RuntimeError(f"No h_cls npy files found in: {hcls_dir}")

        vectors = []
        for path in files:
            vec = np.load(path).astype(np.float32)
            if vec.ndim == 2:
                vec = vec.mean(axis=0)
            else:
                vec = vec.reshape(-1)
            if vec.shape[0] == self.embed_dim:
                vectors.append(vec)

        if len(vectors) == 0:
            raise RuntimeError(
                f"No valid {self.embed_dim}-d h_cls vectors found in: {hcls_dir}"
            )

        lib = torch.from_numpy(np.stack(vectors)).float().to(device)
        self.lib = lib
        self.lib_norm = self._normalize(lib)

    @staticmethod
    def _normalize(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb):
        query = self._normalize(text_emb)
        sim = torch.matmul(query, self.lib_norm.t())

        k = min(self.topk, sim.shape[1])
        top_scores, top_idx = torch.topk(sim, k=k, dim=1)
        top_hcls = self.lib[top_idx]

        if k < self.topk:
            pad_h = torch.zeros(
                text_emb.shape[0],
                self.topk - k,
                self.embed_dim,
                device=text_emb.device,
                dtype=top_hcls.dtype,
            )
            pad_s = torch.full(
                (text_emb.shape[0], self.topk - k),
                -1e6,
                device=text_emb.device,
                dtype=top_scores.dtype,
            )
            top_hcls = torch.cat([top_hcls, pad_h], dim=1)
            top_scores = torch.cat([top_scores, pad_s], dim=1)

        return top_hcls, top_scores


def load_state_strip_module(state_dict):
    out = {}
    for key, value in state_dict.items():
        if key.split(".")[0] == "module":
            out[".".join(key.split(".")[1:])] = value
        else:
            out[key] = value
    return out


def sanitize_text_for_filename(text):
    value = text.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    if not value:
        value = "motion"
    return value[:120]


def load_texts(args):
    texts = []
    if args.text is not None and args.text.strip() != "":
        texts.append(args.text.strip())

    if args.text_file is not None:
        with open(args.text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    if len(texts) == 0:
        raise ValueError("Please provide --text or --text_file with at least one non-empty line.")

    return texts


def build_text_encoder(mode, t5_model_path, device):
    mode = mode.lower()
    if mode == "t5":
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(t5_model_path)
        encoder.eval()
        return encoder

    if mode == "clip":
        import clip

        clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    raise ValueError(f"Unsupported text encoder mode: {mode}")


@torch.no_grad()
def encode_text(text_encoder, mode, text, device):
    mode = mode.lower()
    if mode == "t5":
        arr = np.asarray(text_encoder.encode([text]), dtype=np.float32)
        return torch.from_numpy(arr).float().to(device)

    if mode == "clip":
        import clip

        tokens = clip.tokenize([text], truncate=True).to(device)
        return text_encoder.encode_text(tokens).float()

    raise ValueError(f"Unsupported text encoder mode: {mode}")


def resolve_empty_text_path(args, text_embed_dim):
    candidates = [args.empty_text_path]
    if text_embed_dim == 768:
        candidates.extend(
            [
                "humanml3d_272/text_latents_t5/empty_text_embedding.npy",
                "humanml3d_272/text_latents_t5/empty_cfg_text_t5.npy",
            ]
        )
    elif text_embed_dim == 512:
        candidates.extend(
            [
                "humanml3d_272/text_latents_clip/empty_cfg_text.npy",
                "humanml3d_272/text_latents_clip/empty_cfg_text_clip.npy",
            ]
        )

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Cannot find empty text embedding for dim={text_embed_dim}. checked: {candidates}"
    )


def build_retriever(args, text_embed_dim, device):
    candidates = [args.hcls_dir]
    if text_embed_dim == 768:
        candidates.extend(
            [
                "humanml3d_272/h_cls_latents_msa_vae/exp",
                "humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5",
            ]
        )
    elif text_embed_dim == 512:
        candidates.extend(
            [
                "humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv5_phase2_t2m_272_iter2000_α0",
            ]
        )

    seen = set()
    tried = []
    for path in candidates:
        if path in seen or not path:
            continue
        seen.add(path)
        if not os.path.isdir(path):
            tried.append(f"{path} (not found)")
            continue
        try:
            return RAGRetriever(path, embed_dim=text_embed_dim, topk=args.retrieval_topk, device=device)
        except Exception as exc:
            tried.append(f"{path} ({exc})")

    raise RuntimeError("Failed to build retriever. Tried: " + " | ".join(tried))


@torch.no_grad()
def sample_motion_latents_with_stop(
    rag_model,
    retriever,
    text_emb,
    empty_text_emb,
    reference_end_latent,
    threshold=0.1,
    length=300,
    unit_length=4,
    cfg_scale=4.0,
    latent_dim=16,
    device=torch.device("cuda"),
):
    top_hcls, top_scores = retriever.retrieve(text_emb)
    max_token_len = max(1, int(length) // int(unit_length))

    reference_end_latent = reference_end_latent.reshape(-1)
    if reference_end_latent.numel() != latent_dim:
        raise ValueError(
            f"reference end latent dim mismatch: got {reference_end_latent.numel()}, expected {latent_dim}"
        )
    reference_end_latent = reference_end_latent.view(1, latent_dim)

    xs = None
    generated_steps = 0
    for _ in range(max_token_len):
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
        )

        distance_l2 = torch.linalg.norm(next_token - reference_end_latent)
        next_token = next_token.unsqueeze(1)
        xs = next_token if xs is None else torch.cat([xs, next_token], dim=1)
        generated_steps += 1

        if distance_l2 < threshold:
            break

    if xs is None:
        xs = torch.zeros((1, 1, latent_dim), device=device, dtype=torch.float32)

    return xs, generated_steps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate motions from text using MSA-T2M (RAG + Diffusion-AR + MSA-VAE)."
    )

    parser.add_argument("--text", type=str, default=None, help="Single text prompt.")
    parser.add_argument("--text_file", type=str, default=None, help="Text file with one prompt per line.")

    parser.add_argument(
        "--out_dir",
        type=str,
        default="demo_output/generated_272",
        help="Directory for decoded 272-dim motions.",
    )
    parser.add_argument(
        "--aitviewer_out_dir",
        type=str,
        default="demo_output/aitviewer_ready",
        help="Directory for inverse-converted rotation files (SMPL85).",
    )

    parser.add_argument(
        "--resume_pth",
        type=str,
        default="Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5/net_last.pth",
    )
    parser.add_argument(
        "--resume_trans",
        type=str,
        default="Experiments/MotionStreamer_t2m_272_msa_rag_t5/latest.pth",
    )

    parser.add_argument("--hcls_dir", type=str, default="humanml3d_272/h_cls_latents_msa_vae/exp")
    parser.add_argument(
        "--empty_text_path",
        type=str,
        default="humanml3d_272/text_latents_t5/empty_text_embedding.npy",
    )
    parser.add_argument("--reference_end_latent", type=str, default="reference_end_latent_t2m_272.npy")

    parser.add_argument("--text_encoder", type=str, choices=["t5", "clip"], default="t5")
    parser.add_argument("--t5_model_path", type=str, default="sentencet5-xxl/")
    parser.add_argument("--retrieval_topk", type=int, default=3)
    parser.add_argument(
        "--text_embed_dim",
        type=int,
        default=None,
        help="Optional override. If unset, infer from rag_model cond embedding input dim.",
    )

    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--unit_length", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--end_latent_threshold_ddpm", type=float, default=0.1)
    parser.add_argument("--end_latent_threshold_rf", type=float, default=0.1)

    parser.add_argument("--generative_head_type", type=str, default="ddpm", choices=["ddpm", "rectified_flow"])
    parser.add_argument("--num_flow_steps", type=int, default=20)
    parser.add_argument("--flow_solver", type=str, default="euler", choices=["euler"])
    parser.add_argument("--rf_time_sampling", type=str, default="uniform", choices=["uniform"])
    parser.add_argument("--rf_loss_type", type=str, default="mse", choices=["mse"])

    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--down_t", type=int, default=2)
    parser.add_argument("--stride_t", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation_growth_rate", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--num_diffusion_head_layers", type=int, default=9)

    parser.add_argument("--trans_d_model", type=int, default=768)
    parser.add_argument("--trans_nhead", type=int, default=8)
    parser.add_argument("--trans_enc_layers", type=int, default=4)
    parser.add_argument("--trans_dec_layers", type=int, default=4)
    parser.add_argument("--trans_ff_size", type=int, default=1024)
    parser.add_argument("--trans_dropout", type=float, default=0.1)
    parser.add_argument("--clip_dim", type=int, default=768)

    return parser.parse_args()


def main():
    args = parse_args()
    texts = load_texts(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.aitviewer_out_dir, exist_ok=True)

    clip_range = [-30, 20]
    vae = MSA_HumanVAE(
        hidden_size=args.hidden_size,
        down_t=args.down_t,
        stride_t=args.stride_t,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation="relu",
        latent_dim=args.latent_dim,
        clip_range=clip_range,
        trans_d_model=args.trans_d_model,
        trans_nhead=args.trans_nhead,
        trans_enc_layers=args.trans_enc_layers,
        trans_dec_layers=args.trans_dec_layers,
        trans_ff_size=args.trans_ff_size,
        trans_dropout=args.trans_dropout,
        clip_dim=args.clip_dim,
    ).to(device)

    ckpt_vae = torch.load(args.resume_pth, map_location="cpu")
    vae_state = ckpt_vae["net"] if isinstance(ckpt_vae, dict) and "net" in ckpt_vae else ckpt_vae
    vae.load_state_dict(vae_state, strict=True)
    vae.eval()

    config = LLaMAHFConfig.from_name("Normal_size")
    config.block_size = 78
    base_model = LLaMAHF(
        config,
        args.num_diffusion_head_layers,
        args.latent_dim,
        device,
        generative_head_type=args.generative_head_type,
        num_flow_steps=args.num_flow_steps,
        flow_solver=args.flow_solver,
        rf_time_sampling=args.rf_time_sampling,
        rf_loss_type=args.rf_loss_type,
    )
    rag_model = LLaMARAGWrapper(base_model=base_model, model_dim=config.n_embd).to(device)

    ckpt_rag = torch.load(args.resume_trans, map_location="cpu")
    if "trans" not in ckpt_rag or "rag" not in ckpt_rag:
        raise KeyError("RAG checkpoint must contain both trans and rag keys.")
    rag_model.base_model.load_state_dict(load_state_strip_module(ckpt_rag["trans"]), strict=False)
    rag_model.load_state_dict(load_state_strip_module(ckpt_rag["rag"]), strict=False)
    rag_model.eval()

    inferred_text_dim = int(rag_model.base_model.transformer.cond_embed.in_features)
    text_embed_dim = inferred_text_dim if args.text_embed_dim is None else int(args.text_embed_dim)
    if text_embed_dim != inferred_text_dim:
        raise ValueError(
            f"--text_embed_dim ({text_embed_dim}) mismatches checkpoint cond dim ({inferred_text_dim})."
        )

    text_encoder = build_text_encoder(args.text_encoder, args.t5_model_path, device)

    empty_text_path = resolve_empty_text_path(args, text_embed_dim)
    empty_text_emb = torch.from_numpy(np.load(empty_text_path).astype(np.float32)).reshape(-1).to(device)
    if empty_text_emb.shape[0] != text_embed_dim:
        raise ValueError(
            f"empty text embedding dim mismatch: got {empty_text_emb.shape[0]}, expected {text_embed_dim}"
        )

    if not os.path.exists(args.reference_end_latent):
        raise FileNotFoundError(f"reference end latent not found: {args.reference_end_latent}")
    reference_end_latent = torch.from_numpy(np.load(args.reference_end_latent).astype(np.float32)).to(device)
    retriever = build_retriever(args, text_embed_dim, device)

    if args.threshold is None:
        selected_threshold = args.end_latent_threshold_rf if args.generative_head_type == 'rectified_flow' else args.end_latent_threshold_ddpm
    else:
        selected_threshold = args.threshold

    print(f'[INFO] generative_head_type={args.generative_head_type}, threshold={selected_threshold}')

    for index, text in enumerate(texts):
        text_emb = encode_text(text_encoder, args.text_encoder, text, device)
        if text_emb.shape[-1] != text_embed_dim:
            raise ValueError(
                f"text embedding dim mismatch: got {text_emb.shape[-1]}, expected {text_embed_dim}"
            )

        latents, sampled_steps = sample_motion_latents_with_stop(
            rag_model=rag_model,
            retriever=retriever,
            text_emb=text_emb,
            empty_text_emb=empty_text_emb,
            reference_end_latent=reference_end_latent,
            threshold=selected_threshold,
            length=args.length,
            unit_length=args.unit_length,
            cfg_scale=args.cfg_scale,
            latent_dim=args.latent_dim,
            device=device,
        )

        # MSA-VAE decoder output is the HumanML3D-style 272-dim sequence: (F,272).
        motion_272 = vae.forward_decoder(latents).squeeze(0).detach().cpu().numpy().astype(np.float32)

        # Crucial requirement: call repository built-in inverse solver to convert
        # 272-dim representation into rotation-driven SMPL format (SMPL85).
        smpl_85 = recover_from_local_rotation(motion_272, njoint=22).astype(np.float32)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        stem = sanitize_text_for_filename(text)
        if len(texts) > 1:
            stem = f"{index:03d}_{stem}"

        raw_path = os.path.join(args.out_dir, f"{stem}_{timestamp}_272.npy")
        aitv_path = os.path.join(args.aitviewer_out_dir, f"{stem}_{timestamp}.npy")

        if args.generative_head_type == 'rectified_flow':
            print(f'     flow_steps: {sampled_steps}')
            print('     ddpm_steps: 0')
        else:
            print(f'     ddpm_steps: {sampled_steps}')
            print('     flow_steps: 0')
        np.save(raw_path, motion_272)
        np.save(aitv_path, smpl_85)

        print(f"[OK] text={text}")
        print(f"     272-dim saved: {raw_path}")
        print(f"     inverse-converted rotation saved: {aitv_path}")


if __name__ == "__main__":
    main()
