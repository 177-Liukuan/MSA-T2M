import os
import re
import time
import glob
import random
import warnings
import argparse

import numpy as np
import torch

from models.llama_model import LLaMAHF, LLaMAHFConfig
from models.llama_rag_model import LLaMARAGWrapper
import models.msa_vae as msa_vae
from visualization.recover_visualize import recover_from_local_position
import visualization.plot_3d_global as plot_3d


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================
# User-editable config area
# =========================
cfg_scale = 6
threshold = 0.1
retrieval_topk = 3
max_length = 300
fps = 30
disable_rag = False
seed = 123
deterministic = True
generative_head_type = "auto"  # auto|ddpm|rectified_flow
num_flow_steps = 50
flow_solver = "euler"
rf_time_sampling = "uniform"
rf_loss_type = "mse"

# Fixed paths (same as msa_gen_motion.py)
resume_pth = "Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/net_best_mpjpe.pth"
resume_trans = "Experiments/MotionStreamer_t2m_272_msa_rag_t5_trans662048_vaefulldb_k3_testcode_worker4/net_Iter100000.pth"
hcls_dir = "./humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right"
empty_text_path = "./humanml3d_272/text_latents_t5/empty_text_embedding.npy"
reference_end_latent = "humanml3d_272/t2m_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/reference_end_latent_msa_vae_t2m_272.npy"
t5_model_path = "sentencet5-xxl/"

# Dataset paths
test_split_file = "./humanml3d_272/split/test.txt"
texts_dir = "./humanml3d_272/texts"

# Additional defaults aligned with msa_gen_motion.py
unit_length = 4
latent_dim = 16
text_embed_dim = 768
num_diffusion_head_layers = 9

hidden_size = 1024
down_t = 2
stride_t = 2
depth = 3
dilation_growth_rate = 3

trans_d_model = 768
trans_nhead = 8
trans_enc_layers = 6
trans_dec_layers = 6
trans_ff_size = 2048
trans_dropout = 0.1
clip_dim = 768

mean_path = "humanml3d_272/mean_std/Mean.npy"
std_path = "humanml3d_272/mean_std/Std.npy"

output_dir = "demo_output/MSA-T2M-Batch_02"

use_random_topk_inference = True

# Maximum samples in a single GPU batch during AR generation.
# Lower this if you hit OOM; increase for better GPU utilization.
inference_batch_size = 32


# =========================
# Utility classes & funcs
# =========================

class RAGRetriever:
    """In-memory h_cls retrieval library for inference."""

    def __init__(self, hcls_dir_path, topk=3, embed_dim=768, device=torch.device("cuda")):
        self.topk = int(topk)
        self.embed_dim = int(embed_dim)

        files = sorted(glob.glob(os.path.join(hcls_dir_path, "*.npy")))
        if len(files) == 0:
            raise RuntimeError(f"No h_cls npy files found in: {hcls_dir_path}")

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
                f"No valid {self.embed_dim}-d h_cls vectors found in: {hcls_dir_path}"
            )

        lib = torch.from_numpy(np.stack(vectors)).float().to(device)
        self.lib = lib
        self.lib_norm = self._norm(lib)

    @staticmethod
    def _norm(x, eps=1e-6):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, text_emb):
        """text_emb: [B, D_text] -> top_hcls [B, topk, D], top_scores [B, topk]"""
        query = self._norm(text_emb)
        sim = torch.matmul(query, self.lib_norm.t())

        k = min(self.topk, sim.shape[1])
        top_scores, top_idx = torch.topk(sim, k=k, dim=1)
        top_hcls = self.lib[top_idx]

        if k < self.topk:
            pad_h = torch.zeros(
                text_emb.shape[0], self.topk - k, self.embed_dim,
                device=text_emb.device, dtype=top_hcls.dtype,
            )
            pad_s = torch.full(
                (text_emb.shape[0], self.topk - k), -1e6,
                device=text_emb.device, dtype=top_scores.dtype,
            )
            top_hcls = torch.cat([top_hcls, pad_h], dim=1)
            top_scores = torch.cat([top_scores, pad_s], dim=1)

        return top_hcls, top_scores


def set_reproducibility(seed_value, deterministic_mode=True):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic_mode)
    if deterministic_mode:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def load_state_strip_module(state_dict):
    out = {}
    for key, value in state_dict.items():
        if key.split(".")[0] == "module":
            out[".".join(key.split(".")[1:])] = value
        else:
            out[key] = value
    return out


def infer_head_type_from_ckpt(ckpt, user_choice="auto"):
    valid = {"ddpm", "rectified_flow"}
    user_norm = str(user_choice).strip().lower()
    if user_norm == "auto":
        ckpt_type = None
        if isinstance(ckpt, dict):
            ckpt_type = ckpt.get("generative_head_type", None)
        ckpt_norm = str(ckpt_type).strip().lower() if ckpt_type is not None else None
        return ckpt_norm if ckpt_norm in valid else "ddpm"
    if user_norm in valid:
        return user_norm
    raise ValueError(
        f"Invalid generative_head_type: {user_choice}. Expected one of auto|ddpm|rectified_flow"
    )


def sanitize_text_for_filename(raw_text):
    cleaned = raw_text.strip().lower()
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^\w\-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "motion"
    return cleaned[:60]


def load_test_ids(split_file):
    ids = []
    with open(split_file, "r") as f:
        for line in f:
            sid = line.strip()
            if sid:
                ids.append(sid)
    return ids


def load_text_for_id(texts_dir, motion_id):
    """Load one random text description for the given motion id.

    Each line in the .txt has the format:
        raw_text#pos_tags#start#end
    Returns the raw_text part.
    """
    txt_path = os.path.join(texts_dir, motion_id + ".txt")
    if not os.path.exists(txt_path):
        return None, txt_path
    lines = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("#")
                raw_text = parts[0].strip()
                if raw_text:
                    lines.append(raw_text)
    if not lines:
        return None, txt_path
    return random.choice(lines), txt_path


def check_required_files():
    required_files = [
        ("resume_pth", resume_pth),
        ("resume_trans", resume_trans),
        ("empty_text_path", empty_text_path),
        ("reference_end_latent", reference_end_latent),
        ("mean_path", mean_path),
        ("std_path", std_path),
        ("test_split_file", test_split_file),
    ]
    for name, path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file missing: {name} -> {path}")
    if not os.path.isdir(texts_dir):
        raise FileNotFoundError(f"Texts directory missing: {texts_dir}")
    if not disable_rag and not os.path.isdir(hcls_dir):
        raise FileNotFoundError(f"RAG retrieval directory missing: {hcls_dir}")
    if not os.path.isdir(t5_model_path):
        raise FileNotFoundError(f"T5 model path missing: {t5_model_path}")


@torch.no_grad()
def sample_motion_latents_batch(
    rag_model,
    text_embs,          # [B, D_text] — already on device
    top_hcls,           # [B, topk, D] or None
    top_scores,         # [B, topk]   or None
    empty_text_emb,     # [D_text] on device
    reference_end,      # [1, latent_dim] on device
    disable_rag_flag=False,
    stop_threshold=0.1,
    length=300,
    unit_len=4,
    cfg=4.0,
    token_latent_dim=16,
    device=torch.device("cuda"),
    use_random_topk=False,
):
    """True batched autoregressive generation.

    All B samples share a single transformer forward pass per AR step.
    Per-sample stop conditions are tracked with a boolean mask; generation
    ends when every sample has stopped or max_token_len is reached.

    Returns:
        List[Tensor]: length-B list of tensors, each shaped [T_i, latent_dim]
                      where T_i is the actual (stopped) token length.
    """
    B = text_embs.shape[0]
    max_token_len = int(length) // int(unit_len)

    # Optionally pick one random h_cls per sample for diversity
    if use_random_topk and (not disable_rag_flag) and top_hcls is not None and top_hcls.shape[1] > 1:
        rand_k = torch.randint(0, top_hcls.shape[1], (B,), device=device)  # [B]
        top_hcls = top_hcls[torch.arange(B, device=device), rand_k].unsqueeze(1)    # [B,1,D]
        top_scores = top_scores[torch.arange(B, device=device), rand_k].unsqueeze(1)  # [B,1]

    if disable_rag_flag:
        top_hcls = None
        top_scores = None

    # AR state
    xs = torch.zeros(B, 0, token_latent_dim, device=device, dtype=torch.float32)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    stop_at = torch.full((B,), max_token_len, dtype=torch.long, device=device)

    ref = reference_end.view(1, token_latent_dim).expand(B, -1)  # [B, latent_dim]

    for step in range(max_token_len):
        # --- single forward for all B samples ---
        next_tokens = rag_model.sample_next_with_cfg(
            motion_prefix=xs,
            text_emb=text_embs,
            top3_h_cls=top_hcls,
            top3_sim_scores=top_scores,
            empty_text_emb=empty_text_emb,
            cfg_scale=cfg,
            temperature=1.0,
        )  # [B, latent_dim]

        # Check per-sample stop criterion
        dist = torch.linalg.norm(next_tokens - ref, dim=-1)  # [B]
        newly_done = (~done) & (dist < stop_threshold)
        stop_at[newly_done] = step + 1
        done |= newly_done

        xs = torch.cat([xs, next_tokens.unsqueeze(1)], dim=1)  # [B, step+1, latent_dim]

        if done.all():
            break

    # Slice each sample to its individual stop length
    stop_at_cpu = stop_at.cpu().tolist()
    results = []
    for i in range(B):
        t = max(1, stop_at_cpu[i])
        results.append(xs[i, :t, :])  # [T_i, latent_dim]

    return results


def build_models(device, run_disable_rag):
    """Load VAE and RAG transformer, return (net, rag_model)."""
    clip_range = [-30, 20]
    net = msa_vae.MSA_HumanVAE(
        hidden_size=hidden_size,
        down_t=down_t,
        stride_t=stride_t,
        depth=depth,
        dilation_growth_rate=dilation_growth_rate,
        activation="relu",
        latent_dim=latent_dim,
        clip_range=clip_range,
        trans_d_model=trans_d_model,
        trans_nhead=trans_nhead,
        trans_enc_layers=trans_enc_layers,
        trans_dec_layers=trans_dec_layers,
        trans_ff_size=trans_ff_size,
        trans_dropout=trans_dropout,
        clip_dim=clip_dim,
    ).to(device)

    print(f"Loading VAE checkpoint: {resume_pth}")
    ckpt_vae = torch.load(resume_pth, map_location="cpu")
    state_vae = ckpt_vae["net"] if (isinstance(ckpt_vae, dict) and "net" in ckpt_vae) else ckpt_vae
    net.load_state_dict(state_vae, strict=True)
    net.eval()

    print(f"Loading transformer checkpoint: {resume_trans}")
    ckpt_rag = torch.load(resume_trans, map_location="cpu")
    ckpt_head_type = ckpt_rag.get("generative_head_type", "missing") if isinstance(ckpt_rag, dict) else "missing"
    resolved_head_type = infer_head_type_from_ckpt(ckpt_rag, generative_head_type)
    print(f"[HeadType] checkpoint: {ckpt_head_type}  ->  resolved: {resolved_head_type}")

    config = LLaMAHFConfig.from_name("Normal_size")
    config.block_size = 78
    base_model = LLaMAHF(
        config,
        num_diffusion_head_layers,
        latent_dim,
        device,
        generative_head_type=resolved_head_type,
        num_flow_steps=num_flow_steps,
        flow_solver=flow_solver,
        rf_time_sampling=rf_time_sampling,
        rf_loss_type=rf_loss_type,
    )
    rag_model = LLaMARAGWrapper(
        base_model=base_model,
        model_dim=config.n_embd,
        disable_rag=run_disable_rag,
    ).to(device)

    if "trans" not in ckpt_rag:
        raise KeyError("Checkpoint must contain trans key.")
    rag_model.base_model.load_state_dict(load_state_strip_module(ckpt_rag["trans"]), strict=False)

    if "rag" in ckpt_rag:
        rag_model.load_state_dict(load_state_strip_module(ckpt_rag["rag"]), strict=False)
    elif not run_disable_rag:
        raise KeyError("RAG checkpoint must contain rag key when disable_rag=False.")

    rag_model.eval()
    return net, rag_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch motion generation from test set with visualization."
    )
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=10,
        help="Number of test samples to randomly draw and visualize (2~1000, default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed,
        help=f"Random seed (default: {seed}).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help=f"Output directory (default: {output_dir}).",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=cfg_scale,
        help=f"Classifier-free guidance scale (default: {cfg_scale}).",
    )
    parser.add_argument(
        "--disable_rag",
        action="store_true",
        default=disable_rag,
        help="Disable RAG retrieval (no-RAG mode).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=inference_batch_size,
        help=f"GPU batch size for AR generation (default: {inference_batch_size}). Reduce if OOM.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    n = args.num_samples
    if not (2 <= n <= 1000):
        raise ValueError(f"--num_samples must be between 2 and 1000, got {n}.")

    run_seed = args.seed
    run_output_dir = args.output_dir
    run_cfg_scale = args.cfg_scale
    run_disable_rag = args.disable_rag
    run_batch_size = max(1, args.batch_size)

    try:
        check_required_files()
        os.makedirs(run_output_dir, exist_ok=True)

        set_reproducibility(run_seed, deterministic_mode=deterministic)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # ---- Sample n test IDs ----
        all_test_ids = load_test_ids(test_split_file)
        print(f"Test set size: {len(all_test_ids)}")
        if n > len(all_test_ids):
            raise ValueError(
                f"Requested {n} samples but test set only has {len(all_test_ids)} entries."
            )
        sampled_ids = random.sample(all_test_ids, n)
        print(f"Randomly sampled {n} motion IDs from test set.")

        # ---- Load texts for sampled IDs ----
        samples = []  # list of (motion_id, text)
        skipped = []
        for mid in sampled_ids:
            text_str, txt_path = load_text_for_id(texts_dir, mid)
            if text_str is None:
                print(f"[WARN] No text found for {mid} ({txt_path}), skipping.")
                skipped.append(mid)
            else:
                samples.append((mid, text_str))

        if len(samples) < 2:
            raise RuntimeError(
                f"Too few valid samples after loading texts: {len(samples)}. "
                f"Need at least 2. Skipped: {skipped}"
            )
        print(f"Valid samples with text: {len(samples)} (skipped: {len(skipped)})")

        # ---- Load models ----
        net, rag_model = build_models(device, run_disable_rag)

        # ---- Text encoder ----
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise ImportError(
                "Failed to import sentence_transformers. Please ensure SentenceTransformer is installed."
            ) from exc

        text_encoder = SentenceTransformer(t5_model_path)
        text_encoder.eval()

        empty_text_emb = (
            torch.from_numpy(np.load(empty_text_path).astype(np.float32))
            .reshape(-1)
            .to(device)
        )
        if empty_text_emb.shape[0] != text_embed_dim:
            raise ValueError(
                f"empty text embedding dim mismatch: got {empty_text_emb.shape[0]}, expected {text_embed_dim}"
            )

        # ---- RAG retriever ----
        retriever = None
        if not run_disable_rag:
            retriever = RAGRetriever(
                hcls_dir_path=hcls_dir,
                topk=retrieval_topk,
                embed_dim=text_embed_dim,
                device=device,
            )
        else:
            print("No-RAG mode enabled: retrieval branch is bypassed.")

        # ---- Mean / Std / reference end latent ----
        mean = np.load(mean_path)
        std = np.load(std_path)
        ref_end = torch.from_numpy(np.load(reference_end_latent).astype(np.float32)).to(device)

        # ---- Batch text encoding (one call, all texts) ----
        all_texts = [s[1] for s in samples]
        print(f"Encoding {len(all_texts)} texts...")
        t0_enc = time.time()
        all_text_feats = text_encoder.encode(all_texts, batch_size=run_batch_size, show_progress_bar=True)
        print(f"Text encoding done in {time.time()-t0_enc:.1f}s")

        # ---- Chunked batched AR generation ----
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        txt_log_path = os.path.join(run_output_dir, f"batch_{timestamp}_texts.txt")

        # all_latents[i] = Tensor [T_i, latent_dim] for sample i
        all_latents = []
        num_chunks = (len(samples) + run_batch_size - 1) // run_batch_size
        print(f"\nBatched AR generation: {len(samples)} samples, batch_size={run_batch_size}, chunks={num_chunks}")

        t0_gen = time.time()
        for chunk_idx in range(num_chunks):
            start = chunk_idx * run_batch_size
            end = min(start + run_batch_size, len(samples))
            chunk_size = end - start
            print(f"  Chunk [{chunk_idx+1}/{num_chunks}] samples {start+1}~{end} ...", flush=True)

            chunk_feats = np.asarray(all_text_feats[start:end], dtype=np.float32)
            chunk_text_embs = torch.from_numpy(chunk_feats).to(device)  # [B_chunk, D_text]

            chunk_top_hcls = None
            chunk_top_scores = None
            if not run_disable_rag:
                chunk_top_hcls, chunk_top_scores = retriever.retrieve(chunk_text_embs)

            chunk_latents = sample_motion_latents_batch(
                rag_model=rag_model,
                text_embs=chunk_text_embs,
                top_hcls=chunk_top_hcls,
                top_scores=chunk_top_scores,
                empty_text_emb=empty_text_emb,
                reference_end=ref_end,
                disable_rag_flag=run_disable_rag,
                stop_threshold=threshold,
                length=max_length,
                unit_len=unit_length,
                cfg=run_cfg_scale,
                token_latent_dim=latent_dim,
                device=device,
                use_random_topk=use_random_topk_inference,
            )
            all_latents.extend(chunk_latents)

        print(f"AR generation done in {time.time()-t0_gen:.1f}s")

        # ---- Decode, visualize, save ----
        success_count = 0
        fail_count = 0
        log_lines = []

        print(f"\nDecoding and rendering {len(samples)} motions...")
        for idx, ((motion_id, input_text), latent) in enumerate(zip(samples, all_latents)):
            print(f"  [{idx+1}/{len(samples)}] ID={motion_id} | tokens={latent.shape[0]} | text: {input_text[:60]}")
            try:
                # VAE decode: [1, T, latent_dim]
                motion = (
                    net.forward_decoder(latent.unsqueeze(0))
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                pred_xyz = recover_from_local_position(motion * std + mean, 22)
                xyz = pred_xyz.reshape(1, -1, 22, 3)

                text_token = sanitize_text_for_filename(input_text)
                stem = f"batch_{motion_id}_{text_token}_{timestamp}"
                gif_path = os.path.join(run_output_dir, f"{stem}.gif")
                npy_path = os.path.join(run_output_dir, f"{stem}.npy")

                plot_3d.draw_to_batch(xyz, [input_text], [gif_path], fps=fps)
                np.save(npy_path, motion)

                print(f"    [OK] GIF: {gif_path}")
                print(f"    [OK] NPY: {npy_path}")
                log_lines.append((idx + 1, motion_id, input_text, gif_path, npy_path, "OK"))
                success_count += 1

            except Exception as exc:
                print(f"    [FAIL] {type(exc).__name__}: {exc}")
                log_lines.append((idx + 1, motion_id, input_text, "", "", f"FAIL: {type(exc).__name__}: {exc}"))
                fail_count += 1

        # ---- Write text log ----
        with open(txt_log_path, "w", encoding="utf-8") as f:
            f.write("Batch Motion Generation Log\n")
            f.write(f"Generated at: {timestamp}\n")
            f.write(f"Total samples: {len(samples)}  |  Success: {success_count}  |  Failed: {fail_count}\n")
            f.write(f"Seed: {run_seed}  |  CFG scale: {run_cfg_scale}  |  Disable RAG: {run_disable_rag}\n")
            f.write(f"AR batch size: {run_batch_size}\n")
            f.write("=" * 80 + "\n\n")
            for (i, mid, txt, gif_p, npy_p, status) in log_lines:
                f.write(f"[{i:04d}] Motion ID : {mid}\n")
                f.write(f"       Text      : {txt}\n")
                f.write(f"       GIF       : {gif_p}\n")
                f.write(f"       NPY       : {npy_p}\n")
                f.write(f"       Status    : {status}\n")
                f.write("\n")

        print(f"\n{'='*60}")
        print(f"Batch complete. Success: {success_count}/{len(samples)}, Failed: {fail_count}")
        print(f"Text log saved: {txt_log_path}")
        print(f"Outputs in: {run_output_dir}")

    except Exception as exc:
        print("[ERROR] Batch motion generation failed.")
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
