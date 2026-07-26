"""Reproducible inference-cost benchmark for MSA-T2M, MotionStreamer and ReMoDiffuse.

The runner records stage timings without saving generated motions.  Heavy model
imports are lazy so manifest/statistics tests work in a CPU-only environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .common import (BenchmarkConfig, build_manifest, environment_metadata,
                     load_humanml_test_captions, summarize_samples,
                     timed_call, validate_sample, write_jsonl)


ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str) -> str:
    value = Path(path)
    return str(value if value.is_absolute() else ROOT / value)


class Adapter:
    name = "base"
    streaming = False

    def __init__(self, device: str, cfg: argparse.Namespace):
        self.device = device
        self.cfg = cfg

    def generate(self, caption: str, frames: int) -> Dict[str, Any]:
        raise NotImplementedError


class MSAAdapter(Adapter):
    name = "msa_t2m"
    streaming = True

    def __init__(self, device: str, cfg: argparse.Namespace):
        super().__init__(device, cfg)
        import torch
        import msa_gen_motion as impl
        from sentence_transformers import SentenceTransformer
        self.torch, self.impl = torch, impl
        self.device_obj = torch.device(device)
        self._load()

    def _load(self):
        t, i, d = self.torch, self.impl, self.device_obj
        from sentence_transformers import SentenceTransformer
        import models.msa_vae as msa_vae
        from models.llama_model import LLaMAHF, LLaMAHFConfig
        from models.llama_rag_model import LLaMARAGWrapper
        self.text_encoder = SentenceTransformer(_resolve(self.cfg.t5_model))
        self.text_encoder.eval()
        self.vae = msa_vae.MSA_HumanVAE(hidden_size=1024, down_t=2, stride_t=2,
            depth=3, dilation_growth_rate=3, activation="relu", latent_dim=16,
            clip_range=[-30, 20], trans_d_model=768, trans_nhead=8,
            trans_enc_layers=6, trans_dec_layers=6, trans_ff_size=2048,
            trans_dropout=0.1, clip_dim=768).to(d).eval()
        vae_ckpt = t.load(_resolve(self.cfg.msa_vae), map_location="cpu")
        self.vae.load_state_dict(vae_ckpt.get("net", vae_ckpt), strict=True)
        ckpt = t.load(_resolve(self.cfg.msa_ckpt), map_location="cpu")
        head = ckpt.get("generative_head_type", "ddpm")
        base = LLaMAHF(LLaMAHFConfig.from_name("Normal_size"), 9, 16, d,
                       generative_head_type=head, num_flow_steps=50)
        self.model = LLaMARAGWrapper(base_model=base, model_dim=base.config.n_embd).to(d)
        self.model.base_model.load_state_dict(i.load_state_strip_module(ckpt["trans"]), strict=False)
        if "rag" in ckpt:
            self.model.load_state_dict(i.load_state_strip_module(ckpt["rag"]), strict=False)
        self.model.eval()
        self.empty = t.from_numpy(np.load(_resolve(self.cfg.empty_text))).float().to(d)
        self.retriever = i.RAGRetriever(_resolve(self.cfg.hcls), topk=5, embed_dim=768, device=d)

    def generate(self, caption: str, frames: int) -> Dict[str, Any]:
        t, d = self.torch, self.device_obj
        text_box, text_ms = timed_call(lambda: t.from_numpy(np.asarray(
            self.text_encoder.encode([caption]), dtype=np.float32)).to(d), self.device)
        (hcls, scores), retrieval_ms = timed_call(lambda: self.retriever.retrieve(text_box), self.device)
        latent_box, generation_ms = timed_call(lambda: self.impl.sample_motion_latents_with_stop(
            self.model, self.text_encoder, self.retriever, caption, self.empty,
            t.zeros(16, device=d), False, 768, -1.0, frames, 4, 4.0, 16, d, False), self.device)
        motion, decode_ms = timed_call(lambda: self.vae.forward_decoder(latent_box), self.device)
        return {"output_shape": list(motion.squeeze(0).shape),
                "timings_ms": {"text_ms": text_ms, "retrieval_ms": retrieval_ms,
                               "generation_ms": generation_ms, "decode_ms": decode_ms,
                               "e2e_ms": text_ms + retrieval_ms + generation_ms + decode_ms}}


class MotionStreamerAdapter(Adapter):
    name = "motionstreamer"
    streaming = True

    def __init__(self, device: str, cfg: argparse.Namespace):
        super().__init__(device, cfg)
        import torch
        import models.tae as tae
        from models.llama_model import LLaMAHF, LLaMAHFConfig
        from sentence_transformers import SentenceTransformer
        self.torch, self.device_obj = torch, torch.device(device)
        self.text_encoder = SentenceTransformer(_resolve(cfg.t5_model)).eval()
        self.vae = tae.Causal_HumanTAE(hidden_size=1024, down_t=2, stride_t=2,
            depth=3, dilation_growth_rate=3, activation="relu", latent_dim=16,
            clip_range=[-30, 20]).to(self.device_obj).eval()
        vae = torch.load(_resolve(cfg.ms_vae), map_location="cpu")
        self.vae.load_state_dict(vae["net"], strict=True)
        ckpt = torch.load(_resolve(cfg.ms_ckpt), map_location="cpu")
        self.model = LLaMAHF(LLaMAHFConfig.from_name("Normal_size"), 9, 16,
                             self.device_obj).to(self.device_obj).eval()
        self.model.load_state_dict({k.split("module.", 1)[-1]: v for k, v in ckpt["trans"].items()}, strict=True)

    def generate(self, caption: str, frames: int) -> Dict[str, Any]:
        t = self.torch
        text, text_ms = timed_call(lambda: self.text_encoder.encode(caption), self.device)
        latent, generation_ms = timed_call(lambda: self.model.sample_for_eval_CFG_inference(
            caption, frames, self.text_encoder, self.device_obj, 4, None, -1.0, 4.0, 1.0), self.device)
        motion, decode_ms = timed_call(lambda: self.vae.forward_decoder(latent), self.device)
        return {"output_shape": list(motion.squeeze(0).shape),
                "timings_ms": {"text_ms": text_ms, "retrieval_ms": 0.0,
                               "generation_ms": generation_ms, "decode_ms": decode_ms,
                               "e2e_ms": text_ms + generation_ms + decode_ms}}


class ReMoDiffuseAdapter(Adapter):
    name = "remodiffuse"

    def __init__(self, device: str, cfg: argparse.Namespace):
        super().__init__(device, cfg)
        import torch
        sys.path.insert(0, str(ROOT / "explorations" / "ReMoDiffuse"))
        import mmcv  # noqa: F401
        from mmcv.runner import load_checkpoint
        from mogen.models import build_architecture
        import importlib.util
        source = ROOT / "explorations" / "retrieval_baselines" / "remodiffuse_gen_motion.py"
        spec = importlib.util.spec_from_file_location("benchmark_remodiffuse_impl", source)
        impl = importlib.util.module_from_spec(spec); spec.loader.exec_module(impl)
        self.torch, self.device_obj, self.impl = torch, torch.device(device), impl
        db = _resolve(cfg.remo_db)
        model = build_architecture(impl._build_model_cfg(db))
        load_checkpoint(model, _resolve(cfg.remo_ckpt), map_location="cpu", strict=False)
        model.to(self.device_obj).eval()
        impl._reload_db_into_model(model, db, self.device_obj)
        self.model = model

    def generate(self, caption: str, frames: int) -> Dict[str, Any]:
        t, d = self.torch, self.device_obj
        motion_input = t.zeros(1, frames, 272, device=d)
        batch = {"motion": motion_input, "motion_mask": t.ones(1, frames, device=d),
                 "motion_length": t.LongTensor([frames]).to(d),
                 "motion_metas": [{"text": caption}], "inference_kwargs": {}}
        result, generation_ms = timed_call(lambda: self.model(**batch)[0]["pred_motion"], self.device)
        return {"output_shape": list(result.shape),
                "timings_ms": {"text_ms": 0.0, "retrieval_ms": 0.0,
                               "generation_ms": generation_ms, "decode_ms": 0.0,
                               "e2e_ms": generation_ms}}


def _metadata(args: argparse.Namespace) -> Dict[str, Any]:
    result = environment_metadata()
    result.update({"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                   "device": args.device, "method": args.method})
    try:
        import torch
        result.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
    except Exception:
        pass
    return result


def _check_gpu_guard(device: str, allow_busy: bool) -> None:
    if not str(device).startswith("cuda") or allow_busy:
        return
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return
    pids = [line.strip() for line in output.splitlines() if line.strip()]
    if pids:
        raise RuntimeError("GPU compute processes detected (%s); use --allow-busy-gpu only for an intentional shared run" % ",".join(pids))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["msa_t2m", "motionstreamer", "remodiffuse"], required=True)
    parser.add_argument("--data-root", default="humanml3d_272")
    parser.add_argument("--output", default="benchmark_results/inference")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", default="60,120,196")
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t5-model", default="sentencet5-xxl")
    parser.add_argument("--msa-ckpt", default="Experiments/MotionStreamer_t2m_272_msa_rag_t5_trans662048_vaefulldb_k5/net_Iter100000.pth")
    parser.add_argument("--msa-vae", default="Experiments/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right/net_best_mpjpe.pth")
    parser.add_argument("--ms-ckpt", default="Experiments/explorations/motionstreamer_baselines/MotionStreamer_t2m_272_cached_embeddings_8gpu_bf16/latest.pth")
    parser.add_argument("--ms-vae", default="Experiments/causal_TAE_t2m_272_h100_20260203/net_last.pth")
    parser.add_argument("--hcls", default="humanml3d_272/h_cls_latents_msa_vae/MSA_VAEv6_phase2_t2m_272_phase1_alpha0_t5_trans662048_fulldb_right")
    parser.add_argument("--empty-text", default="humanml3d_272/text_latents_t5/empty_text_embedding.npy")
    parser.add_argument("--remo-ckpt", default="explorations/ReMoDiffuse/epoch_20.pth")
    parser.add_argument("--remo-db", default="explorations/ReMoDiffuse/data/database/t2m_text_train_272.npz")
    parser.add_argument("--manifest-only", action="store_true", help="write the deterministic manifest without loading models")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    args = parser.parse_args(argv)
    args.frames = tuple(int(v) for v in args.frames.split(","))
    if args.method != "remodiffuse" and 300 not in args.frames:
        args.frames = args.frames + (300,)
    random.seed(args.seed); np.random.seed(args.seed)
    captions = load_humanml_test_captions(_resolve(args.data_root))
    manifest = build_manifest(captions, args.frames, args.num_runs, args.warmups, args.seed)
    if args.manifest_only:
        output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
        write_jsonl(str(output / (args.method + "_manifest.jsonl")), manifest)
        print(str(output / (args.method + "_manifest.jsonl")))
        return
    _check_gpu_guard(args.device, args.allow_busy_gpu)
    adapter = {"msa_t2m": MSAAdapter, "motionstreamer": MotionStreamerAdapter,
               "remodiffuse": ReMoDiffuseAdapter}[args.method](args.device, args)
    samples = []
    for item in manifest:
        result = adapter.generate(item["caption"], item["frames"])
        sample = {"schema_version": 1, **item, "method": adapter.name, **result}
        validate_sample(sample); samples.append(sample)
    out = Path(args.output) / (adapter.name + "_" + time.strftime("%Y%m%d_%H%M%S")); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out / "samples.jsonl"), samples)
    summary = summarize_samples(samples); summary["metadata"] = _metadata(args)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["method", "frames", "count", "e2e_mean_ms", "e2e_p95_ms", "fps_mean", "rtf_mean"])
        for key, row in summary["groups"].items():
            writer.writerow([key.split("@", 1)[0], row["frames"], row["count"], row["e2e_ms"]["mean"], row["e2e_ms"]["p95"], row["effective_fps"]["mean"], row["rtf"]["mean"]])
    def write_tex(path, keys):
        with (out / path).open("w", encoding="utf-8") as handle:
            handle.write("% Generated by benchmarks/inference/run_benchmark.py\n")
            handle.write("\\begin{tabular}{lrrrr}\\toprule\nMethod & Frames & E2E (ms) & FPS & RTF \\\\ \\midrule\n")
            for key, row in summary["groups"].items():
                if keys is not None and row["frames"] not in keys:
                    continue
                handle.write("%s & %d & %.2f & %.2f & %.3f \\\\ \n" %
                             (key.split("@", 1)[0], row["frames"], row["e2e_ms"]["mean"],
                              row["effective_fps"]["mean"], row["rtf"]["mean"]))
            handle.write("\\bottomrule\\end{tabular}\n")
    write_tex("table_main.tex", set((60, 120, 196)))
    write_tex("table_streaming.tex", None)
    print(str(out))


if __name__ == "__main__":
    main()
