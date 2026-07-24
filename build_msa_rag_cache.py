#!/usr/bin/env python
"""Build or validate the packed cache used by official MSA-T2M RAG training."""

import argparse
import json
import time

from humanml3d_272.msa_rag_cache import build_cache, validate_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="t2m_272")
    parser.add_argument("--motion-latent-dir", required=True)
    parser.add_argument("--text-latent-dir", required=True)
    parser.add_argument("--hcls-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--text-embed-dim", type=int, required=True)
    parser.add_argument("--retrieval-batch-size", type=int, default=256)
    parser.add_argument("--exclude-self", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    common = {
        "cache_dir": args.cache_dir,
        "dataset_name": args.dataset_name,
        "motion_latent_dir": args.motion_latent_dir,
        "text_latent_dir": args.text_latent_dir,
        "hcls_dir": args.hcls_dir,
        "topk": args.topk,
        "text_embed_dim": args.text_embed_dim,
        "exclude_self": args.exclude_self,
    }
    if args.validate_only:
        manifest = validate_cache(**common)
        action = "validated"
    else:
        manifest = build_cache(
            **common,
            force=args.force,
            retrieval_batch_size=args.retrieval_batch_size
        )
        action = "ready"
    summary = {
        "action": action,
        "cache_dir": args.cache_dir,
        "sample_count": manifest["sample_count"],
        "caption_count": manifest["caption_count"],
        "topk": manifest["topk"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
