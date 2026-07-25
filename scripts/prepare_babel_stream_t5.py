#!/usr/bin/env python3
"""Precompute BABEL stream SentenceT5 frame targets without training."""

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from humanml3d_272.babel_stream_t5_cache import build_cache


DEFAULTS = {
    "train": {
        "motion_dir": "babel_272_stream/train_stream",
        "text_dir": "babel_272_stream/train_stream_text",
        "output_dir": "babel_272_stream/t5_enc_single/train",
    },
    "val": {
        "motion_dir": "babel_272_stream/val_stream",
        "text_dir": "babel_272_stream/val_stream_text",
        "output_dir": "babel_272_stream/t5_enc_single/val",
    },
}


class SentenceT5Encoder:
    """Adapt SentenceTransformer to the small cache-builder encoder interface."""

    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def encode(self, texts):
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute frame-level SentenceT5 targets for BABEL stream data."
    )
    parser.add_argument("--split", choices=sorted(DEFAULTS), required=True)
    parser.add_argument("--motion-dir")
    parser.add_argument("--text-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--t5-model-path", default="sentencet5-xxl/")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    defaults = DEFAULTS[args.split]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.t5_model_path, device=args.device)
    encoder = SentenceT5Encoder(model, args.batch_size)
    manifest = build_cache(
        split=args.split,
        motion_dir=args.motion_dir or defaults["motion_dir"],
        text_dir=args.text_dir or defaults["text_dir"],
        output_dir=args.output_dir or defaults["output_dir"],
        encoder=encoder,
        model_signature=str(Path(args.t5_model_path).expanduser().resolve()),
        overwrite=args.overwrite,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
