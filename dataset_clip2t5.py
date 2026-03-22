"""
Convert frame-level CLIP text embeddings (T, 512) to SentenceT5 embeddings (T, 768)
for HumanML3D-BABEL intersection data.

Pipeline:
  CLIP frame embedding -> nearest lookup row -> action label text -> SentenceT5 -> T5 embedding

Default lookup files:
  - humanml3d_272/pca/clip_embeddings.tsv  (N, 512)
  - humanml3d_272/pca/label_to_id.json     (label -> row index)

Default input directory:
  - humanml3d_272/clip_enc_single

Output directory:
  - auto-derived by replacing "clip" with "t5" in the final folder name,
    or adding "_t5" suffix if no "clip" substring exists.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def infer_output_dir(input_dir: str) -> str:
    """Infer output dir from input dir by renaming the last path component."""
    p = Path(input_dir)
    last = p.name
    if "clip" in last:
        new_last = last.replace("clip", "t5")
    else:
        new_last = f"{last}_t5"
    return str(p.with_name(new_last))


def load_lookup(clip_tsv_path: str, label_to_id_path: str) -> Tuple[np.ndarray, List[str]]:
    """Load CLIP lookup matrix and id->label list."""
    clip_table = np.loadtxt(clip_tsv_path).astype(np.float32)  # (N, 512)

    with open(label_to_id_path, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)

    num_rows = clip_table.shape[0]
    id_to_label = [None] * num_rows
    for label, idx in label_to_id.items():
        if 0 <= idx < num_rows:
            id_to_label[idx] = label

    missing = sum(x is None for x in id_to_label)
    if missing > 0:
        raise ValueError(
            f"label_to_id.json has missing ids: {missing} ids do not map to clip_embeddings rows"
        )

    return clip_table, id_to_label


def encode_labels_with_t5(
    labels: List[str],
    t5_model_path: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Encode lookup labels once with SentenceT5, returning (N, 768)."""
    model = SentenceTransformer(t5_model_path, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        embeddings = model.encode(
            labels,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Unexpected T5 embedding shape: {embeddings.shape}")
    return embeddings


def build_label_index(clip_table: np.ndarray, device: str) -> torch.Tensor:
    """Build normalized CLIP lookup tensor for cosine nearest search."""
    clip_table_t = torch.from_numpy(clip_table).to(device)
    clip_table_t = F.normalize(clip_table_t, dim=1)
    return clip_table_t


def map_clip_to_label_ids(
    clip_frames: np.ndarray,
    clip_table_norm: torch.Tensor,
    device: str,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map frame-level CLIP vectors to nearest lookup row ids by cosine similarity."""
    x = torch.from_numpy(clip_frames.astype(np.float32)).to(device)
    x = F.normalize(x, dim=1)

    all_ids = []
    all_sims = []
    with torch.no_grad():
        for st in range(0, x.shape[0], chunk_size):
            ed = min(st + chunk_size, x.shape[0])
            sims = x[st:ed] @ clip_table_norm.T
            best_sim, best_id = torch.max(sims, dim=1)
            all_ids.append(best_id.cpu().numpy())
            all_sims.append(best_sim.cpu().numpy())

    return np.concatenate(all_ids, axis=0), np.concatenate(all_sims, axis=0)


def collect_input_files(input_dir: str, split_file: str = "", max_files: int = 0) -> List[Path]:
    """Collect input npy files from split list or entire directory."""
    in_dir = Path(input_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    files: List[Path] = []

    if split_file:
        with open(split_file, "r", encoding="utf-8") as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        files = [in_dir / f"{sid}.npy" for sid in ids]
    else:
        files = sorted(in_dir.glob("*.npy"))

    files = [p for p in files if p.exists()]

    if max_files > 0:
        files = files[:max_files]

    return files


def process_dataset(
    input_dir: str,
    output_dir: str,
    clip_tsv_path: str,
    label_to_id_path: str,
    t5_model_path: str,
    device: str,
    t5_batch_size: int,
    sim_chunk_size: int,
    split_file: str,
    max_files: int,
    overwrite: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/5] Loading lookup table: {clip_tsv_path}")
    clip_table, id_to_label = load_lookup(clip_tsv_path, label_to_id_path)
    print(f"  lookup rows: {clip_table.shape[0]}, clip dim: {clip_table.shape[1]}")

    print(f"[2/5] Encoding lookup labels with SentenceT5: {t5_model_path}")
    t5_lookup = encode_labels_with_t5(
        labels=id_to_label,
        t5_model_path=t5_model_path,
        device=device,
        batch_size=t5_batch_size,
    )
    print(f"  t5 lookup shape: {t5_lookup.shape}")

    print("[3/5] Building cosine-search index")
    clip_table_norm = build_label_index(clip_table, device=device)

    files = collect_input_files(input_dir, split_file=split_file, max_files=max_files)
    print(f"[4/5] Processing files: {len(files)} files")

    out_dir = Path(output_dir)
    sims_min = []
    sims_mean = []
    converted = 0
    skipped = 0

    for src in tqdm(files, desc="Converting"):
        dst = out_dir / src.name
        if dst.exists() and not overwrite:
            skipped += 1
            continue

        clip_frames = np.load(src)
        if clip_frames.ndim != 2 or clip_frames.shape[1] != 512:
            print(f"[WARN] Skip {src.name}: expected (T, 512), got {clip_frames.shape}")
            skipped += 1
            continue

        label_ids, best_sims = map_clip_to_label_ids(
            clip_frames=clip_frames,
            clip_table_norm=clip_table_norm,
            device=device,
            chunk_size=sim_chunk_size,
        )

        t5_frames = t5_lookup[label_ids]  # (T, 768)
        np.save(dst, t5_frames.astype(np.float32))

        sims_min.append(float(best_sims.min()))
        sims_mean.append(float(best_sims.mean()))
        converted += 1

    print("[5/5] Done")
    print(f"  output_dir      : {output_dir}")
    print(f"  converted_files : {converted}")
    print(f"  skipped_files   : {skipped}")
    if sims_min:
        print(f"  cosine min(avg) : {np.mean(sims_min):.6f}")
        print(f"  cosine mean(avg): {np.mean(sims_mean):.6f}")
        print("  Note: values close to 1.0 indicate reliable CLIP->label lookup.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace frame-level CLIP embeddings with SentenceT5 embeddings for HumanML3D-BABEL intersection."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./humanml3d_272/clip_enc_single",
        help="Input folder containing CLIP npy files with shape (T, 512)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output folder for T5 npy files. If empty, auto-rename last folder clip->t5.",
    )
    parser.add_argument(
        "--clip-tsv",
        type=str,
        default="./humanml3d_272/pca/clip_embeddings.tsv",
        help="Path to CLIP lookup embeddings TSV (N, 512)",
    )
    parser.add_argument(
        "--label-to-id",
        type=str,
        default="./humanml3d_272/pca/label_to_id.json",
        help="Path to label->id json matching clip-tsv rows",
    )
    parser.add_argument(
        "--t5-model-path",
        type=str,
        default="sentencet5-xxl/",
        help="Local SentenceT5 model path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device for lookup search and T5 encoding",
    )
    parser.add_argument(
        "--t5-batch-size",
        type=int,
        default=8,
        help="Batch size for SentenceT5 label encoding (reduce if OOM)",
    )
    parser.add_argument(
        "--sim-chunk-size",
        type=int,
        default=2048,
        help="Chunk size for frame->lookup cosine matching",
    )
    parser.add_argument(
        "--split-file",
        type=str,
        default="./humanml3d_272/split/train_ft.txt",
        help="Optional split file with sample ids. Empty string means process all npy in input-dir.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most K files for smoke testing. 0 means all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output npy files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.strip() if args.output_dir else ""
    if not output_dir:
        output_dir = infer_output_dir(args.input_dir)

    print("==== dataset_process_t5_embeddings ====")
    print(f"input_dir     : {args.input_dir}")
    print(f"output_dir    : {output_dir}")
    print(f"clip_tsv      : {args.clip_tsv}")
    print(f"label_to_id   : {args.label_to_id}")
    print(f"t5_model_path : {args.t5_model_path}")
    print(f"device        : {args.device}")

    with torch.no_grad():
        process_dataset(
            input_dir=args.input_dir,
            output_dir=output_dir,
            clip_tsv_path=args.clip_tsv,
            label_to_id_path=args.label_to_id,
            t5_model_path=args.t5_model_path,
            device=args.device,
            t5_batch_size=args.t5_batch_size,
            sim_chunk_size=args.sim_chunk_size,
            split_file=args.split_file,
            max_files=args.max_files,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
