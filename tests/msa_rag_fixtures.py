from pathlib import Path

import numpy as np


def create_rag_fixture(root: Path):
    sample_ids = ["000001", "000002", "000003"]
    texts = {
        "000001": np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
        "000002": np.array([[0.0, 1.0]], dtype=np.float32),
        "000003": np.array([[0.6, 0.8], [-1.0, 0.0]], dtype=np.float32),
    }
    hcls = {
        "000001": np.array([1.0, 0.0], dtype=np.float32),
        "000002": np.array([0.0, 1.0], dtype=np.float32),
        "000003": np.array([0.6, 0.8], dtype=np.float32),
    }
    motions = {
        "000001": np.arange(8, dtype=np.float32).reshape(4, 2),
        "000002": np.arange(6, dtype=np.float32).reshape(3, 2),
        "000003": np.arange(10, dtype=np.float32).reshape(5, 2),
    }

    split_dir = root / "humanml3d_272" / "split"
    motion_dir = root / "motion"
    text_dir = root / "text"
    hcls_dir = root / "hcls"
    for directory in (split_dir, motion_dir, text_dir, hcls_dir):
        directory.mkdir(parents=True, exist_ok=True)

    (split_dir / "train.txt").write_text("\n".join(sample_ids) + "\n")
    for sample_id in sample_ids:
        np.save(motion_dir / f"{sample_id}.npy", motions[sample_id])
        np.save(text_dir / f"{sample_id}.npy", texts[sample_id])
        np.save(hcls_dir / f"{sample_id}.npy", hcls[sample_id])

    return {
        "root": root,
        "sample_ids": sample_ids,
        "motion_dir": motion_dir,
        "text_dir": text_dir,
        "hcls_dir": hcls_dir,
        "motions": motions,
        "texts": texts,
        "hcls": hcls,
    }
