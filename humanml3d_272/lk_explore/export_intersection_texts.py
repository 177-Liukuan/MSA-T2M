"""
导出 HumanML3D 与 BABEL 数据集交集中每个 motion 的所有文本描述。

交集由 split/*_ft.txt 文件定义（train_ft / val_ft / test_ft）。
- HumanML3D 文本：humanml3d_272/texts/<motion_id>.txt
  每行格式：文本描述#词性标注#起始时间#结束时间
- BABEL 文本：babel_272/texts/<motion_id>.txt
  每行格式：动作标签#词性#起始时间#结束时间

输出文件格式：
    motion_id: <motion_id>  [split: train/val/test]
    [HumanML3D]
      1. <文本描述>
      ...
    [BABEL]
      1. <动作标签>
      ...
"""

from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent          # humanml3d_272/
BABEL_DIR   = BASE_DIR.parent / "babel_272"                  # babel_272/
SPLIT_DIR   = BASE_DIR / "split"
HML_TEXTS   = BASE_DIR / "texts"
BABEL_TEXTS = BABEL_DIR / "texts"
OUTPUT_FILE = Path(__file__).resolve().parent / "intersection_motion_texts.txt"

# ── 读取交集 motion ID ────────────────────────────────────
split_files = {
    "train": SPLIT_DIR / "train_ft.txt",
    "val":   SPLIT_DIR / "val_ft.txt",
    "test":  SPLIT_DIR / "test_ft.txt",
}

motion_split = {}   # motion_id -> split name
for split_name, split_path in split_files.items():
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            mid = line.strip()
            if mid:
                motion_split[mid] = split_name

print(f"交集 motion 总数: {len(motion_split)}")

# ── 按 split 顺序排列：train -> val -> test ───────────────
order = {"train": 0, "val": 1, "test": 2}
sorted_ids = sorted(motion_split.keys(),
                    key=lambda x: (order[motion_split[x]], x))


def parse_texts(text_file):
    """从文本文件中提取原始描述（取 # 分隔的第一段）。"""
    results = []
    if not text_file.exists():
        return results
    with open(text_file, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            desc = raw.split("#")[0].strip()
            if desc:
                results.append(desc)
    return results


# ── 写出结果 ─────────────────────────────────────────────
hml_missing   = []
babel_missing = []
total_hml     = 0
total_babel   = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    out.write("# HumanML3D ∩ BABEL 数据集交集文本描述\n")
    out.write(f"# 共 {len(sorted_ids)} 个 motion\n\n")

    for motion_id in sorted_ids:
        split_tag = motion_split[motion_id]

        hml_descs   = parse_texts(HML_TEXTS   / f"{motion_id}.txt")
        babel_descs = parse_texts(BABEL_TEXTS  / f"{motion_id}.txt")

        if not hml_descs:
            hml_missing.append(motion_id)
        if not babel_descs:
            babel_missing.append(motion_id)

        out.write(f"motion_id: {motion_id}  [split: {split_tag}]\n")

        out.write("  [HumanML3D]\n")
        if hml_descs:
            for i, desc in enumerate(hml_descs, 1):
                out.write(f"    {i}. {desc}\n")
            total_hml += len(hml_descs)
        else:
            out.write("    （未找到 HumanML3D 文本文件）\n")

        out.write("  [BABEL]\n")
        if babel_descs:
            for i, desc in enumerate(babel_descs, 1):
                out.write(f"    {i}. {desc}\n")
            total_babel += len(babel_descs)
        else:
            out.write("    （未找到 BABEL 文本文件）\n")

        out.write("\n")

# ── 统计摘要 ─────────────────────────────────────────────
print(f"输出文件: {OUTPUT_FILE}")
print(f"HumanML3D 文本总条数: {total_hml}")
print(f"BABEL     文本总条数: {total_babel}")
if hml_missing:
    n = len(hml_missing)
    print(f"缺失 HumanML3D 文本的 motion（{n} 个）: {hml_missing[:5]}{'...' if n>5 else ''}")
else:
    print("所有 motion 均找到 HumanML3D 文本文件。")
if babel_missing:
    n = len(babel_missing)
    print(f"缺失 BABEL 文本的 motion（{n} 个）: {babel_missing[:5]}{'...' if n>5 else ''}")
else:
    print("所有 motion 均找到 BABEL 文本文件。")
