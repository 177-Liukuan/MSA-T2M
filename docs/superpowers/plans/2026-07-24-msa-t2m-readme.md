# MSA-T2M 中文 README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将根目录的 MotionStreamer README 替换为准确描述 MSA-T2M 正式方法、复现流程和历史代码谱系的中文 README。

**Architecture:** README 采用“论文展示 + 正式复现 + 代码状态表”的单文档结构。所有方法判断以批准的设计文档、论文草稿、Git 历史和当前调用链为依据，并将正式主线与消融、后续探索及失败路线显式分离。

**Tech Stack:** GitHub-Flavored Markdown、Bash 命令示例、PyTorch/Accelerate 项目脚本

## Global Constraints

- README 使用中文，方法名、脚本名和状态标签保留英文。
- 正式 MSA-VAE 必须描述为 Causal TAE 预训练、冻结 CNN 训练 Semantic AE、全量联合微调三个阶段。
- 正式 RAG-Diffusion-AR 必须描述为 T5 查询、全局 `[CLS]` Top-K 检索、softmax 单 token 融合、`[Text, RAG, Motion]` 前缀和 DDPM diffusion head。
- CLIP、Rectified Flow、cross-attention、Q-Former 和早期一次性训练入口不得表述为正式方法。
- 不删除或重构遗留代码，不修改嵌套子模块。
- 论文结果必须标注为 NeurIPS 2026 草稿结果、AAAI 2027 前待复核。
- 当前代码和论文超参数冲突必须显式披露。

---

## File Structure

- Modify: `README.md` — MSA-T2M 项目总览、正式复现指南、结果和代码谱系。
- Reference: `docs/superpowers/specs/2026-07-24-msa-t2m-readme-design.md` — 已批准的方法边界和验收标准，不修改。
- Reference: `paper writing/mypaper/2026_KuanLiu_Text2Motion_NeurIPS/neurips_2026_draft.tex` — 方法与当前结果来源，不修改。

### Task 1: 重写 MSA-T2M README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: 当前仓库脚本、模型文件、论文草稿和已批准设计。
- Produces: 一个可在 GitHub 渲染的中文项目入口文档；后续验证任务依赖其标题、状态标签、路径和命令块。

- [ ] **Step 1: 保存替换前的可验证基线**

Run:

```bash
rg -n "MotionStreamer: Streaming Motion Generation|MSA-T2M|RAG-Diffusion-AR" README.md
```

Expected: 找到 MotionStreamer 标题；找不到完整的 MSA-T2M 正式方法说明。

- [ ] **Step 2: 用批准的信息架构完整替换 README**

使用 `apply_patch` 将 `README.md` 改写为以下有序章节：

1. `MSA-T2M` 标题、论文全名、AAAI 2027 完善中状态和 MotionStreamer 上游说明；
2. 项目简介、核心观察和两个正式模块；
3. 文本流程图：`Motion → Causal CNN-VAE → Semantic AE → [CLS] DB` 与 `Text → Top-K retrieval → RAG token → Diffusion-AR`；
4. `OFFICIAL` 快速导航表，逐项列出正式模型、数据、训练、评估和推理文件；
5. 环境安装与 HumanML3D-272、BABEL 局部标签、T5-XXL 和 TMR evaluator 的准备要求；
6. MSA-VAE 三阶段训练命令，并解释项目 Stage 编号与脚本 Phase 编号；
7. T5 全局/局部文本特征、MSA motion latent 和 `[CLS]` 检索库的预计算说明；
8. `GENERATIVE_HEAD_TYPE=ddpm bash TRAIN_t2m_rag.sh <NUM_GPUS>` 正式训练命令；
9. `bash EVAL_t2m_rag_t5.sh <NUM_GPUS>` 正式评估入口和 `msa_gen_motion.py` 推理入口；
10. NeurIPS 2026 草稿结果表及待复核声明；
11. `OFFICIAL`、`ABLATION`、`EXPERIMENTAL`、`LEGACY / NEGATIVE RESULT` 四类代码谱系表；
12. 缺失 `prepare_text_embeddings.py`、无效 `DISABLE_RAG` launcher、迭代数冲突、硬编码路径和当前环境依赖冲突等已知问题；
13. MotionStreamer、HumanML3D、BABEL、TMR 致谢，匿名投稿阶段不虚构作者信息的引用说明和 MIT license。

正式导航至少包含以下现存路径：

```text
models/msa_vae.py
models/llama_rag_model.py
humanml3d_272/dataset_msa_vae.py
humanml3d_272/dataset_msa_rag.py
TRAIN_causal_TAE.sh
TRAIN_msa_vae_phase1.sh
TRAIN_msa_vae_phase2.sh
get_text_latent_t5.py
get_msa_latent.py
TRAIN_t2m_rag.sh
train_t2m_rag.py
EVAL_t2m_rag_t5.sh
eval_msa_t2m_rag_t5.py
msa_gen_motion.py
```

- [ ] **Step 3: 运行内容边界检查**

Run:

```bash
rg -n "^# MSA-T2M|三阶段|TRAIN_causal_TAE|TRAIN_msa_vae_phase1|TRAIN_msa_vae_phase2|RAG-Diffusion-AR|GENERATIVE_HEAD_TYPE=ddpm|OFFICIAL|ABLATION|EXPERIMENTAL|LEGACY / NEGATIVE RESULT|Rectified Flow|CLIP|交叉注意力|NeurIPS 2026|AAAI 2027" README.md
```

Expected: 所有正式阶段、状态标签和失败路线均可命中；标题为 MSA-T2M。

- [ ] **Step 4: 检查 Markdown 和本地路径**

Run:

```bash
git diff --check
python - <<'PY'
import pathlib
import re

text = pathlib.Path("README.md").read_text(encoding="utf-8")
paths = set(re.findall(r"`([^`\\n]+\\.(?:py|sh|yaml|md))`", text))
missing = sorted(p for p in paths if not pathlib.Path(p).exists())
assert not missing, f"README references missing local files: {missing}"
assert text.count("```") % 2 == 0, "unbalanced fenced code blocks"
print(f"validated {len(paths)} local file references")
PY
```

Expected: `git diff --check` 无输出；Python 检查打印验证数量且退出码为 0。

- [ ] **Step 5: 人工核对正式方法不被实验代码污染**

Run:

```bash
sed -n '1,420p' README.md
```

Expected:

- 正式方法只使用 T5、全局 `[CLS]` 单 RAG token 和 DDPM；
- Cross-attention、RF、CLIP、Q-Former 只出现在非正式代码表或已知问题；
- `TRAIN_msa_vae.sh` 不出现在正式三阶段命令中；
- 原始 MotionStreamer 被描述为上游基线。

- [ ] **Step 6: 提交 README**

```bash
git add README.md
git commit -m "docs: rewrite README for MSA-T2M"
```

### Task 2: 最终仓库级验证

**Files:**
- Verify: `README.md`
- Verify: `docs/superpowers/specs/2026-07-24-msa-t2m-readme-design.md`

**Interfaces:**
- Consumes: Task 1 生成的 README 和设计文档验收标准。
- Produces: 可追溯的验证结果和干净的目标文件 diff；不改变嵌套子模块状态。

- [ ] **Step 1: 验证首屏与正式入口**

Run:

```bash
sed -n '1,80p' README.md
rg -n "TRAIN_causal_TAE.sh|TRAIN_msa_vae_phase1.sh|TRAIN_msa_vae_phase2.sh|TRAIN_t2m_rag.sh|EVAL_t2m_rag_t5.sh|msa_gen_motion.py" README.md
```

Expected: 首屏为 MSA-T2M；六个关键入口均出现且位于正式流程。

- [ ] **Step 2: 验证设计验收标准**

Run:

```bash
python - <<'PY'
from pathlib import Path

text = Path("README.md").read_text(encoding="utf-8")
checks = {
    "msa_title": text.lstrip().startswith("# MSA-T2M"),
    "three_stage": all(x in text for x in [
        "TRAIN_causal_TAE.sh",
        "TRAIN_msa_vae_phase1.sh",
        "TRAIN_msa_vae_phase2.sh",
    ]),
    "rag_prefix": all(x in text for x in [
        "[Text, RAG, Motion]",
        "DDPM",
        "Top-K",
        "softmax",
    ]),
    "status_taxonomy": all(x in text for x in [
        "OFFICIAL",
        "ABLATION",
        "EXPERIMENTAL",
        "LEGACY / NEGATIVE RESULT",
    ]),
    "draft_caveat": "NeurIPS 2026 草稿" in text and "AAAI 2027" in text,
}
failed = [name for name, ok in checks.items() if not ok]
assert not failed, f"README acceptance checks failed: {failed}"
print("README acceptance checks passed")
PY
```

Expected: 输出 `README acceptance checks passed`。

- [ ] **Step 3: 确认只修改目标文件且保留用户子模块状态**

Run:

```bash
git status --short --branch
git log -2 --oneline
```

Expected:

- README 提交和设计文档提交位于当前分支顶部；
- 仅保留任务开始前已经存在的 `MotionCLIP`、`VideoVAEPlus` 和 `paper writing/Research-Paper-Writing-Skills` 嵌套仓库状态；
- 没有未提交的 README 或计划文件改动。
