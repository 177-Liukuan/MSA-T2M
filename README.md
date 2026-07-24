# MSA-T2M

## Align to Retrieve: Multi-Scale Semantic Latent Alignment for Retrieval-Augmented Text-to-Motion Generation

> **项目状态：** 本仓库是 MSA-T2M 的研究代码，当前正在补充实验并准备转投 AAAI 2027。论文尚未发表，checkpoint、最终配置和完整数据准备流程尚未公开。
>
> **上游关系：** MSA-T2M 基于 [MotionStreamer](https://github.com/zju3dv/MotionStreamer) 的连续潜空间自回归生成框架开发。本 README 只将 MSA-T2M 的正式主线标为 `OFFICIAL`；仓库中保留的 MotionStreamer 原始代码、失败路线和后续探索不会被误列为本文方法。

MSA-T2M 面向文本驱动人体运动生成中的一个核心矛盾：加强文本—运动语义对齐通常有利于文本可控性，却可能扭曲运动流形并降低生成真实感。我们的方案将多尺度语义对齐与检索增强生成作为一组相互补偿的组件：

- **MSA-VAE** 在 MotionStreamer 的 Causal CNN-VAE 上增加 Transformer Semantic AE，通过局部动作语义和全局文本语义组织连续运动潜空间；
- **RAG-Diffusion-AR** 从该语义潜空间检索真实运动先验，将检索结果融合为一个 RAG token，并在保留 MotionStreamer DDPM 自回归框架的基础上提升生成质量。

## 方法总览

### 1. MSA-VAE

MSA-VAE 包含两个互补轨道：

- **Representation VAE（TAE）**：Causal CNN 编码器将 272 维运动序列压缩为连续局部潜变量，CNN 解码器负责运动重建；
- **Semantic AE**：Transformer encoder 在局部潜变量前加入可学习的 `[CLS]`，Transformer decoder 从全局语义表示重建潜变量。

训练中使用两种语义监督：

- **局部对齐**：将潜变量时间步与 HumanML3D–BABEL 交集中的帧级动作标签 T5 特征对齐；
- **全局对齐**：将 `[CLS]` 与 HumanML3D sequence-level caption 的 T5 特征对齐。`[CLS]` 同时用作后续检索数据库的 key/value。

### 2. RAG-Diffusion-AR

正式生成主线只在 MotionStreamer 的 AR Model with Diffusion Head 上做轻量修改：

1. 使用 Sentence-T5-XXL 文本特征作为查询；
2. 在训练集 MSA-VAE `[CLS]` 库中执行余弦 Top-K 检索；
3. 依据相似度 softmax 权重将 Top-K `[CLS]` 融合为一个 RAG token；
4. 将 `[Text, RAG, Motion]` token 序列输入 LLaMA 风格 causal Transformer；
5. 使用 MotionStreamer 的 **DDPM diffusion head** 逐步生成下一个连续运动潜变量；
6. 使用 MSA-VAE Representation Decoder 将生成潜变量还原为 272 维运动。

```text
离线表示学习：
Motion-272 → Causal CNN-VAE → local motion latents → Semantic AE → [CLS] retrieval DB
                                  │                         │
                           local T5 alignment        global T5 alignment

文本到运动生成：
Text → Sentence-T5 → Top-K [CLS] retrieval → softmax fusion → RAG token
                                                                    │
previous motion latents ───────────────────────→ [Text, RAG, Motion] prefix
                                                                    │
                                               Causal Transformer + DDPM head
                                                                    │
                                                     MSA-VAE decoder → Motion
```

## 正式代码导航

下表是当前论文方法的推荐入口。除非在做消融或继续探索，请优先使用这些文件。

| 模块 | 状态 | 文件 | 作用 |
|---|---|---|---|
| MSA-VAE | `OFFICIAL` | `models/msa_vae.py` | Causal CNN-VAE、Semantic AE 与多尺度语义投影 |
| MSA-VAE 数据 | `OFFICIAL` | `humanml3d_272/dataset_msa_vae.py` | 运动、全局 caption T5 特征及局部动作 T5 特征 |
| TAE 预训练 | `OFFICIAL` | `TRAIN_causal_TAE.sh` | 三阶段流程的第一阶段 |
| Semantic AE 训练 | `OFFICIAL` | `TRAIN_msa_vae_phase1.sh` | 冻结 CNN，训练 Transformer AE 与语义对齐 |
| 联合微调 | `OFFICIAL` | `TRAIN_msa_vae_phase2.sh` | 解冻所有组件并采用差分学习率 |
| T5 全局特征 | `OFFICIAL` | `get_text_latent_t5.py` | 预计算 caption 和空文本的 Sentence-T5 特征 |
| MSA 潜变量库 | `OFFICIAL` | `get_msa_latent.py` | 导出运动潜变量、全局 `[CLS]` 和终止潜变量 |
| RAG 模型 | `OFFICIAL` | `models/llama_rag_model.py` | 全局 `[CLS]` 检索融合与 RAG 前缀包装 |
| RAG 数据 | `OFFICIAL` | `humanml3d_272/dataset_msa_rag.py` | 加载 T5、motion latent 和 Top-K `[CLS]` |
| RAG 训练 | `OFFICIAL` | `train_t2m_rag.py`、`TRAIN_t2m_rag.sh` | T5 + global RAG + DDPM 正式训练 |
| RAG 评估 | `OFFICIAL` | `eval_msa_t2m_rag_t5.py`、`EVAL_t2m_rag_t5.sh` | HumanML3D-272/TMR 正式评估 |
| 单条推理 | `OFFICIAL` | `msa_gen_motion.py` | 全局 RAG 文本到运动生成与可视化 |

`models/llama_rag_model.py` 后续也兼容局部检索扩展；本文的正式路径只使用其中的全局 `[CLS]` 单 token 融合，不包含局部 cross-attention。

## 环境安装

仓库继承了 MotionStreamer 的环境：

```bash
conda env create -f environment.yaml
conda activate mgpt
```

Sentence-T5-XXL 需要放在默认目录 `sentencet5-xxl/`，或在各脚本中通过 `T5_MODEL_PATH` 指定：

```bash
huggingface-cli download sentence-transformers/sentence-t5-xxl \
  --local-dir sentencet5-xxl/
```

当前 `environment.yaml` 固定了较旧的 PyTorch/Python 组合。请避免直接在已有系统 Python 环境中混装依赖；已观察到 NumPy ABI、`sentence-transformers` 缺失以及 `transformers`/`huggingface_hub` 版本不兼容等问题。

## 数据与外部资源

### HumanML3D-272

正式实验使用 MotionStreamer 发布的 272 维 HumanML3D 数据：

```bash
huggingface-cli download --repo-type dataset lxxiao/272-dim-HumanML3D \
  --local-dir ./humanml3d_272

cd humanml3d_272
unzip texts.zip
unzip motion_data.zip
cd ..
```

基础目录应至少包含：

```text
humanml3d_272/
├── mean_std/
│   ├── Mean.npy
│   └── Std.npy
├── motion_data/
├── texts/
└── split/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### 局部语义监督

MSA-VAE 的局部对齐使用 HumanML3D 与 BABEL 的交集及帧级动作标签。当前代码期望：

```text
humanml3d_272/
├── split/train_ft.txt
├── clip_enc_single/    # 历史预处理产物
└── t5_enc_single/      # 正式 MSA-VAE 使用的 768-D 局部 T5 特征
```

`dataset_clip2t5.py` 可将历史 `clip_enc_single` 中的标签索引映射为 T5 特征：

```bash
python dataset_clip2t5.py \
  --input-dir ./humanml3d_272/clip_enc_single \
  --output-dir ./humanml3d_272/t5_enc_single \
  --split-file ./humanml3d_272/split/train_ft.txt \
  --t5-model-path sentencet5-xxl/
```

该脚本只是复用历史 CLIP 特征恢复动作标签的**数据迁移工具**；正式模型的语义编码器仍为 T5，不代表 CLIP 路线属于本文方法。完整的 BABEL 对齐预处理尚待整理发布。

### TMR evaluator

评估沿用 MotionStreamer 的 TMR evaluator。可通过上游下载脚本准备：

```bash
python humanml3d_272/prepare/download_evaluator_ckpt.py
```

默认 checkpoint 路径为：

```text
Evaluator_272/experiments/temos/EXP1/checkpoints/epoch=99.ckpt
```

## 正式训练流程

### Stage 1：预训练 Causal TAE

先训练 MotionStreamer 风格的 Causal CNN-VAE，仅建立稳定的物理运动潜空间：

```bash
bash TRAIN_causal_TAE.sh <NUM_GPUS> t2m_272
```

训练完成后，将选定 checkpoint 传给下一阶段：

```bash
export CNN_CKPT=Experiments/<causal_tae_exp>/net_best_mpjpe.pth
```

### Stage 2：冻结 TAE，训练 Semantic AE

`TRAIN_msa_vae_phase1.sh` 中的“Phase 1”对应整个 MSA-VAE 流程的第二阶段。该阶段加载并冻结 CNN encoder、CNN decoder 和 decode projection，仅优化 Transformer Semantic AE、局部投影和全局 `[CLS]` 投影：

```bash
CNN_CKPT="$CNN_CKPT" \
TEXT_ENCODER_TYPE=t5 \
bash TRAIN_msa_vae_phase1.sh <NUM_GPUS> t2m_272
```

主要目标为 Transformer latent reconstruction、local T5 alignment 和 global T5 alignment。

### Stage 3：联合微调 MSA-VAE

`TRAIN_msa_vae_phase2.sh` 中的“Phase 2”对应整个流程的第三阶段。它从 Stage 2 checkpoint 恢复、解冻所有组件，并为 CNN 使用较小学习率：

```bash
export PHASE1_DIR=Experiments/<msa_vae_phase1_exp>

PHASE1_DIR="$PHASE1_DIR" \
TEXT_ENCODER_TYPE=t5 \
bash TRAIN_msa_vae_phase2.sh <NUM_GPUS> t2m_272
```

> **不要将 `explorations/representation_experiments/TRAIN_msa_vae.sh`
> 当作正式入口。** 它是早期一次性训练/CLIP 配置的遗留脚本，没有体现最终的三阶段渐进训练。

### 迭代数说明

论文草稿记载三阶段训练为 2000K / 25K / 5K iterations，但当前 `TRAIN_msa_vae_phase1.sh` 与 `TRAIN_msa_vae_phase2.sh` 默认均为 50K。现阶段 README 不声称二者一致；转投 AAAI 2027 前应根据保存的日志与 checkpoint 重新确认最终报告配置。

## 构建 T5 特征与检索库

### 1. 全局 caption T5 特征

`get_text_latent_t5.py` 会为训练集每个 motion 的所有 caption 保存特征，并额外生成 CFG 使用的 `empty_text_embedding.npy`：

```bash
python get_text_latent_t5.py \
  --dataset-name t2m_272 \
  --output-dir ./humanml3d_272/text_latents_t5 \
  --t5-model-path sentencet5-xxl/ \
  --batch-size 64
```

### 2. MSA-VAE motion latent 与 `[CLS]`

从最终 MSA-VAE checkpoint 导出用于生成的 motion latent 和用于检索的 `[CLS]`：

```bash
python get_msa_latent.py \
  --resume-pth Experiments/<msa_vae_phase2_exp>/net_best_mpjpe.pth \
  --exp-name <msa_vae_phase2_exp> \
  --latent_dir ./humanml3d_272/t2m_latents_msa_vae/<msa_vae_phase2_exp> \
  --dataname t2m_272 \
  --text_encoder_type t5 \
  --text_embed_dim 768 \
  --trans_d_model 768 \
  --trans_nhead 8 \
  --trans_enc_layers 6 \
  --trans_dec_layers 6 \
  --trans_ff_size 2048
```

脚本同时写出：

- `t2m_latents_msa_vae/<exp>/`：用于 RAG-Diffusion-AR 训练的局部运动潜变量；
- `h_cls_latents_msa_vae/<exp>/`：全局 `[CLS]` 检索数据库；
- `mu_latents_msa_vae/<exp>/`：后续局部检索实验使用的确定性 `mu`，不属于论文正式生成主线；
- `reference_end_latent_msa_vae_t2m_272.npy`：自回归生成的终止参考潜变量。

## 训练 RAG-Diffusion-AR

先将 shell 默认目录替换为实际导出目录，或使用同名环境变量覆盖。正式方法固定使用 DDPM：

```bash
export MOTION_LATENT_DIR=./humanml3d_272/t2m_latents_msa_vae/<msa_vae_phase2_exp>
export HCLS_DIR=./humanml3d_272/h_cls_latents_msa_vae/<msa_vae_phase2_exp>
export TEXT_LATENT_DIR=./humanml3d_272/text_latents_t5
export EMPTY_TEXT_PATH=./humanml3d_272/text_latents_t5/empty_text_embedding.npy

GENERATIVE_HEAD_TYPE=ddpm bash TRAIN_t2m_rag.sh <NUM_GPUS>
```

正式 launcher 默认使用 `RAG_CACHE_MODE=packed`。首次启动时，它会在
`accelerate launch` 之前，将静态 motion latent、caption T5 特征和
`[CLS]` 库打包，并为每条 caption 预计算 Top-K 检索；后续实验会先校验
源文件和配置，再复用该缓存。缓存位于
`humanml3d_272/msa_rag_cache/`，属于本地生成产物，不应提交到 Git。

如果更换或重新导出了 MSA-VAE/T5 特征，应显式重建：

```bash
REBUILD_RAG_CACHE=true \
GENERATIVE_HEAD_TYPE=ddpm \
bash TRAIN_t2m_rag.sh <NUM_GPUS>
```

慢速 reference 路径仍保留用于数值等价性复核和紧急回退：

```bash
RAG_CACHE_MODE=reference \
GENERATIVE_HEAD_TYPE=ddpm \
bash TRAIN_t2m_rag.sh <NUM_GPUS>
```

可通过 `RAG_CACHE_DIR`、`RETRIEVAL_TOPK` 和 `NUM_WORKERS` 覆盖默认缓存
目录、Top-K 和 DataLoader worker 数。packed 模式检测到缓存不完整、
源文件变化或配置不兼容时会直接停止，不会静默退回在线检索。

核心调用链为：

```text
TRAIN_t2m_rag.sh
├── build_msa_rag_cache.py
└── train_t2m_rag.py
    ├── humanml3d_272/dataset_msa_rag.py
    ├── models/rag_training.py
    ├── models/llama_model.py
    └── models/llama_rag_model.py
```

正式配置使用 T5 768-D embedding、Top-K 全局 `[CLS]`、joint condition dropout 和 DDPM diffusion loss。当前 launcher 默认 `retrieval_topk=5`；论文中的 K 消融应通过该参数单独记录。

## 评估与推理

### 正式评估

为评估脚本提供最终 MSA-VAE、RAG-Diffusion-AR checkpoint 和离线特征目录：

```bash
MSA_VAE_CKPT=Experiments/<msa_vae_phase2_exp>/net_best_mpjpe.pth \
RAG_CKPT=Experiments/<rag_exp>/net_Iter100000.pth \
MOTION_LATENT_DIR=./humanml3d_272/t2m_latents_msa_vae/<msa_vae_phase2_exp> \
HCLS_DIR=./humanml3d_272/h_cls_latents_msa_vae/<msa_vae_phase2_exp> \
TEXT_LATENT_DIR=./humanml3d_272/text_latents_t5 \
bash EVAL_t2m_rag_t5.sh 1
```

正式评估入口是 `eval_msa_t2m_rag_t5.py`。它使用 HumanML3D-272/TMR 协议报告 FID、R-Precision、MM-Dist 和 Diversity。

### 单条文本推理

`msa_gen_motion.py` 是当前全局 RAG 正式推理脚本，但 checkpoint、prompt 和输出目录仍在文件顶部配置：

```bash
python msa_gen_motion.py
```

运行前需要修改其中的 `text`、`resume_pth`、`resume_trans`、`hcls_dir`、`t5_model_path` 和 `output_dir`。输出包含 `.npy` 运动及 GIF 可视化。

## 当前论文草稿结果

以下数值来自 **NeurIPS 2026 草稿**，尚未经过 AAAI 2027 补充实验与最终复核：

| Method | FID ↓ | R@1 ↑ | R@2 ↑ | R@3 ↑ | MM-Dist ↓ | Diversity → |
|---|---:|---:|---:|---:|---:|---:|
| Real motion | 0.002 | 0.702 | 0.864 | 0.914 | 15.151 | 27.492 |
| MotionStreamer | 11.790 | 0.631 | 0.802 | 0.859 | 16.081 | 27.284 |
| MSA-T2M | **10.826** | **0.659** | **0.813** | **0.877** | **15.820** | **27.459** |

当前草稿中的主结果与消融表存在 `10.832` / `10.826` 的书写差异；这里采用消融表和结论中的 `10.826`，最终投稿前必须用保存的评估输出统一。

## 代码谱系与状态

根目录仅保留论文正式复现入口。消融、负结果、上游基线、早期 demo
和诊断脚本统一归档在 [`explorations/`](explorations/README.md)；归档脚本应从
仓库根目录按该索引中的新命令启动。

### `OFFICIAL`：论文正式主线

| 路线 | 关键文件 | 说明 |
|---|---|---|
| T5 MSA-VAE | `train_msa_vae.py`、`models/msa_vae.py` | 三阶段训练的 Stage 2–3 共用实现 |
| Global RAG + DDPM | `train_t2m_rag.py`、`models/llama_rag_model.py` | 单个全局 RAG token 前缀注入 |
| T5 evaluation | `eval_msa_t2m_rag_t5.py` | 论文正式评估路径 |
| Global RAG inference | `msa_gen_motion.py` | 正式单条生成路径 |

### `ABLATION`：论文消融或主线对照

| 路线 | 文件/参数 | 状态 |
|---|---|---|
| 去除局部/全局对齐 | `--local_align_weight`、`--global_align_weight` | MSA-VAE 组件消融 |
| 去除 dual-track decoupling | `--disable_decoupling` | MSA-VAE 结构消融 |
| 不同检索数量 K | `--retrieval_topk` | RAG 超参数消融 |
| No-RAG | `--disable_rag` | 使用 `explorations/ablations/no_rag/TRAIN_t2m_no_rag.sh` |

### `EXPERIMENTAL`：论文之后或尚未进入正式主线

| 路线 | 关键文件 | 说明 |
|---|---|---|
| Local RAG cross-attention | `explorations/cross_attention/local_rag/` | 使用检索到的局部 `mu` token；不是当前论文方法 |
| Multi-text-token MCA | `explorations/cross_attention/mca/`、`models/llama_rag_model_mca.py` | token-level T5 cross-attention |
| Latent retrieval CA | `explorations/cross_attention/latent_retrieval/`、`models/llama_rag_model_latent_retr.py` | 检索完整 motion latent 后做 cross-attention |
| MCA/local inference | `explorations/cross_attention/` | 对应探索分支的推理 |

### `LEGACY / NEGATIVE RESULT`：早期或效果不佳路线

| 路线 | 关键文件 | 结论 |
|---|---|---|
| CLIP 文本编码 | `explorations/clip/` | 早期版本，最终方法改用 T5 |
| Rectified Flow head | `explorations/rectified_flow/`、`models/diffloss.py` | 替换 DDPM 后效果不佳，正式方法保留 DDPM |
| 生成器 cross-attention | `explorations/cross_attention/mca/`、`models/llama_rag_model_mca.py` | MCA 路线效果未达到主线；不属于论文 |
| Q-Former RAG | `explorations/qformer/` | 独立检索表示尝试，未进入当前方法 |
| 一次性 MSA-VAE | `explorations/representation_experiments/TRAIN_msa_vae.sh` | 早期 CLIP/phase-0 入口，已由三阶段训练替代 |
| 原始 MotionStreamer | `explorations/motionstreamer_baselines/` | 上游基线与继承代码，不是 MSA-T2M |

`EXPERIMENTAL` 与 `LEGACY / NEGATIVE RESULT` 的区别是：前者可作为 AAAI 2027 补充实验候选，后者已有负面结果或已经被正式设计替代。交叉注意力相关代码跨越多个时间点，均不应混入正式复现命令。

## 已知问题与复现注意事项

1. **历史 T5 launcher 失效：** `explorations/qformer/PREPARE_text_embeddings.sh`
   调用仓库中不存在的 `prepare_text_embeddings.py`。正式路线请直接使用
   `get_text_latent_t5.py`。
2. **论文与 shell 迭代数不一致：** 草稿写为 2000K / 25K / 5K，当前 MSA-VAE phase shell 为 50K / 50K。
3. **K 的默认值发生过变化：** 当前 `TRAIN_t2m_rag.sh` 默认 K=5，部分 checkpoint 名与评估脚本仍包含 K=3。
4. **大量路径仍硬编码：** shell 和 demo 中的实验名是作者工作区快照，运行前必须改为自己的 checkpoint/feature 目录。
5. **推理脚本尚未 CLI 化：** `msa_gen_motion.py` 需在文件顶部修改配置。
6. **环境文件需要复核：** 当前运行环境出现过 NumPy ABI 及 Hugging Face 依赖版本冲突，完整 clean-environment smoke test 尚未完成。
7. **尚未发布完整资产：** MSA-T2M checkpoint、完整局部标签预处理和最终 AAAI 配置尚未公开。

## 与 MotionStreamer 的关系

本项目继承并保留了 MotionStreamer 的以下基础组件：

- HumanML3D/BABEL 的 272 维 SMPL motion representation；
- Causal Temporal Autoencoder；
- LLaMA 风格 causal Transformer、Two-Forward training 和 DDPM diffusion head；
- TMR-based HumanML3D-272 evaluator；
- 可视化与运动导出工具。

MSA-T2M 的新增核心是：

- 在 Causal CNN-VAE 上构建多尺度语义对齐的 Semantic AE；
- 使用 `[CLS]` 建立紧凑、可检索的真实运动先验库；
- 在 text token 与已生成 motion token 之间增加融合后的 RAG token。

## 致谢

本项目建立在以下工作与资源之上：

- [MotionStreamer](https://github.com/zju3dv/MotionStreamer)
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
- [BABEL](https://babel.is.tue.mpg.de/)
- [TMR](https://github.com/Mathux/TMR)
- [Sentence-T5](https://huggingface.co/sentence-transformers/sentence-t5-xxl)

请同时遵守 AMASS、HumanML3D、BABEL 及相关模型和数据的原始许可证。

## 引用

MSA-T2M 仍处于匿名投稿与实验完善阶段，因此这里暂不虚构作者信息或提供不准确的 BibTeX。论文信息确定后将补充正式引用：

```text
Align to Retrieve: Multi-Scale Semantic Latent Alignment for
Retrieval-Augmented Text-to-Motion Generation.
```

使用本仓库的上游组件时，也请引用 MotionStreamer 及其依赖工作。

## License

本仓库沿用根目录 `LICENSE` 中的 MIT License。数据、预训练模型和嵌套第三方项目可能适用各自的许可证。
