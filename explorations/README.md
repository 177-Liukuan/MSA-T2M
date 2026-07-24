# MSA-T2M 探索路线归档

这里保存未进入论文正式主线的消融、负结果、上游基线、早期 demo
和诊断入口。正式复现请使用仓库根目录的 Causal TAE、MSA-VAE、
T5 特征、Global-RAG DDPM 训练与评估脚本。

所有命令均从仓库根目录运行。Python 入口使用 `python -m`，Shell
入口直接传给 `bash`。归档只移动了入口文件，共享实现仍位于
`models/`、`humanml3d_272/`、`options/`、`utils/` 和
`visualization/`。

状态含义：

- `ABLATION`：正式方法的消融。
- `EXPERIMENTAL`：可继续研究，但未进入当前论文方法。
- `NEGATIVE RESULT`：已有负面结果或已被正式方法替代。
- `BASELINE`：比较方法或上游 MotionStreamer。
- `HISTORY`：历史 demo、诊断或项目记录。

## Experiment results

Existing and future exploration results live under
`Experiments/explorations/<route>/`. Official Causal TAE, MSA-VAE,
Global-RAG DDPM, and formal ablation results remain directly under
`Experiments/`. Exploration scripts may still consume an official checkpoint
from the root; only their own output belongs under the exploration tree.

## No-RAG

状态：`ABLATION`。使用正式 T5/DDPM 训练实现，但关闭检索 token。
需要与正式训练相同的 MSA-VAE checkpoint、T5 特征、motion latent
和 h-cls 检索库。

| 原根路径 | 新命令 |
|---|---|
| `demo_msa_t2m_no_rag_t5.py` | `python -m explorations.ablations.no_rag.demo_msa_t2m_no_rag_t5` |
| `DEMO_msa_t2m_no_rag_t5.sh` | `bash explorations/ablations/no_rag/DEMO_msa_t2m_no_rag_t5.sh` |
| `eval_msa_t2m_no_rag_t5.py` | `python -m explorations.ablations.no_rag.eval_msa_t2m_no_rag_t5` |
| `EVAL_t2m_no_rag_t5.sh` | `bash explorations/ablations/no_rag/EVAL_t2m_no_rag_t5.sh` |
| `TRAIN_t2m_no_rag.sh` | `bash explorations/ablations/no_rag/TRAIN_t2m_no_rag.sh` |

## CLIP conditioning

状态：`NEGATIVE RESULT`。这是早期 512 维 CLIP 条件路线；正式方法使用
离线 Sentence-T5-XXL 768 维特征。需要 CLIP 权重、早期 TAE latent
及对应 checkpoint。

| 原根路径 | 新命令 |
|---|---|
| `demo_msa_t2m_clip.py` | `python -m explorations.clip.demo_msa_t2m_clip` |
| `eval_t2m_clip_baseline.py` | `python -m explorations.clip.eval_t2m_clip_baseline` |
| `EVAL_t2m_clip_baseline.sh` | `bash explorations/clip/EVAL_t2m_clip_baseline.sh` |
| `eval_t2m_rag.py` | `python -m explorations.clip.eval_t2m_rag` |
| `EVAL_t2m_rag.sh` | `bash explorations/clip/EVAL_t2m_rag.sh` |
| `get_text_latent_clip.py` | `python -m explorations.clip.get_text_latent_clip` |
| `train_t2m_baseline_clip.py` | `python -m explorations.clip.train_t2m_baseline_clip` |
| `TRAIN_t2m_baseline_clip.sh` | `bash explorations/clip/TRAIN_t2m_baseline_clip.sh` |

## Local-RAG cross-attention

状态：`NEGATIVE RESULT`。该路线把检索到的局部 motion 表示作为
cross-attention token；它不是论文中的单个全局 RAG token。需要局部
标签、局部 latent、T5 特征与匹配结构的 checkpoint。

| 原根路径 | 新命令 |
|---|---|
| `eval_msa_t2m_rag_local.py` | `python -m explorations.cross_attention.local_rag.eval_msa_t2m_rag_local` |
| `EVAL_t2m_rag_local.sh` | `bash explorations/cross_attention/local_rag/EVAL_t2m_rag_local.sh` |
| `msa_gen_motion_local.py` | `python -m explorations.cross_attention.local_rag.msa_gen_motion_local` |
| `train_t2m_rag_local.py` | `python -m explorations.cross_attention.local_rag.train_t2m_rag_local` |
| `TRAIN_t2m_rag_local.sh` | `bash explorations/cross_attention/local_rag/TRAIN_t2m_rag_local.sh` |
| `TRAIN_THEN_EVAL_t2m_rag_local.sh` | `bash explorations/cross_attention/local_rag/TRAIN_THEN_EVAL_t2m_rag_local.sh` |

## Multi-token MCA

状态：`NEGATIVE RESULT`。Flamingo 风格的多文本 token
cross-attention 未优于正式 Global-RAG 路线。需要 token-level T5
特征及 MCA checkpoint。

| 原根路径 | 新命令 |
|---|---|
| `eval_msa_t2m_rag_mca.py` | `python -m explorations.cross_attention.mca.eval_msa_t2m_rag_mca` |
| `EVAL_t2m_rag_mca.sh` | `bash explorations/cross_attention/mca/EVAL_t2m_rag_mca.sh` |
| `get_text_token_latent_t5.py` | `python -m explorations.cross_attention.mca.get_text_token_latent_t5` |
| `msa_gen_motion_mca.py` | `python -m explorations.cross_attention.mca.msa_gen_motion_mca` |
| `msa_gen_motion_mca_op.py` | `python -m explorations.cross_attention.mca.msa_gen_motion_mca_op` |
| `train_t2m_rag_multi_text_token.py` | `python -m explorations.cross_attention.mca.train_t2m_rag_multi_text_token` |
| `Train_t2m_rag_multi_text_token.sh` | `bash explorations/cross_attention/mca/Train_t2m_rag_multi_text_token.sh` |

## Latent-retrieval cross-attention

状态：`EXPERIMENTAL`。检索完整 motion latent 后做 cross-attention，
含普通 CFG 与 additive CFG 评估。需要预构建 latent retrieval library、
T5 特征和匹配 checkpoint。

| 原根路径 | 新命令 |
|---|---|
| `build_latent_retr_library.py` | `python -m explorations.cross_attention.latent_retrieval.build_latent_retr_library` |
| `eval_msa_t2m_rag_latent_retr.py` | `python -m explorations.cross_attention.latent_retrieval.eval_msa_t2m_rag_latent_retr` |
| `EVAL_t2m_rag_latent_retr.sh` | `bash explorations/cross_attention/latent_retrieval/EVAL_t2m_rag_latent_retr.sh` |
| `eval_msa_t2m_rag_latent_retr_addcfg.py` | `python -m explorations.cross_attention.latent_retrieval.eval_msa_t2m_rag_latent_retr_addcfg` |
| `EVAL_t2m_rag_latent_retr_addcfg.sh` | `bash explorations/cross_attention/latent_retrieval/EVAL_t2m_rag_latent_retr_addcfg.sh` |
| `precompute_latent_retr_lookup.py` | `python -m explorations.cross_attention.latent_retrieval.precompute_latent_retr_lookup` |
| `train_t2m_rag_latent_retr.py` | `python -m explorations.cross_attention.latent_retrieval.train_t2m_rag_latent_retr` |
| `Train_t2m_rag_latent_retr.sh` | `bash explorations/cross_attention/latent_retrieval/Train_t2m_rag_latent_retr.sh` |

## Rectified Flow

状态：`NEGATIVE RESULT`。用 Rectified Flow 替换 DDPM head 的结果不佳，
正式方法保留 DDPM。需要正式 Global-RAG 数据资产和 RF checkpoint；
训练 Shell 仍复用根目录 `train_t2m_rag.py` 的可配置生成 head。

| 原根路径 | 新命令 |
|---|---|
| `eval_msa_t2m_rag_t5_rf.py` | `python -m explorations.rectified_flow.eval_msa_t2m_rag_t5_rf` |
| `EVAL_t2m_rag_t5_rf.sh` | `bash explorations/rectified_flow/EVAL_t2m_rag_t5_rf.sh` |
| `Train_t2m_rag_rf.sh` | `bash explorations/rectified_flow/Train_t2m_rag_rf.sh` |

## Q-Former retrieval

状态：`EXPERIMENTAL`。独立的 Q-Former 运动—文本对齐尝试，未进入正式
方法。需要 Causal TAE checkpoint、本地 Sentence-T5 模型和文本特征。

`PREPARE_text_embeddings.sh` 在移动前已经不可运行：它调用仓库中不存在的
`prepare_text_embeddings.py`。请勿将其当作正式预处理入口；正式 T5
特征应由根目录 `get_text_latent_t5.py` 生成。

| 原根路径 | 新命令 |
|---|---|
| `build_rag_db.py` | `python -m explorations.qformer.build_rag_db` |
| `PREPARE_text_embeddings.sh` | `bash explorations/qformer/PREPARE_text_embeddings.sh`（历史失效） |
| `train_qformer_rag.py` | `python -m explorations.qformer.train_qformer_rag` |
| `TRAIN_qformer_rag.sh` | `bash explorations/qformer/TRAIN_qformer_rag.sh` |

## MotionStreamer baselines

状态：`BASELINE`。保留上游 MotionStreamer 及离线文本缓存版本，用于
对照和训练流水线诊断。需要原始 Causal TAE latent、CLIP/T5 资产和对应
checkpoint；`TRAIN_evaluator_272.sh` 还依赖 `Evaluator_272` 的环境。

| 原根路径 | 新命令 |
|---|---|
| `demo_t2m.py` | `python -m explorations.motionstreamer_baselines.demo_t2m` |
| `eval_t2m.py` | `python -m explorations.motionstreamer_baselines.eval_t2m` |
| `EVAL_t2m.sh` | `bash explorations/motionstreamer_baselines/EVAL_t2m.sh` |
| `get_latent.py` | `python -m explorations.motionstreamer_baselines.get_latent` |
| `motionstreamer_gen_motion.py` | `python -m explorations.motionstreamer_baselines.motionstreamer_gen_motion` |
| `train_motionstreamer.py` | `python -m explorations.motionstreamer_baselines.train_motionstreamer` |
| `TRAIN_motionstreamer.sh` | `bash explorations/motionstreamer_baselines/TRAIN_motionstreamer.sh` |
| `train_t2m.py` | `python -m explorations.motionstreamer_baselines.train_t2m` |
| `TRAIN_t2m.sh` | `bash explorations/motionstreamer_baselines/TRAIN_t2m.sh` |
| `Train_t2m_multi.sh` | `bash explorations/motionstreamer_baselines/Train_t2m_multi.sh` |
| `train_t2m_cached.py` | `python -m explorations.motionstreamer_baselines.train_t2m_cached` |
| `TRAIN_t2m_cached.sh` | `bash explorations/motionstreamer_baselines/TRAIN_t2m_cached.sh` |
| `TRAIN_evaluator_272.sh` | `bash explorations/motionstreamer_baselines/TRAIN_evaluator_272.sh` |

## Representation-learning experiments

状态：`EXPERIMENTAL` / `NEGATIVE RESULT`。包含一次性 MSA-VAE、
SAE-v1 和 TAE-GAN-v1；正式 MSA-VAE 应使用根目录的两阶段 Shell
配合预训练 Causal TAE。需要各路线自己的 checkpoint 和 HumanML3D。

| 原根路径 | 新命令 |
|---|---|
| `demo_msa_vae_sample.py` | `python -m explorations.representation_experiments.demo_msa_vae_sample` |
| `eval_sae_v1.py` | `python -m explorations.representation_experiments.eval_sae_v1` |
| `EVAL_sae_v1.sh` | `bash explorations/representation_experiments/EVAL_sae_v1.sh` |
| `train_sae_v1.py` | `python -m explorations.representation_experiments.train_sae_v1` |
| `TRAIN_sae_v1.sh` | `bash explorations/representation_experiments/TRAIN_sae_v1.sh` |
| `train_tae_gan_v1.py` | `python -m explorations.representation_experiments.train_tae_gan_v1` |
| `TRAIN_tae_gan_v1.sh` | `bash explorations/representation_experiments/TRAIN_tae_gan_v1.sh` |
| `EVAL_tae_gan_v1.sh` | `bash explorations/representation_experiments/EVAL_tae_gan_v1.sh` |
| `TRAIN_msa_vae.sh` | `bash explorations/representation_experiments/TRAIN_msa_vae.sh` |
| `TRAIN_msa_vae_multi.sh` | `bash explorations/representation_experiments/TRAIN_msa_vae_multi.sh` |

## Retrieval baselines

状态：`BASELINE`。包含直接检索、RAG2Motion 和 ReMoDiffuse 推理对照。
这些入口依赖各自的索引、checkpoint，以及未随本仓库发布的外部工程资产。

| 原根路径 | 新命令 |
|---|---|
| `demo_retrieval.py` | `python -m explorations.retrieval_baselines.demo_retrieval` |
| `RAG2Motion.py` | `python -m explorations.retrieval_baselines.RAG2Motion` |
| `remodiffuse_gen_motion.py` | `python -m explorations.retrieval_baselines.remodiffuse_gen_motion` |

## Demos and diagnostics

状态：`HISTORY`。这些脚本用于阶段性可视化、格式转换和检查，不是正式
训练或评估入口。按脚本需要准备 checkpoint、SMPL/body model、BVH
工具或 GUI 渲染依赖。

`smoke_test.py` 是早期 CLIP/MSA 开发检查，其中提到的
`train_t2m_msa.py` 和 `TRAIN_t2m_msa.sh` 在归档前已经不存在；当前会将
缺失项报告为历史跳过，不代表正式主线 smoke suite。

| 原根路径 | 新命令 |
|---|---|
| `demo_msa_t2m_t5.py` | `python -m explorations.demos_and_diagnostics.demo_msa_t2m_t5` |
| `demo_msa_t2m_t5_02.py` | `python -m explorations.demos_and_diagnostics.demo_msa_t2m_t5_02` |
| `demo_verify_dataset.py` | `python -m explorations.demos_and_diagnostics.demo_verify_dataset` |
| `demo_verify_t5_conversion.py` | `python -m explorations.demos_and_diagnostics.demo_verify_t5_conversion` |
| `generate_motion.py` | `python -m explorations.demos_and_diagnostics.generate_motion` |
| `inspect_latent_shapes.py` | `python -m explorations.demos_and_diagnostics.inspect_latent_shapes` |
| `msa_gen_motion_batch.py` | `python -m explorations.demos_and_diagnostics.msa_gen_motion_batch` |
| `render_smpl_aitviewer_pos.py` | `python -m explorations.demos_and_diagnostics.render_smpl_aitviewer_pos` |
| `render_smpl_aitviewer_rot.py` | `python -m explorations.demos_and_diagnostics.render_smpl_aitviewer_rot` |
| `representation_272_to_bvh.py` | `python -m explorations.demos_and_diagnostics.representation_272_to_bvh` |
| `smoke_test.py` | `python -m explorations.demos_and_diagnostics.smoke_test` |
| `verify_setup.py` | `python -m explorations.demos_and_diagnostics.verify_setup` |
| `visualize_t2m_generation.py` | `python -m explorations.demos_and_diagnostics.visualize_t2m_generation` |

## Project history

状态：`HISTORY`。这是历史实现摘要、工作流草稿和备份脚本，不应作为
当前实验入口。`run.sh` 与 `run_training.sh` 在移动前已经调用不存在的
`train_t2m_msa.py`；`sedbash` 为空文件，均只作记录保留。

| 原根路径 | 新位置或命令 |
|---|---|
| `IMPLEMENTATION_SUMMARY.py` | `python -m explorations.project_history.IMPLEMENTATION_SUMMARY` |
| `WORKFLOW_GUIDE.py` | `python -m explorations.project_history.WORKFLOW_GUIDE` |
| `run.sh` | `explorations/project_history/run.sh`（历史失效） |
| `run_training.sh` | `explorations/project_history/run_training.sh`（历史失效） |
| `sedbash` | `explorations/project_history/sedbash`（空历史文件） |
| `TRAIN_msa_vae_phase1.sh.bak` | `explorations/project_history/TRAIN_msa_vae_phase1.sh.bak` |
| `TRAIN_msa_vae_phase2.sh.bak` | `explorations/project_history/TRAIN_msa_vae_phase2.sh.bak` |

## 维护规则

- 新的正式入口只有在属于论文主线时才放在根目录。
- 探索训练、评估、demo 和预处理入口必须归入相应路线。
- 移动归档入口时同步更新本索引、Shell 的 `REPO_ROOT` 解析及
  `tests/test_exploration_layout.py`。
- 不要为历史失效脚本伪造缺失实现；应明确记录其原有状态。
