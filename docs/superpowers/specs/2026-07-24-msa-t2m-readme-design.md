# MSA-T2M 中文 README 设计

日期：2026-07-24

## 目标

将仓库根目录仍沿用 MotionStreamer 的 README 改写为 MSA-T2M 项目说明，并解决当前工作区中正式方法、论文消融、后续探索和失败路线混杂的问题。README 应同时服务于：

1. 研究者快速理解 MSA-T2M 的论文贡献；
2. 作者在 AAAI 2027 补充实验期间定位正式训练与评估入口；
3. 后续公开仓库时避免误用 CLIP、Rectified Flow、交叉注意力等非正式分支。

README 使用中文，保留必要的英文方法名、脚本名和状态标签。

## 事实来源与判定原则

方法判定同时参考：

- 论文草稿 `paper writing/mypaper/2026_KuanLiu_Text2Motion_NeurIPS/neurips_2026_draft.tex`；
- Git 历史中 CLIP、T5、RAG、RF 和交叉注意力分支的演化顺序；
- 当前模型实现、数据集代码、训练脚本和评估脚本之间的实际调用关系；
- 作者确认的两条正式方法主线。

当论文文字、shell 默认值与代码实现不一致时：

1. 不伪造一致性；
2. README 的可执行入口以当前代码为准；
3. 单独提示最终论文超参数仍需复核；
4. 不把仅因兼容参数而存在的实现写成正式方法。

## 正式方法边界

### MSA-VAE

正式 MSA-VAE 采用三阶段渐进训练：

1. 通过 `TRAIN_causal_TAE.sh` 预训练 MotionStreamer 风格的 Causal CNN-VAE，建立物理运动潜空间；
2. 通过 `TRAIN_msa_vae_phase1.sh` 加载并冻结 CNN 编解码器，训练 Transformer Semantic AE、全局 `[CLS]` 表征和局部/全局语义对齐；
3. 通过 `TRAIN_msa_vae_phase2.sh` 解冻全部组件，以差分学习率联合微调。

由于 MSA-VAE 脚本自身将后两步命名为 Phase 1 和 Phase 2，README 必须同时给出“整个项目的 Stage 1–3”和“脚本内部 Phase 1–2”两套编号，避免读者误以为缺少第一阶段。

### RAG-Diffusion-AR

正式生成器：

- 使用 Sentence-T5-XXL 的 768 维文本嵌入；
- 使用冻结 MSA-VAE 预编码训练集运动潜变量和全局 `[CLS]`；
- 以文本嵌入为查询，在 `[CLS]` 库中做余弦 Top-K 检索；
- 用相似度 softmax 对检索结果加权，融合为单个 RAG token；
- 将 `[Text token, RAG token, generated motion latent tokens]` 输入 LLaMA 风格因果 Transformer；
- 保留 MotionStreamer 的 DDPM diffusion head 和自回归生成范式。

正式入口为：

- `get_text_latent_t5.py`
- `get_msa_latent.py`
- `train_t2m_rag.py` / `TRAIN_t2m_rag.sh`
- `eval_msa_t2m_rag_t5.py` / `EVAL_t2m_rag_t5.sh`
- `msa_gen_motion.py`
- `models/msa_vae.py`
- `models/llama_rag_model.py` 中的全局 RAG 路径
- `humanml3d_272/dataset_msa_vae.py`
- `humanml3d_272/dataset_msa_rag.py`

## README 信息架构

README 采用“论文展示 + 复现指南 + 代码谱系”的混合结构：

1. 项目标题、状态和上游关系；
2. 简短摘要与核心贡献；
3. 方法总览；
4. 正式代码导航；
5. 环境、数据和外部资源准备；
6. MSA-VAE 三阶段训练；
7. T5、运动潜变量及检索库预计算；
8. RAG-Diffusion-AR 训练；
9. 正式评估与推理；
10. 当前论文结果；
11. 正式、消融、实验和遗留代码状态表；
12. 已知问题与复现注意事项；
13. 致谢、引用和许可证。

## 代码状态标记

README 使用以下标签：

- `OFFICIAL`：当前论文正式方法及推荐复现入口；
- `ABLATION`：服务于论文论证的去组件或超参数实验；
- `EXPERIMENTAL`：论文之后仍可继续研究，但不属于当前论文主线；
- `LEGACY / NEGATIVE RESULT`：早期实现或已验证效果不佳的路线。

具体归类：

### OFFICIAL

- T5 版本的 MSA-VAE 三阶段训练；
- 全局 `[CLS]` 检索和单 RAG token 前缀注入；
- DDPM diffusion head；
- T5 正式评估与全局 RAG 推理。

### ABLATION

- 禁用局部或全局对齐损失；
- 禁用 RAG；
- 不同检索数量 K；
- 论文中明确报告的正式组件消融。

若现有 launcher 未正确传递消融参数，README 只说明其研究用途和风险，不提供未经验证的“一键复现”承诺。

### EXPERIMENTAL

- `train_t2m_rag_local.py` 等局部检索扩展；
- MCA / multi-text-token cross-attention；
- latent retrieval cross-attention；
- 论文提交后新增但未进入当前正式主线的推理和评估变体。

### LEGACY / NEGATIVE RESULT

- CLIP 文本编码器与 CLIP baseline；
- Rectified Flow generation head；
- 为生成器引入的各类交叉注意力路线；
- Q-Former RAG；
- 早期一次性 `TRAIN_msa_vae.sh`；
- 原始 MotionStreamer 训练与评估入口。

原始 MotionStreamer 代码仍需保留，并明确标为上游基线，而非删除。

## 复现安全与已知不一致

README 将显式记录：

- `PREPARE_text_embeddings.sh` 当前引用缺失的 `prepare_text_embeddings.py`，因此推荐直接使用仓库现有 T5 特征脚本，并要求执行前核对参数；
- `TRAIN_t2m_no_rag.sh` 仅设置 `DISABLE_RAG` 环境变量，但 `TRAIN_t2m_rag.sh` 当前未将其转换为 `--disable_rag`，不能当作已验证入口；
- MSA-VAE shell 默认迭代数与论文草稿中的 2000K / 25K / 5K 不完全一致；
- 若 checkpoint、数据目录或硬编码模型路径尚未公开，README 使用环境变量占位并明确要求用户替换；
- 不承诺当前 README 命令已在没有本地数据和 checkpoint 的全新环境完成端到端运行。

## 结果呈现

README 可引用当前论文草稿中已报告的 HumanML3D-272 主结果，并注明这些数值来自 NeurIPS 2026 草稿、仍将在 AAAI 2027 补充实验中复核。避免将未最终核验的结果描述为已经正式发表。

## 非目标

本次改写不包含：

- 删除或重构遗留代码；
- 修复所有历史 launcher；
- 发布 checkpoint 或数据；
- 修改论文方法或实验结论；
- 将后续局部 RAG 扩展并入正式方法；
- 改写上游 MotionStreamer 的许可证。

## 验收标准

1. README 首屏不再显示 MotionStreamer 的论文标题和作者；
2. MSA-VAE 的三阶段流程清楚且与代码冻结逻辑一致；
3. RAG-Diffusion-AR 被准确描述为全局检索单 token 前缀注入和 DDPM head；
4. 正式训练、评估和推理入口可在 30 秒内从 README 定位；
5. CLIP、RF、cross-attention、Q-Former 等路线均不会被误认为正式方法；
6. 所有本地 Markdown 链接和列出的文件路径均存在，或被明确标注为外部资源；
7. README 不覆盖或修改嵌套子模块中的用户改动。
