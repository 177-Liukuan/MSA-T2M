

为了提升重建保真度，我们首先对 MotionStreamer 中的因果时序自编码器（TAE）进行改进，引入重复填充策略以优化重建 FID，得到增强版本 TAE‑α。在此基础上，我们在 TAE‑α 构建的连续因果潜空间中对运动表示进行双路语义建模：

- 全局语义建模：提取用于检索增强生成的全局 RAG Token，表征运动序列的整体语义与风格。
- 局部语义建模：提取一组局部 RAG Token，捕捉细粒度的运动动态与原子动作模式。

在生成阶段，我们采用基于检索增强的自回归扩散生成（RAG + DDPM Head）。具体地：

- 输入文本首先用于检索全局 RAG Token，该 Token 作为前缀拼接在输入序列的文本嵌入之后、运动潜变量 (z) 之前，为生成器提供稳定的全局语义引导。
- 同时，从文本中解析出若干动词（或动作短语），分别检索对应的局部 RAG Token，构成一个动态的临时检索库，这类似Dynamic Retrieval Attention所使用的memory，不同的是它的memory来自历史帧，我的“memory”来自VAE构造的局部RAG Token检索库。
- 自回归骨干从llama转变为Flamingo，以便为模型加入门控交叉注意力Block，融入RAG Tokens
- 在自回归生成过程中，需要类似设计一个 HyDRA 的Dynamic Retrieval Attention的检索模块，地位等同于Flamingo的Perceiver Resampler ，作用类似Dynamic Retrieval Attention，将每次生成所检索出的数量不等局部RAG Tokens转换成固定数量的RAG Tokens，作为外部记忆融入自回归模型骨干，即作为交叉注意力Block的KV
