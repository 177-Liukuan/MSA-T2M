# MSA-T2M v2 Research Memory Document



我们在 MotionStreamer TAE 构建的连续因果潜空间中对运动表示进行双路语义建模：

- **全局语义建模**：提取用于检索增强生成的**全局 RAG Token**，表征运动序列的整体语义与风格。
- **局部语义建模**：提取一组**局部 RAG Token**，捕捉细粒度的运动动态与原子动作模式。

在生成阶段，我们采用基于检索增强的自回归扩散生成（RAG + DDPM Head）。具体地：

- 输入文本首先用于检索**全局 RAG Token**，该 Token 作为前缀拼接在输入序列的文本嵌入之后、运动潜变量 \(z\) 之前，为生成器提供稳定的全局语义引导（与v1保持一致）。
- 同时，用文本直接检索Motion Latent作为局部RAG Token（来自VAE构造的局部RAG Token检索库），局部RAG Token 通过交叉注意力注入到自回归生成模型种（与v1的不同，增加了局部RAG Token增强运动生成）

 