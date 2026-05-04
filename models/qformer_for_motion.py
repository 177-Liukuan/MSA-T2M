"""
qformer_for_motion.py
=====================
BLIP-2 风格的 Q-Former，适配运动时序特征。

输入
----
x_motion : [B, T, 1024]   TAE 编码器最后一个 MLP 之前的 1024-dim 中间特征
captions  : List[str]      训练时传入的文本描述

输出
----
Z_out    : [B, num_queries, query_dim]   全局 RAG Token
text_logits : [B, seq_len, vocab_size]  训练时用于 MTG loss

三种注意力掩码
--------------
build_mtc_mask : 单模态（Q token 只 attend 自身）
build_mtm_mask : 双向（Q token + text token 互相 attend）
build_mtg_mask : 多模态因果（text token causal，但可 attend Q token）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Motion -> query 的交叉注意力投影层
# ---------------------------------------------------------------------------

class MotionCrossAttn(nn.Module):
    """单层 cross-attention: query token (Q) attend motion 特征 (KV)."""

    def __init__(self, query_dim: int, motion_dim: int = 1024,
                 num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert query_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(query_dim,  query_dim, bias=False)
        self.k_proj  = nn.Linear(motion_dim, query_dim, bias=False)
        self.v_proj  = nn.Linear(motion_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim,  query_dim)
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(query_dim)

    def forward(self, queries: torch.Tensor,
                motion: torch.Tensor,
                motion_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        queries : [B, Nq, Dq]
        motion  : [B, T,  Dm]
        motion_key_padding_mask : [B, T]  True = ignore
        """
        residual = queries
        B, Nq, Dq = queries.shape
        _, T,  _  = motion.shape

        Q = self.q_proj(queries).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(motion ).view(B, T,  self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(motion ).view(B, T,  self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, Nq, T]
        if motion_key_padding_mask is not None:
            # [B, 1, 1, T]
            mask = motion_key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                 # [B, H, Nq, Dh]
        out = out.transpose(1, 2).contiguous().view(B, Nq, Dq)
        out = self.out_proj(out)
        return self.norm(out + residual)


# ---------------------------------------------------------------------------
# Q-Former 层：self-attn (with optional text) -> cross-attn (motion) -> FFN
# ---------------------------------------------------------------------------

class QFormerLayer(nn.Module):
    def __init__(self, query_dim: int = 768, motion_dim: int = 1024,
                 num_heads: int = 8, ffn_ratio: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(query_dim, num_heads,
                                                 dropout=dropout,
                                                 batch_first=True)
        self.cross_attn = MotionCrossAttn(query_dim, motion_dim,
                                          num_heads, dropout)
        dim_ffn = query_dim * ffn_ratio
        # Q token 专属 FFN（BLIP-2 双 FFN 设计：Q/text 各独立参数）
        self.ffn_query = nn.Sequential(
            nn.Linear(query_dim, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, query_dim),
            nn.Dropout(dropout),
        )
        # text token 专属 FFN
        self.ffn_text = nn.Sequential(
            nn.Linear(query_dim, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, query_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2_query = nn.LayerNorm(query_dim)
        self.norm2_text  = nn.LayerNorm(query_dim)

    def forward(self, queries: torch.Tensor,
                motion: torch.Tensor,
                attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None,
                motion_key_padding_mask: torch.Tensor = None,
                num_queries: int = None) -> torch.Tensor:
        """
        queries            : [B, Nq (+Nt), Dq]   Nq=query token 数, Nt=text token 数(MTM/MTG)
        motion             : [B, T, Dm]
        attn_mask          : [Nq+Nt, Nq+Nt] 自注意力掩码
        key_padding_mask   : [B, Nq+Nt]     padding mask (True=ignore)
        motion_key_padding_mask : [B, T]    motion padding mask
        num_queries        : 实际 Q token 数量；MTM/MTG 时 queries 额外拼接了 text token
        """
        # 1. self-attention（全部 token 参与，Q 与 text 互相 attend）
        sa_out, _ = self.self_attn(queries, queries, queries,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask)
        queries = self.norm1(queries + sa_out)

        # 2. cross-attention to motion：严格只让 Q token 部分 attend motion
        #    text token 不应直接看到 motion，必须通过 Q token 瓶颈（BLIP-2 设计意图）
        Nq = num_queries if num_queries is not None else queries.shape[1]
        q_part = self.cross_attn(queries[:, :Nq, :], motion,
                                  motion_key_padding_mask)  # [B, Nq, Dq]
        if queries.shape[1] > Nq:
            # MTM/MTG 模式：将更新后的 Q token 与 text token 重新拼接
            queries = torch.cat([q_part, queries[:, Nq:, :]], dim=1)  # [B, Nq+Nt, Dq]
        else:
            queries = q_part  # MTC / features 模式

        # 3. FFN（Q token / text token 各用独立 FFN，对齐 BLIP-2 双 FFN 设计）
        q_only = queries[:, :Nq, :]
        q_out  = self.norm2_query(q_only + self.ffn_query(q_only))
        if queries.shape[1] > Nq:
            t_part = queries[:, Nq:, :]
            t_out  = self.norm2_text(t_part + self.ffn_text(t_part))
            return torch.cat([q_out, t_out], dim=1)
        return q_out


# ---------------------------------------------------------------------------
# 文本 Tokenizer 简单封装（推断时不需要）
# ---------------------------------------------------------------------------

class T5XXLFrozenTextEncoder(nn.Module):
    """
    冻结的 sentence-T5-XXL 编码器 + 可学习投影层。

    - T5 encoder（FP16，全部冻结）：输出 [B, L, d_model] 上下文表示
    - 可学习 proj：d_model(1024) → query_dim(768)，与 Q-Former 联合训练
    - 可学习 lm_head：query_dim → vocab_size，用于 MTG 损失
    - MTG token 空间与 T5 tokenizer 一致
    """

    def __init__(self,
                 t5_model_path: str = 'sentencet5-xxl/',
                 query_dim: int = 768,
                 dropout: float = 0.1,
                 max_len: int = 64):
        super().__init__()
        try:
            from transformers import T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
        except Exception:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(t5_model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 冻结 T5 编码器（FP16，不计入可训练参数）────────────────────────
        from transformers import T5EncoderModel
        self.t5 = T5EncoderModel.from_pretrained(
            t5_model_path, torch_dtype=torch.float16)
        for p in self.t5.parameters():
            p.requires_grad = False

        t5_hidden   = self.t5.config.d_model    # 1024
        vocab_size  = self.t5.config.vocab_size  # 32128
        self.vocab_size  = vocab_size
        self.hidden_size = query_dim

        # ── 可学习投影：t5_hidden → query_dim ───────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(t5_hidden, query_dim, bias=False),
            nn.LayerNorm(query_dim),
            nn.Dropout(dropout),
        )

        # ── LM head（MTG 用，可训练）────────────────────────────────────────
        self.lm_head = nn.Linear(query_dim, vocab_size, bias=False)

    def tokenize(self, captions, max_length: int = 64, device='cpu'):
        enc = self.tokenizer(
            captions,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        return {k: v.to(device) for k, v in enc.items()}

    def get_token_embeddings(self, input_ids: torch.Tensor,
                              attention_mask: torch.Tensor = None) -> torch.Tensor:
        """冻结 T5 编码器前向 + 可学习投影。[B,L] → [B,L,query_dim]"""
        with torch.no_grad():
            out = self.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        # FP16 → float32 再投影（proj 是 float32）
        return self.proj(out.last_hidden_state.float())

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self.get_token_embeddings(input_ids, attention_mask)

    def encode_to_hidden(self, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        只运行冻结 T5 encoder，返回 last_hidden_state（FP16）。
        不经过 proj，用于预计算缓存。
        [B, L] → [B, L, t5_hidden=1024] FP16
        """
        with torch.no_grad():
            out = self.t5(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state   # FP16，保留原始 dtype

    def proj_hidden(self, t5_hidden: torch.Tensor) -> torch.Tensor:
        """
        对预计算的 T5 hidden state 应用可学习 proj。
        [B, L, 1024] FP16 → [B, L, query_dim] FP32
        proj 是可训练模块，每步训练都需要执行。
        """
        return self.proj(t5_hidden.float())



# ---------------------------------------------------------------------------
# 主模型：MotionQFormer
# ---------------------------------------------------------------------------

class MotionQFormer(nn.Module):
    """
    BLIP-2 风格 Q-Former，适配运动潜变量序列。

    Args
    ----
    num_queries   : 可学习查询 token 数量
    query_dim     : 查询 token 维度（同时作为 Q-Former hidden size）
    motion_dim    : 运动特征维度（TAE 中间特征，1024）
    num_layers    : Q-Former 层数
    num_heads     : 多头注意力头数
    t5_model_path : 本地 sentence-T5 模型目录（用于 MTM/MTG tokenizer）
    text_emb_dim  : 文本对比嵌入维度（sentence-T5 输出维度，默认 768）
    max_text_len  : T5 tokenizer 最大长度
    temp          : MTC softmax 温度参数（可学习）
    """

    def __init__(
        self,
        num_queries   : int   = 4,
        query_dim     : int   = 768,
        motion_dim    : int   = 1024,
        num_layers    : int   = 6,
        num_heads     : int   = 8,
        ffn_ratio     : int   = 4,
        dropout       : float = 0.1,
        t5_model_path : str   = 'sentencet5-xxl/',
        text_emb_dim  : int   = 768,   # sentence-T5 = 768
        max_text_len  : int   = 64,
        temp_init     : float = 0.07,
        queue_size    : int   = 0,     # 0 = 不使用 queue；推荐 4096
    ):
        super().__init__()
        self.num_queries  = num_queries
        self.query_dim    = query_dim
        self.motion_dim   = motion_dim
        self.max_text_len = max_text_len

        # ── 可学习查询 token ───────────────────────────────────────────────
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_queries, query_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # ── Q-Former 层堆叠 ───────────────────────────────────────────────
        self.layers = nn.ModuleList([
            QFormerLayer(query_dim, motion_dim, num_heads, ffn_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(query_dim)

        # ── MTC 投影头（对齐 Q-Former 输出与 T5/文本 embedding）────────────
        self.motion_proj = nn.Linear(query_dim, text_emb_dim)
        self.temp = nn.Parameter(torch.tensor(temp_init))

        # ── MTM 二元分类头 ─────────────────────────────────────────────────
        self.itm_head = nn.Linear(query_dim, 2)

        # ── 文本编码器（冻结 T5-XXL + 可学习投影，MTM/MTG）────────────────
        self.text_encoder = T5XXLFrozenTextEncoder(
            t5_model_path=t5_model_path,
            query_dim=query_dim,
            dropout=dropout,
            max_len=max_text_len,
        )

        # ── MTG 语言模型头（可学习，绑定在 T5XXLFrozenTextEncoder.lm_head）──
        self.lm_head = self.text_encoder.lm_head

        # ── Negative Queue（FIFO，用于 MTC 更多负样本）────────────────────
        self.queue_size = queue_size
        if queue_size > 0:
            # motion: [Q, Nq, text_emb_dim]，text: [Q, text_emb_dim]
            mq = F.normalize(torch.randn(queue_size, num_queries, text_emb_dim), dim=-1)
            tq = F.normalize(torch.randn(queue_size, text_emb_dim), dim=-1)
            self.register_buffer('motion_queue', mq)
            self.register_buffer('text_queue',   tq)
            self.register_buffer('queue_ptr',    torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------
    # Negative Queue 操作
    # ------------------------------------------------------------------

    @torch.no_grad()
    def dequeue_and_enqueue(self, motion_feat: torch.Tensor, text_feat: torch.Tensor):
        """FIFO queue 更新。motion_feat: [B, Nq, D], text_feat: [B, D]（已 L2-norm）"""
        if self.queue_size == 0:
            return
        B = motion_feat.size(0)
        ptr = int(self.queue_ptr)
        # FIFO 循环写入（支持 B > 剩余空间时绕回）
        end = ptr + B
        if end <= self.queue_size:
            self.motion_queue[ptr:end] = motion_feat.detach()
            self.text_queue[ptr:end]   = text_feat.detach()
        else:
            first = self.queue_size - ptr
            self.motion_queue[ptr:]    = motion_feat[:first].detach()
            self.text_queue[ptr:]      = text_feat[:first].detach()
            rem = B - first
            self.motion_queue[:rem]    = motion_feat[first:].detach()
            self.text_queue[:rem]      = text_feat[first:].detach()
        self.queue_ptr[0] = end % self.queue_size

    # ------------------------------------------------------------------
    # 注意力掩码工厂函数
    # ------------------------------------------------------------------

    @staticmethod
    def build_mtc_mask(num_queries: int) -> None:
        """MTC: 纯查询 token，无额外掩码（全 attend）。返回 None 即可。"""
        return None

    @staticmethod
    def build_mtm_mask(num_queries: int, text_len: int) -> torch.Tensor:
        """
        MTM: query token + text token 双向全 attend。
        返回 [Nq+Nt, Nq+Nt] 全 False 掩码。
        """
        N = num_queries + text_len
        return torch.zeros(N, N, dtype=torch.bool)

    @staticmethod
    def build_mtg_mask(num_queries: int, text_len: int) -> torch.Tensor:
        """
        MTG: text token 因果掩码，但 text token 可 attend query token。
        [Nq+Nt, Nq+Nt] 矩阵，True = 不可见。

        格式：
          rows = query token 部分: 全 attend（全 False）
          rows = text  token 部分:
            - 前 num_queries 列（Q token）: False（可见）
            - 后 Nt 列: 下三角可见（因果）
        """
        N = num_queries + text_len
        mask = torch.ones(N, N, dtype=torch.bool)
        # query token 行：全 False（query 可 attend 所有位置）
        mask[:num_queries, :] = False
        # text token 行：query 列 = False，text 列 = causal
        mask[num_queries:, :num_queries] = False
        causal = torch.triu(
            torch.ones(text_len, text_len, dtype=torch.bool), diagonal=1)
        mask[num_queries:, num_queries:] = causal
        return mask

    # ------------------------------------------------------------------
    # 前向传播工具
    # ------------------------------------------------------------------

    def _forward_query(self, x_motion: torch.Tensor,
                       motion_pad_mask: torch.Tensor = None,
                       extra_tokens: torch.Tensor = None,
                       attn_mask: torch.Tensor = None,
                       key_padding_mask: torch.Tensor = None
                       ) -> torch.Tensor:
        """
        x_motion : [B, T, 1024]
        extra_tokens : [B, Nt, Dq]  文本 token embedding（MTM/MTG 时使用）
        返回 [B, Nq (+Nt), Dq]
        """
        B = x_motion.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # [B, Nq, Dq]

        if extra_tokens is not None:
            queries = torch.cat([queries, extra_tokens], dim=1)  # [B, Nq+Nt, Dq]

        if attn_mask is not None:
            attn_mask = attn_mask.to(x_motion.device)

        for layer in self.layers:
            queries = layer(queries, x_motion,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            motion_key_padding_mask=motion_pad_mask,
                            num_queries=self.num_queries)  # 告知每层实际 Q token 数

        return self.norm(queries)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def forward_features(self, x_motion: torch.Tensor,
                         motion_pad_mask: torch.Tensor = None
                         ) -> torch.Tensor:
        """
        仅提取 query 特征（推理 / RAG 构建时使用）。
        返回 Z_out : [B, num_queries, query_dim]
        """
        return self._forward_query(x_motion, motion_pad_mask)

    def forward_mtc(self, x_motion: torch.Tensor,
                    text_embeds: torch.Tensor,
                    motion_pad_mask: torch.Tensor = None):
        """
        MTC 前向：返回 motion embedding 和 text embedding 用于对比损失。

        text_embeds : [B, text_emb_dim]  预计算的句子级嵌入（已 L2-norm 可选）
        返回
        -----
        motion_feat : [B, text_emb_dim]  Q-Former 输出后投影的平均 embedding
        text_feat   : [B, text_emb_dim]  原始输入 text_embeds
        temp        : 标量温度
        """
        Z_out = self._forward_query(x_motion, motion_pad_mask,
                                    attn_mask=self.build_mtc_mask(self.num_queries))
        # Max Pool MTC：投影所有 Nq token，保留 per-token 特征 [B, Nq, text_emb_dim]
        # 对比损失中用 einsum + max 取最大相似度（BLIP-2 设计）
        motion_feat = self.motion_proj(Z_out)           # [B, Nq, text_emb_dim]
        motion_feat = F.normalize(motion_feat, dim=-1)  # per-token L2 归一化
        text_feat   = F.normalize(text_embeds, dim=-1)
        # 温度钳位 [0.01, 0.5]：防止坍缩（过小→loss 饱和→梯度消失）
        return motion_feat, text_feat, self.temp.clamp(0.01, 0.5)

    def forward_mtm(self, x_motion: torch.Tensor,
                    captions, device,
                    motion_pad_mask: torch.Tensor = None):
        """
        MTM 前向：返回匹配概率 logits。
        B 中前半部分为正样本，后半部分为负样本（外部负采样后拼接）。
        返回 [B, 2]
        """
        enc = self.text_encoder.tokenize(captions, self.max_text_len, device)
        txt_emb = self.text_encoder.get_token_embeddings(
            enc['input_ids'], enc.get('attention_mask'))  # [B, L, 768]
        Nt = txt_emb.size(1)
        Nq = self.num_queries

        mask = self.build_mtm_mask(Nq, Nt).to(device)
        # key_padding_mask: True = 忽略位置
        # query token 全不 padding，text token padding 来自 attention_mask
        q_pad    = torch.zeros(x_motion.size(0), Nq, dtype=torch.bool, device=device)
        txt_pad  = (enc['attention_mask'] == 0)
        kp_mask  = torch.cat([q_pad, txt_pad], dim=1)

        out = self._forward_query(x_motion, motion_pad_mask,
                                  extra_tokens=txt_emb,
                                  attn_mask=mask,
                                  key_padding_mask=kp_mask)
        # 取 [CLS] 等价位置 = 第一个 query token
        cls_feat = out[:, 0, :]          # [B, Dq]
        return self.itm_head(cls_feat)   # [B, 2]

    def forward_mtg(self, x_motion: torch.Tensor,
                    captions, device,
                    motion_pad_mask: torch.Tensor = None):
        """
        MTG 前向：运动引导的文本生成，返回 logits 用于 cross-entropy。
        返回 logits : [B, Nt, vocab_size]
        """
        enc = self.text_encoder.tokenize(captions, self.max_text_len, device)
        input_ids = enc['input_ids']
        txt_emb   = self.text_encoder.get_token_embeddings(
            input_ids, enc.get('attention_mask'))  # [B, L, 768]
        Nt = txt_emb.size(1)
        Nq = self.num_queries

        mask   = self.build_mtg_mask(Nq, Nt).to(device)
        q_pad  = torch.zeros(x_motion.size(0), Nq, dtype=torch.bool, device=device)
        txt_pad = (enc['attention_mask'] == 0)
        kp_mask = torch.cat([q_pad, txt_pad], dim=1)

        out = self._forward_query(x_motion, motion_pad_mask,
                                  extra_tokens=txt_emb,
                                  attn_mask=mask,
                                  key_padding_mask=kp_mask)
        text_out = out[:, Nq:, :]          # [B, Nt, Dq]
        logits   = self.lm_head(text_out)  # [B, Nt, vocab_size]
        return logits, input_ids


    def forward_mtm_from_emb(self, x_motion: torch.Tensor,
                              txt_emb: torch.Tensor,
                              attention_mask: torch.Tensor = None,
                              motion_pad_mask: torch.Tensor = None):
        """
        MTM forward（预计算嵌入版本）：跳过 T5 encoder 调用。
        txt_emb        : [B, L, query_dim]  已过 text_encoder.proj_hidden
        attention_mask : [B, L]
        """
        Nq = self.num_queries
        Nt = txt_emb.size(1)
        B  = x_motion.size(0)
        mask    = self.build_mtm_mask(Nq, Nt).to(x_motion.device)
        q_pad   = torch.zeros(B, Nq, dtype=torch.bool, device=x_motion.device)
        txt_pad = (attention_mask == 0) if attention_mask is not None else                   torch.zeros(B, Nt, dtype=torch.bool, device=x_motion.device)
        kp_mask = torch.cat([q_pad, txt_pad], dim=1)
        out     = self._forward_query(x_motion, motion_pad_mask,
                                      extra_tokens=txt_emb,
                                      attn_mask=mask,
                                      key_padding_mask=kp_mask)
        return self.itm_head(out[:, 0, :])   # [B, 2]

    def forward_mtg_from_emb(self, x_motion: torch.Tensor,
                              input_ids: torch.Tensor,
                              txt_emb: torch.Tensor,
                              attention_mask: torch.Tensor = None,
                              motion_pad_mask: torch.Tensor = None):
        """
        MTG forward（预计算嵌入版本）：跳过 T5 encoder 调用。
        input_ids : [B, L]  用于计算 MTG 损失的 target tokens
        txt_emb   : [B, L, query_dim]
        """
        Nq = self.num_queries
        Nt = txt_emb.size(1)
        B  = x_motion.size(0)
        mask    = self.build_mtg_mask(Nq, Nt).to(x_motion.device)
        q_pad   = torch.zeros(B, Nq, dtype=torch.bool, device=x_motion.device)
        txt_pad = (attention_mask == 0) if attention_mask is not None else                   torch.zeros(B, Nt, dtype=torch.bool, device=x_motion.device)
        kp_mask = torch.cat([q_pad, txt_pad], dim=1)
        out      = self._forward_query(x_motion, motion_pad_mask,
                                       extra_tokens=txt_emb,
                                       attn_mask=mask,
                                       key_padding_mask=kp_mask)
        return self.lm_head(out[:, Nq:, :]), input_ids   # [B, Nt, vocab], [B, L]

    def forward_mtm_from_hidden(self, x_motion: torch.Tensor,
                                 t5_hidden: torch.Tensor,
                                 attention_mask: torch.Tensor = None,
                                 motion_pad_mask: torch.Tensor = None):
        """
        MTM forward（T5 hidden 预计算版本）：proj 仍参与梯度。
        t5_hidden : [B, L, 1024] FP16（预计算缓存）
        """
        txt_emb = self.text_encoder.proj_hidden(t5_hidden)
        return self.forward_mtm_from_emb(x_motion, txt_emb, attention_mask, motion_pad_mask)

    def forward_mtg_from_hidden(self, x_motion: torch.Tensor,
                                 input_ids: torch.Tensor,
                                 t5_hidden: torch.Tensor,
                                 attention_mask: torch.Tensor = None,
                                 motion_pad_mask: torch.Tensor = None):
        """
        MTG forward（T5 hidden 预计算版本）：proj 仍参与梯度。
        """
        txt_emb = self.text_encoder.proj_hidden(t5_hidden)
        return self.forward_mtg_from_emb(x_motion, input_ids, txt_emb, attention_mask, motion_pad_mask)

    def forward(self, x_motion: torch.Tensor,
                captions=None, device=None,
                motion_pad_mask: torch.Tensor = None,
                mode: str = 'features'):
        """
        mode : 'features' | 'mtc' | 'mtm' | 'mtg'
        对于训练，请直接调用 forward_mtc / forward_mtm / forward_mtg。
        """
        if mode == 'features':
            return self.forward_features(x_motion, motion_pad_mask)
        elif mode == 'mtc':
            assert captions is not None
            return self.forward_mtc(x_motion, captions, motion_pad_mask)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use forward_mtc/mtm/mtg directly.")
