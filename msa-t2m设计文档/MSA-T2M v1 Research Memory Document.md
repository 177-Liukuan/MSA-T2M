# MSA-T2M v1 Research Memory Document

## 1. Research Overview

This project focuses on **text-driven human motion generation (Text-to-Motion, T2M)** in a **continuous latent space**, aiming to simultaneously improve:

- **Motion realism** (physical plausibility, smooth dynamics)
- **Text alignment** (fine-grained and compositional semantic fidelity)

The core hypothesis is:

> **Semantic alignment in latent space is not only beneficial for text-motion matching, but also enables a retrieval-friendly structure that can be exploited to improve motion generation quality.**

------

## 2. Core Problem

Existing latent-space T2M methods suffer from two fundamental limitations:

### (1) Lack of explicit semantic structure

- Latent spaces are optimized for **reconstruction / generation**, not semantics
- Semantic information is **entangled with motion dynamics**
- Leads to:
  - weak controllability
  - poor compositional generalization

### (2) Underutilization of motion priors

- Most methods rely **only on text conditioning**

- Ignore the fact that:

  > motion datasets already contain rich, reusable motion patterns

- Retrieval-based methods (e.g., ReMoDiffuse) operate in **raw motion space**, which is:

  - noisy
  - redundant
  - not semantically structured

------

## 3. Key Insight

### 💡 Insight 1: Text-motion alignment is inherently **multi-scale**

- **Local scale** → atomic actions (e.g., “raise arm”)
- **Global scale** → compositional intent (e.g., “walk then sit”)

Single-scale alignment is insufficient:

- Local-only → good detail, poor coherence
- Global-only → good intent, poor fine control

------

### 💡 Insight 2: Alignment reshapes latent space structure

A well-aligned latent space:

- clusters semantically similar motions
- reduces modality gap
- becomes naturally **retrieval-friendly**

------

### 💡 Insight 3: Retrieval can compensate alignment-induced degradation

Strong alignment may hurt realism (distort motion manifold), but:

> **Retrieval provides real motion priors that restore realism**

------

## 4. Overall Framework: MSA-T2M

A two-stage pipeline:

```
Stage 1: MSA-VAE        → build structured latent space
Stage 2: RAG-Diffusion  → generate motion with retrieval priors
```

------

## 5. Stage 1: MSA-VAE (Multi-Scale Semantic Alignment VAE)

### 5.1 Objective

Learn a **continuous causal latent space** that is:

- physically meaningful (for generation)
- semantically structured (for alignment & retrieval)

------

### 5.2 Architecture

#### (A) Physical Latent (Bottom layer)

- **Causal CNN VAE (TAE-style)**
- Input: motion (X \in \mathbb{R}^{T \times 272})
- Output:
  - (\mu, \log\sigma^2)
  - latent (z \in \mathbb{R}^{n \times d})

Key properties:

- temporal causality
- noise reduction
- compact representation

------

#### (B) Semantic Latent (Top layer)

- **Transformer Autoencoder**
- Input: (\mu) (deterministic latent)
- Add `[CLS]` token
- Output:
  - global feature: (h_{\text{cls}})

Constraint:

- decoder reconstructs (\mu) from (h_{\text{cls}})

👉 ensures:

> global token encodes both **semantics + motion structure**

------

### 5.3 Multi-Scale Alignment

#### (1) Local alignment

- supervision: **BABEL frame-level labels**
- each latent step aligned with local text

[
\mathcal{L}_{local} = 1 - \cos(\text{Proj}(\mu_i), l_i)
]

------

#### (2) Global alignment

- supervision: **HumanML3D captions**

Problem:

- training uses 64-frame window
- caption describes full sequence

Solution:
👉 **dynamic interpolation**

[
g_{target} = (1-\alpha) g_{global} + \alpha l_{pooled}, \quad \alpha = T/L
]

------

### 5.4 Training Strategy (3-stage)

1. Train CNN-VAE (physical space)
2. Freeze CNN, train Transformer + alignment
3. Joint fine-tuning

------

### 5.5 Output

MSA-VAE produces:

- local latent sequence: (z_{1:n})
- global semantic token: (h_{\text{cls}})

------

## 6. Stage 2: RAG-Diffusion-AR

### 6.1 Motivation

Use structured latent space to:

> retrieve relevant motion priors and guide generation

------

### 6.2 Retrieval Database

Offline construction:

- key: (h_{\text{cls}})
- value: motion latent sequence

------

### 6.3 Retrieval Process

Given text embedding (t):

1. compute similarity with all (h_{\text{cls}})
2. retrieve Top-K
3. soft fusion:

[
r = \sum_k \text{softmax}(s_k) \cdot \text{Proj}(h_k)
]

------

### 6.4 Generation Model

#### Autoregressive latent generation

[
p(z_{1:n} | t, r) = \prod_i p(z_i | z_{<i}, t, r)
]

------

#### Input structure (prefix)

```
[BOS, text tokens, retrieval token r, z_<i]
```

------

#### Backbone

- Causal Transformer

------

#### Diffusion Head

- MLP predicts noise

[
\mathcal{L}*{diff} = ||\epsilon - \epsilon*\theta(z_i^t | t, c_i)||^2
]

------

### 6.5 Synergistic CFG

Drop conditions during training:

- text
- retrieval

Inference:

[
\epsilon = \epsilon_{uncond} + s(\epsilon_{cond} - \epsilon_{uncond})
]

👉 jointly controls:

- semantic alignment
- motion realism

------

## 7. Key Contributions

### Contribution 1

First to identify:

> **text-motion alignment is inherently multi-scale in latent space**

------

### Contribution 2

MSA-VAE:

> joint local-global alignment in continuous causal latent space

------

### Contribution 3

Key paradigm shift:

> alignment → structured latent → retrieval → improved generation

------

### Contribution 4

RAG-Diffusion-AR:

> retrieval-augmented autoregressive diffusion in latent space

------

## 8. Empirical Findings

### Key observations:

1. **RAG significantly improves FID**
2. **Local alignment is fundamental**
3. **Global alignment mainly improves retrieval quality**
4. **Alignment + RAG has synergy (1+1 > 2)**

------

## 9. Limitations

- BABEL labels are coarse (no fine body-part control)
- only single-segment offline generation
- no streaming / long motion yet

------

## 10. Future Directions

- fine-grained semantic annotation (LLM-based)
- local RAG tokens (frame-level retrieval)
- streaming / infinite motion generation
- flow-based generation (Rectified Flow)

------

## 11. One-Sentence Summary

> MSA-T2M reshapes the motion latent space via multi-scale semantic alignment, making it retrieval-friendly, and leverages this structure through RAG to improve motion generation quality.

------

