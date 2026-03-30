<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2009%20%E2%80%94%20Self-Attention%20with%20Trainable%20Weights&fontSize=24&fontColor=ffffff&fontAlignY=55&desc=Scaled%20Dot-Product%20Attention%20%7C%20Stage%202%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-09%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-Scaled%20Dot--Product%20Attention-2E75B6?style=for-the-badge)](.)
[![Used In](https://img.shields.io/badge/Used%20In-GPT--2%20%7C%20GPT--3%20%7C%20All%20Modern%20LLMs-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

This is the most important topic in Stage 2. Topic 8 implemented simplified self-attention — useful for understanding the mathematics, but limited because it could only attend to tokens with similar raw embeddings. Topic 9 implements the real mechanism: **Scaled Dot-Product Attention** with three trainable weight matrices W_q, W_k, and W_v. This is the exact self-attention used in the original Transformer (Vaswani et al. 2017), GPT-2, GPT-3, and every modern large language model.

---

## The Core Innovation — Three Trainable Matrices

```
Topic 8 (simplified):  score(i,j) = x(i) · x(j)          ← raw embeddings
Topic 9 (trainable):   score(i,j) = q(i) · k(j)           ← projected vectors
                       where q(i) = x(i) @ W_q
                             k(j) = x(j) @ W_k
                             v(j) = x(j) @ W_v

QUERY  (W_q): What am I looking for?         ← current token's perspective
KEY    (W_k): What do I contain?             ← every token's identity
VALUE  (W_v): What do I contribute?          ← every token's information content

These three matrices are LEARNED during training.
The model adjusts W_q, W_k, W_v via backpropagation to produce
context vectors that are genuinely useful for predicting the next token.
```

---

## The Four Steps

### Step 1 — Project inputs to Q, K, V

```python
x_2 = inputs[1]   # 'journey' — the query token
d_in, d_out = 3, 2

W_query = nn.Parameter(torch.rand(d_in, d_out))  # [3, 2]
W_key   = nn.Parameter(torch.rand(d_in, d_out))  # [3, 2]
W_value = nn.Parameter(torch.rand(d_in, d_out))  # [3, 2]

query_2 = x_2 @ W_query   # → [0.4306, 1.4551]
queries = inputs @ W_query  # [6,3] @ [3,2] = [6,2]
keys    = inputs @ W_key    # [6,3] @ [3,2] = [6,2]
values  = inputs @ W_value  # [6,3] @ [3,2] = [6,2]
```

### Step 2 — Attention Scores (Query · Key)

```python
attn_score_22 = query_2.dot(keys[1])   # 1.8524  ← 'journey' attends to itself
attn_scores_2 = query_2 @ keys.T       # [1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]

attn_scores = queries @ keys.T         # [6,2] @ [2,6] = [6,6]  ← full matrix
```

### Step 3 — Scale by √d_k then Softmax

```python
d_k = keys.shape[-1]   # d_k = 2
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

# WHY sqrt(d_k)?
# Reason 1: Without scaling, softmax becomes "peaky" → unstable gradients
#   softmax([0.1,-0.2,0.3,-0.2,0.5])     → [0.19, 0.14, 0.24, 0.14, 0.29]  balanced
#   softmax([0.1,-0.2,0.3,-0.2,0.5] * 8) → [0.03, 0.00, 0.16, 0.00, 0.80]  peaky ❌
#
# Reason 2: Dot product variance grows with dimension
#   dim=5:   variance before=4.63,  after scaling=0.93  ✅
#   dim=100: variance before=95.1,  after scaling=0.95  ✅
#   Dividing by sqrt(dim) keeps variance ≈ 1 → stable learning
```

### Step 4 — Context Vector (Weighted Sum of VALUES)

```python
context_vec_2 = attn_weights_2 @ values
# tensor([0.3061, 0.8210])

# KEY DISTINCTION: weighted sum of VALUE vectors, not raw input vectors
# values = inputs @ W_v  ← projected through a separate trainable matrix
```

---

## Complete Shapes — Every Tensor

| Tensor | Shape | Description |
|--------|-------|-------------|
| `inputs X` | `[6, 3]` | 6 tokens, d_in=3 |
| `W_q, W_k, W_v` | `[3, 2]` | Trainable weight matrices |
| `Q, K, V` | `[6, 2]` | Projected query/key/value matrices |
| `attn_scores` | `[6, 6]` | Q @ K.T — one score per pair |
| `attn_weights` | `[6, 6]` | Softmax normalized, rows sum to 1 |
| `context_vecs Z` | `[6, 2]` | Final enriched embeddings |

---

## The Compact Class — Two Versions

```python
# v1: nn.Parameter with random init (for understanding)
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    def forward(self, x):
        keys    = x @ self.W_key
        queries = x @ self.W_query
        values  = x @ self.W_value
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values

# v2: nn.Linear with Kaiming init (for training — preferred)
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values
```

---

## Key Insight

> The three weight matrices W_q, W_k, W_v are the **entire intelligence** of self-attention. Everything else — dot products, softmax, weighted sums — is fixed arithmetic. Only these three matrices are learned. When the model improves at language generation, it does so by adjusting these weights. None of the context vectors are optimized right now. When the LLM is trained, W_q, W_k, and W_v will be learned end-to-end via backpropagation, causing the context vectors to be perfectly optimized.

---

## Research Connection

This topic is a direct from-scratch implementation of **Vaswani et al. (2017) — Section 3.2.1: Scaled Dot-Product Attention**. The Q, K, V notation, the √d_k scaling, and the softmax normalization are all from that section. GPT-2, GPT-3, and every major LLM built since 2017 uses this exact mechanism at its core.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — sharp summary with complete code |
| `Trainable_Self_Attention.ipynb` | Full implementation — W_q/W_k/W_v initialization, single query computation, full matrix multiplication, scaling demonstration, variance demo, SelfAttention_v1, SelfAttention_v2 |
| `Topic9_SelfAttentionQKV.docx` | Complete documentation — 14 sections, 5 architecture diagrams embedded, full shape tracking, two variance/stability demonstrations, both class implementations |

---

## Next Topic

**[Topic 10 → Causal Self-Attention](../10_causal_attention/README.md)**
*Adds the causal mask — future tokens are set to −∞ before softmax, ensuring the model only attends to previous and current tokens. Required for autoregressive language generation.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
