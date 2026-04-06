<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2012%20%E2%80%94%20Multi-Head%20Attention%20Part%202&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Efficient%20Parallel%20Implementation%20%7C%20Stage%202%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-12%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-Weight%20Split%20Implementation-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-One%20Matmul%2C%20All%20Heads-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topic 11 built `MultiHeadAttentionWrapper` — multiple independent `CausalAttention` instances run sequentially via `[head(x) for head in self.heads]`. Mathematically correct, but computationally wasteful: `num_heads` separate matrix multiplications per Q/K/V. Topic 12 replaces all of that with a single `MultiHeadAttention` class that uses **one** large weight matrix per Q/K/V, splits the result via `.view()` and `.transpose()`, and processes all heads simultaneously. Same output. Dramatically fewer operations. This is the implementation that actually goes into GPT-2.

---

## The Core Shift

```
TOPIC 11 — MultiHeadAttentionWrapper:
   num_heads separate W_Q matrices, each shape [d_in, head_dim]
   num_heads matrix multiplications to compute queries
   heads processed sequentially: [head(x) for head in self.heads]

TOPIC 12 — MultiHeadAttention:
   ONE W_Q matrix, shape [d_in, d_out]   where d_out = num_heads × head_dim
   ONE matrix multiplication to compute all queries at once
   split into heads via .view() — no loop, no separate classes

   Mathematical result: identical.
   Computational cost: dramatically lower in Topic 12.
```

---

## The Key Idea — Weight Splitting

Instead of `Q1 = X @ W_q1` and `Q2 = X @ W_q2` (two multiplications), the model computes `Q = X @ W_q` once and reshapes:

```python
# Topic 11 — two separate multiplications:
Q1 = X @ W_q1   # W_q1 shape [d_in, head_dim]
Q2 = X @ W_q2   # W_q2 shape [d_in, head_dim]

# Topic 12 — one multiplication + reshape:
Q  = X @ W_q    # W_q  shape [d_in, d_out]   d_out = head_dim × num_heads
Q  = Q.view(b, num_tokens, num_heads, head_dim)
#  shape: [b, num_tokens, d_out] → [b, num_tokens, num_heads, head_dim]

head_dim = d_out // num_heads    # 6 // 2 = 3
assert d_out % num_heads == 0    # must be exactly divisible
```

`.view()` is a zero-copy reshape — no data is moved, only the interpretation of the tensor changes.

---

## Numerical Trace (torch.manual_seed(123), d\_in=6, d\_out=6, num\_heads=2, head\_dim=3)

Input: "the", "cat", "sleeps" — each a 6-dimensional embedding.

**Step 1 — Input shape:**
```python
batch = torch.stack((inputs, inputs), dim=0)
batch.shape  →  torch.Size([2, 3, 6])   # batch, num_tokens, d_in
```

**Step 2 — Project through W_Q, W_K, W_V (one multiplication each):**
```python
queries = self.W_query(x)   # [2, 3, 6]  — all heads still merged in d_out
keys    = self.W_key(x)     # [2, 3, 6]
values  = self.W_value(x)   # [2, 3, 6]
```

**Step 3 — Unroll: split d_out into (num_heads, head_dim) via .view():**
```python
queries = queries.view(b, num_tokens, num_heads, head_dim)
# [2, 3, 6]  →  [2, 3, 2, 3]
#                     ↑   ↑
#               num_heads  head_dim

# Queries after .view() — batch 0:
# token "the":    [[-0.2705, -0.3346, -0.4582],  ← Head 1
#                  [ 0.6342, -0.0257,  0.6895]]  ← Head 2
# token "cat":    [[-0.2823, -0.6065, -0.6140],
#                  [ 0.4528,  0.0178,  0.6942]]
# token "sleeps": [[-0.2511, -0.3628, -0.4498],
#                  [ 0.2973, -0.0592,  0.5851]]
```

**Step 4 — Transpose: bring num_heads before num_tokens:**
```python
queries = queries.transpose(1, 2)
# [2, 3, 2, 3]  →  [2, 2, 3, 3]
#  b toks h hd     b  h toks hd

# Queries after .transpose(1,2) — batch 0:
# Head 1: [[-0.2705, -0.3346, -0.4582],  ← "the"
#          [-0.2823, -0.6065, -0.6140],  ← "cat"
#          [-0.2511, -0.3628, -0.4498]]  ← "sleeps"
# Head 2: [[ 0.6342, -0.0257,  0.6895],
#          [ 0.4528,  0.0178,  0.6942],
#          [ 0.2973, -0.0592,  0.5851]]
```

**Step 5 — Attention scores, causal mask, scale by 1/√head\_dim, softmax:**
```python
attn_scores = queries @ keys.transpose(2, 3)
# shape: [2, 2, 3, 3] — batch, heads, tokens, tokens

# After causal mask (-inf future) and ÷ sqrt(3) — batch 0:
# Head 1: [[-0.0530,   -inf,   -inf],
#          [-0.1003, -0.2265,   -inf],
#          [-0.0573, -0.1379, -0.0738]]
# Head 2: [[-0.0337,   -inf,   -inf],
#          [-0.0145, -0.1459,   -inf],
#          [ 0.0262, -0.0939,  0.0533]]

attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
# Head 1: [[1.0000, 0.0000, 0.0000],  "the" sees only itself
#          [0.5315, 0.4685, 0.0000],  "cat" sees the + cat
#          [0.3441, 0.3174, 0.3385]]  "sleeps" sees all 3
# Head 2: [[1.0000, 0.0000, 0.0000],
#          [0.5328, 0.4672, 0.0000],
#          [0.3431, 0.3043, 0.3526]]
```

**Step 6 — Context vectors, transpose back, merge heads via .contiguous().view():**
```python
context_vec = (attn_weights @ values).transpose(1, 2)
# [2, 2, 3, 3] → [2, 3, 2, 3]

context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
# [2, 3, 2, 3] → [2, 3, 6]
# .contiguous() required: .transpose() breaks memory contiguity, .view() needs it back

# After merge — batch 0:
# [[-0.1354,  0.2538,  0.1353,  0.0993,  0.5164,  0.1103],  "the"
#  [-0.2447,  0.3172, -0.0060,  0.1318,  0.6144,  0.0032],  "cat"
#  [-0.2898,  0.2501,  0.0070,  0.0092,  0.5663,  0.0665]]  "sleeps"
#   ←──── Head 1 (3 dims) ────→  ←──── Head 2 (3 dims) ────→

context_vec = self.out_proj(context_vec)   # learned mixing across heads
```

> **Why the two tensors are different:** The "After merge" values above (`[-0.1354, 0.2538, ...]`) are the raw concatenated head outputs — Head 1's 3 values followed by Head 2's 3 values, directly from `attn_weights @ values`. The final `context_vecs` tensor (`[0.1569, -0.0873, ...]`) is what comes out **after** `out_proj` multiplies by a learned 6×6 weight matrix and adds a bias. They are not errors — they are two different stages of the same forward pass. `out_proj` is what changes the numbers.

**Final output:**
```python
context_vecs.shape  →  torch.Size([2, 3, 6])

context_vecs = tensor([[[ 0.1569, -0.0873,  0.0210,  0.0215, -0.3243, -0.2518],  # the
                         [ 0.1117, -0.0547,  0.0406, -0.0213, -0.3251, -0.2993],  # cat
                         [ 0.1196, -0.0491,  0.0318, -0.0635, -0.2788, -0.2578]], # sleeps
                        [identical — same input stacked twice]])
```

---

## The Complete MultiHeadAttention Class

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)   # ONE matrix per Q/K/V
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)                  # output projection
        self.dropout  = nn.Dropout(dropout)
        self.register_buffer('mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys    = self.W_key(x)      # [b, num_tokens, d_out]
        queries = self.W_query(x)
        values  = self.W_value(x)

        # Split d_out into (num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Bring num_heads before num_tokens
        keys    = keys.transpose(1, 2)      # [b, num_heads, num_tokens, head_dim]
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn_scores = queries @ keys.transpose(2, 3)    # [b, num_heads, num_tokens, num_tokens]
        mask_bool   = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Combine heads
        context_vec = (attn_weights @ values).transpose(1, 2)   # [b, num_tokens, num_heads, head_dim]
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec

# Test
torch.manual_seed(123)
mha = MultiHeadAttention(d_in=6, d_out=6, context_length=3, dropout=0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs.shape)   # torch.Size([2, 3, 6])
```

---

## Three Critical Additions vs Topic 11

| Addition | What It Does | Why It Matters |
|----------|-------------|----------------|
| `.view(b, num_tokens, num_heads, head_dim)` | Splits d_out into (num_heads, head_dim) — zero-copy reshape | Replaces num_heads separate weight matrices with one reshape |
| `.transpose(1,2)` and `.transpose(2,3)` | (1,2): brings heads before tokens. (2,3): transposes last two dims for Q×Kᵀ | Two distinct transpositions — getting either wrong silently produces wrong shapes |
| `out_proj = nn.Linear(d_out, d_out)` | Learned linear transformation after combining heads | Allows model to mix information across heads — used in all GPT-2 layers |

---

## Efficiency at GPT-2 Scale

```
GPT-2 small: 12 heads, d_model=768, head_dim=64
GPT-2 large: 25 heads, d_model=1600, head_dim=64   (d_in = d_out in all GPT models)

Topic 11 style (Wrapper):  12 × 3 = 36 matrix multiplications per attention block
Topic 12 style (this):          3 matrix multiplications per attention block

GPT-2 small has 12 attention blocks:
  Topic 11 style: 432 matrix multiplications
  Topic 12 style:  36 matrix multiplications   ← same output, 12× fewer operations
```

---

## Key Insight

> `.view()` and `.transpose()` together implement what Topic 11 achieved with `num_heads` separate classes — but without a loop, without separate weight matrices, and without repeated matrix multiplications. The `d_out` dimension in the single `W_Q` matrix implicitly contains all heads: the first `head_dim` values belong to Head 1, the next `head_dim` to Head 2, and so on. `.view()` makes this split explicit. `.contiguous().view()` at the end reverses it — merging all head outputs back into a single `d_out`-dimensional vector per token. The scaling factor is `√head_dim` (not `√d_out`) because after the split each head operates in a `head_dim`-dimensional subspace, and dot products scale with that dimension.

---

## Research Connection

This topic implements **Vaswani et al. (2017) — Section 3.2.2** in its complete form, including the output projection matrix W^O (`out_proj`). GPT-2 (Radford et al., 2019) uses `MultiHeadAttention` (Topic 12 style) throughout all 12 layers — not the wrapper from Topic 11. The `d_in = d_out` constraint and `head_dim=64` design for GPT-2 small (12 heads, 768-dim) and large (25 heads, 1600-dim) are direct applications of the mathematics in this topic.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — core shift, weight splitting concept, full 6-step numerical trace with actual values, complete class, efficiency comparison |
| `Multi_Head_Attention.ipynb` | Full implementation — MultiHeadAttention class, all intermediate tensor shapes verified, out_proj, batched test with actual output |
| `Topic12_MultiHeadAttention_P2.docx` | Complete deep dive — 10 sections, 10-step numerical walkthrough with every intermediate tensor value, .view()/.transpose()/.contiguous() mechanics, efficiency argument with GPT-2 scale numbers, shape tracking table, .contiguous() requirement explained |

---

## Next Topic

**[Topic 13 → Bird's Eye View of LLM Architecture](../13_architecture_overview/README.md)**
*Stage 2 is complete. Topic 13 opens Stage 3 — showing how MultiHeadAttention connects with Layer Normalization, GELU, Feed-Forward Network, and Shortcut connections to form one complete GPT-2 Transformer block.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
