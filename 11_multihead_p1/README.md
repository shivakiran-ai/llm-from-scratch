<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2011%20%E2%80%94%20Multi-Head%20Attention%20Part%201&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Wrapper%20Implementation%20%7C%20Stage%202%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-11%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-Wrapper%20Implementation-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-Parallel%20Specialized%20Heads-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Multi-head attention is the mechanism that transforms a single attention perspective into many simultaneous specialized ones. Instead of one set of W_Q, W_K, W_V matrices, the model runs `num_heads` independent attention operations in parallel — each developing its own specialization — and concatenates all outputs. This topic implements the first and most intuitive approach: a wrapper that stacks multiple `CausalAttention` instances from Topic 10 using `nn.ModuleList` and combines their outputs via `torch.cat`.

---

## The Core Extension

```
CAUSAL ATTENTION (Topic 10):
   One W_Q, W_K, W_V set — one attention matrix — one context vector per token

MULTI-HEAD ATTENTION (Topic 11):
   num_heads independent sets of W_Q, W_K, W_V
   num_heads causal attention matrices (each masked, each with dropout)
   num_heads context vectors per token — concatenated into one final output

   The causal mask from Topic 10 applies INSIDE each head independently.
   Every head still obeys the same rule: cannot look to the right.
```

---

## Why Multiple Heads? — The GPT-2 Anatomy

Using "The cat sat" as example — in GPT-2 small (768-dim, 12 heads), the token "sat" is processed through three phases:

```
THE SLICE (INPUT)
   "sat" → 768-dimensional vector
   Model cuts it into 12 segments of 64 dimensions each
   Head 1  gets dimensions   1 –  64
   Head 2  gets dimensions  65 – 128   ...   Head 12 gets 705 – 768

THE MULTI-TASKING (PROCESSING)
   Head 1 (Grammar):  Is "sat" the verb for the noun "cat"?
   Head 2 (Tense):    Is this past, present, or future?
   Head 3 (Logic):    Can a cat physically perform the action of sitting?
   ...all 12 run in parallel on their 64-dim slices...

THE RE-STITCH (OUTPUT)
   [Head1(64)] + [Head2(64)] + ... + [Head12(64)] = 768 total dimensions
   "sat" is now a 768-wide vector containing 12 expert opinions about its role.
```

> **Basically:** The 12 heads provide the **diversity** of what to look at. The causal mask provides the **direction** (past only).

---

## Numerical Trace (torch.manual_seed(123), d\_in=3, d\_out=2)

```python
# Weight matrices — actual values from the implementation
W_query:  [[0.2961, 0.5166],    W_key:  [[0.1366, 0.1025],
            [0.2517, 0.6886],             [0.1841, 0.7264],
            [0.0740, 0.8665]]             [0.3153, 0.6871]]

# query_2 = x_2 @ W_query  ("journey" token, x_2 = [0.55, 0.87, 0.66])
query_2  →  tensor([0.4306, 1.4551])

# Attention score between "journey" and itself:
attn_score_22 = query_2.dot(keys[1])  →  tensor(1.8524)

# All scores for query_2, scaled by 1/sqrt(d_k=2), softmax applied:
attn_weights_2 = torch.softmax(attn_scores_2 / 2**0.5, dim=-1)
              →  tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
#                        Your   journey  starts   with    one    step

# Context vector for "journey":
context_vec_2 = attn_weights_2 @ values  →  tensor([0.3061, 0.8210])
```

---

## The Complete MultiHeadAttentionWrapper

```python
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Each head has its own W_query, W_key, W_value and causal mask
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

# Test
batch = torch.stack((inputs, inputs), dim=0)   # [2, 6, 3]

torch.manual_seed(123)
mha = MultiHeadAttentionWrapper(d_in=3, d_out=2, context_length=6,
                                 dropout=0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs.shape)   # torch.Size([2, 6, 4])

print(context_vecs)
# tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],   # Your
#          [-0.5874,  0.0058,  0.5891,  0.3257],   # journey
#          [-0.6300, -0.0632,  0.6202,  0.3860],   # starts
#          [-0.5675, -0.0843,  0.5478,  0.3589],   # with
#          [-0.5526, -0.0981,  0.5321,  0.3428],   # one
#          [-0.5299, -0.1081,  0.5077,  0.3493]],  # step
#         [identical — same input stacked twice]])
# First 2 values per row = Head 1 output (Z1)
# Last  2 values per row = Head 2 output (Z2)
```

---

## Two Critical New Additions vs Topic 10

| Addition | What It Is | Why It Matters |
|----------|-----------|----------------|
| `nn.ModuleList` | Registers each head as a proper PyTorch submodule | Plain Python list makes parameters invisible to optimizer — training silently fails |
| `torch.cat(..., dim=-1)` | Concatenates head outputs along embedding dimension | `dim=-1` stacks along embedding dim — [2,6,2]+[2,6,2] → [2,6,4] not [2,12,2] |

---

## Dimension Mathematics

```
This implementation:   d_in=3, d_out=2, num_heads=2
   Per head:    [2, 6, 3] × W_Q/K/V [3,2]  →  [2, 6, 2]
   After cat:   [2, 6, 2] + [2, 6, 2]       →  [2, 6, 4]
   Final dim  = d_out × num_heads = 2 × 2 = 4

GPT-2 small:   d_model=768, num_heads=12, d_head=64
   Each head:   [batch, tokens, 768] → 64-dim slice → [batch, tokens, 64]
   After cat:   [batch, tokens, 64 × 12] = [batch, tokens, 768]   ✅
   Output dim = input dim — enables stacking 12 identical transformer layers
```

---

## Key Insight

> Multi-head attention is not just running attention multiple times. Each head operates in a **different learned subspace** — a 64-dimensional slice in GPT-2. The wrapper makes this explicit: `num_heads` independent `CausalAttention` instances each learn their own `W_Q`, `W_K`, `W_V`, and the final output is the horizontal concatenation of all context vectors. The sequential processing `[head(x) for head in self.heads]` is mathematically correct but not optimal — Topic 12 derives the parallel weight-splitting implementation that processes all heads in a single matrix multiplication.

---

## Research Connection

This topic implements **Vaswani et al. (2017) — Section 3.2.2: Multi-Head Attention**. The paper states *"Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions"* and defines h parallel heads with d_k = d_v = d_model / h. GPT-2 (Radford et al., 2019) uses exactly this mechanism across all 12 Transformer layers with 12 heads each at 64 dimensions.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — concept, GPT-2 anatomy, numerical trace, complete code, dimension mathematics, key insight |
| `Multi_Head_Attention.ipynb` | Full implementation — MultiHeadAttentionWrapper class, nn.ModuleList demonstration, batch testing, output concatenation verified end to end |
| `Topic11_MultiHeadAttention_P1.docx` | Complete deep dive — 11 sections, verified numerical trace with actual weight matrix values and full 6×6 attention score matrix, GPT-2 anatomy, causal constraint across heads, both architecture diagrams, shape tracking table, key observations, research paper connections |

---

## Next Topic

**[Topic 12 → Multi-Head Attention Part 2](../12_multihead_p2/README.md)**
*The wrapper runs heads sequentially. Part 2 derives the mathematically identical but efficient parallel implementation — all heads processed simultaneously through weight splitting and tensor reshaping. Also introduces the output projection matrix W^O from Vaswani et al.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
