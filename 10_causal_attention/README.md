<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2010%20%E2%80%94%20Causal%20Self-Attention&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Masked%20Attention%20%7C%20Stage%202%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-10%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-Masked%20Attention-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-Autoregressive%20Constraint-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Causal attention, also known as **masked attention**, is a special form of self-attention that restricts each token to only attend to previous and current tokens in the sequence. This is the autoregressive constraint that makes language generation possible. Without it, the model would see future tokens during training and learn to cheat rather than genuinely predict — causing complete failure at inference time.

---

## The Core Constraint

```
SELF-ATTENTION (Topic 9):
   Every token attends to ALL positions — past, present, and future
   'Your'    can see: Your, journey, starts, with, one, step
   'journey' can see: Your, journey, starts, with, one, step

CAUSAL ATTENTION (Topic 10):
   Every token attends ONLY to current and previous positions
   'Your'    can see: Your
   'journey' can see: Your, journey
   'starts'  can see: Your, journey, starts
   'step'    can see: Your, journey, starts, with, one, step
```

---

## Two Masking Approaches

### Approach 1 — Mask After Softmax (Naive)

```python
# Step 1: Compute attention weights normally
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)

# Step 2: Create lower triangular mask
mask_simple = torch.tril(torch.ones(context_length, context_length))

# Step 3: Zero out future positions
masked = attn_weights * mask_simple

# Step 4: Renormalize rows to sum to 1
masked_norm = masked / masked.sum(dim=1, keepdim=True)
```

### Approach 2 — Upper Triangular -∞ Mask (Efficient — used in practice)

```python
# Step 1: Create upper triangular mask (1s = future positions to mask)
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)

# Step 2: Fill future positions with -inf BEFORE softmax
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

# Step 3: Apply softmax — e^(-inf) = 0, no renormalization needed!
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
# Each row already sums to 1. ✅
```

> **Why -∞ is better:** `exp(-∞) = 0`, so softmax assigns exactly zero probability to masked positions and automatically renormalizes the rest — no separate step needed.

---

## The Causal Attention Weight Matrix

```
           Your   journey  starts   with    one    step
Your    [ 1.000   0.000   0.000   0.000   0.000   0.000 ]  ← only sees itself
journey [ 0.552   0.448   0.000   0.000   0.000   0.000 ]
starts  [ 0.380   0.310   0.310   0.000   0.000   0.000 ]
with    [ 0.276   0.246   0.246   0.232   0.000   0.000 ]
one     [ 0.218   0.198   0.198   0.189   0.197   0.000 ]
step    [ 0.194   0.166   0.167   0.154   0.167   0.153 ]  ← sees everything

Each row sums to 1.0. Weights are redistributed among VISIBLE tokens only.
```

---

## Dropout — Additional Regularization

```python
# Dropout randomly zeros out attention weights during training
dropout = nn.Dropout(0.5)   # 50% dropout rate

# Remaining values are SCALED UP by 1/0.5 = 2 to maintain balance
# dropout(attn_weights) → some positions zeroed, rest multiplied by 2

# Applied in TWO areas in Transformer architectures:
# 1. After calculating attention scores
# 2. After attention weights (most common — this is what we implement)
```

---

## The Complete CausalAttention Class

```python
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # register_buffer: not trainable, moves with model to GPU
        self.register_buffer('mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape           # batch dimension
        keys    = self.W_key(x)                  # [b, num_tokens, d_out]
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)   # transpose dims 1&2 only
        attn_scores.masked_fill_(                       # in-place: saves memory
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values   # [b, num_tokens, d_out]

# Test
batch = torch.stack((inputs, inputs), dim=0)   # [2, 6, 3]
ca = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
context_vecs = ca(batch)
print(context_vecs.shape)   # torch.Size([2, 6, 2])
```

---

## Three Critical New Additions vs Topic 9

| Addition | What It Is | Why It Matters |
|----------|-----------|----------------|
| `register_buffer` | Fixed mask, not trainable | Mask moves to GPU but optimizer never updates it |
| `keys.transpose(1, 2)` | Skip batch dimension in transpose | Handles 3D batch tensors correctly |
| `masked_fill_` (in-place) | Trailing underscore = no memory copy | Memory efficient for large models |

---

## Key Insight

> Causal masking is not an optional feature — it is the architectural requirement that makes autoregressive generation possible. The -∞ trick works because `exp(-∞) = 0`, so softmax assigns exactly zero probability to masked positions and automatically redistributes the weights among visible tokens. The causal mask must be registered as a buffer, not a parameter, because it is a fixed mathematical constraint — not something the model should learn.

---

## Research Connection

This topic implements **Vaswani et al. (2017) — Section 3.2.3: Masked Multi-Head Attention**. The paper describes preventing positions from attending to subsequent positions. The dropout follows **Srivastava et al. (2014) — Dropout: A Simple Way to Prevent Neural Networks from Overfitting**. GPT-2 uses this exact causal attention mechanism throughout all 12 of its Transformer layers.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — sharp summary with complete code |
| `Causal_Attention.ipynb` | Full implementation — two masking approaches, tril/triu exploration, dropout demonstration, complete CausalAttention class, batch testing |
| `Topic10_CausalAttention.docx` | Complete documentation — 11 sections, 2 architecture diagrams, both masking approaches compared, dropout explained, register_buffer vs nn.Parameter, in-place operations |

---

## Next Topic

**[Topic 11 → Multi-Head Attention Part 1](../11_multihead_p1/README.md)**
*The CausalAttention class built here becomes the foundation. Multiple instances run in parallel — each learning to attend to different aspects of the input.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
