<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=130&text=Topic%2012%20%E2%80%94%20Multi-Head%20Attention%20(Weight-Split)&fontSize=22&fontColor=ffffff&fontAlignY=52&desc=Stage%202%20Closing%20Topic%20%7C%20The%20Production%20Attention%20Module%20of%20GPT-2&descSize=12&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-12%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Closes%20Here-1A56A0?style=for-the-badge)](.)
[![Document](https://img.shields.io/badge/Document-Implementation%20%26%20Derivation-2E75B6?style=for-the-badge)](.)
[![Production Form](https://img.shields.io/badge/Production%20Form-GPT--2%20Tensor--Equal-22C55E?style=for-the-badge)](.)
[![Speed Up](https://img.shields.io/badge/Speed--up-12%C3%97%20fewer%20matmuls-F59E0B?style=for-the-badge)](.)

**[← Back to the Building LLMs from Scratch series](../README.md)**

</div>

---

## Abstract

This topic closes Stage 2 by replacing the wrapper formulation of multi-head attention developed in Topic 11 with the weight-split formulation used in production GPT-2. The wrapper instantiates `num_heads` independent `CausalAttention` modules and runs them sequentially in Python, performing 3 × `num_heads` matrix multiplications to produce queries, keys, and values. The weight-split formulation derived here uses a single W_Q, W_K, W_V matrix of shape `[d_in, d_out]` with `d_out = num_heads × head_dim`, performs one matrix multiplication per projection, and recovers the per-head structure through tensor reshape and transpose operations. Mathematically the two formulations are identical; computationally the weight-split form reduces a 12-layer GPT-2 forward pass from 432 to 36 attention-projection matmuls, an order-of-magnitude reduction in kernel launches without altering any output. The full pipeline is traced numerically on a three-token toy sentence with d_in = d_out = 6 and num_heads = 2 (head_dim = 3); every intermediate tensor is reported with verified PyTorch values. The complete `MultiHeadAttention` `nn.Module` is then assembled with the production output projection W_O. By the end of this topic the attention mechanism is feature-complete and tensor-equal to the GPT-2 production module.

---

## At a Glance

| | |
|---|---|
| **Topic** | 12 of 36 — **Stage 2 closing topic** |
| **Document type** | Implementation & Derivation — production multi-head attention |
| **Also known as** | Weight-Split Multi-Head Attention |
| **Core question** | How are `num_heads` independent attention computations performed in a single batched matrix multiplication, and why is this the production form? |
| **Builds on** | Topic 11 — Multi-Head Attention (Wrapper) |
| **Running example** | 3 tokens, batched ×2, d_in = d_out = 6, num_heads = 2, head_dim = 3 |
| **Used in** | GPT-2, GPT-3, GPT-4, LLaMA, Claude — every modern decoder-only LLM |
| **Efficiency gain** | 432 → 36 matmuls per GPT-2 Small forward pass (12× fewer kernel launches) |
| **Main artifact** | `Topic12_MultiHeadAttention_P2.docx` — 18-page technical report with embedded diagram, six-step numerical trace, and full production class |
| **Version** | 1.0 · May 2026 |

---

## The Core Shift

```
TOPIC 11 — MultiHeadAttentionWrapper:
   num_heads separate W_Q matrices, each shape [d_in, head_dim]
   num_heads matrix multiplications to compute queries
   heads processed sequentially: [head(x) for head in self.heads]

TOPIC 12 — MultiHeadAttention (this topic):
   ONE W_Q matrix, shape [d_in, d_out]   where d_out = num_heads × head_dim
   ONE matrix multiplication produces all queries at once
   per-head structure recovered via .view() and .transpose() — no Python loop

   Mathematical result:  identical (tensor-equal at floating-point precision)
   Computational cost:   dramatically lower in Topic 12
```

---

## The Key Idea — Weight Splitting

```python
# Topic 11 — num_heads independent multiplications:
Q1 = X @ W_q1   # W_q1 shape [d_in, head_dim]
Q2 = X @ W_q2   # W_q2 shape [d_in, head_dim]
# ... up to num_heads of these ...

# Topic 12 — one multiplication + reshape:
Q  = X @ W_q                                  # W_q shape [d_in, d_out]
Q  = Q.view(b, num_tokens, num_heads, head_dim)

# Constraint:   d_out = num_heads × head_dim
head_dim = d_out // num_heads                 # in this topic, 6 // 2 = 3
assert d_out % num_heads == 0                 # must be exactly divisible
```

`.view()` is a **zero-copy reshape** — no data is moved, only the interpretation of the tensor changes. The consolidated `W_q` has the same total parameter count as the `num_heads` separate matrices it replaces; what changes is that they are now stored as one contiguous block of memory.

---

## Numerical Trace — Six Steps

`torch.manual_seed(123)` · d_in = d_out = 6 · num_heads = 2 · head_dim = 3 · 3 tokens (`"the"`, `"cat"`, `"sleeps"`) · batched ×2.

### Step 1 — Input shape

```python
batch.shape  →  torch.Size([2, 3, 6])     # batch, num_tokens, d_in
```

### Step 2 — Project through W_Q, W_K, W_V (one matmul each)

```python
queries = self.W_query(x)   # [2, 3, 6]  — all heads still merged in d_out
keys    = self.W_key(x)     # [2, 3, 6]
values  = self.W_value(x)   # [2, 3, 6]
```

### Step 3 — Unroll: split d_out into (num_heads, head_dim) via `.view()`

```python
queries = queries.view(b, num_tokens, num_heads, head_dim)
# [2, 3, 6]  →  [2, 3, 2, 3]
#                     ↑  ↑
#               num_heads, head_dim

# After .view() — batch 0:
# 'the':    [[-0.2705, -0.3346, -0.4582],   ← Head 1
#            [ 0.6342, -0.0257,  0.6895]]   ← Head 2
# 'cat':    [[-0.2823, -0.6065, -0.6140],
#            [ 0.4528,  0.0178,  0.6942]]
# 'sleeps': [[-0.2511, -0.3628, -0.4498],
#            [ 0.2973, -0.0592,  0.5851]]
```

### Step 4 — Transpose: bring `num_heads` before `num_tokens`

```python
queries = queries.transpose(1, 2)
# [2, 3, 2, 3]  →  [2, 2, 3, 3]
#  b toks h hd     b   h toks hd
```

### Step 5 — Causal attention with per-head √head_dim scaling

```python
attn_scores = queries @ keys.transpose(2, 3)              # [2, 2, 3, 3]
attn_scores.masked_fill_(mask_bool, -torch.inf)
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

# attn_weights — batch 0:
# Head 1: [[1.0000, 0.0000, 0.0000],   'the'    sees only itself
#          [0.5315, 0.4685, 0.0000],   'cat'    sees the+cat
#          [0.3441, 0.3174, 0.3385]]   'sleeps' sees all three
# Head 2: [[1.0000, 0.0000, 0.0000],
#          [0.5328, 0.4672, 0.0000],
#          [0.3431, 0.3043, 0.3526]]
```

> **Why √head_dim, not √d_out:** after the `.view()` split, each head operates entirely within its own head_dim-dimensional subspace. The dot product `q · k` inside a single head sums over `head_dim` products, giving `Var(q · k) = head_dim`. Dividing by `√head_dim` restores unit variance per head.

### Step 6 — Recombine heads, then `out_proj`

```python
context_vec = (attn_weights @ values).transpose(1, 2)
# [2, 2, 3, 3]  →  [2, 3, 2, 3]

context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
# [2, 3, 2, 3]  →  [2, 3, 6]
# .contiguous() required: .transpose() breaks memory contiguity, .view() needs it back

# After merge (before out_proj) — batch 0:
# [[-0.1354,  0.2538,  0.1353,  0.0993,  0.5164,  0.1103],   'the'
#  [-0.2447,  0.3172, -0.0060,  0.1318,  0.6144,  0.0032],   'cat'
#  [-0.2898,  0.2501,  0.0070,  0.0092,  0.5663,  0.0665]]   'sleeps'
#   ←──── Head 1 (3 dims) ────→  ←──── Head 2 (3 dims) ────→

context_vec = self.out_proj(context_vec)   # learned mixing across heads
```

> **Two stages, not in conflict:** The pre-`out_proj` values (`[-0.1354, 0.2538, ...]`) are the raw concatenation of the two heads' outputs. The final values (`[0.1569, -0.0873, ...]`) are what comes out **after** `out_proj` multiplies by a learned 6×6 matrix and adds a bias. `out_proj` is precisely what mixes information across heads.

### Final output

```python
context_vecs.shape  →  torch.Size([2, 3, 6])

context_vecs = tensor([[[ 0.1569, -0.0873,  0.0210,  0.0215, -0.3243, -0.2518],   # the
                        [ 0.1117, -0.0547,  0.0406, -0.0213, -0.3251, -0.2993],   # cat
                        [ 0.1196, -0.0491,  0.0318, -0.0635, -0.2788, -0.2578]],  # sleeps
                       [identical — same input stacked twice]])
```

---

## The Complete `MultiHeadAttention` Class

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out     = d_out
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        # ONE matrix per Q/K/V — replaces num_heads separate matrices
        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)              # NEW: W_O

        self.dropout  = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Step 2: project through Q, K, V
        keys    = self.W_key(x)                                       # [b, n_t, d_out]
        queries = self.W_query(x)
        values  = self.W_value(x)

        # Step 3: split d_out into (num_heads, head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Step 4: bring heads before tokens
        keys    = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values  = values.transpose(1, 2)

        # Step 5: causal scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool   = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 6: recombine heads and project
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
```

---

## Three Production Patterns Introduced Here

| Pattern | What It Does | Why It Matters |
|---|---|---|
| `.view(b, num_tokens, num_heads, head_dim)` | Splits d_out into (num_heads, head_dim) — zero-copy reshape | Replaces num_heads separate weight matrices with one reshape; introducing the head dimension is free at runtime |
| Two distinct `.transpose()` calls | `.transpose(1, 2)` lifts heads in front of tokens; `.transpose(2, 3)` swaps last two dims of keys for Q · Kᵀ | Operate on different dimension pairs and serve different roles — getting either wrong silently produces wrong shapes |
| `out_proj = nn.Linear(d_out, d_out)` | Learned linear layer applied after head concatenation | The only operation in the layer that mixes information across heads; specified by Vaswani et al. as W_O |

> **And one further requirement:** `.contiguous()` before `.view()` after a `.transpose()`. The transpose breaks memory contiguity; `.view()` requires contiguity to compute valid strides. Skipping `.contiguous()` raises a runtime error rather than producing a silent bug, but it is still a recurring source of friction.

---

## Efficiency at GPT-2 Scale

| Quantity | Wrapper (Topic 11) | Weight-Split (this topic) |
|---|---:|---:|
| Q/K/V matmuls per attention block | 3 × num_heads = 36 | 3 |
| Layers in GPT-2 Small | 12 | 12 |
| Total Q/K/V matmuls per forward pass | **432** | **36** |
| GPU kernel launches | one per matmul + Python loop overhead | one per matmul, no Python loop |
| Output values | identical (given matched init) | identical (given matched init) |

The **mathematical content is identical**; only the way operations are batched differs. For GPT-2 Large with 25 heads × 36 layers, the gap widens further. This is why the weight-split form is the universal choice for any production attention implementation.

---

## Key Technical Observations

Seven takeaways. Full argument for each in Section 9 of the main document.

1. **Weight splitting consolidates `num_heads` matmuls into one.** The total parameter count is unchanged; only the way parameters are stored and accessed differs.
2. **`.view()` is metadata-only.** No data is moved when splitting d_out into (num_heads, head_dim) or merging them back. The same memory is reinterpreted under a new index pattern.
3. **Two distinct `.transpose()` calls.** `.transpose(1, 2)` lifts heads in front of tokens; `.transpose(2, 3)` swaps last two dims for Q · Kᵀ. Easy to confuse; getting either wrong silently produces incorrect tensors.
4. **The scaling factor is `√head_dim`, not `√d_out`.** Each head operates within its own `head_dim`-dim subspace. Variance algebra fixes the scaling factor as `√head_dim`.
5. **`.contiguous()` before `.view()` is a hard requirement after `.transpose()`.** Skipping it produces a runtime error rather than a silent bug, but is still a recurring source of friction.
6. **`out_proj` is the learned cross-head mixer.** Without it, no operation in the layer combines information across heads. Specified by Vaswani et al. as W_O.
7. **The wrapper and weight-split forms produce identical outputs.** With matched weight initialization the two are tensor-equal at floating-point precision — the wrapper of Topic 11 serves as a unit-test reference for the weight-split form's correctness.

---

## Experiments and Open Questions

- **Hand-verified equivalence** with the wrapper formulation — copy each per-head W_Q, W_K, W_V from the wrapper into the corresponding slice of the consolidated weight matrix; `torch.allclose(y_wrapper, y_weight_split, atol=1e-6)` returns `True`. This is the strongest correctness check available for the weight-split form (Section 10.1).
- **Open question** — why `head_dim ≈ 64` across the GPT family? GPT-2 Small (12 × 64), GPT-2 Large (25 × 64), GPT-3 (96 × 128). Two partial explanations: GPU efficiency at sizes ÷ 32, and variance-argument breakdown at very small `head_dim`. Whether this is fundamentally optimal or an unchallenged local minimum is open (Section 10.2).
- **Proposed future work** — head-dim ablation: hold d_model = 256 fixed, train identical decoder-only Transformers with `(num_heads, head_dim)` ∈ `{(4, 64), (8, 32), (16, 16), (32, 8)}` on the same corpus, measure final perplexity (Section 10.3).

---

## References

References are listed in IEEE numeric format. Each is cited inline in the main document and included in the References section at the end of the docx.

| # | Authors | Title | Venue | Year |
|:-:|---------|-------|-------|:-:|
| 1 | A. Vaswani et al. | Attention Is All You Need (Section 3.2.2 — Multi-Head Attention) | *NeurIPS* | 2017 |
| 2 | A. Radford et al. | Language Models are Unsupervised Multitask Learners (GPT-2) | *OpenAI Tech. Report* | 2019 |
| 3 | A. Paszke et al. | PyTorch: An Imperative Style, High-Performance Deep Learning Library | *NeurIPS* | 2019 |
| 4 | T. B. Brown et al. | Language Models are Few-Shot Learners (GPT-3) | *NeurIPS* | 2020 |

---

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This document — conceptual summary and reader entry point |
| `Multi_Head_Attention.ipynb` | Full PyTorch implementation — `MultiHeadAttention` class, all intermediate tensor shapes verified, `out_proj` included, batched test with actual numerical output |
| `Topic12_MultiHeadAttention_P2.docx` | Primary artifact (v1.0) — 18-page technical report comprising title page with abstract and keywords, table of contents with page numbers, 12 numbered sections, embedded wrapper-vs-weight-split diagram (Figure 1), six-step numerical trace from input through to final output, complete production class, three production patterns explained, `.contiguous()` requirement derivation, GPT-2 efficiency analysis, complete tensor inventory, hand-verified wrapper equivalence protocol, Experiments & Open Questions, Discussion on relationship to prior work, and IEEE-format references |

> **Note on the docx:** the Table of Contents is statically generated with real page numbers and dot leaders. No fields need to be updated on open; the document renders identically in Microsoft Word, Google Docs, LibreOffice, and PDF exports.

---

## Stage 2 Closes — What Comes Next

This topic closes Stage 2. The `MultiHeadAttention` class developed here is the production attention module used in GPT-2 and is feature-complete: causal masking, dropout, batched processing, weight-split consolidation, and the output projection are all in place.

**Previous:** &nbsp; [Topic 11 — Multi-Head Attention (Wrapper)](../11_multihead_p1/README.md) &nbsp;·&nbsp; *The pedagogical baseline against which the weight-split form is verified*

**Next:** &nbsp; [Topic 13 — Bird's Eye View of LLM Architecture](../13_architecture_overview/README.md) &nbsp;·&nbsp; *Stage 3 opens. Topic 13 surveys the full GPT-2 Transformer block: how `MultiHeadAttention` combines with Layer Normalization, GELU activation, the feed-forward network, and shortcut (residual) connections. Topics 14 through 18 then implement each of those components from scratch, culminating in the complete 124M-parameter GPT-2 model.*

---

<div align="center">

**Building LLMs from Scratch**  
A technical report series on the ground-up implementation of a GPT-2-class language model.

**Shiva Kiran Dadishetty** &nbsp;·&nbsp; Independent Research &nbsp;·&nbsp; Texas, USA  
[github.com/shivakiran-ai/llm-from-scratch](https://github.com/shivakiran-ai/llm-from-scratch)

*Document version 1.0 &nbsp;·&nbsp; Last updated May 2026*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
