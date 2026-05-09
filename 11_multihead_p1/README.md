<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=130&text=Topic%2011%20%E2%80%94%20Multi-Head%20Attention%20(Wrapper)&fontSize=24&fontColor=ffffff&fontAlignY=52&desc=Technical%20Report%20%7C%20Stage%202%3A%20The%20Attention%20Mechanism%20%7C%20Building%20LLMs%20from%20Scratch&descSize=12&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-11%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Document](https://img.shields.io/badge/Document-Implementation%20%26%20Derivation-2E75B6?style=for-the-badge)](.)
[![Approach](https://img.shields.io/badge/Approach-Wrapper%20Implementation-22C55E?style=for-the-badge)](.)
[![References](https://img.shields.io/badge/References-3%20%7C%20IEEE-F59E0B?style=for-the-badge)](.)

**[← Back to the Building LLMs from Scratch series](../README.md)**

</div>

---

## Abstract

This topic implements multi-head attention in its most pedagogically transparent form: a wrapper class that stacks `num_heads` independent `CausalAttention` instances developed in Topic 10 and concatenates their outputs. Multi-head attention is the architectural extension that allows each Transformer block to learn several distinct representational subspaces simultaneously rather than a single global view. Each head receives the same input but is initialized with its own independent W_Q, W_K, W_V projections and its own causal mask buffer; in the forward pass, each head is run in turn and the resulting context vectors are concatenated along the embedding dimension. The full computation is traced numerically on the same six-token toy sentence used throughout the series, with d_in = 3, d_out = 2, and num_heads = 2; every intermediate tensor — the per-head W_Q, W_K, W_V matrices, the attention scores, the per-head context vectors, and the concatenated output of shape `[2, 6, 4]` — is reported to four decimal places. The class uses two production patterns introduced for the first time in this topic: `nn.ModuleList` for proper submodule registration, and `torch.cat` with `dim=-1` for concatenation along the embedding axis.

---

## At a Glance

| | |
|---|---|
| **Topic** | 11 of 36 — Stage 2 |
| **Document type** | Implementation & Derivation — multi-head, wrapper form |
| **Also known as** | Multi-Head Self-Attention (Wrapper Implementation) |
| **Core question** | How do `num_heads` independent attention computations combine into a single layer output, and why is each head an independent learned subspace? |
| **Builds on** | Topic 10 — Causal Self-Attention |
| **Running example** | `"Your journey starts with one step"` — d_in = 3, d_out = 2, num_heads = 2, batched ×2 |
| **In GPT-2 Small** | d_model = 768, num_heads = 12, d_head = 64 across all 12 transformer layers |
| **Trainable parameters** | num_heads × (3 × d_in × d_out) = 36 in this toy run; ~2.36M per attention block in GPT-2 |
| **Main artifact** | `Topic11_MultiHeadAttention_P1.docx` — 13-section technical report with 2 embedded diagrams |
| **Version** | 1.0 · May 2026 |

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

Multi-head provides the **diversity** of what to look at. The causal mask provides the **direction** (past only).

---

## Why Multiple Heads — The GPT-2 Anatomy

Using "The cat sat" as example. In GPT-2 Small (d_model = 768, num_heads = 12), the token "sat" goes through three phases:

```
THE SLICE  (input)
   'sat' arrives as a 768-dim vector.
   12 heads collectively cover all 768 dimensions:
     Head  1 owns dimensions    1 –  64
     Head  2 owns dimensions   65 – 128
     ...
     Head 12 owns dimensions  705 – 768

THE MULTI-TASK  (parallel processing)
   Head 1 (Grammar):  Is 'sat' the verb for the noun 'cat'?
   Head 2 (Tense):    Is this past, present, or future?
   Head 3 (Logic):    Can a cat physically perform sitting?
   ...all 12 run independently with their own W_Q, W_K, W_V...

THE RE-STITCH  (output)
   [Head1 (64)] + [Head2 (64)] + ... + [Head12 (64)] = 768 dim
   'sat' is now a 768-wide vector containing 12 expert opinions.
```

---

## Numerical Trace of One Head — `"journey"` Through the Pipeline

Using `torch.manual_seed(123)`, d_in = 3, d_out = 2.

### Step 1 — Three weight matrices

```python
W_query = [[0.2961, 0.5166],    W_key = [[0.1366, 0.1025],    W_value = [[0.0756, 0.1966],
           [0.2517, 0.6886],            [0.1841, 0.7264],               [0.3164, 0.4017],
           [0.0740, 0.8665]]            [0.3153, 0.6871]]               [0.1186, 0.8274]]
# shape [3, 2]                  # shape [3, 2]                  # shape [3, 2]
```

### Step 2 — Project `x_2 = [0.55, 0.87, 0.66]` ('journey') into query space

```python
query_2 = x_2 @ W_query       # tensor([0.4306, 1.4551])
```

### Step 3 — Attention score of 'journey' against itself, full row, scaled softmax

```python
attn_score_22 = query_2.dot(keys[1])   # tensor(1.8524)

attn_scores_2 = query_2 @ keys.T
# tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])

attn_weights_2 = torch.softmax(attn_scores_2 / 2**0.5, dim=-1)
# tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])  sum = 1.0
```

### Step 4 — Context vector for the query

```python
context_vec_2 = attn_weights_2 @ values
# tensor([0.3061, 0.8210])   # shape [2] — one 2-dim context vector for this head
```

### Step 5 — Two heads, two batches, concatenated

```python
batch = torch.stack((inputs, inputs), dim=0)   # [2, 6, 3]

torch.manual_seed(123)
mha = MultiHeadAttentionWrapper(d_in=3, d_out=2, context_length=6,
                                dropout=0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs.shape)   # torch.Size([2, 6, 4])

print(context_vecs)
#                       ← Head 1 →   ← Head 2 →
# tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],   # Your
#          [-0.5874,  0.0058,  0.5891,  0.3257],   # journey
#          [-0.6300, -0.0632,  0.6202,  0.3860],   # starts
#          [-0.5675, -0.0843,  0.5478,  0.3589],   # with
#          [-0.5526, -0.0981,  0.5321,  0.3428],   # one
#          [-0.5299, -0.1081,  0.5077,  0.3493]],  # step
#         [identical for batch 2 — same input stacked twice]])
```

First 2 values per token = Head 1 output (Z₁); last 2 values = Head 2 output (Z₂).

---

## The Complete `MultiHeadAttentionWrapper` Class

```python
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        # Each head has its own W_query, W_key, W_value, and mask buffer
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
```

Two lines — but every detail matters. Each `CausalAttention` instance has its own independent `W_Q`, `W_K`, `W_V`, and mask. The list comprehension runs heads sequentially in Python — correct but inefficient. Topic 12 derives the equivalent batched-matrix-multiplication form.

---

## Two Production Patterns Introduced Here

| Pattern | Where | Why It Matters |
|---|---|---|
| `nn.ModuleList` | `__init__`: stores the heads | A plain Python list silently fails to register parameters with PyTorch — the model trains without errors but head weights never receive gradient updates. `nn.ModuleList` is the only correct container |
| `torch.cat([...], dim=-1)` | `forward`: combining outputs | Concatenates along the embedding axis. Using `dim=1` would stack along the *token* axis instead, breaking every downstream operation. `dim=-1` is safe regardless of how many leading dimensions exist |

---

## Dimension Mathematics — Why d_out × num_heads = d_model

| Quantity | Toy Implementation | GPT-2 Small |
|---|---:|---:|
| d_in (input embedding) | 3 | 768 |
| num_heads | 2 | 12 |
| d_out (per head) | 2 | 64 |
| d_out × num_heads (output) | 2 × 2 = 4 | 64 × 12 = 768 |
| Output matches input? | No (intentional, for clarity) | Yes (768 = 768) |
| Stackable layers? | Not without an extra projection | Yes — 12 identical blocks stacked |

In any production model d_out is chosen specifically as `d_model / num_heads` so that the concatenated output has the same dimensionality as the input. This is what allows N identical Transformer blocks to be stacked.

---

## Key Technical Observations

Six takeaways. Full argument for each in Section 10 of the main document.

1. **Multi-head attention is independent attention computations, not a different attention computation.** Each head runs the full causal scaled dot-product mechanism from Topic 10. The arithmetic of any one head is unchanged.
2. **Each head learns a distinct representational subspace.** Independent W_Q, W_K, W_V projections allow different heads to specialize on syntax, semantics, position, coreference. Diversity from heads; constraint from the shared causal mask.
3. **Output dimension grows as d_out × num_heads.** In production d_out = d_model / num_heads so the concatenated output matches the input dimension exactly. This dimensional symmetry is what makes Transformer blocks stackable.
4. **`nn.ModuleList` is required, not optional.** A plain Python list silently fails to register parameters. The model trains without errors but its head parameters never receive gradient updates.
5. **`dim = -1` is the safe choice for embedding-axis concatenation.** Always points to the last dimension regardless of how many leading dimensions a tensor has.
6. **The wrapper is mathematically correct but operationally inefficient.** The Python list comprehension forces sequential execution. Topic 12 derives an equivalent implementation that processes all heads in a single batched matrix multiplication.

---

## Experiments and Open Questions

- **Hand-verified concatenation** for the `'journey'` token — running each head independently must reproduce the corresponding 2-dim slices of the wrapper output exactly. This is the unit test that protects against the two silent-failure modes (Section 11.1).
- **Open question** — do different heads of a trained Transformer genuinely specialize? Empirical literature is mixed: some studies find clear syntactic-role specialization in early layers; others find that majorities of heads can be pruned with little performance loss. Likely depends on training data scale and layer depth (Section 11.2).
- **Proposed future work** — head-pruning experiment after training a small Transformer to convergence: zero out individual heads one at a time, measure validation loss increase. Hypothesis predicts a power-law distribution of head importance (Section 11.3).

---

## References

References are listed in IEEE numeric format. Each is cited inline in the main document and included in the References section at the end of the docx.

| # | Authors | Title | Venue | Year |
|:-:|---------|-------|-------|:-:|
| 1 | A. Vaswani et al. | Attention Is All You Need (Section 3.2.2 — Multi-Head Attention) | *NeurIPS* | 2017 |
| 2 | A. Radford et al. | Language Models are Unsupervised Multitask Learners (GPT-2) | *OpenAI Tech. Report* | 2019 |
| 3 | A. Paszke et al. | PyTorch: An Imperative Style, High-Performance Deep Learning Library | *NeurIPS* | 2019 |

---

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This document — conceptual summary and reader entry point |
| `Multi_Head_Attention.ipynb` | Full PyTorch implementation — `MultiHeadAttentionWrapper` class, `nn.ModuleList` demonstration, batch testing, output concatenation verified end-to-end |
| `Topic11_MultiHeadAttention_P1.docx` | Primary artifact (v1.0) — 16-page technical report with title page, abstract, keywords, table of contents with page numbers, 13 numbered sections, two embedded architecture diagrams (Figures 1–2), full numerical trace of one head, dimension-mathematics comparison table, hand-verified concatenation, Experiments & Open Questions, Discussion on relationship to prior work, and IEEE-format references |

> **Note on the docx:** the Table of Contents is statically generated with real page numbers and dot leaders. No fields need to be updated on open; the document renders identically in Microsoft Word, Google Docs, LibreOffice, and PDF exports.

---

## Navigation

**Previous:** &nbsp; [Topic 10 — Causal Self-Attention](../10_causal_attention/README.md) &nbsp;·&nbsp; *Adds the causal mask and dropout*

**Next:** &nbsp; [Topic 12 — Multi-Head Attention (Weight-Split)](../12_multihead_p2/README.md) &nbsp;·&nbsp; *Replaces the wrapper's `nn.ModuleList` of independent `CausalAttention` modules with a single set of large W_Q, W_K, W_V projections of shape `[d_model, d_model]`, processed in a single batched matrix multiplication. The heads are recovered via reshape and transpose rather than via separate Python instances. An additional output projection W_O is added, completing the production multi-head attention block as defined in Vaswani et al. and used in GPT-2.*

---

<div align="center">

**Building LLMs from Scratch**  
A technical report series on the ground-up implementation of a GPT-2-class language model.

**Shiva Kiran Dadishetty** &nbsp;·&nbsp; Independent Research &nbsp;·&nbsp; Texas, USA  
[github.com/shivakiran-ai/llm-from-scratch](https://github.com/shivakiran-ai/llm-from-scratch)

*Document version 1.0 &nbsp;·&nbsp; Last updated May 2026*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
