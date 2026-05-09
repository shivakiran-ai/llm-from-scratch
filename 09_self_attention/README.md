<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=130&text=Topic%2009%20%E2%80%94%20Self-Attention%20with%20Trainable%20Weights&fontSize=24&fontColor=ffffff&fontAlignY=52&desc=Technical%20Report%20%7C%20Stage%202%3A%20The%20Attention%20Mechanism%20%7C%20Building%20LLMs%20from%20Scratch&descSize=12&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-09%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Document](https://img.shields.io/badge/Document-Implementation%20%26%20Derivation-2E75B6?style=for-the-badge)](.)
[![Used In](https://img.shields.io/badge/Used%20In-GPT--2%20%7C%20GPT--3%20%7C%20All%20Modern%20LLMs-22C55E?style=for-the-badge)](.)
[![References](https://img.shields.io/badge/References-4%20%7C%20IEEE-F59E0B?style=for-the-badge)](.)

**[← Back to the Building LLMs from Scratch series](../README.md)**

</div>

---

## Abstract

This topic implements scaled dot-product attention from first principles — the trainable self-attention mechanism introduced by Vaswani et al. (2017) and used as the architectural core of GPT-2, GPT-3, and every modern large language model. Building on the simplified mechanism of Topic 8, the implementation introduces three trainable projection matrices W_Q, W_K, and W_V that map each input embedding into separate query, key, and value subspaces. The full computation is traced numerically on the same six-token toy sentence used throughout the series — "Your journey starts with one step" — with d_in = 3 and d_out = 2; every intermediate tensor is reported to four decimal places. Two independent justifications for the 1/√d_k scaling factor are derived: softmax saturation under large input magnitudes, and dot-product variance growth with key dimensionality. Two equivalent class implementations are presented — `SelfAttention_v1` using `nn.Parameter` for pedagogical clarity, and `SelfAttention_v2` using `nn.Linear` for production-quality Kaiming initialization.

---

## At a Glance

| | |
|---|---|
| **Topic** | 09 of 36 — Stage 2 |
| **Document type** | Implementation & Derivation — first trainable attention |
| **Also known as** | Scaled Dot-Product Attention |
| **Used in** | Original Transformer · GPT-2 · GPT-3 · All modern LLMs |
| **Core question** | How do three trainable matrices W_Q, W_K, W_V transform a fixed mathematical operation into a learning system? |
| **Builds on** | Topic 08 — Simplified Self-Attention |
| **Running example** | `"Your journey starts with one step"` — 6 tokens, d_in = 3, d_out = 2 |
| **Trainable parameters** | 3 × (d_in × d_out) = 18 in this toy run; ~1.77M per attention block in GPT-2 Small |
| **Main artifact** | `Topic9_SelfAttentionQKV.docx` — 15-section technical report with 4 embedded diagrams |
| **Version** | 1.0 · April 2026 |

---

## Topic Overview

Topic 8 implemented simplified self-attention without trainable weights — useful for understanding the math, but limited because it could only attend to tokens with similar raw embeddings. **Topic 9 implements the real mechanism**: scaled dot-product attention with three trainable weight matrices W_Q, W_K, W_V. This is the exact self-attention used in the original Transformer, GPT-2, GPT-3, and every modern large language model. Stages 3 onward build the surrounding architecture, but the attention computation itself is complete (in single-head form) by the end of this topic.

---

## The Core Innovation — Three Trainable Matrices

```
Topic 8 (simplified):  score(i, j) = x(i) · x(j)        ← raw embeddings
Topic 9 (trainable):   score(i, j) = q(i) · k(j)        ← projected vectors
                       where  q(i) = x(i) @ W_Q
                              k(j) = x(j) @ W_K
                              v(j) = x(j) @ W_V

QUERY  (W_Q): What am I looking for?    ← current token's perspective
KEY    (W_K): What do I contain?        ← every token's identity
VALUE  (W_V): What do I contribute?     ← every token's information

Three matrices. Three roles. All learned via backpropagation.
```

Because W_Q and W_K are independent learnable matrices, the model can learn alignments that have no counterpart in raw embedding similarity. This is the source of all expressive power that simplified attention lacked.

---

## The Four Steps

### Step 1 — Project inputs to Q, K, V

```python
x_2 = inputs[1]              # 'journey' — the query token
d_in, d_out = 3, 2

W_query = nn.Parameter(torch.rand(d_in, d_out))   # [3, 2]
W_key   = nn.Parameter(torch.rand(d_in, d_out))   # [3, 2]
W_value = nn.Parameter(torch.rand(d_in, d_out))   # [3, 2]

query_2 = x_2 @ W_query      # tensor([0.4306, 1.4551])
queries = inputs @ W_query   # [6, 3] @ [3, 2] = [6, 2]
keys    = inputs @ W_key     # [6, 3] @ [3, 2] = [6, 2]
values  = inputs @ W_value   # [6, 3] @ [3, 2] = [6, 2]
```

### Step 2 — Attention Scores (Query · Key)

```python
attn_score_22  = query_2.dot(keys[1])   # tensor(1.8524)
attn_scores_2  = query_2 @ keys.T       # [1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]
attn_scores    = queries @ keys.T       # [6, 2] @ [2, 6] = [6, 6]
```

### Step 3 — Scale by √d_k, then Softmax

```python
d_k = keys.shape[-1]   # 2
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
# tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

# Why √d_k specifically?
# Reason 1 — softmax saturation:
#   softmax([0.1, -0.2, 0.3, -0.2, 0.5])      → [0.19, 0.14, 0.24, 0.14, 0.29]  balanced
#   softmax([0.1, -0.2, 0.3, -0.2, 0.5] * 8)  → [0.03, 0.00, 0.16, 0.00, 0.80]  peaky ❌
#
# Reason 2 — variance growth with dimension:
#   d=5:    var(q · k) before = 4.63   after = 0.93
#   d=100:  var(q · k) before = 95.1   after = 0.95
#   Dividing by √d_k keeps variance ≈ 1 → stable gradients.
```

### Step 4 — Context Vector (Weighted Sum of *Values*)

```python
context_vec_2 = attn_weights_2 @ values
# tensor([0.3061, 0.8210])
```

The critical change from Topic 8: the weighted sum operates on **value vectors V**, not raw inputs. Three independent linear projections, three independent learned roles.

---

## The Complete Pipeline — Five Lines

```python
queries      = inputs @ W_query                                # [6, 2]
keys         = inputs @ W_key                                  # [6, 2]
values       = inputs @ W_value                                # [6, 2]
attn_scores  = queries @ keys.T                                # [6, 6]
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)   # [6, 6]
context_vecs = attn_weights @ values                           # [6, 2]
```

This block is canonical scaled dot-product attention as defined in Section 3.2.1 of Vaswani et al. (2017). Causal masking (Topic 10), wrapper multi-head (Topic 11), and weight-split multi-head (Topic 12) are layered on top of this exact computation without modifying it.

---

## Complete Shape Inventory

| Tensor | Shape | Description |
|--------|-------|-------------|
| `inputs X` | `[6, 3]` | 6 tokens, d_in = 3 |
| `W_Q, W_K, W_V` | `[3, 2]` | Trainable weight matrices |
| `Q, K, V` | `[6, 2]` | Projected query / key / value matrices |
| `attn_scores` | `[6, 6]` | Q · K^T — one score per pair |
| `attn_weights` | `[6, 6]` | Softmax-normalized — each row sums to 1 |
| `context_vecs Z` | `[6, 2]` | Final enriched embeddings |

In **GPT-2 Small**, the same pipeline runs at d_in = d_out = d_model = 768, with 12 parallel heads of d_k = 64 each (Topic 12).

---

## The Two Class Implementations

### `SelfAttention_v1` — `nn.Parameter` (pedagogical)

```python
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
```

### `SelfAttention_v2` — `nn.Linear` with Kaiming init (production)

```python
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

The two are forward-equivalent. `nn.Linear` is preferred for any production use because Kaiming uniform initialization preserves activation variance across layers — uniform random does not.

---

## Key Technical Observations

Six takeaways. Full argument for each in Section 12 of the main document.

1. **The three weight matrices W_Q, W_K, W_V are the entire intelligence of self-attention.** Everything else — dot products, softmax, weighted sums — is fixed arithmetic. Only these three matrices are learned.
2. **Separating query, key, and value lets three different relationships be learned independently.** What to look for, what to match against, what to extract — three parameter sets answering three different questions about the same input.
3. **Dividing by √d_k has two independent justifications.** Softmax saturates with large inputs; dot-product variance grows linearly with d_k. The factor √d_k uniquely addresses both.
4. **None of the context vectors are optimized in this isolated trace.** During training, W_Q, W_K, W_V are learned end-to-end via backpropagation; the context vectors then become useful representations for next-token prediction.
5. **In GPT-like models d_in = d_out = d_model.** d_in = 3, d_out = 2 here is illustrative; in GPT-2 Small d_model = 768, all weight matrices are 768 × 768.
6. **`nn.Linear(bias=False)` is the production form of a learnable projection.** Mathematically equivalent to `nn.Parameter(W)` followed by `x @ W`, but with Kaiming initialization that preserves variance across layers.

---

## Experiments and Open Questions

- **Hand-verified attention score** for q(2) · k(2) — direct arithmetic confirms the PyTorch tensor output to three decimal places, demonstrating the diagnostic discipline used throughout the series (Section 13.1).
- **Open question** — only d_q = d_k is mathematically required (because q and k must be dot-producted); d_v has no algebraic constraint linking it. Why does production almost always set d_v = d_k? (Section 13.2)
- **Proposed future work** — controlled empirical comparison between simplified attention (Topic 8) and trainable attention (this topic) on a synthetic task where optimal alignment is intentionally orthogonal to embedding similarity (Section 13.3).

---

## References

References are listed in IEEE numeric format. Each is cited inline in the main document and included in the References section at the end of the docx.

| # | Authors | Title | Venue | Year |
|:-:|---------|-------|-------|:-:|
| 1 | A. Vaswani et al. | Attention Is All You Need | *NeurIPS* | 2017 |
| 2 | A. Radford et al. | Language Models are Unsupervised Multitask Learners | *OpenAI Tech. Report* | 2019 |
| 3 | T. Brown et al. | Language Models are Few-Shot Learners | *NeurIPS* | 2020 |
| 4 | K. He, X. Zhang, S. Ren, J. Sun | Delving Deep into Rectifiers (Kaiming Init) | *ICCV* | 2015 |

---

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This document — conceptual summary and reader entry point |
| `Trainable_Self_Attention.ipynb` | Full PyTorch implementation — W_Q/W_K/W_V initialization, single-query trace, full matrix multiplication, scaling demonstration, variance demo, both class implementations |
| `Topic9_SelfAttentionQKV.docx` | Primary artifact (v1.0) — technical report comprising title page with abstract and keywords, table of contents with page numbers, 15 numbered sections, four embedded architecture diagrams (Figures 1–4), hand-verified attention score, Experiments & Open Questions, Discussion on relationship to prior work, and IEEE-format references |

> **Note on the docx:** the Table of Contents is statically generated with real page numbers and dot leaders. No fields need to be updated on open; the document renders identically in Microsoft Word, Google Docs, LibreOffice, and PDF exports.

---

## Navigation

**Previous:** &nbsp; [Topic 08 — Simplified Self-Attention](../08_simplified_attention/README.md) &nbsp;·&nbsp; *Attention without trainable weights*

**Next:** &nbsp; [Topic 10 — Causal Self-Attention](../10_causal_attention/README.md) &nbsp;·&nbsp; *Adds the causal mask required for autoregressive generation. Before softmax, every attention score from a query at position i to a key at position j > i is set to −∞ so that the corresponding weight is zero. Dropout on attention weights is also introduced for regularization.*

---

<div align="center">

**Building LLMs from Scratch**  
A technical report series on the ground-up implementation of a GPT-2-class language model.

**Shiva Kiran Dadishetty** &nbsp;·&nbsp; Independent Research &nbsp;·&nbsp; Texas, USA  
[github.com/shivakiran-ai/llm-from-scratch](https://github.com/shivakiran-ai/llm-from-scratch)

*Document version 1.0 &nbsp;·&nbsp; Last updated April 2026*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
