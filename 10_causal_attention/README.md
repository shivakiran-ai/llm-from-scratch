<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=130&text=Topic%2010%20%E2%80%94%20Causal%20Self-Attention&fontSize=28&fontColor=ffffff&fontAlignY=52&desc=Technical%20Report%20%7C%20Stage%202%3A%20The%20Attention%20Mechanism%20%7C%20Building%20LLMs%20from%20Scratch&descSize=12&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-10%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Document](https://img.shields.io/badge/Document-Implementation%20%26%20Derivation-2E75B6?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-Masked%20Self--Attention-22C55E?style=for-the-badge)](.)
[![References](https://img.shields.io/badge/References-3%20%7C%20IEEE-F59E0B?style=for-the-badge)](.)

**[← Back to the Building LLMs from Scratch series](../README.md)**

</div>

---

## Abstract

This topic implements causal self-attention — the masked variant of the trainable attention mechanism developed in Topic 9 — and adds the architectural details required for production-quality usage: dropout regularization, batch processing, and proper buffer registration for non-trainable mask tensors. Causal masking is the operation that prevents each position from attending to positions to its right, enforcing the autoregressive constraint that next-token prediction relies on. Two equivalent masking strategies are presented and compared: a naive post-softmax multiplicative mask, and the production-standard pre-softmax fill with negative infinity that exploits `exp(-∞) = 0` to handle masking and renormalization in a single softmax pass. The complete `CausalAttention` `nn.Module` is then assembled, demonstrating three production patterns new to the series — `register_buffer` for fixed mask tensors, `transpose(1, 2)` for batched key transposition, and the in-place `masked_fill_` operator for memory-efficient masking. By the end of this topic the attention mechanism is ready for parallelization across multiple heads in Topics 11 and 12.

---

## At a Glance

| | |
|---|---|
| **Topic** | 10 of 36 — Stage 2 |
| **Document type** | Implementation & Derivation — adds masking and dropout |
| **Also known as** | Masked Self-Attention |
| **Core question** | How does the model attend only to current and past tokens, and why is this required for autoregressive generation? |
| **Builds on** | Topic 09 — Self-Attention with Trainable Weights |
| **Running example** | `"Your journey starts with one step"` — 6 tokens, batched ×2, d_in = 3, d_out = 2 |
| **Used in** | Every decoder-only LLM — GPT-2, GPT-3, GPT-4, LLaMA, Claude |
| **Trainable parameters** | Same as Topic 9 (causal mask + dropout add zero parameters) |
| **Main artifact** | `Topic10_CausalAttention.docx` — 14-section technical report with 2 embedded diagrams |
| **Version** | 1.0 · May 2026 |

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

Without this constraint, the model trains by peeking at future tokens — a behavior it cannot reproduce at inference time when those tokens do not yet exist. Every decoder-only LLM uses causal masking; it is part of the architectural definition.

---

## Two Masking Approaches

### Approach 1 — Mask After Softmax (Naive)

```python
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
mask_simple  = torch.tril(torch.ones(context_length, context_length))
masked       = attn_weights * mask_simple
masked_norm  = masked / masked.sum(dim=1, keepdim=True)   # explicit renorm
```

### Approach 2 — Upper-Triangular -∞ Mask (Production)

```python
mask         = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked       = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)   # rows already sum to 1
```

**Why -∞ is better:** `exp(-∞) = 0`, so softmax assigns exactly zero probability to masked positions and automatically renormalizes the rest in a single pass. No separate normalization step needed.

---

## The Causal Attention Weight Matrix

```
           Your   journey  starts   with    one    step
Your    [ 1.000   0.000   0.000   0.000   0.000   0.000 ]   ← only sees itself
journey [ 0.552   0.448   0.000   0.000   0.000   0.000 ]
starts  [ 0.380   0.310   0.310   0.000   0.000   0.000 ]
with    [ 0.276   0.246   0.246   0.232   0.000   0.000 ]
one     [ 0.218   0.198   0.198   0.189   0.197   0.000 ]
step    [ 0.194   0.166   0.167   0.154   0.167   0.153 ]   ← sees everything

Each row sums to 1.0. Weights are redistributed among VISIBLE tokens only.
```

---

## Dropout — Additional Regularization

```python
dropout = nn.Dropout(0.5)   # 50% dropout rate
# Random ~50% of values → 0
# Surviving values × 1/(1-0.5) = 2 (so expected value is preserved)

# At inference: model.eval() disables dropout entirely.
# Training and inference operate on tensors of the same expected magnitude;
# only the variance differs.
```

Applied to the attention weights inside the forward pass, **after** softmax.

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
        self.register_buffer(                               # fixed, not trainable
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape                       # batch dim
        keys    = self.W_key(x)                             # [b, num_tokens, d_out]
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)        # transpose dims 1&2 only
        attn_scores.masked_fill_(                           # in-place: saves memory
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values                        # [b, num_tokens, d_out]

# Test
batch = torch.stack((inputs, inputs), dim=0)                # [2, 6, 3]
ca = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
context_vecs = ca(batch)
print(context_vecs.shape)                                   # torch.Size([2, 6, 2])
```

---

## Three Production Patterns Introduced Here

| Pattern | Where | Why It Matters |
|---|---|---|
| `register_buffer` | `__init__`: stores the mask | Fixed tensor that moves with `.to(device)` and is saved in the state dict, but the optimizer never updates it |
| `keys.transpose(1, 2)` | `forward`: score computation | Flips only the last two dimensions; the bare `.T` would scramble the batch dimension |
| `masked_fill_` (in-place) | `forward`: applying the mask | Trailing underscore = no tensor allocation; halves the memory footprint of the masking step |

---

## Key Technical Observations

Six takeaways. Full argument for each in Section 11 of the main document.

1. **Causal masking is an architectural constraint, not a regularizer.** It aligns information at training with information at inference. Without it, the model has no way to reproduce its training-time behavior at generation.
2. **The pre-softmax negative-infinity trick is the production form.** Filling future positions with `-∞` before softmax produces a valid distribution in a single softmax pass — `exp(-∞) = 0` contributes nothing to either the numerator or the denominator.
3. **Dropout regularizes by stochastic gradient routing.** Random subsets of attention weights are zeroed each training step, forcing the model to learn distributed representations rather than relying on any single attention edge.
4. **`register_buffer` is the right home for fixed tensors that travel with the model.** The causal mask must live on whichever device the model lives on but is never updated by the optimizer.
5. **`transpose(1, 2)` is the batched equivalent of `.T`.** The bare `.T` transposes all dimensions and silently scrambles the batch dimension when applied to 3D tensors.
6. **The single-head module is now feature-complete for production.** Topics 11 and 12 only add parallelism — they do not change the internal computation of any individual head.

---

## Experiments and Open Questions

- **Hand-verified causal attention row** for `'journey'` — direct softmax arithmetic over the two visible positions confirms `[0.5517, 0.4483, 0, 0, 0, 0]` to four decimal places, demonstrating that the mask zeros contribute exactly nothing to the row sum (Section 12.1).
- **Open question** — does dropout on attention weights still help at long context lengths? Several modern architectures have removed it. Empirical answer likely depends on training data scale and head specialization (Section 12.2).
- **Proposed future work** — controlled ablation training two identical decoder-only models (with and without attention dropout) on a small corpus, measuring next-token cross-entropy on a held-out validation set as a function of training steps (Section 12.3).

---

## References

References are listed in IEEE numeric format. Each is cited inline in the main document and included in the References section at the end of the docx.

| # | Authors | Title | Venue | Year |
|:-:|---------|-------|-------|:-:|
| 1 | A. Vaswani et al. | Attention Is All You Need (Section 3.2.3 — Masked Multi-Head Attention) | *NeurIPS* | 2017 |
| 2 | N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov | Dropout: A Simple Way to Prevent Neural Networks from Overfitting | *JMLR* | 2014 |
| 3 | A. Radford et al. | Language Models are Unsupervised Multitask Learners (GPT-2) | *OpenAI Tech. Report* | 2019 |

---

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This document — conceptual summary and reader entry point |
| `Causal_Attention.ipynb` | Full PyTorch implementation — both masking approaches, `tril`/`triu` exploration, dropout demonstration, complete `CausalAttention` class, batch testing |
| `Topic10_CausalAttention.docx` | Primary artifact (v1.0) — technical report comprising title page with abstract and keywords, table of contents with page numbers, 14 numbered sections, two embedded architecture diagrams (Figures 1–2), comparison table of the two masking approaches, hand-verified causal row, Experiments & Open Questions, Discussion on relationship to prior work, and IEEE-format references |

> **Note on the docx:** the Table of Contents is statically generated with real page numbers and dot leaders. No fields need to be updated on open; the document renders identically in Microsoft Word, Google Docs, LibreOffice, and PDF exports.

---

## Navigation

**Previous:** &nbsp; [Topic 09 — Self-Attention with Trainable Weights](../09_self_attention/README.md) &nbsp;·&nbsp; *Scaled dot-product attention with W_Q, W_K, W_V*

**Next:** &nbsp; [Topic 11 — Multi-Head Attention (Wrapper)](../11_multihead_p1/README.md) &nbsp;·&nbsp; *Runs multiple instances of the `CausalAttention` class built here in parallel, each with its own learned projections W_Q, W_K, W_V. The outputs of all heads are concatenated to form the layer output. The wrapper implementation uses an explicit `nn.ModuleList` of independent `CausalAttention` modules — conceptually transparent but inefficient because it cannot batch the per-head computations.*

---

<div align="center">

**Building LLMs from Scratch**  
A technical report series on the ground-up implementation of a GPT-2-class language model.

**Shiva Kiran Dadishetty** &nbsp;·&nbsp; Independent Research &nbsp;·&nbsp; Texas, USA  
[github.com/shivakiran-ai/llm-from-scratch](https://github.com/shivakiran-ai/llm-from-scratch)

*Document version 1.0 &nbsp;·&nbsp; Last updated May 2026*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
