<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=130&text=Topic%2008%20%E2%80%94%20Simplified%20Self-Attention&fontSize=28&fontColor=ffffff&fontAlignY=52&desc=Technical%20Report%20%7C%20Stage%202%3A%20The%20Attention%20Mechanism%20%7C%20Building%20LLMs%20from%20Scratch&descSize=12&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-08%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Document](https://img.shields.io/badge/Document-Implementation%20%26%20Derivation-2E75B6?style=for-the-badge)](.)
[![Version](https://img.shields.io/badge/Version-1.0-22C55E?style=for-the-badge)](.)
[![References](https://img.shields.io/badge/References-3%20%7C%20IEEE-F59E0B?style=for-the-badge)](.)

**[← Back to the Building LLMs from Scratch series](../README.md)**

</div>

---

## Abstract

This topic presents the first full implementation of self-attention from first principles — without any trainable parameters. Three operations are defined in sequence: computation of attention scores by dot product between a query token and every input token, normalization of those scores into a valid probability distribution by softmax, and construction of a context vector as the weighted sum of all input embeddings with those normalized weights as coefficients. The mechanism is demonstrated numerically on a six-token toy sentence — "Your journey starts with one step" — using three-dimensional embeddings; every intermediate tensor is reported to four decimal places and verified against both a hand-coded loop and the equivalent matrix-multiplication form. A direct comparison between naive division normalization and softmax normalization is carried out to expose the gradient-stability argument for softmax. The topic closes with a limitation analysis that motivates the introduction of trainable query, key, and value projections in Topic 9.

---

## At a Glance

| | |
|---|---|
| **Topic** | 08 of 36 — Stage 2 |
| **Document type** | Implementation & Derivation — first attention implementation |
| **Core question** | How do three simple operations — dot product, softmax, weighted sum — produce context-aware representations? |
| **Builds on** | Topic 07 — Introduction to Attention |
| **Running example** | `"Your journey starts with one step"` — 6 tokens, 3-dim embeddings |
| **Trainable parameters** | Zero — the mechanism uses raw embeddings throughout |
| **Main artifact** | `Topic8_SimplifiedAttention.docx` — 14-section technical report |
| **Version** | 1.0 · April 2026 |

---

## Topic Overview

Topic 7 answered **why** attention exists. Topic 8 implements it — from scratch, with no trainable weights. The goal is to understand the three fundamental operations of self-attention before adding the full trainable machinery in Topic 9: compute **attention scores** via dot products, normalize into **attention weights** via softmax, and produce **context vectors** via weighted sums. These three steps are the computational skeleton of every attention variant that follows in Stage 2 and of the attention module used in GPT-2.

---

## The Core Problem — Embedding Vectors Are Context-Free

```python
# Sentence: "Your journey starts with one step"
inputs = torch.tensor([
  [0.43, 0.15, 0.89],  # 'Your'    x(1)
  [0.55, 0.87, 0.66],  # 'journey' x(2)
  [0.57, 0.85, 0.64],  # 'starts'  x(3)
  [0.22, 0.58, 0.33],  # 'with'    x(4)
  [0.77, 0.25, 0.10],  # 'one'     x(5)
  [0.05, 0.80, 0.55],  # 'step'    x(6)
])

# [0.55, 0.87, 0.66] captures what 'journey' MEANS.
# It contains NO information about how 'journey' relates to
# the other tokens in the sentence. A context vector fixes this.
```

Each row of `inputs` is a semantic embedding of a token in isolation. These embeddings do not encode the relational structure between tokens — which word is relevant to which, which pair forms a subject-verb dependency, which token should be attended to when predicting the next word. The goal of self-attention is to produce, for every input position, a second vector that preserves the original meaning and additionally encodes exactly this relational structure.

---

## The Three Steps

Every attention mechanism in Stage 2 is built on these three operations applied in sequence.

### Step 1 — Attention Scores (Dot Products)

```python
query = inputs[1]                      # x(2) = 'journey'
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
#         'Your'  'journey' 'starts' 'with'  'one'   'step'
#
# Dot product quantifies alignment in embedding space.
# Higher dot product = more relevant = stronger attention.
```

### Step 2 — Attention Weights (Softmax)

```python
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

# tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# Sum = 1.0000   (valid probability distribution)
#
# Softmax is used instead of simple division because it
# exponentially sharpens the distribution: small scores
# approach zero and large scores approach one — which is
# what the optimizer needs for stable gradient flow.
```

### Step 3 — Context Vector (Weighted Sum)

```python
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

# tensor([0.4419, 0.6515, 0.5683])
#
# Original 'journey' embedding: [0.55, 0.87, 0.66]
# Context vector z(2):          [0.44, 0.65, 0.57]   <- enriched
#
# z(2) preserves the meaning of 'journey' AND encodes how every
# other token in the sentence relates to it.
```

---

## The Complete Pipeline — Three Lines

Once the loop form of each step is understood, the entire mechanism collapses into three vectorized operations.

```python
attn_scores   = inputs @ inputs.T                    # [6, 6]
attn_weights  = torch.softmax(attn_scores, dim=-1)   # [6, 6], each row sums to 1
context_vecs  = attn_weights @ inputs                # [6, 3]
```

All six context vectors are computed in a single pass. This is the exact computational skeleton retained by every subsequent attention variant in Stage 2; the only thing that changes is what the inputs to the matrix multiplications are.

---

## The Full 6 × 6 Attention Weight Matrix

```
             Your    journey  starts  with    one     step
Your      [ 0.2098  0.2006   0.1981  0.1242  0.1220  0.1452 ]
journey   [ 0.1385  0.2379   0.2333  0.1240  0.1082  0.1581 ]   <- query row for z(2)
starts    [ 0.1390  0.2369   0.2326  0.1242  0.1108  0.1565 ]
with      [ 0.1435  0.2074   0.2046  0.1462  0.1263  0.1720 ]
one       [ 0.1526  0.1958   0.1975  0.1367  0.1879  0.1295 ]
step      [ 0.1385  0.2184   0.2128  0.1420  0.0988  0.1896 ]
```

Each row is one query token's attention distribution over all input positions. Each row sums to exactly 1.0000. The matrix is **not symmetric** — attention from *Your* to *journey* is not the same as attention from *journey* to *Your*, because each row is normalized independently.

---

## Why Trainable Weights Are Needed — The `warm` Limitation

```
Sentence:  "The cat sat on the mat because it was warm."
Query:     "warm"

WITHOUT trainable weights (this topic):
  The dot product measures similarity in the RAW embedding space.
  'warm' is most similar to itself.
  'mat' is not a near neighbour of 'warm' in embedding space,
  so it receives LOW attention — even though it is the key
  contextual word: the MAT is what is warm.

WITH trainable weights (Topic 9):
  W_Q, W_K, W_V project the inputs into learned subspaces.
  The model can LEARN that 'warm' should attend to 'mat'
  in this context, regardless of raw-embedding similarity.
```

Simplified attention can only amplify relationships that are already present in the embedding space. Contextual relationships that exist in the *sentence* but not in the *embeddings* are invisible to it. Topic 9 lifts this ceiling by introducing trainable projections.

---

## Key Technical Observations

Six takeaways motivate the work that follows. The full argument for each appears in Section 11 of the main document.

1. **The dot product is the correct compatibility function.** It quantifies alignment in the embedding space, which is the signal attention uses to decide how much weight each candidate input deserves. Any refinement — cosine similarity, scaled dot product, or bilinear form — is a variation of this same principle.
2. **Softmax is architecturally necessary, not conventional.** Simple summation produces a distribution that sums to one but fails to suppress small scores sufficiently, destabilizing gradient flow. The exponential sharpening of softmax is what enables stable training.
3. **The context vector is an enrichment, not a replacement.** z(i) preserves the semantic signature of x(i) while additionally encoding a weighted summary of every other position. This dual information content is what makes z(i) more useful than x(i) for next-token prediction.
4. **Semantic similarity is not the same as contextual relevance.** Two tokens can be unrelated in embedding space yet critical to each other in a specific sentence. This is the central limitation of simplified attention and the motivation for trainable projections.
5. **Matrix multiplication replaces explicit loops without changing outputs.** The operations `inputs @ inputs.T` and `attn_weights @ inputs` compute all scores and all context vectors simultaneously. This is the source of the parallelism that makes attention practical at scale.
6. **The three-line pipeline is the stable skeleton.** Trainable projections, causal masking, and multi-head parallelism are all modifications layered on top of the same three operations: scores → weights → weighted sum.

---

## Experiments and Open Questions

Beyond the numerical implementation, the main document contributes three items that move it from exposition toward engagement with the mechanism:

- **A hand-verified context vector for `'journey'`** — arithmetic performed manually against each of the three output components, matching the PyTorch tensor to four decimal places (Section 12.1). This is the simplest diagnostic against transcription and orientation bugs.
- **An open question** — quantifying when raw dot-product attention is insufficient: on what fraction of natural language is the gap between semantic similarity and contextual relevance actually large? A systematic study would convert a qualitative claim into a quantitative answer (Section 12.2).
- **Proposed future work** — a controlled empirical comparison between simplified attention and the full trainable mechanism of Topic 9 on a synthetic dataset designed to exhibit the `warm` phenomenon, holding all other hyperparameters fixed (Section 12.3).

---

## References

References are listed in IEEE numeric format. Each is cited inline in the main document and included in the References section at the end of the docx.

| # | Authors | Title | Venue | Year |
|:-:|---------|-------|-------|:-:|
| 1 | A. Vaswani et al. | Attention Is All You Need | *NeurIPS* | 2017 |
| 2 | D. Bahdanau, K. Cho, and Y. Bengio | Neural Machine Translation by Jointly Learning to Align and Translate | *ICLR* | 2015 |
| 3 | A. Radford et al. | Language Models are Unsupervised Multitask Learners | *OpenAI Technical Report* | 2019 |

---

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This document — conceptual summary and reader entry point |
| `Attention_Mechanism.ipynb` | Fully annotated PyTorch implementation — all three steps, both loop and matrix forms, with verified numerical outputs |
| `Topic8_SimplifiedAttention.docx` | Primary artifact (v1.0) — technical report comprising title page with abstract and keywords, table of contents with page numbers, 14 numbered sections, hand-verified context vector, Experiments & Open Questions, Discussion on relationship to prior work, and IEEE-format references |

> **Note on the docx:** the Table of Contents is statically generated with real page numbers. No fields need to be updated on open; the document renders identically in Microsoft Word, Google Docs, LibreOffice, and PDF exports.

---

## Navigation

**Previous:** &nbsp; [Topic 07 — Introduction to Attention](../07_attention_intro/README.md) &nbsp;·&nbsp; *Opens Stage 2 — why attention exists*

**Next:** &nbsp; [Topic 09 — Self-Attention with Q/K/V](../09_self_attention/README.md) &nbsp;·&nbsp; *The first trainable version of self-attention — introduces the three projection matrices W_Q, W_K, W_V and the 1/√d_k scaling factor. The model now learns what to query, what to attend to, and what to extract.*

---

<div align="center">

**Building LLMs from Scratch**  
A technical report series on the ground-up implementation of a GPT-2-class language model.

**Shiva Kiran Dadishetty** &nbsp;·&nbsp; Independent Research &nbsp;·&nbsp; Texas, USA  
[github.com/shivakiran-ai/llm-from-scratch](https://github.com/shivakiran-ai/llm-from-scratch)

*Document version 1.0 &nbsp;·&nbsp; Last updated April 2026*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
