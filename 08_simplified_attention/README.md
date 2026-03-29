<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2008%20%E2%80%94%20Simplified%20Self-Attention&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Stage%202%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-08%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Type](https://img.shields.io/badge/Type-No%20Trainable%20Weights-2E75B6?style=for-the-badge)](.)
[![Math](https://img.shields.io/badge/Math-Dot%20Products%20%7C%20Softmax%20%7C%20Context%20Vectors-F59E0B?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topic 7 answered **why** attention exists. Topic 8 implements it — from scratch, with no trainable weights. The goal is to understand the three fundamental operations of self-attention before adding the full trainable machinery: compute **attention scores** via dot products, normalize into **attention weights** via softmax, and produce **context vectors** via weighted sums. These three steps are the computational skeleton of every attention variant that follows.

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
# But it contains NO information about:
# → How important 'your' is relative to 'journey'
# → Which word in the sentence helps predict the next token
# Solution: compute a CONTEXT VECTOR — enriched embedding with relational info
```

---

## The Three Steps

### Step 1 — Attention Scores (Dot Products)

```python
query = inputs[1]  # x(2) = 'journey' — the query token

attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
#         'your'  'journey' 'starts' 'with'  'one'  'step'
#
# Dot product quantifies how aligned two vectors are.
# Higher dot product = more relevant = more attention.
```

### Step 2 — Attention Weights (Softmax)

```python
# PyTorch softmax — numerically stable
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

# tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# Sum = 1.0000 ✅
#
# Why softmax, not simple division?
# With scores like [1, 2, 3, ..., 400]:
# Simple division: 1/400 = 0.0025 — confuses optimizer in backprop
# Softmax: small values → 0, large values → 1 — stable gradients
```

### Step 3 — Context Vector (Weighted Sum)

```python
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

# tensor([0.4419, 0.6515, 0.5683])
#
# Original 'journey' embedding: [0.55, 0.87, 0.66]
# Context vector z(2):          [0.44, 0.65, 0.57]  ← enriched version
#
# z(2) contains: what 'journey' means + how it relates to all other tokens
```

---

## The Complete Pipeline — Three Lines

```python
# Step 1: Attention scores — all pairs via matrix multiplication
attn_scores  = inputs @ inputs.T          # [6, 6]

# Step 2: Attention weights — softmax per row
attn_weights = torch.softmax(attn_scores, dim=-1)   # [6, 6]

# Step 3: Context vectors — weighted sum for all tokens
context_vecs = attn_weights @ inputs      # [6, 3]

# All 6 context vectors computed simultaneously.
# Each row is an enriched embedding for the corresponding input token.
```

---

## The Full 6×6 Attention Weight Matrix

```
         your  journey  starts  with   one    step
your   [ 0.21   0.20    0.20   0.12   0.12   0.15 ]
journey[ 0.14   0.24    0.23   0.12   0.11   0.16 ]  ← query = 'journey'
starts [ 0.14   0.24    0.23   0.12   0.11   0.16 ]
with   [ 0.14   0.21    0.20   0.15   0.13   0.17 ]
one    [ 0.15   0.20    0.20   0.14   0.19   0.13 ]
step   [ 0.14   0.22    0.21   0.14   0.10   0.19 ]

Each row sums to 1.0. Each row is one token's attention distribution over all others.
```

---

## Key Insight — Why Dot Products Measure Relevance

> In the context of self-attention mechanisms, the dot product determines the extent to which elements of a sequence attend to one another. **Higher dot product = higher similarity = higher attention score between two elements.** The dot product actually encodes how aligned or not aligned the vectors are — so using dot products to find attention scores directly quantifies which tokens are relevant to which.

---

## The Limitation — Why Trainable Weights Are Needed

```
Sentence: "The cat sat on the mat because it was warm."
Query: "warm"

WITHOUT trainable weights:
  → Dot products on raw embeddings: "warm" is most similar to itself
  → "mat" might get some attention — but only if raw embeddings capture this
  → "cat", "sat" get LOW similarity scores — not semantically related to "warm"
  → But "mat" is the KEY contextual word: the mat IS warm

WITH trainable weights (Topic 9):
  → Model LEARNS that "warm" should attend to "mat" even without semantic similarity
  → Model learns "warm" often follows "mat" in this type of context
  → Captures long-range dependency through learned representations
```

---

## Research Connection

This topic implements the computational skeleton of **Vaswani et al. (2017) — Section 3.2 Scaled Dot-Product Attention**. The same three steps — dot product scores, softmax normalization, weighted sum — are at the core of every Transformer model. Topic 9 adds the Q/K/V projection matrices and the √d_k scaling factor described in the same section.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — sharp summary |
| `Attention_Mechanism.ipynb` | Full implementation — attention scores, simple normalization, softmax naive, PyTorch softmax, single context vector, full 6×6 matrix, all context vectors |
| `Topic8_SimplifiedAttention.docx` | Complete documentation — 13 sections, context vector concept, dot product theory, softmax derivation, numerical stability, the 'warm' limitation, full 3-line pipeline |

---

## Next Topic

**[Topic 09 → Self-Attention with Keys, Queries & Values](../09_self_attention/README.md)**
*Introduces trainable projection matrices W_Q, W_K, W_V — the model learns what to query, what to attend to, and what to extract. This is the full, trainable self-attention used in GPT-2.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
