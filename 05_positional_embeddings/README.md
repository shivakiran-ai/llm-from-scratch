<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2005%20%E2%80%94%20Positional%20Embeddings&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-05%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Concept](https://img.shields.io/badge/Concept-Positional%20Embeddings-1A56A0?style=for-the-badge)](.)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](.)
[![Builds On](https://img.shields.io/badge/Builds%20On-Topic%2004%20Token%20Embeddings-F59E0B?style=for-the-badge)](../04_token_embeddings/README.md)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Token embeddings tell the model **what** each token means. But the embedding layer is completely position-blind — the same token ID always maps to the same vector, regardless of where it appears in the sequence. This topic solves that: positional embeddings inject location information so the model knows token 1 comes before token 2. Together, **token embeddings + positional embeddings = input embeddings** — the final, complete input to the LLM.

---

## Where This Fits

```
Topic 4:  Token IDs → Token embeddings      [8, 4, 256]  (WHAT each token means)
Topic 5:  + Positional embeddings           [4, 256]     (WHERE each token appears)
          = Input embeddings                [8, 4, 256]  ← final input to LLM
```

---

## The Core Problem

```python
# The cat sat on the mat   → token_ids: [464, 3797, 3332, 319, 262, 2603]
# On the mat the cat sat   → token_ids: [319, 262, 2603, 464, 3797, 3332]

# Token ID 464 ("The") ALWAYS retrieves the same row from embedding matrix
# The model has NO way to distinguish position 1 from position 4
# → "The cat sat on the mat" and "On the mat the cat sat" look identical
```

---

## What Was Built

- **Problem analysis** — why token embeddings alone are position-blind, with the exact weight matrix example from notes
- **Two types comparison** — Absolute vs Relative positional embeddings, use cases, which models use each
- **Full implementation** — token embedding layer (50,257 × 256), DataLoader batch (8 × 4), token embeddings (8 × 4 × 256), positional embedding layer (4 × 256), final addition with broadcasting
- **Complete dimension walkthrough** — every shape traced from token IDs to final input embeddings
- **GPT-2 connection** — learned absolute positional embeddings optimised during training

---

## The Implementation

```python
# Token embedding layer
token_embedding_layer = torch.nn.Embedding(50257, 256)
token_embeddings = token_embedding_layer(inputs)
# shape: [8, 4, 256]

# Positional embedding layer — same output_dim, context_length rows
pos_embedding_layer = torch.nn.Embedding(context_length, 256)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
# shape: [4, 256]

# Add — PyTorch broadcasts [4, 256] → [8, 4, 256]
input_embeddings = token_embeddings + pos_embeddings
# shape: [8, 4, 256] ← final input to LLM
```

---

## Key Insight

> The same positional embeddings are applied to every sample in the batch — because position 0 always means "first token in this sequence" regardless of which text sample it belongs to. PyTorch's broadcasting handles this automatically, converting `[4, 256]` to `[8, 4, 256]`. Both token and positional embedding weights are **trainable parameters** — the model jointly learns what each token means AND where each position matters.

---

## Absolute vs Relative

| | Absolute | Relative |
|--|---------|---------|
| What | Unique embedding per position | Encodes distance between tokens |
| Best for | Fixed-order generation (GPT) | Long sequences, variable lengths |
| GPT-2 uses | ✅ Learned during training | ❌ |
| Modern variant | — | RoPE (LLaMA), ALiBi (BLOOM) |

---

## Research Connection

This topic implements the learned absolute positional embedding design from **Radford et al. (2019) — GPT-2**, which improved on the fixed sinusoidal encodings from **Vaswani et al. (2017) — Attention Is All You Need**. The original Transformer paper established the fundamental insight: self-attention is permutation-invariant — it has no built-in notion of order — so position information must be explicitly injected.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — sharp topic summary |
| `Positional_Embeddings.ipynb` | Full implementation notebook — token embedding layer, DataLoader integration, positional embedding layer, broadcasting addition, shape verification at every step |
| `Topic5_PositionalEmbeddings.docx` | Complete technical documentation — position-blind problem, absolute vs relative comparison, full dimension walkthrough, broadcasting explanation, GPT-2 scale, research connections |

---

## Next Topic

**[Topic 06 → Full Data Preprocessing Pipeline](../06_data_pipeline/README.md)**
*Combines all of Topics 1–5 into a single end-to-end pipeline: raw text → input embeddings ready for the Transformer.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
