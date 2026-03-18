<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2003%20%E2%80%94%20Input-Target%20Data%20Pairs%20%26%20DataLoader&fontSize=24&fontColor=ffffff&fontAlignY=55&desc=Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-03%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Concept](https://img.shields.io/badge/Concept-Sliding%20Window%20%7C%20DataLoader-1A56A0?style=for-the-badge)](.)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](.)
[![Builds On](https://img.shields.io/badge/Builds%20On-Topic%2002%20BPE-F59E0B?style=for-the-badge)](../02_bpe/README.md)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

We have token IDs from BPE. But the model cannot train on raw IDs — it needs structured **(input, target) pairs** that define the prediction task. This topic builds the complete data pipeline: from token IDs to batched PyTorch tensors ready for LLM training. It is the **last step before vector embeddings**.

---

## Where This Fits

```
Topic 1-2:  Raw Text → Token IDs          (tokenizer + BPE)
Topic 3:    Token IDs → (Input, Target)   ← THIS TOPIC
Topic 4:    Token IDs → Embeddings        (next topic)
```

---

## What Was Built

- **Manual input-target pairs** — sliding window applied to The Verdict (5,145 BPE tokens), showing all 4 prediction tasks per context window
- **Self-supervised learning** — how sentence structure itself generates labels with zero human annotation
- **Teacher forcing** — why ground truth is always used as input during training, never the model's own predictions
- **Parallel training intuition** — how masking allows all positions to be trained simultaneously in one matrix calculation
- **GPTDatasetV1** — PyTorch Dataset class with sliding window, max_length, and stride
- **create_dataloader_v1** — complete factory function tested at batch size 1 and batch size 8

---

## The Core Idea

```
Sentence: "LLM Learn to predict one word at a time"

Iteration 1:  [LLM]                          → Learn
Iteration 2:  [LLM, Learn]                   → to
Iteration 3:  [LLM, Learn, to]               → predict
Iteration 4:  [LLM, Learn, to, predict]      → one

# One sentence → many training pairs. No labels needed.
# Words after the target are masked. Model cannot see the future.
```

---

## The Sliding Window

```python
# context_size = 4, stride = 4  (no overlap — stride equals max_length)

x = [['In',  'the',     'heart',   'of'    ],  # row 1: positions 0-3
     ['a',   'city',    'stood',   'the'   ],  # row 2: positions 4-7  ← no overlap
     ['old', 'library', 'a',       'relic' ]]  # row 3: positions 8-11

y = [['the',     'heart',   'of',    'a'    ],  # row 1 target: shifted by 1
     ['city',    'stood',   'the',   'old'  ],  # row 2 target
     ['library', 'a',       'relic', 'from' ]]  # row 3 target

# stride = max_length → zero overlap → every token in exactly one batch
# stride < max_length → overlap → more training data, risk of overfitting
# Production default: stride=128, max_length=256 → 50% overlap
```

---

## Key Insight

> In regression and classification, one (input, output) pair = one prediction task. In LLMs, one (input, output) pair = **N prediction tasks**, where N = context size. A context size of 256 means every single batch delivers 256 simultaneous training signals. This efficiency — combined with parallel matrix computation — is what makes LLM training at scale possible.

---

## Research Connection

The sliding window DataLoader implemented here is the exact data pipeline used in **Radford et al. (2019) — GPT-2** (max_length=1024, 40GB WebText) and scaled unchanged in **Brown et al. (2020) — GPT-3** (300 billion tokens). The causal masking that enables parallel training of all positions simultaneously is the core contribution of **Vaswani et al. (2017) — Attention Is All You Need**, implemented in Topic 10.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — sharp topic summary |
| `Input_Output_Target_Pairs_DataLoader.ipynb` | Full implementation notebook — manual pairs, FruitDataset illustration, GPTDatasetV1, create_dataloader_v1, all batch size tests |
| `Topic3_DataLoader.docx` | Complete technical documentation — autoregressive model, self-supervised learning, teacher forcing, parallel training magic, sliding window, full DataLoader implementation and parameter reference |

---

## Next Topic

**[Topic 04 → Token Embeddings](../04_token_embeddings/README.md)**
*Converts the integer token IDs produced by this DataLoader into dense continuous vectors — the final transformation before the Transformer processes the data.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
