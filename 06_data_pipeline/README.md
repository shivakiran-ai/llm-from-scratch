<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2006%20%E2%80%94%20Full%20Data%20Preprocessing%20Pipeline&fontSize=26&fontColor=ffffff&fontAlignY=55&desc=Stage%201%20Capstone%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-06%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%201-COMPLETE-22C55E?style=for-the-badge)](.)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](.)
[![Synthesizes](https://img.shields.io/badge/Synthesizes-Topics%2001--05-2E75B6?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

This is the **Stage 1 Capstone**. Topics 1 through 5 each built one component of the data pipeline in isolation. Topic 6 assembles all five into a single, unified, end-to-end system — exactly the input embedding pipeline shown in the GPT-2 architecture diagram. Raw text enters. Input embeddings exit. Ready for the Transformer.

---

## The Complete Pipeline

```
Input text:   "This is an example."
      ↓  Topic 1+2 — Tokenizer + BPE (tiktoken)
Token IDs:    [40134, 2052, 133, 389, 12]
      ↓  Topic 3 — DataLoader (sliding window, batch_size=8, context=4)
Batched:      inputs.shape = [8, 4]
      ↓  Topic 4 — Token Embeddings (nn.Embedding(50257, 256))
Token emb:    token_embeddings.shape = [8, 4, 256]
      ↓  Topic 5 — Positional Embeddings (nn.Embedding(4, 256))
Pos emb:      pos_embeddings.shape = [4, 256]  → broadcast → [8, 4, 256]
      +  (element-wise addition)
Input emb:    input_embeddings.shape = [8, 4, 256]  ← final input to LLM
      ↓
GPT-like Transformer  →  Output text
```

---

## The One-Function Pipeline

```python
def preprocess_pipeline(raw_text):
    # Step 1+2: Text → Batched token ID tensors (Topics 1, 2, 3)
    dataloader = create_dataloader_v1(raw_text, batch_size=8,
                                      max_length=4, stride=4)
    inputs, targets = next(iter(dataloader))
    # inputs.shape = [8, 4]

    # Step 3: Token IDs → Token embeddings (Topic 4)
    token_embeddings = token_embedding_layer(inputs)
    # token_embeddings.shape = [8, 4, 256]

    # Step 4+5: Positions → Positional embeddings → Add (Topic 5)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    # pos_embeddings.shape = [4, 256]

    input_embeddings = token_embeddings + pos_embeddings
    # input_embeddings.shape = [8, 4, 256]

    return input_embeddings, targets
```

---

## Shape Summary

| Data | Shape | What It Represents |
|------|-------|-------------------|
| `inputs` | `[8, 4]` | 8 samples × 4 token IDs |
| `token_embeddings` | `[8, 4, 256]` | Semantic meaning of each token |
| `pos_embeddings` | `[4, 256]` | Location of each position |
| `input_embeddings` | `[8, 4, 256]` | **WHAT + WHERE — final LLM input** |

---

## Trainable Parameters in Stage 1

| Component | Shape | Parameters |
|-----------|-------|-----------|
| Token embedding layer | 50,257 × 256 | 12,865,792 |
| Positional embedding layer | 4 × 256 | 1,024 |
| **Total** | — | **12,866,816** |

Both layers are trained jointly with the LLM — they are not frozen.

---

## Key Insight

> The `context_length` parameter is the single thread connecting Topics 3, 4, and 5. It is `max_length` in the DataLoader, the number of rows in the positional embedding layer, and the length dimension in every tensor shape. Changing one requires changing all three — they represent the same physical quantity: how many tokens the model sees at one time.

---

## Stage 1 Summary — What Was Built

| Topic | Component | Core Insight |
|-------|-----------|-------------|
| 01 | Regex Tokenizer | KeyError in V1 is not a bug — it exposes closed-vocabulary limitation |
| 02 | BPE via tiktoken | Strategic decision — don't reimplement, understand the algorithm deeply |
| 03 | DataLoader | 1 input-target pair = N prediction tasks where N = context_size |
| 04 | Token Embeddings | Embedding layer = linear layer on one-hot but ~50,000× more efficient |
| 05 | Positional Embeddings | Broadcasting [4,256] → [8,4,256] is correct by design, not by convenience |
| 06 | Full Pipeline | Stage 1 complete — raw text → input embeddings in one function |

---

## Research Connection

This pipeline implements **Section 2 of Radford et al. (2019) — GPT-2** exactly. The vocabulary (50,257), learned positional embeddings, and the token + position addition are all design choices from the original paper. When scaled to `d_model=768` and `context_length=1,024`, this becomes the actual GPT-2 input embedding module built in Topic 18.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — Stage 1 capstone summary |
| `Full_Data_Pipeline.ipynb` | End-to-end runnable notebook — all 5 components assembled, shapes verified at every step |
| `Topic6_FullDataPipeline.docx` | Complete documentation — pipeline diagram walkthrough, shape table, trainable parameter counts, GPT-2 scale comparison, research connections |

---

## Stage 1 Complete ✅

**[Stage 2 → Topic 07: Introduction to Attention](../07_attention_intro/README.md)**

*The `input_embeddings` tensor produced by this pipeline feeds directly into the attention mechanism. Stage 2 begins with the question that drives the entire Transformer architecture: how does the model decide which tokens are relevant to each other?*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
