<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2004%20%E2%80%94%20Token%20Embeddings&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-04%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Concept](https://img.shields.io/badge/Concept-Vector%20Embeddings-1A56A0?style=for-the-badge)](.)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](.)
[![Builds On](https://img.shields.io/badge/Builds%20On-Topic%2003%20DataLoader-F59E0B?style=for-the-badge)](../03_dataloader/README.md)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

The DataLoader from Topic 3 produces batches of integer token IDs. But a Transformer cannot operate on integers — it needs continuous, dense vector representations. Token embeddings are that final transformation: converting each token ID into a rich numerical vector that encodes semantic meaning. This topic answers the fundamental question: **how do we represent words as numbers in a way that is useful for learning?**

---

## Where This Fits

```
Topic 1-2:  Raw Text → Token IDs           (tokenizer + BPE)
Topic 3:    Token IDs → Batched tensors    (DataLoader)
Topic 4:    Token IDs → Dense vectors      ← THIS TOPIC
Topic 5:    + Positional information       (next topic)
```

---

## What Was Built

- **Three failed approaches** — why random numbers, raw token IDs, and one-hot encoding all fail to capture semantic relationships between words
- **5-feature vector example** — showing how dog/cat cluster together and apple/banana cluster together when represented as feature vectors
- **Word2Vec demonstration** — proving that trained embeddings capture semantic similarity, relational structure, and the famous `king - man + woman = queen` result
- **torch.nn.Embedding** — PyTorch implementation showing the lookup operation on a weight matrix
- **The linear layer equivalence** — proof that an embedding layer is mathematically identical to a linear layer on one-hot inputs, but orders of magnitude more efficient

---

## The Core Insight — Why Embeddings Work

```
Random:   "cat" → 34,   "kitten" → -80   ← no relationship captured
One-hot:  "cat" → [0,1,0,0,...],  "kitten" → [0,0,0,1,...]  ← equidistant
Embedding:"cat" → [31, 3, 21, 18, 31],   "kitten" → [29, 3, 20, 17, 30]
                                                        ↑ similar vectors ✅
```

Semantically similar words → similar vectors → close in embedding space.

---

## The Embedding Layer — A Lookup Table

```python
# vocab_size=4, embedding_dim=5
embedding = torch.nn.Embedding(4, 5)

# Token IDs for "fox jumps dog"
idx = torch.tensor([2, 3, 1])

# Retrieves rows 2, 3, 1 from the weight matrix — no computation
output = embedding(idx)   # shape: [3, 5]
```

The embedding layer does **not** perform matrix multiplication. It directly retrieves the row corresponding to each token ID — a pure lookup operation.

---

## Key Insight

> The embedding layer is mathematically equivalent to a neural network linear layer applied to one-hot encoded inputs — but the embedding layer skips all the zero multiplications. For GPT-2's vocabulary of 50,257 tokens, one-hot vectors have 50,256 zeros per token. The embedding layer ignores all of them and goes directly to the one relevant row. Same result, ~50,000× fewer operations.

---

## GPT-2 Embedding Scale

| Model | Vocab | Dimension | Matrix Shape | Parameters |
|-------|-------|-----------|--------------|-----------|
| GPT-2 Small | 50,257 | 768 | 50,257 × 768 | 38.6M |
| GPT-2 Medium | 50,257 | 1,024 | 50,257 × 1,024 | 51.5M |
| GPT-2 XL | 50,257 | 1,600 | 50,257 × 1,600 | 80.4M |

All embedding weights are **trainable parameters** — initialized randomly and optimized during training.

---

## Research Connection

This topic implements the embedding design from **Radford et al. (2019) — GPT-2** (`vocab_size=50,257`, `embedding_dim=768`). The theoretical foundation was established by **Mikolov et al. (2013) — Word2Vec**, which first proved that linear relationships between words emerge naturally from training on text co-occurrence statistics.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — sharp topic summary |
| `Token_Embeddings.ipynb` | Full implementation notebook — Word2Vec demo, similarity scores, vector arithmetic, torch.nn.Embedding, linear layer equivalence proof |
| `Topic4_TokenEmbeddings.docx` | Complete technical documentation — three failed approaches, 5-feature example, semantic space, embedding weight matrix, GPT-2 scale, research connections |

---

## Next Topic

**[Topic 05 → Positional Embeddings](../05_positional_embeddings/README.md)**
*Token embeddings encode what a token means but not where it appears. Positional embeddings add location information — together they form the complete input to the Transformer.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
