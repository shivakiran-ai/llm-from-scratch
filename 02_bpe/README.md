<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2002%20%E2%80%94%20The%20GPT%20Tokenizer%3A%20Byte%20Pair%20Encoding&fontSize=26&fontColor=ffffff&fontAlignY=55&desc=Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-02%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Concept](https://img.shields.io/badge/Concept-Byte%20Pair%20Encoding-1A56A0?style=for-the-badge)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![Library](https://img.shields.io/badge/Library-tiktoken%20(OpenAI)-22C55E?style=for-the-badge)](.)
[![Builds On](https://img.shields.io/badge/Builds%20On-Topic%2001%20Tokenizer-F59E0B?style=for-the-badge)](../01_tokenizer/README.md)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

In Topic 1, we built a tokenizer that crashed with `KeyError` on any word not seen during training. The fix — `<|unk|>` — was a patch, not a solution. This topic covers the real solution: **Byte Pair Encoding (BPE)** — the subword tokenization algorithm used to train GPT-2, GPT-3, and the original ChatGPT. With BPE, unknown words are structurally impossible.

---

## Direct Connection to Topic 1

```
SimpleTokenizerV2 (Topic 1)        tiktoken BPE (Topic 2)
─────────────────────────────      ─────────────────────────────
Vocabulary:  1,132 tokens          Vocabulary:  50,257 tokens
Unknown words: <|unk|> patch       Unknown words: impossible
Training data: 1 short story       Training data: 40GB WebText
Used in: educational exercise      Used in: GPT-2, GPT-3, ChatGPT
```

---

## Three Tokenization Strategies

| Strategy | Example | Tokens | Problem |
|----------|---------|--------|---------|
| Word-Based | `"dinosaur"` | `["dinosaur"]` | OOV → `KeyError` |
| Character-Based | `"dinosaur"` | `["d","i","n","o","s","a","u","r"]` | Meaning lost, 8× longer |
| **Subword BPE ✓** | `"dinosaur"` | `["dino","saur"]` | **None — best of both** |

---

## What Was Built

- **BPE algorithm walkthrough** — full merge iterations on a real dataset `{"old":7, "older":3, "finest":9, "lowest":4}`, every frequency table traced step by step from raw characters to final 11-token vocabulary
- **Data compression origin** — BPE traced to Gage (1994), original compression example fully walked through: `aaabdaaabac` → `wdwac`
- **GPT-2 vocabulary breakdown** — 256 base bytes + 50,000 learned merges + 1 special token = 50,257
- **tiktoken implementation** — encode, decode, and unknown word fallback demonstrated with `"Akwirw ier"`

---

## The BPE Algorithm — Core Idea

```
Dataset: {"old<w>":7, "older<w>":3, "finest<w>":9, "lowest<w>":4}

Iteration 1:  e + s  → es       (freq: 13)  finest + lowest share this
Iteration 2:  es + t → est      (freq: 13)  common suffix discovered
Iteration 3:  est + <w> → est<w>(freq: 13)  word-ending token
Iteration 4:  o + l  → ol       (freq: 10)
Iteration 5:  ol + d → old      (freq: 10)  full word becomes 1 token

Final vocab:  11 tokens  ← vocabulary for LLM training
```

---

## Key Insight

> A word-level tokenizer sees `"finest"` and `"lowest"` as completely unrelated. BPE discovers through pure frequency statistics that both end in `"est"` — automatically learning the superlative suffix without any linguistic rules hardcoded. This is why subword tokenization helps models learn morphological relationships.

---

## The Fallback Chain — Why Unknown Words Are Impossible

```
Input word → Try full word token
           → Try learned subword merges
           → Try individual characters
           → Fall back to individual bytes (256 always available)
           → Always succeeds. Never crashes.
```

```python
# Proof — completely made-up word
tokenizer.encode("Akwirw ier")   # [33901, 86, 343, 86, 220, 959] ✅
tokenizer.decode([33901, 86, 343, 86, 220, 959])  # "Akwirw ier" ✅
# Perfect round-trip. No <|unk|>. No crash.
```

---

## Research Connection

This topic directly implements the tokenization design in **Radford et al. (2019) — GPT-2**, built on the NLP adaptation of BPE by **Sennrich et al. (2016)**. The identical tokenizer is used unchanged in **GPT-3 (175B parameters)**, proving this design scales without modification to the largest language models ever built.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — sharp topic summary |
| `Byte_Pair_Encoder.ipynb` | Full implementation notebook with tiktoken |
| `Topic2_BPE_ShivaKiranDadishetty.docx` | Complete technical documentation — BPE origins, all merge iterations, frequency tables, GPT-2 vocabulary breakdown, research connections |

---

## Next Topic

**[Topic 03 → Input-Target Data Pairs — DataLoader](../03_dataloader/README.md)**
*Takes the BPE token IDs produced here and builds the sliding-window (input, target) pairs that train the LLM.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

**[← Back to Main Repository](../README.md)**

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
