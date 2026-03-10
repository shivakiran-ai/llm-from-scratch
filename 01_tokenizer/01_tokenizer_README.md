<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2001%20%E2%80%94%20LLM%20Tokenizer%20from%20Scratch&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-01%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Concept](https://img.shields.io/badge/Concept-Tokenization-1A56A0?style=for-the-badge)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![Dataset](https://img.shields.io/badge/Dataset-The%20Verdict%20%7C%20Edith%20Wharton-F59E0B?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

LLMs are neural networks — they process numbers, not words. Tokenization is the bridge between raw human language and the numerical world a model can learn from. This topic builds that bridge entirely from scratch — no external libraries, pure Python — and exposes every design decision that tools like HuggingFace hide from you.

---

## The Pipeline

```
Raw Text  →  Tokens  →  Token IDs  →  Embeddings (Topic 04)
                  ↑ This topic covers these two stages ↑
```

---

## What Was Built

- **Regex tokenizer** — iteratively developed to handle words, punctuation, quotes, dashes, and all edge cases in real text
- **Vocabulary** — 1,130 unique tokens from Edith Wharton's *The Verdict*, sorted alphabetically, each mapped to a unique integer ID
- **SimpleTokenizerV1** — `encode()` and `decode()` methods on a closed vocabulary
- **SimpleTokenizerV2** — production-ready tokenizer with `<|unk|>` for unknown words and `<|endoftext|>` for multi-document training

---

## Key Insight

> The `KeyError` crash in V1 when encountering an unknown word is not a bug — it is the model exposing the fundamental limitation of closed-vocabulary tokenization. This exact failure is what motivated Byte Pair Encoding in GPT-2, where unknown words are impossible by design.

---

## Research Connection

This implementation directly reflects the tokenization design described in **Radford et al. (2019) — GPT-2**, including the use of `<|endoftext|>` as a document boundary marker. The same tokenization philosophy scales unchanged to **GPT-3's 175B parameters** *(Brown et al., 2020)*.

---

## Files

| File | Description |
|------|-------------|
| `LLM_Tokenizer.ipynb` | Full implementation notebook |
| `Topic1_LLM_Tokenizer_ShivaKiranDadishetty.docx` | Complete technical documentation — every concept, iteration, and design decision in depth |

---

## Next Topic

**[Topic 02 → The GPT Tokenizer: Byte Pair Encoding](../02_bpe/README.md)**
*Eliminates `<|unk|>` entirely — any word in any language, no unknown tokens possible.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
