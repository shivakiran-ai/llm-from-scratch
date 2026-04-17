<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=130&text=Topic%2007%20%E2%80%94%20Introduction%20to%20Attention&fontSize=28&fontColor=ffffff&fontAlignY=52&desc=Technical%20Report%20%7C%20Stage%202%3A%20The%20Attention%20Mechanism%20%7C%20Building%20LLMs%20from%20Scratch&descSize=12&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-07%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Document](https://img.shields.io/badge/Document-Conceptual%20%26%20Historical-2E75B6?style=for-the-badge)](.)
[![Version](https://img.shields.io/badge/Version-1.0-22C55E?style=for-the-badge)](.)
[![References](https://img.shields.io/badge/References-7%20%7C%20IEEE-F59E0B?style=for-the-badge)](.)

**[← Back to the Building LLMs from Scratch series](../README.md)**

</div>

---

## Abstract

This topic establishes the conceptual and historical foundation for the attention mechanism — the architectural innovation at the heart of every modern large language model. Beginning with the long-term dependency problem in natural language, the document traces the thirty-seven-year progression of sequence modelling: recurrent neural networks and their information bottleneck, long short-term memory networks and their partial mitigation of vanishing gradients, the Bahdanau attention mechanism as the first direct-access formulation, and the Transformer architecture which eliminates recurrence entirely. A toy hand-computed alignment matrix illustrates Bahdanau attention numerically, and an open question is raised regarding whether the original thesis of self-attention supplanting recurrence holds up under the quadratic-complexity limits of long-context models. This topic is a conceptual opener for Stage 2; implementation of the mechanisms described here begins in Topic 8.

---

## At a Glance

| | |
|---|---|
| **Topic** | 07 of 36 — Stage 2 Opening |
| **Document type** | Conceptual and historical — no implementation |
| **Core question** | Why does attention exist? What architectural failure did it replace? |
| **Historical arc** | RNNs (1980) → LSTMs (1997) → Bahdanau (2014) → Transformers (2017) |
| **Implementation status** | No attention code introduced here; begins in Topic 8 |
| **Main artifact** | `Topic7_AttentionIntro.docx` — 17-page technical report |
| **Version** | 1.0 · April 2026 |

---

## Topic Overview

Every modern large language model is built on a single architectural idea: every token in a sequence should have direct access to every other token when computing its own representation. This idea — self-attention — did not appear in a single paper. It is the outcome of a thirty-seven-year argument between architectures, each addressing the failures of the previous one while revealing new limitations of its own.

This document traces that argument. It is a conceptual rather than an implementation topic: no PyTorch code is introduced here, because the purpose is to make the design decisions that drive Topics 8 through 18 legible. Every choice made in the GPT-2 architecture — causal masking, the 4× feed-forward expansion ratio, the specific placement of LayerNorm, the decision to use 12 parallel attention heads rather than a single larger one — is a response to a failure documented in this document. Implementation begins in the next topic.

---

## The Motivating Problem

Sequential models fail on long-term dependencies. The canonical example:

> *"The cat that was sitting on the mat, which was next to the dog, jumped."*

Processing the word `jumped` correctly requires direct access to the subject `cat`, eight tokens earlier across a relative clause. A recurrent architecture must route this information through every intermediate hidden state — and no fixed-size vector can reliably preserve all such relationships as sequence length grows. The bottleneck is architectural, not a matter of more data or larger hidden dimensions. This is the specific failure that attention was invented to fix.

---

## The Thirty-Seven-Year Timeline

| Year | Architecture | Innovation | Remaining limitation |
|:----:|--------------|------------|----------------------|
| **1980** | Recurrent Neural Networks | Hidden state preserves memory across time steps | Vanishing gradients; single context vector bottleneck |
| **1997** | Long Short-Term Memory | Cell state and three-gate mechanism allow gradients to flow across long sequences | Sequential by construction; bottleneck unchanged |
| **2014** | Bahdanau Attention | Decoder accesses all encoder hidden states via learned attention weights | Still requires an RNN backbone |
| **2017** | Transformer | Self-attention replaces recurrence entirely; fully parallel | *Current state of the art — basis of every modern LLM* |

The pace of progress accelerated: seventeen years from RNN to LSTM, seventeen from LSTM to Bahdanau, but only three from Bahdanau to the full Transformer. The Transformer addressed all three prior limitations simultaneously — no vanishing gradients, no context bottleneck, fully parallelisable — which is why every modern decoder-only language model, from GPT-2 onwards, is built on it.

> A detailed visual timeline appears as **Figure 1** in `Topic7_AttentionIntro.docx`.

---

## The Four Attention Types — Stage 2 Roadmap

Stage 2 implements the full attention mechanism across six topics. Each topic introduces exactly one new architectural capability on top of the previous:

| # | Topic | Capability introduced |
|:-:|-------|------------------------|
| 07 | Introduction to Attention | *(This topic)* Conceptual and historical motivation |
| 08 | Simplified Self-Attention | Attention without trainable weights — dot products, softmax, context vectors |
| 09 | Self-Attention with Q / K / V | Trainable projection matrices `W_Q`, `W_K`, `W_V`; scaled dot-product attention |
| 10 | Causal Self-Attention | Causal masking for autoregressive generation; dropout on attention weights |
| 11 | Multi-Head Attention — Wrapper | Parallel heads via `nn.ModuleList` (conceptual implementation) |
| 12 | Multi-Head Attention — Weight-Split | Single-matrix production formulation used in GPT-2 |

By Topic 12, the exact multi-head attention module used in GPT-2 is complete and verified.

---

## Cross-Attention vs Self-Attention

| Property | Cross-Attention *(Bahdanau, 2014)* | Self-Attention *(Transformer, 2017)* |
|---|---|---|
| Sequences involved | Two different sequences | One sequence attending to itself |
| Primary use case | Machine translation (source → target) | Language generation (GPT family) |
| Access pattern | Decoder attends to encoder states | Every token attends to every token in the same sequence |
| Used in GPT-2 | No | Yes — with causal masking |

Cross-attention operates *between* two sequences; self-attention operates *within* one sequence. Modern autoregressive language models use only self-attention, combined with a causal mask that prevents any token from attending to positions to its right.

---

## Key Technical Observations

Six conceptual takeaways motivate the implementation work that follows. The full argument for each appears in Section 10 of the main document.

1. **The bottleneck is architectural, not statistical.** No amount of data or compute can fix an RNN's inability to route information across a single fixed-size context vector. Only a change in architecture can.
2. **Attention weights are learned, not designed.** The alignment between source and target positions emerges from training on parallel corpora; it is never imposed by hand.
3. **The 'self' in self-attention is load-bearing.** Cross-attention operates between two sequences; self-attention operates within one. GPT-style autoregressive generation requires only the latter, plus a causal mask.
4. **Dynamic focus changes the shape of computation.** At every decoding step the attention distribution is recomputed from scratch, allowing variable-length inputs to produce variable-length outputs without loss of fidelity.
5. **LSTMs solved the vanishing gradient, not the bottleneck.** These are often confused. The additive cell-state update prevents gradients from diminishing; it does not widen the single-vector information channel between encoder and decoder. Two different failures, only the first addressed by LSTMs.
6. **Parallelism is the key computational advantage.** RNNs must process tokens strictly sequentially — token *t* cannot be computed until token *t−1* is done. Transformers process all tokens simultaneously. This is why Transformers train orders of magnitude faster than RNNs on modern GPU hardware.

---

## Experiments and Open Questions

Beyond the historical narrative, the main document contributes three items that move it from summary to engagement with the literature:

- **A hand-computed Bahdanau alignment matrix** for a toy French → English translation, demonstrating numerically how the learned alignment distribution handles words with no direct cross-language counterpart (Section 11.1).
- **An open question on the Transformer thesis** — whether *"Attention Is All You Need"* remains defensible given the quadratic complexity of self-attention in long-context regimes, and what the revival of recurrence in state-space models means for the original claim (Section 11.2).
- **Proposed future work** — an empirical protocol for demonstrating the RNN information bottleneck on a synthetic reverse-string task, comparing accuracy as a function of sequence length against a matched Transformer (Section 11.3).

---

## References

References are listed in IEEE numeric format. Each is cited inline in the main document and included in the References section at the end of the docx.

| # | Authors | Title | Venue | Year |
|:-:|---------|-------|-------|:-:|
| 1 | S. Hochreiter and J. Schmidhuber | Long Short-Term Memory | *Neural Computation* | 1997 |
| 2 | D. Bahdanau, K. Cho, and Y. Bengio | Neural Machine Translation by Jointly Learning to Align and Translate | *ICLR* | 2015 |
| 3 | A. Vaswani et al. | Attention Is All You Need | *NeurIPS* | 2017 |
| 4 | A. Radford et al. | Language Models are Unsupervised Multitask Learners | *OpenAI Technical Report* | 2019 |
| 5 | J. L. Elman | Finding Structure in Time | *Cognitive Science* | 1990 |
| 6 | Y. Bengio, P. Simard, and P. Frasconi | Learning Long-Term Dependencies with Gradient Descent is Difficult | *IEEE Trans. Neural Networks* | 1994 |
| 7 | I. Sutskever, O. Vinyals, and Q. V. Le | Sequence to Sequence Learning with Neural Networks | *NeurIPS* | 2014 |

---

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This document — conceptual summary, stage roadmap, and reader entry point |
| `Attention_Intro.ipynb` | Cumulative notebook containing all code from Stage 1 (Topics 1–6). No attention code is introduced here; implementation begins in Topic 8 |
| `Topic7_AttentionIntro.docx` | Primary artifact (v1.0) — 17-page technical report comprising title page with abstract and keywords, static table of contents with page numbers, 13 numbered sections, embedded 37-year timeline figure (author's illustration), Experiments & Open Questions, Discussion on relationship to prior work, and IEEE-format references |

> **Note on the docx:** the Table of Contents is statically generated with real page numbers. No fields need to be updated on open; the document renders identically in Microsoft Word, Google Docs, LibreOffice, and PDF exports.

---

## Navigation

**Previous:** &nbsp; [Topic 06 — Full Data Preprocessing Pipeline](../06_data_pipeline/README.md) &nbsp;·&nbsp; *Closes Stage 1*

**Next:** &nbsp; [Topic 08 — Simplified Self-Attention](../08_simplified_attention/README.md) &nbsp;·&nbsp; *The first implementation of attention from scratch — dot products, softmax normalisation, and context vectors, without any trainable weights. Establishes the mathematical core of the mechanism before the full Q/K/V formulation in Topic 9.*

---

<div align="center">

**Building LLMs from Scratch**  
A technical report series on the ground-up implementation of a GPT-2-class language model.

**Shiva Kiran Dadishetty** &nbsp;·&nbsp; Independent Research &nbsp;·&nbsp; Texas, USA  
[github.com/shivakiran-ai/llm-from-scratch](https://github.com/shivakiran-ai/llm-from-scratch)

*Document version 1.0 &nbsp;·&nbsp; Last updated April 2026*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
