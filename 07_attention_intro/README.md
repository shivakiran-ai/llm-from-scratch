<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2007%20%E2%80%94%20Introduction%20to%20Attention&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Stage%202%20Opens%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-07%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%202-Attention%20Mechanism-1A56A0?style=for-the-badge)](.)
[![Type](https://img.shields.io/badge/Type-Conceptual%20Foundation-2E75B6?style=for-the-badge)](.)
[![Builds Toward](https://img.shields.io/badge/Builds%20Toward-Multi--Head%20Attention-F59E0B?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Stage 2 opens with the most important question in modern AI: **why does attention exist?** This topic does not implement attention yet — it builds the complete conceptual and historical case for why attention had to be invented. RNNs fail on long sequences. LSTMs help but do not solve the bottleneck. Bahdanau (2014) introduces selective access. Transformers (2017) eliminate the RNN entirely. Every design decision in GPT-2 traces back to the failures documented here.

---

## The Sentence That Motivates Everything

```
"The cat that was sitting on the mat, which was next to the dog, jumped."
     ↑                                                             ↑
  subject                                                        verb

When the model processes 'cat' — it must pay maximum attention to 'jumped'.
These two semantically connected words are separated by an 8-word relative clause.
This is a LONG-TERM DEPENDENCY. Sequential models break here.
```

---

## The Historical Journey — 37 Years

```
RNNs (1980)
→ Innovation:  Hidden state captures memory across time steps
→ Limitation:  Vanishing gradients · Entire input compressed into ONE context vector
                              ↓
LSTMs (1997)
→ Innovation:  Cell state + 3 gates (forget, input, output) — better long-term memory
→ Limitation:  Still sequential · Still single context vector bottleneck
                              ↓
Bahdanau Attention (2014)
→ Innovation:  Decoder accesses ALL encoder hidden states selectively
→ Limitation:  Still requires RNN backbone
                              ↓
Transformers (2017)
→ Innovation:  Self-attention only — no RNN required · Fully parallel
→ This is what GPT-2 uses ← built in Topics 8–12
```

---

## The RNN Bottleneck — Why It Fails

```
Encoding stage:
[Kannst][du][mir][helfen][diesen][Satz][zu][übersetzen]
  h1  →  h2  →  h3  →  h4  →  h5  →  h6  →  h7  →  h8
                                                        ↓
                                         SINGLE CONTEXT VECTOR
                                         (h1 through h7 discarded)
                                                        ↓
Decoding stage: decoder receives ONLY h8

Problem: For long sentences, one vector cannot encode all information.
Decoder has no direct access to earlier hidden states.
This leads to loss of context — especially for long-range dependencies.
```

---

## The LSTM Internal Architecture

```
At each time step t, an LSTM maintains TWO states:
  h_t  = hidden state  (short-term memory)
  c_t  = cell state    (long-term memory)

Three gating mechanisms control the flow:

Forget gate:  f_t = sigmoid(W_f · [h_{t-1}, x_t])   ← what to erase
Input gate:   i_t = sigmoid(W_i · [h_{t-1}, x_t])   ← what to write
Output gate:  o_t = sigmoid(W_o · [h_{t-1}, x_t])   ← what to output

Cell state update:
  c_t = f_t * c_{t-1}  +  i_t * tanh(W_c · [h_{t-1}, x_t])
         ↑ forget old       ↑ write new

The additive update is what solves vanishing gradients —
gradients flow back through c_t without diminishing.
```

---

## Bahdanau Attention — The Fix

```python
# WITHOUT attention — old RNN decoder sees only final hidden state:
context = h_final   # single fixed vector for ALL output tokens

# WITH Bahdanau attention — decoder sees ALL hidden states:
# For each output token, compute attention weights dynamically:

# Generating 'I' from 'Je suis étudiant':
attention_weights = [0.85, 0.10, 0.05]   # h1('Je') dominates
context = 0.85*h1 + 0.10*h2 + 0.05*h3

# Generating 'am':
attention_weights = [0.05, 0.88, 0.07]   # h2('suis') dominates
context = 0.05*h1 + 0.88*h2 + 0.07*h3

# The model is NOT mindlessly aligning position 1 with position 1.
# It LEARNED from training data how to align words in French-English.
# The alignment is data-driven — discovered through the training process.
```

---

## Cross-Attention vs Self-Attention

| | Cross-Attention (Bahdanau) | Self-Attention (Transformers) |
|--|--------------------------|-------------------------------|
| Sequences | 2 different sequences | 1 sequence attending to itself |
| Use case | Translation (source → target) | Language generation (GPT) |
| Access | Decoder attends to encoder states | Every token attends to all tokens |
| GPT-2 uses | ❌ | ✅ |

> In self-attention, all attention is given **within** a particular sequence. The sequence attends to itself — every token can attend to every other token in the same sequence when computing its representation.

---

## The Four Attention Types — Progression Through Stage 2

| Topic | Type | What It Adds |
|-------|------|-------------|
| 07 | Introduction to Attention | Historical motivation — why attention had to be invented |
| 08 | Simplified Self-Attention | Core mathematics — dot products, softmax, context vectors |
| 09 | Self-Attention with Q/K/V | Trainable W_Q, W_K, W_V projection matrices |
| 10 | Causal Self-Attention | Masking — only attend to previous tokens |
| 11–12 | Multi-Head Attention | Parallel heads — different representation subspaces |

---

## Key Insight

> The problem attention solves is not a performance limitation — it is a **fundamental architectural impossibility**. For long-range dependencies, sequential models cannot maintain the required information across the single context vector bottleneck regardless of model size or training data. Attention changes the architecture so that every token has direct access to every other token. No bottleneck. No information loss. This single change enables the entire GPT architecture.

---

## Research Connection

| Paper | Year | Contribution |
|-------|------|-------------|
| Hochreiter & Schmidhuber — *Long Short-Term Memory* | 1997 | Cell state + gating mechanisms — solved vanishing gradients |
| Bahdanau et al. — *Neural Machine Translation by Jointly Learning to Align and Translate* | 2014 | First attention mechanism — selective access to all encoder states |
| Vaswani et al. — *Attention Is All You Need* | 2017 | Transformer — self-attention only, no RNN required |
| Radford et al. — *GPT-2* | 2019 | Causal self-attention for language generation |

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — conceptual summary |
| `Attention_Intro.ipynb` | Notebook — simplified attention, context vectors, attention scores |
| `Topic7_AttentionIntro.docx` | Complete documentation — 12 sections, RNN failure, LSTM gates, Bahdanau 2014, self-attention definition, 37-year timeline |

---

## Next Topic

**[Topic 08 → Simplified Self-Attention](../08_simplified_attention/README.md)**
*The mathematical implementation — dot products, softmax normalization, context vectors. No trainable weights yet — pure attention mathematics from scratch.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
