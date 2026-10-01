<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2020%20-%20LLM%20Loss%20Function&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Cross-Entropy%20Loss%20%7C%20Perplexity%20%7C%20Stage%204%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-20%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%204-Pretraining-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-Cross--Entropy%20Loss%20and%20Perplexity-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-loss%3D10.7722%20%7C%20perplexity%3D48725-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topic 19 built `generate_text_simple` — converting logits to text. Topic 20 defines how to **measure how wrong those predictions are** — the loss function. Cross-entropy loss quantifies the gap between what the model predicted and what it should have predicted. Perplexity makes that number interpretable: a loss of 10.7722 means the model is as uncertain as randomly choosing from 48,725 tokens. This is the training signal that drives all 163M parameters toward better predictions.

---

## Stage 4 Roadmap

```
Topic 19  ✅  Text generation          (generate_text_simple)
Topic 20  ←   Text evaluation          (loss function)          ← THIS TOPIC
Topic 21      Training/validation      (entire dataset through model)
Topic 22      LLM training function    (backpropagation + weight update)
Topic 23      Text generation          (temperature scaling)
              strategies
Topic 24      Top-k sampling
Topic 25      Weight saving/loading
Topic 26      Pre-trained weights from OpenAI
```

---

## Inputs and Targets

```python
# Inputs: token IDs the model receives
inputs = torch.tensor([[16833, 3626, 6100],   # "every effort moves"
                       [40,    1107,  588]])  # "I really like"

# Targets: the correct next token at each position (true prediction values)
targets = torch.tensor([[3626, 6100,  345],   # "effort moves you"
                        [1107,  588, 11311]]) # "really like chocolate"

# Targets = inputs shifted by 1 position (same sliding window as Stage 1 DataLoader)
# "every"  (16833) → correct next = "effort" (3626)
# "effort" (3626)  → correct next = "moves"  (6100)
# "moves"  (6100)  → correct next = "you"    (345)

# Goal of training: make the model predict target IDs with high probability.
```

---

## The 3-Step Process — Logits to Loss

**Step 1 — Get logits from GPTModel:**
```python
with torch.no_grad():
    logits = model(inputs)

probas    = torch.softmax(logits, dim=-1)   # [2, 3, 50257]
token_ids = torch.argmax(probas, dim=-1, keepdim=True)

# Before training — predicted outputs are wrong (random weights):
# Targets: " effort moves you"
# Outputs: " Armed heNetMessage"   ← gibberish
```

**Step 2 — Target probabilities (small vocabulary illustration from notes):**
```
Vocabulary: {"a":0, "effort":1, "every":2, "forward":3, "moves":4, "you":5, "zoo":6}

"every"  → [0.10, (0.60), 0.26, 0.05, 0.000, 0.02, 0.01] → "effort"  = 0.60
"effort" → [0.06, 0.07, 0.01, 0.26, (0.35), 0.13, 0.12]  → "moves"   = 0.35
"moves"  → [0.01, 0.10, 0.10, 0.20, 0.12, (0.34), 0.13]  → "you"     = 0.34

Target probabilities = [0.60, 0.35, 0.34]
Goal of training: get these as close to 1.0 as possible.
```

**Step 3 — Full model output (verified notebook values):**
```
(1) Logits            = [[[0.113, -0.1057, -0.3666, ...]]]
(2) Probabilities     = [[[1.8849e-05, 1.5172e-05, ...]]]
(3) Target probs      = [7.0451e-05, 3.1061e-05, 1.1563e-05, ...]
(4) Log probabilities = [-9.5012, -10.3796, -11.3677, ...]
(5) Average log prob  = -10.7722
(6) Negative average  =  10.7722  ← cross-entropy loss
```

---

## Cross-Entropy Loss — PyTorch Implementation

```python
# Check shapes:
print("Logits shape:", logits.shape)
# Logits shape: torch.Size([2, 3, 50257])
print("Targets shape:", targets.shape)
# Targets shape: torch.Size([2, 3])

# Flatten: combine batch and token dimensions
logits_flat  = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
# Flattened logits: torch.Size([6, 50257])   (2×3 = 6 rows)
print("Flattened targets:", targets_flat.shape)
# Flattened targets: torch.Size([6])

# Compute loss — VERY IMPORTANT
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
# tensor(10.7722)
```

> **⚠️ Do NOT apply softmax before `cross_entropy`** — PyTorch handles it internally. `cross_entropy` expects raw logits. It internally does: (1) softmax → (2) negative log-likelihood. These are fused for numerical stability. Applying softmax twice gives wrong results.

**For 2 batches — flatten both together:**
```python
# Batch 1 targets: [i11, i12, i13] → [3626, 6100, 345]
# Batch 2 targets: [i21, i22, i23] → [1107,  588, 11311]

# targets.flatten() → [3626, 6100, 345, 1107, 588, 11311]
# Merged probabilities: [p11, p12, p13, p21, p22, p23]
# Goal: all 6 values as close to 1 as possible.
```

---

## Perplexity

```python
perplexity = torch.exp(loss)
print(perplexity)
# tensor(48725.4219)
```

**What perplexity = 48,725 means:**
```
The model is roughly as uncertain as if it had to choose the next token
randomly from about 48,725 tokens in the vocabulary.

GPT-2 vocabulary = 50,257 tokens.
Perplexity 48,725 ≈ nearly as confused as random guessing from full vocab.
This is expected — the model has random weights (not trained yet).

Lower perplexity score = Better predictions.

perplexity = torch.exp(loss)   →   e^10.7722 = 48,725

If we just calculate cross-entropy loss, we don't even know how it relates
to vocabulary size. Perplexity solves this — much more interpretable.
```

| Metric | Value | Meaning |
|--------|-------|---------|
| Cross-entropy loss | 10.7722 | Raw training signal — used for backpropagation |
| Perplexity | 48,725 | Effective vocabulary size model is uncertain about |
| After training | Much lower | Loss → 0, perplexity → small number |

---

## Full Numerical Trace

```
Loss formula:  Loss = -log(p)  where p = probability of correct token
               Loss = 0 when p = 1.0 (perfect prediction)
               Loss = ∞ when p = 0.0 (impossible prediction)

Step by step:
(1) Logits            → raw model output scores
(2) Probabilities     → softmax(logits) — values between 0 and 1
(3) Target probs      → probability at correct token index
(4) Log probabilities → log(target_probs) — negative values
(5) Average           → mean of all log probabilities
(6) Negative average  → -mean = cross-entropy loss = 10.7722

PyTorch shortcut: torch.nn.functional.cross_entropy(logits_flat, targets_flat)
→ does all 6 steps automatically, loss = 10.7722 ✅

Perplexity: torch.exp(loss) = e^10.7722 = 48,725 ✅
```

---

## Key Insight

> The loss function is the bridge between prediction and learning. Without it, the model has no training signal — no way to know which direction to update its 163M parameters. Cross-entropy loss measures exactly how far the current prediction is from the correct answer by comparing two probability distributions: the model's output (spread across 50,257 tokens) and the truth (probability 1 on the target token, 0 everywhere else). Minimizing this loss — from 10.7722 toward 0 — is what training does. Perplexity makes this concrete: from "as confused as random guessing from 48,725 tokens" toward "confident prediction from a small set of likely words."

---

## Research Connection

**Shannon (1948) — A Mathematical Theory of Communication** — foundational paper defining cross-entropy H(p,q) = -Σ p(x) log q(x). Minimizing cross-entropy = maximizing likelihood of correct tokens. **Jelinek et al. (1977) — Perplexity** — introduced perplexity as the standard evaluation metric for language models. Still used today to compare GPT-2, LLaMA, and every modern LLM. **Radford et al. (2019) — GPT-2** — trained using exactly this cross-entropy loss on next-token prediction. GPT-2 XL achieves perplexity 18.34 on WikiText-103.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — Stage 4 roadmap, inputs/targets, 3-step pipeline, cross-entropy implementation, flatten requirement, perplexity explained (48,725), full numerical trace |
| `LLM_Loss_function.ipynb` | Full implementation — inputs/targets, logits, probas, token_ids, cross_entropy(logits_flat, targets_flat), loss=10.7722, perplexity=48725 verified |
| `Topic20_LLMLossFunction.docx` | Complete deep dive — 9 sections, Stage 4 roadmap, input-target relationship, 3-step logits-to-loss with small vocabulary illustration, cross_entropy internal steps (softmax+NLL), flatten mechanics, perplexity interpretation, Shannon/perplexity/GPT-2 paper connections |

---

## Next Topic

**[Topic 21 → Evaluation on Real Dataset](../21_evaluation/README.md)**
*calc\_loss\_batch and calc\_loss\_loader apply this loss function systematically across the entire training and validation dataset, producing the loss curves needed to monitor training progress.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
