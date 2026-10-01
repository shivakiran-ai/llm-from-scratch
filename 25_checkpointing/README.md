<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2025%20-%20Saving%20and%20Loading%20Model%20Weights&fontSize=24&fontColor=ffffff&fontAlignY=55&desc=Checkpointing%20%7C%20state_dict%20%7C%20Stage%204%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-25%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%204-Pretraining-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-Checkpointing-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-torch.save%20%2B%20state__dict-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topics 19–24 completed the full pretraining workflow. The trained model now generates coherent text — but only in memory. When the Python session ends, all 163M trained weights are lost. Topic 25 solves this: `torch.save` + `state_dict` persists the model to disk. Saving the **optimizer state** is equally important — AdamW stores historical gradient data per parameter that is essential for continued training convergence.

---

## Why We Need This

```
Stage 4 so far:
  Topic 22 ✅  Trained model — loss 9.781→0.391, 10 epochs, 13.80 min
  Topic 23 ✅  Temperature scaling
  Topic 24 ✅  Top-k sampling, generate() function

Issue: overfitting — model is memorizing.
Training done on very small amount of data (5K tokens).
Model exists only in RAM — lost when session ends.

Solution: save model weights to disk → reload anytime → continue or deploy.
```

---

## What Is a state\_dict?

```python
# state_dict = dictionary mapping each layer name to its parameter tensor:
model.state_dict()
# {
#   "tok_emb.weight":                    tensor([50257, 768]),
#   "pos_emb.weight":                    tensor([1024, 768]),
#   "trf_blocks.0.norm1.scale":          tensor([768]),
#   "trf_blocks.0.att.W_query.weight":   tensor([768, 768]),
#   ... (all 163M parameters)
# }

# state_dict does NOT contain:
#   → model architecture (the GPTModel class)
#   → hyperparameters (GPT_CONFIG_124M)
#   → optimizer state
# To load: you need BOTH the model class + saved state_dict
```

---

## Method 1 — Save Model Weights Only

```python
# ── SAVE ──────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "model.pth")
# model.state_dict() → dictionary mapping each layer to its parameters
# "model.pth"        → filename (.pth is convention, any extension works)

# ── LOAD ──────────────────────────────────────────────────────────────────
model = GPTModel(GPT_CONFIG_124M)        # create fresh model instance
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()                             # set to evaluation mode

# weights_only=True → secure loading (prevents arbitrary code execution)
# After loading: generate exactly as before — weights fully preserved
```

**Use when:** inference only (text generation, evaluation, deployment).

---

## Method 2 — Save Model AND Optimizer (Recommended for Continued Training)

### Why the Optimizer State Must Be Saved

```
AdamW stores additional parameters for EACH of the 163M model weights:
  → Hyperparameters: lr, weight_decay, betas, eps
  → Historical data: past gradients (m_t, v_t) per parameter

AdamW uses historical data to adjust learning rates for each model
parameter dynamically.

Without saving optimizer state:
  → Optimizer RESETS when loaded
  → Historical gradient data is lost
  → Model may learn suboptimally or fail to converge properly
  → Will LOSE the ability to generate coherent text
```

### Saving Both

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

torch.save({
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
},
"model_and_optimizer.pth")
# Single checkpoint file containing both state dicts
```

### Loading and Resuming

```python
checkpoint = torch.load("model_and_optimizer.pth", weights_only=False)

# Restore model
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])

# Restore optimizer (with all historical gradient data)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

model.train()   # set to training mode before resuming

# Continue training — optimizer picks up exactly where it left off
```

---

## Method 1 vs Method 2

| | Model only (`model.pth`) | Model + Optimizer (`checkpoint.pth`) |
|---|---|---|
| **Use when** | Inference, deployment | Resuming training |
| **File size** | ~622 MB | ~1.8 GB |
| **Optimizer state** | Not saved | Saved (m_t, v_t per param) |
| **weights\_only** | `True` (secure) | `False` (loads dict) |
| **After loading** | `model.eval()` | `model.train()` |

---

## Key Insight

> `torch.save(model.state_dict(), "model.pth")` is the entire save operation. But for continued training, the optimizer state is equally important. AdamW maintains a running average of gradient history for all 163M parameters — this is what makes its adaptive learning rates work. Without saving and restoring this history, the optimizer restarts cold and the model may fail to converge from where it left off. Saving both state dicts in a single checkpoint file costs slightly more disk space but preserves all the information needed to resume training seamlessly.

---

## Research Connection

**Kingma and Ba (2015) — Adam** — the adaptive optimizer whose state (first + second moment estimates) must be saved. **Loshchilov and Hutter (2019) — AdamW** — decoupled weight decay; optimizer state includes the per-parameter momentum history that drives adaptive learning rates.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — state\_dict explained, Method 1 (model only), Method 2 (model + optimizer), why AdamW state must be saved, comparison table |
| `Model_Weights_Loaded.ipynb` | Full implementation — torch.save, torch.load, state\_dict, optimizer state\_dict, checkpoint save/load, weights\_only=True |
| `Topic25_SavingLoadingWeights.docx` | Complete deep dive — 8 sections, state\_dict structure, both save/load methods with full code, AdamW historical data explanation, weights\_only security note, Adam/AdamW paper connections |

---

## Next Topic

**[Topic 26 → Loading OpenAI GPT-2 Pretrained Weights](../26_pretrained_weights/README.md)**
*Stage 4 closes: load OpenAI's publicly released GPT-2 weights into our exact GPTModel. The model trained on 5K tokens becomes a model trained on 40GB of text.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
