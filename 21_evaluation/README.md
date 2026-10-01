<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2021%20-%20Training%20and%20Validation%20Loss&fontSize=26&fontColor=ffffff&fontAlignY=55&desc=Evaluating%20LLM%20Performance%20on%20Real%20Dataset%20%7C%20Stage%204%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-21%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%204-Pretraining-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-Training%20and%20Validation%20Loss-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-calc__loss__batch%20and%20calc__loss__loader-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topic 20 computed cross-entropy loss on a small handcrafted example. Topic 21 applies the same loss to a **real dataset**: "the-verdict.txt" — a short story tokenized to 5,145 BPE tokens. The dataset is split 90/10 into training and validation portions, DataLoaders are created for each, and two utility functions — `calc_loss_batch` and `calc_loss_loader` — systematically compute loss across all batches. Result: Training loss = **10.9876**, Validation loss = **10.9811** — both high and close because the model has not been trained yet.

---

## Why Training vs Validation Loss

```
Training loss:    loss on the 90% of data the model trains ON
Validation loss:  loss on the 10% of data held OUT during training

Training loss is NOT what really matters.
What matters: how the LLM performs on text it has NOT seen before.

Training ≈ Validation (both high): model is equally bad on both — no overfitting
Train << Validation: model memorized training data — overfitting
Both decreasing together: model is learning real language patterns ✅
```

---

## Step 1 — Split Dataset 90/10

```python
train_ratio = 0.90
split_idx   = int(train_ratio * len(text_data))
train_data  = text_data[:split_idx]   # first 90%
val_data    = text_data[split_idx:]   # last  10%

# The verdict dataset:
# Total BPE tokens: 5,145
# Training:   ~4,630 characters → 4,608 tokens (after DataLoader)
# Validation: ~514 characters   →   512 tokens
```

**The stride = context_length design (no overlap):**
```
stride = 4, context_size = 4:
  Chunk 1: [I had always thought]    → target: [had always thought Jack]
  Chunk 2: [Jack Gisburn rather a]   → target: [Gisburn rather a cheap]
  Chunk 3: [cheap genius ...]        → target: [genius ...]
  No token is skipped. stride = context_size → no overlap between chunks.
```

---

## Step 2 — Create DataLoaders

```python
torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],   # 256
    stride=GPT_CONFIG_124M["context_length"],       # 256 (no overlap)
    drop_last=True,    # discard last incomplete batch
    shuffle=True,      # randomize order each epoch
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,   # keep last (possibly smaller) batch
    shuffle=False,     # no shuffling for evaluation
    num_workers=0
)

# Verify shapes:
# Train loader: torch.Size([2, 256]) torch.Size([2, 256]) × 9 batches
# Val loader:   torch.Size([2, 256]) torch.Size([2, 256]) × 1 batch

# Token count:
print("Training tokens:", train_tokens)    # Training tokens:   4608
print("Validation tokens:", val_tokens)   # Validation tokens:   512
print("All tokens:", train_tokens + val_tokens)   # All tokens: 5120
```

---

## Step 3 — Loss Pipeline on Real Data

From the notes — first input "I had always thought" through the model:

```
"I had always thought" → GPTModel
         ↓ logits [4, 50257]
         ↓ softmax → probabilities
         ↓ Get output prob for target tokens

Target: "had always thought Jack"
  had     → token ID 23
  always  → token ID 3881
  thought → token ID 11233
  Jack    → token ID 15

Cross-entropy = -(1/4)(log p1 + log p2 + log p3 + log p4)
NLL graph: loss high when p is small, drops to 0 when p → 1
```

---

## Step 4 — calc\_loss\_batch and calc\_loss\_loader

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss   = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),    # [batch×tokens, vocab_size]
        target_batch.flatten()   # [batch×tokens]
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)   # use all batches
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()   # .item() → Python float
        else:
            break
    return total_loss / num_batches   # average loss
```

---

## Step 5 — Compute Training and Validation Loss

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)   # reproducibility (DataLoader shuffling)

with torch.no_grad():    # disable gradient tracking — evaluation only
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss   = calc_loss_loader(val_loader,   model, device)

print("Training loss:", train_loss)
# Training loss: 10.98758347829183

print("Validation loss:", val_loss)
# Validation loss: 10.981106758117676
```

**Understanding the results:**
```
Training loss:   10.9876
Validation loss: 10.9811   ← almost identical

Both losses are ~11 — very high.
This is expected: model has RANDOM weights, not trained yet.
Training ≈ Validation confirms: no overfitting (equally bad on both).
Perplexity: e^10.98 ≈ 59,000 tokens equally uncertain.

After training (Topic 22):
  Training loss decreases → model learns from training data
  Validation loss should also decrease → model generalizes
```

---

## Key Insight

> Training loss alone tells you nothing about generalization. A model can achieve zero training loss by memorizing every example — this is overfitting. The validation loss on the held-out 10% is the true measure of model quality. At this stage, both losses are ~10.98 because the model has random weights — neither dataset has been "seen" meaningfully. Once training begins, watching training and validation loss together tells the complete story: both should decrease together toward a minimum.

---

## Research Connection

**Radford et al. (2019) — GPT-2** — uses the same train/validation split methodology. `calc_loss_loader` implements exactly what tracks GPT-2's training progress — the only difference is scale (40GB vs 5KB of text). **Brown et al. (2020) — GPT-3** — same infrastructure at 175B scale; batch size 3.2M tokens vs our 2 samples × 256 tokens.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — train/val split, DataLoader setup, loss pipeline on real data, calc\_loss\_batch, calc\_loss\_loader, results (10.9876 / 10.9811) |
| `LLM_Training_Validation_Loss.ipynb` | Full implementation — 90/10 split, train/val loaders, shape verification, token count (4608+512=5120), calc\_loss\_batch, calc\_loss\_loader, final results |
| `Topic21_TrainingValidationLoss.docx` | Complete deep dive — 10 sections, dataset split, DataLoader shapes, loss pipeline with token IDs (had=23, always=3881, thought=11233, Jack=15), both utility functions, device setup, results analysis, overfitting explanation |

---

## Next Topic

**[Topic 22 → Full Pretraining Loop](../22_pretraining/README.md)**
*The training loop: calc\_loss\_batch → loss.backward() → optimizer.step(). Training and validation losses are tracked to produce loss curves. The model finally learns from real text.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
