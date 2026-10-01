<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2022%20-%20Full%20LLM%20Pretraining%20Loop&fontSize=26&fontColor=ffffff&fontAlignY=55&desc=Coding%20the%20Entire%20LLM%20Pre-Training%20Loop%20%7C%20Stage%204%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-22%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%204-Pretraining-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-train__model__simple-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-Loss%209.781%20to%200.391%20in%2010%20Epochs-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topics 19–21 built the evaluation tools. Topic 22 assembles the **complete pretraining loop**: `train_model_simple` — 8 steps that take the model from random weights (loss=9.781) to coherent English text generation (loss=0.391) in 10 epochs and 13.80 minutes. The model evolves from `"Every effort moves you,,,,,,,,,,,"` to `"Yes--quite insensible to the irony."` — overfitting to the training story verbatim by the end, as expected for a tiny dataset.

---

## The 8-Step Pretraining Loop Schematic

```
(1) for each training epoch           ← one epoch = one complete pass over training set
(2)   for each batch in training set  ← batches = train_set_size / batch_size
(3)     optimizer.zero_grad()         ← reset gradients from previous batch
(4)     loss = calc_loss_batch()      ← calculate loss on current batch
(5)     loss.backward()               ← backward pass: compute all gradients
(6)     optimizer.step()              ← update ALL 163M weights using gradients
(7)     evaluate_model()              ← optional: print train/val losses
(8)     generate_and_print_sample()   ← optional: generate text for visual inspection

Main step: find loss gradients using loss.backward()
Steps 3-6 are the standard PyTorch training pattern for all deep neural networks.
```

---

## The Differentiable Pipeline

```
Inputs → GPTModel → Logits → Softmax → Cross-entropy loss → Loss → Backpropagation
                               (output, target)

This whole operation is differentiable.
→ loss.backward() propagates gradients through all 163M parameters.
→ optimizer.step() updates all 163M parameters simultaneously.

Parameter count:
  Embedding: 50257×768 + 1024×768 = 38.4M  (not known initially, optimized during training)
  12 TransformerBlocks:
    Multi-head attention: 3×768×768 (Q,K,V) + 768×768 (out_proj) = 2.36M per block
    Feed-forward:         768×(4×768) + (4×768)×768              = 4.72M per block
    Total per block: 2.36 + 4.72 = 7.08M
    Total 12 blocks: 12 × 7.08M = 85.02M
  Final layer (softmax output): 768×50257 = 38.4M

Total to optimize: 38.4M + 85.02M + 38.4M ≈ 162M
(124M as they applied weight tying in original GPT-2)
```

---

## Gradient Descent — Weight Update Rule

```
w = w - α × (∂L/∂w)

where:
  w     = current weight value
  α     = learning rate (0.0004)
  ∂L/∂w = gradient computed in backward pass

Optimizer: AdamW (adapts lr per parameter + momentum + weight decay)
→ Just for ease of view, showing vanilla gradient descent above
```

---

## The Complete train\_model\_simple Function

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                                     # Step 3: reset gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # Step 4: calc loss
            loss.backward()                                           # Step 5: compute gradients
            optimizer.step()                                          # Step 6: update weights
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:                         # Step 7: optional eval
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)  # Step 8

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded,
                                         max_new_tokens=50, context_size=context_size)
    print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()
```

---

## Running the Training

```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=10, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
# Training completed in 13.80 minutes.
```

---

## Training Results — All 10 Epochs

| Ep | Step | Train Loss | Val Loss | Generated Text |
|----|------|-----------|---------|----------------|
| 1 | 0 | 9.781 | 9.933 | — |
| 1 | 5 | 8.111 | 8.339 | `Every effort moves you,,,,,,,,,,,.` |
| 2 | 10 | 6.661 | 7.048 | — |
| 2 | 15 | 5.961 | 6.616 | `Every effort moves you, and, and, and, ...` |
| 3 | 20 | 5.726 | 6.600 | — |
| 3 | 25 | 5.201 | 6.348 | `Every effort moves you, and I had been.` |
| 5 | 40 | 3.732 | 6.160 | `Every effort moves you know it was not that the picture...` |
| 6 | 50 | 2.427 | 6.141 | `Every effort moves you know," was one of the picture...` |
| 9 | 80 | 0.541 | 6.393 | `Every effort moves you?" "Yes--quite insensible to the irony...` |
| 10 | 85 | **0.391** | **6.452** | `Every effort moves you know," was one of the axioms he laid down across the Sevres and silver...` |

**Training completed in 13.80 minutes.**

---

## Overfitting — What It Means

```
Epoch 1-2:  Train loss ↓  AND  Val loss ↓  → both improving together
Epoch 3+:   Train loss ↓↓↓  BUT  Val loss stagnates ~6.1-6.5  → overfitting

Final: Train = 0.391  |  Val = 6.452  (massive divergence)
→ Model memorizes training text verbatim
→ "quite insensible to the irony" is a direct quote from the story

Why this is expected:
  Training dataset: 4,608 tokens (tiny)
  10 epochs: each token seen 10 times
  Real LLM pretraining: 40GB+ of text, usually 1 epoch only
  Larger dataset + fewer epochs → better generalization
```

---

## Key Insight

> `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` is the heartbeat of every neural network training loop. Each iteration, the backward pass computes `∂L/∂w` for all 163M parameters through the entire differentiable GPT pipeline. AdamW then updates each weight in the direction that reduces loss. The model evolves from outputting random commas to quoting the training story verbatim — which proves both that learning is happening and that we need a much larger dataset to achieve genuine generalization.

---

## Research Connection

**Radford et al. (2019) — GPT-2** — `train_model_simple` implements the same procedure: forward pass → cross-entropy loss → backward pass → AdamW weight update. GPT-2 used lr=2.5e-4, trained on 40GB WebText for 800K steps. **Loshchilov and Hutter (2019) — AdamW** — decouples weight decay from gradient update; standard optimizer for all modern LLMs including GPT-2, GPT-3, LLaMA, and Claude.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — 8-step schematic, differentiable pipeline, parameter count breakdown, train\_model\_simple complete code, all 10 epochs with loss values and generated text, overfitting analysis |
| `LLM_Pretraining.ipynb` | Full implementation — train\_model\_simple, evaluate\_model, generate\_and\_print\_sample, AdamW optimizer, 10 epochs training, loss 9.781→0.391, val loss 9.933→6.452, 13.80 minutes |
| `Topic22_FullPretrainingLoop.docx` | Complete deep dive — 10 sections, 8-step schematic, gradient descent math, differentiable pipeline, parameter breakdown (38.4M+85M+38.4M), complete training function, all epoch results table, overfitting analysis, AdamW paper connection |

---

## Next Topic

**[Topic 23 → Temperature Scaling](../23_temperature/README.md)**
*Greedy decoding always picks the most likely token — deterministic but repetitive. Temperature scaling controls randomness in generation: lower temperature = conservative, higher = creative.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
