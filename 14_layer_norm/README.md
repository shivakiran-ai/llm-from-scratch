<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2014%20%E2%80%94%20Layer%20Normalization&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Replacing%20DummyLayerNorm%20%7C%20Stage%203%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-14%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%203-LLM%20Architecture-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-LayerNorm-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-Mean%3D0%20Variance%3D1%20Per%20Token-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topic 13 built `DummyGPTModel` with a `DummyLayerNorm` placeholder that returned its input unchanged. Topic 14 replaces that placeholder with a real `LayerNorm` class. Layer Normalization adjusts every layer's output to have **mean zero and variance one** — keeping activations in a controlled range so gradients remain stable across all 12 transformer blocks during backpropagation.

---

## Where It Sits in the Architecture

```
Final GPT Architecture
     └── Transformer Block (×12)
             ├── (2) Layer Normalization   ← THIS TOPIC
             ├── Masked Multi-Head Attention (Topics 10–12)
             ├── (3) GELU Activation        (Topic 15)
             ├── (4) Feed-Forward Network   (Topic 16)
             └── (5) Shortcut Connections   (Topic 17)

Locations:
  1. Before masked multi-head attention    (LayerNorm 1, inside each block)
  2. After masked multi-head attention     (LayerNorm 2, inside each block)
  3. After all 12 transformer blocks       (final LayerNorm, before output head)

Total: 2 × 12 blocks + 1 final = 25 LayerNorm layers in GPT-2 small.
```

---

## Why It Is Needed — The Gradient Problem

Training deep neural networks is challenging due to two gradient failure modes:

```
Vanishing gradients:
   Layer output too small → gradient magnitudes shrink at each backprop step
   → By the time gradient reaches Layer 1, it is near zero
   → Early layer weights barely update — model stops learning

Exploding gradients:
   Gradient values are large → multiply through layers during backprop
   → By the time gradient reaches Layer 1, it has grown uncontrollably
   → Weight updates are enormous — model diverges

Weight update rule:
   w_new = w_old - (learning_rate × gradient)
   ∂Loss/∂weights tells the network exactly how to adjust weights to reduce future errors.
   If gradient is too small or too large, this update becomes useless or destructive.

Internal Covariate Shift:
   As training proceeds, inputs to each layer keep changing as upstream weights update.
   → Delayed convergence.
   → Layer Normalization prevents this by stabilizing each layer's input distribution.
```

---

## The Main Idea — Numerical Example

Normalize each layer's output to have mean zero and variance one:

```python
# x = output of one specific layer of the neural network
x = [1.0, 0.8, 2.3, 4.4]

# Step 1: Mean
μ = (1.0 + 0.8 + 2.3 + 4.4) / 4 = 2.125

# Step 2: Variance
var = (1/4) × [(-1.125)² + (-1.325)² + (0.175)² + (2.275)²] = 2.0569

# Step 3: Normalize
x_norm = (x - μ) / √var = [-0.784, -0.924, 0.122, 1.586]

# Result: mean = 0.0,  variance = 1.0  ✅
```

---

## Numerical Trace (torch.manual_seed(123))

**Step 1 — Pass through a neural network layer:**
```python
torch.manual_seed(123)
batch_example = torch.randn(2, 5)   # 2 samples, 5 features
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
# tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
#         [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]])
```

**Step 2 — Mean and variance BEFORE normalization:**
```python
mean = out.mean(dim=-1, keepdim=True)
var  = out.var(dim=-1,  keepdim=True)
# Mean:     [[0.1324], [0.2170]]   ← not zero
# Variance: [[0.0231], [0.0398]]   ← not one
```

**Step 3 — Manual normalization:**
```python
out_norm = (out - mean) / torch.sqrt(var)
# tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
#         [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]])

# After normalization:
# Mean:     [[0.0000], [0.0000]]   ✅
# Variance: [[1.0000], [1.0000]]   ✅
```

> `dim=-1`: normalize across the feature dimension (not across tokens or batches).
> `keepdim=True`: result stays `[2, 1]` so it broadcasts correctly against `[2, 6]`.

---

## The LayerNorm Class — Complete Implementation

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps   = 1e-5                               # prevents division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))  # γ — learnable, init=1
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # β — learnable, init=0

    def forward(self, x):
        mean   = x.mean(dim=-1, keepdim=True)
        var    = x.var(dim=-1, keepdim=True, unbiased=False)  # divide by n, not n-1
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# Test
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)

mean = out_ln.mean(dim=-1, keepdim=True)
var  = out_ln.var(dim=-1, unbiased=False, keepdim=True)
# Mean:     [[-0.0000], [0.0000]]   ✅
# Variance: [[ 1.0000], [1.0000]]   ✅
```

---

## Three Critical Details

| Detail | What It Does | Why It Matters |
|--------|-------------|----------------|
| `eps = 1e-5` | Added to variance before `sqrt` | Prevents division by zero if all values in a layer are identical |
| `scale, shift` | Learnable parameters (γ, β) | Allow the model to undo normalization where it helps — model trains scale and shift via backprop |
| `unbiased=False` | Divides variance by n, not n-1 | Matches GPT-2's original TensorFlow implementation — required for loading OpenAI weights in Topic 26 |

---

## Pre-LayerNorm vs Post-LayerNorm

```
Post-LayerNorm (Vaswani 2017 original):
   Input → MultiHeadAttention → LayerNorm → + shortcut
   Gradients must pass through full attention before normalization.
   Less stable for very deep networks.

Pre-LayerNorm (GPT-2, modern LLMs):
   Input → LayerNorm → MultiHeadAttention → + shortcut
   Shortcut provides direct gradient path bypassing attention.
   More stable — preferred for 12+ stacked transformer blocks.
```

GPT-2 uses **Pre-LayerNorm** throughout. This is why `DummyLayerNorm` in Topic 13 sits *before* the attention and feed-forward operations.

---

## Key Insight

> Layer Normalization solves two independent problems simultaneously. It stabilizes gradient magnitudes by keeping activations in a controlled range, and it eliminates Internal Covariate Shift by normalizing each layer's input distribution before processing. The `scale` and `shift` parameters give the model an escape hatch — if the normalized distribution hurts performance, the model can learn to restore the original distribution. Starting at `scale=1, shift=0` means LayerNorm begins as pure normalization and adapts from there.

---

## Research Connection

This topic implements **Ba et al. (2016) — Layer Normalization**. The `scale` (γ), `shift` (β), and `eps` parameters come directly from Algorithm 1 of that paper. **Radford et al. (2019) — GPT-2** adopted Pre-LayerNorm (Section 2.3), which is the architectural choice implemented here. The `unbiased=False` decision matches GPT-2's TensorFlow implementation — required for weight compatibility in Topic 26.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — gradient problem, numerical example, full implementation, three critical details, Pre vs Post-LayerNorm |
| `LLM_Architecture.ipynb` | Full implementation — LayerNorm class, manual normalization trace, scale/shift/eps explained, mean=0 var=1 verified |
| `Topic14_LayerNormalization.docx` | Complete deep dive — 11 sections, gradient theory with weight update rule, vanishing/exploding gradient mechanics, Internal Covariate Shift, numerical walkthrough with all actual values, LayerNorm class implementation, Pre vs Post-LayerNorm comparison, Ba et al. paper connection |

---

## Next Topic

**[Topic 15 → GELU Activation Function](../15_gelu/README.md)**
*The feed-forward network inside each transformer block uses GELU rather than ReLU. Topic 15 derives why — GELU is smooth and differentiable everywhere, avoiding the dying ReLU problem — and implements it.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
