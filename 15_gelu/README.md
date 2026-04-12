<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2015%20%E2%80%94%20GELU%20Activation%20%26%20Feed-Forward%20Network&fontSize=24&fontColor=ffffff&fontAlignY=55&desc=GPT%20Architecture%20Part%203%20%7C%20Stage%203%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-15%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%203-LLM%20Architecture-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-FeedForward%20with%20GELU-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-768→3072→768%20Expansion%20Contraction-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topic 14 implemented `LayerNorm`. Topic 15 implements the other half of the transformer block: the **Feed-Forward Network**. Every transformer block in GPT-2 contains a small neural network sub-module — three layers: `Linear(768→3072) → GELU → Linear(3072→768)`. It expands each token's representation into a 4× larger space, applies a smooth non-linear activation, and contracts back — without ever changing the output shape. This is GPT Architecture Part 3.

---

## Where It Sits

```
Transformer Block (×12)
     ├── Layer Normalization 1         (Topic 14)
     ├── Masked Multi-Head Attention   (Topics 10–12)
     ├── Shortcut connection  +
     ├── Layer Normalization 2         (Topic 14)
     ├── Feed-Forward Network          ← THIS TOPIC
     │         Linear(768 → 3072)      expand ×4
     │         GELU activation
     │         Linear(3072 → 768)      contract ÷4
     └── Shortcut connection  +
```

---

## ReLU — Why Not Used in GPT-2

```python
ReLU(x) = max(0, x)   # x > 0 → x,  x ≤ 0 → exactly 0

# Problem 1: not differentiable at x = 0
# Problem 2: dead neuron problem

# If layer output is negative → ReLU output = 0
# → gradient through that neuron = 0
# → weights cannot update during backpropagation
# → neuron permanently dies — never contributes to learning again

ReLU(-1) = 0.0000   # hard zero
ReLU(-3) = 0.0000   # hard zero — gradient completely blocked
```

> Once a ReLU neuron dies, it can never recover. With many dead neurons, large portions of the network stop learning entirely.

---

## GELU — The Formula

GELU multiplies `x` by the CDF of the Standard Gaussian distribution:

```
GELU(x) = x × Φ(x)

where Φ(x) = ∫ from -∞ to x of (1/√(2π)) × e^(-t²/2) dt
             = CDF of the Standard Gaussian distribution N(0,1)

For positive x → Φ(x) → 1  →  GELU(x) ≈ x
For negative x → Φ(x) → 0  →  GELU(x) ≈ small negative value (NOT exactly zero)

→ Gradient is never exactly zero.
→ No dead neurons.
```

**Tanh approximation used to train GPT-2:**

```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))

The constant 0.044715 was determined empirically.
tanh is differentiable everywhere → smooth gradients throughout.
```

---

## GELU vs ReLU — Actual Values

```python
x     = [-3.0,   -2.0,   -1.0,   -0.5,   0.0,   0.5,   1.0,   2.0,   3.0]
GELU  = [-0.0036, -0.0454, -0.1588, -0.1543, 0.0, 0.3457, 0.8412, 1.9546, 2.9964]
ReLU  = [ 0.0,    0.0,    0.0,    0.0,    0.0,  0.5,   1.0,   2.0,   3.0]

At x = -1:  GELU = -0.1588   ← non-zero. Gradient can flow.
            ReLU =  0.0000   ← hard zero. Dead neuron.

At x = -3:  GELU = -0.0036   ← very small, but still non-zero.
            ReLU =  0.0000   ← always zero for ANY negative input.

For x > 1:  GELU ≈ ReLU      ← nearly identical for positive values.
At x = 0:   both = 0.0000
```

---

## GELU Implementation

```python
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# torch.pi = π ≈ 3.14159
# torch.sqrt(tensor(2/π)) ≈ 0.7979
# Differentiable everywhere — no kinks, no dead neurons
```

---

## The FeedForward Network

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768 → 3072  expand ×4
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 3072 → 768  contract ÷4
        )

    def forward(self, x):
        return self.layers(x)

# Test
ffn = FeedForward(GPT_CONFIG_124M)
x   = torch.rand(2, 3, 768)   # batch=2, tokens=3, emb_dim=768
out = ffn(x)
print(out.shape)
# torch.Size([2, 3, 768])   ← identical to input ✅
```

---

## Shape Tracking

| Layer | Shape | Description |
|-------|-------|-------------|
| Input | `[2, 3, 768]` | batch=2, tokens=3, emb\_dim=768 |
| After Linear 1 | `[2, 3, 3072]` | ×4 expansion — each token's 768-dim vector → 3072-dim |
| After GELU | `[2, 3, 3072]` | Non-linear activation applied element-wise, shape unchanged |
| After Linear 2 | `[2, 3, 768]` | ÷4 contraction — back to original embedding dimension |
| Output | `[2, 3, 768]` | Identical to input — shortcut connection requires this |

---

## Parameter Count

```
Linear 1 (768 → 3072):   768 × 3072 + 3072  =  2,362,368 parameters
Linear 2 (3072 → 768):  3072 × 768  +  768  =  2,360,064 parameters

Total per FeedForward block:                     4,722,432 parameters
× 12 transformer blocks:                        56,669,184 parameters

≈ 46% of GPT-2 small's 124M total parameters come from FFN layers alone.
```

---

## Key Insight

> GELU never outputs exactly zero for negative inputs. At `x = -1`, GELU outputs `-0.1588` — small, but the gradient is non-zero and learning can continue. This single property eliminates the dead neuron problem entirely. The 4× expansion-contraction design (`768 → 3072 → 768`) allows the model to explore a richer representation space without changing the residual stream shape — the shortcut connection requires input and output to be identical in shape, and this design guarantees it.

---

## Research Connection

**Hendrycks and Gimpel (2016) — Gaussian Error Linear Units (GELUs)** — the paper introducing GELU. The tanh approximation (`0.044715`) is from Equation 2 of that paper. **Vaswani et al. (2017) — Attention Is All You Need — Section 3.3** — the origin of the 4× expansion ratio and the Position-wise Feed-Forward Network design. GPT-2 replaced their ReLU with GELU but kept the 4× ratio. **Radford et al. (2019) — GPT-2** — uses GELU throughout all 12 transformer blocks. The `FeedForward` class built here is what goes into every one of those blocks.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — ReLU dead neuron problem, GELU formula and approximation, GELU vs ReLU actual values, FeedForward implementation, shape tracking, parameter count |
| `LLM_Architecture.ipynb` | Full implementation — GELU class, FeedForward class, ffn(x) verified output `[2, 3, 768]`, GPT_CONFIG_124M used throughout |
| `Topic15_GELU_FeedForward.docx` | Complete deep dive — 11 sections, ReLU dead neuron problem, GELU exact formula (CDF of Standard Gaussian), tanh approximation, GELU vs ReLU full comparison table, GELU class, FeedForward class, 3 architecture diagrams, shape tracking, parameter count (~46% of GPT-2 params), Hendrycks 2016 + Vaswani 2017 + GPT-2 paper connections |

---

## Next Topic

**[Topic 16 → Shortcut Connections](../16_shortcuts/README.md)**
*The FeedForward block and MultiHeadAttention both have shortcut (residual) connections wrapped around them. Topic 16 derives why this is essential for gradient flow through 12 stacked transformer blocks and implements the mechanism.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
