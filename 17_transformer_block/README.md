<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2017%20-%20Full%20Transformer%20Block&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Stage%203%20Assembly%20%7C%20All%20Components%20Integrated%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-17%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%203-LLM%20Architecture-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-TransformerBlock-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-8%20Steps%20Assembled%20in%20Sequence-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topics 14, 15, and 16 built `LayerNorm`, `FeedForward` (with GELU), and shortcut connections. Topic 17 assembles all of these — together with `MultiHeadAttention` from Topics 10–12 — into a single `TransformerBlock` class that replaces `DummyTransformerBlock` from Topic 13. In GPT-2 small, this block is stacked exactly 12 times to form the complete 124M parameter model.

---

## The Complete Architecture

```
Final GPT Architecture
─────────────────────────────────────────────────────────
Token Embedding Layer          [vocab_size=50257, emb_dim=768]
Positional Embedding Layer     [context_length=1024, emb_dim=768]
Dropout
─────────────────────────────────────────────────────────
TransformerBlock × 12          ← THIS TOPIC
│
├── Step 1: LayerNorm 1        Pre-LN — before attention    (Topic 14)
├── Step 2: MultiHeadAttention  analyzes token relationships  (Topics 10–12)
├── Step 3: Dropout
├── Step 4: + Shortcut 1       x = x + attention_output      (Topic 16)
│
├── Step 5: LayerNorm 2        Pre-LN — before FFN           (Topic 14)
├── Step 6: FeedForward        Linear(768→3072)→GELU→Linear(3072→768) (Topic 15)
├── Step 7: Dropout
└── Step 8: + Shortcut 2       x = x + ffn_output            (Topic 16)
─────────────────────────────────────────────────────────
Final LayerNorm                                             (Topic 14)
Output Linear Layer → logits   [batch, tokens, 50257]

Input: "Every effort moves you"
```

---

## Step 1 — Layer Normalization 1 (Pre-LN Before Attention)

LayerNorm is the first operation inside every block. Without it, deep networks explode:

```
Each layer makes numbers 1.5× larger (no normalization):
   After 40 layers:  1.5^40 =   12,089
   After 100 layers: 1.5^100 = 406,561,177
   → NaN errors → AI breakdown

LayerNorm fixes this:
   Input:  -0.11, 0.12, -0.36, -0.24, -1.19  → mean=0.13, var=0.39
   Output:  0.60, 1.11,  0.37,  0.57,  0.87  → mean=0.00, var=1.00  ✅

Three operations:
   1. Shift the mean:     subtract average → mean = 0
   2. Scale the variance: divide by std   → variance = 1
   3. Reset distribution: data back to stable bell curve

LayerNorm acts like a strict security guard at every layer boundary.
Before output of Layer 1 touches weights of Layer 2, LayerNorm
grabs those numbers, shrinks them, and centers them around 0.
```

**Placement:** GPT-2 uses Pre-LayerNorm (normalize BEFORE each sub-block). Original Vaswani et al. used Post-LN. Pre-LN is more stable for deep networks. Also applied after all 12 blocks (final LayerNorm before output head).

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps   = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # γ — learnable
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # β — learnable
    def forward(self, x):
        mean   = x.mean(dim=-1, keepdim=True)
        var    = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
# Verified: Mean=-0.0000, Variance=1.0000  ✅
```

---

## Step 2 — Masked Multi-Head Attention

After LayerNorm 1, normalized tokens enter multi-head attention. Self-attention **analyzes relationships BETWEEN input elements** — how much "effort" relates to "Every", how "sat" connects to "cat". Output shape is always identical to input shape.

```python
# Wired into TransformerBlock via GPT_CONFIG_124M:
self.att = MultiHeadAttention(
    d_in=cfg["emb_dim"],              # 768
    d_out=cfg["emb_dim"],             # 768 — d_in = d_out always
    context_length=cfg["context_length"],  # 1024
    num_heads=cfg["n_heads"],         # 12
    dropout=cfg["drop_rate"],         # 0.1
    qkv_bias=cfg["qkv_bias"]          # False
)
# head_dim = 768 // 12 = 64 per head
# Shape in: [batch, tokens, 768] → Shape out: [batch, tokens, 768]
```

Internally: ONE W_Q, W_K, W_V matrix each → split into 12 heads via `.view()` + `.transpose()` → causal mask (future tokens = -inf) → scale by 1/√64 → softmax → context vectors → recombine heads → `out_proj`. *(Full derivation in Topics 10–12.)*

---

## Steps 3 and 7 — Dropout

Dropout appears after attention (Step 3) and after FFN (Step 7). The same `self.drop_shortcut` instance is called twice — it generates a fresh random mask on each call during training.

```python
# Notebook verified output — 50% dropout on tensor of ones:
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))
# tensor([[2., 2., 0., 2., 2., 0.],
#         [0., 0., 0., 2., 0., 2.],  ← active values = 2.0, not 1.0
#         [2., 2., 2., 2., 0., 2.],
#         ...])
```

**Why active values = 2.0 (not 1.0) — Inverted Dropout:**

```
NAIVE DROPOUT (old — required manual fix at inference):
  Training (50% dropout):   5 active neurons × 1.0 = total signal 5.0
  Inference (all 10 on):    10 neurons × 1.0        = total signal 10.0  ← DOUBLED

INVERTED DROPOUT (PyTorch — no inference correction needed):
  Scaling factor = 1/(1-p) = 1/(1-0.5) = 2.0

  Training (50% dropout with inverted scaling):
    5 active neurons × 2.0 = 10.0   ← already scaled
    5 deactivated neurons  = 0.0
    Total signal = 10.0

  Inference (all 10 neurons, no scaling):
    10 neurons × 1.0 = 10.0

  Signal = 10.0 in BOTH training AND inference.
  Inference requires ZERO extra operations.
  This is why PyTorch outputs 2.0 (not 1.0) for active neurons.

Dropout active only during training.
During inference: all neurons on → full network power.
```

**Why dropout is applied — 4 reasons:**
1. **Prevents co-adaptation** — stops neurons relying on specific neighbours
2. **Forces robust features** — every neuron must learn independently (lazy neurons fixed)
3. **Improves generalization** — prevents memorizing training examples
4. **Ensemble effect** — trains many smaller networks simultaneously, averages results

---

## Steps 4 and 8 — Shortcut Connections

After dropout, the block adds the pre-sub-block input directly to the output. Happens twice — once around attention (Step 4), once around FFN (Step 8).

```python
# Without shortcut:
x = sub_block(x)          # gradient flows only through the sub-block

# With shortcut:
shortcut = x               # save input BEFORE sub-block
x = sub_block(x)
x = x + shortcut           # ADD original input back — two gradient paths
```

**Why shortcuts are essential — gradient comparison from notebook:**

```python
# ExampleDeepNeuralNetwork, layer_sizes=[3,3,3,3,3,1], torch.manual_seed(123)

WITHOUT shortcuts:
   layers.0.0.weight: 0.00020  ← Layer 1: effectively zero (vanishingly small)
   layers.1.0.weight: 0.00012
   layers.2.0.weight: 0.00072
   layers.3.0.weight: 0.00140
   layers.4.0.weight: 0.00505  ← Layer 5

WITH shortcuts:
   layers.0.0.weight: 0.22170  ← Layer 1: ×1100 larger — PRESERVED
   layers.1.0.weight: 0.20694
   layers.2.0.weight: 0.32897
   layers.3.0.weight: 0.26657
   layers.4.0.weight: 1.32585  ← Layer 5
```

**The mathematics — why "+1" saves gradients:**

```
Forward:  y_{L+1} = f(y_L) + y_L

Backward: ∂L/∂y_L = (∂L/∂y_{L+1}) × (∂f(y_L)/∂y_L + 1)

The "+1" is ALWAYS present regardless of ∂f(y_L)/∂y_L.
Even if ∂f(y_L)/∂y_L → 0:
   ∂L/∂y_L = (∂L/∂y_{L+1}) × (0 + 1) = ∂L/∂y_{L+1}

→ Gradient passes through unchanged on the shortcut path.
→ Vanishing gradients become impossible on the shortcut path.
```

---

## Step 5 — Layer Normalization 2 (Pre-LN Before FFN)

After shortcut 1, LayerNorm 2 normalizes again before the feed-forward network. Same operation as LayerNorm 1 — different instance (`self.norm2`), same effect: mean=0, variance=1.

---

## Step 6 — Feed-Forward Network (Linear → GELU → Linear)

After LayerNorm 2, each token's vector is processed **individually and independently**. The FFN modifies data individually at each position — no cross-token communication (unlike attention).

```
Self-attention:        analyzes relationships BETWEEN tokens
Feed-forward network:  modifies data INDIVIDUALLY at each position

Together: attention says WHAT to focus on, FFN says HOW to transform it.
```

**4× expansion-contraction:**

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768→3072 expand ×4
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 3072→768 contract
        )
    def forward(self, x):
        return self.layers(x)

# Inputs projected into 4× larger space — richer exploration of representation space.
# Contracts back to 768 — dimensionality preserved for shortcut addition.
# Verified: [2, 3, 768] → [2, 3, 768]  ✅
```

---

## The Complete TransformerBlock Class — All 8 Steps

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff            = FeedForward(cfg)
        self.norm1         = LayerNorm(cfg["emb_dim"])   # before attention
        self.norm2         = LayerNorm(cfg["emb_dim"])   # before FFN
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # ── Sub-block 1: Attention ─────────────────────────────────────
        shortcut = x                     # save for shortcut 1
        x = self.norm1(x)               # Step 1: LayerNorm 1
        x = self.att(x)                 # Step 2: MultiHeadAttention
        x = self.drop_shortcut(x)       # Step 3: Dropout (inverted)
        x = x + shortcut                # Step 4: Shortcut 1

        # ── Sub-block 2: FeedForward ───────────────────────────────────
        shortcut = x                     # save for shortcut 2
        x = self.norm2(x)               # Step 5: LayerNorm 2
        x = self.ff(x)                  # Step 6: FeedForward
        x = self.drop_shortcut(x)       # Step 7: Dropout
        x = x + shortcut                # Step 8: Shortcut 2

        return x

# Verified (from notebook):
torch.manual_seed(123)
x      = torch.rand(2, 4, 768)
block  = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape: ", x.shape)     # torch.Size([2, 4, 768])
print("Output shape:", output.shape) # torch.Size([2, 4, 768])  ✅
```

---

## Shape Tracking — All 8 Steps

| Step | Shape | Description |
|------|-------|-------------|
| Input | `[2, 4, 768]` | Enters block: batch=2, 4 tokens, emb\_dim=768 |
| Step 1: norm1 | `[2, 4, 768]` | LayerNorm 1 — mean=0, var=1. Shape unchanged |
| Step 2: att | `[2, 4, 768]` | MultiHeadAttention — d\_in=d\_out=768 |
| Step 3: dropout | `[2, 4, 768]` | Dropout — values zeroed and scaled. Shape unchanged |
| Step 4: + shortcut | `[2, 4, 768]` | Adds saved pre-attention input. Both must match |
| Step 5: norm2 | `[2, 4, 768]` | LayerNorm 2 — second normalization |
| Step 6: ff | `[2, 4, 768]` | FFN — internally 768→3072→768. Shape restored |
| Step 7: dropout | `[2, 4, 768]` | Dropout — second application |
| Step 8: + shortcut | `[2, 4, 768]` | Adds sub-block 1 output. Final output |

> Every operation preserves `[batch, tokens, 768]`. This is what makes stacking 12 identical blocks possible — no reshaping between blocks ever.

---

## The Dual Protection

```
LayerNorm protects the FORWARD pass:
   → Signal reaches the final layer without exploding or vanishing
   → Every layer receives stable mean=0, var=1 inputs

Shortcut connections protect the BACKWARD pass:
   → Gradient ×1100 larger at Layer 1 with shortcuts
   → Weight updates never freeze in early layers

Together: training 12 stacked transformer blocks is possible.
```

---

## Key Insight

> The transformer block is not a new invention — it is the assembly of five solutions to five problems: LayerNorm solves signal explosion (Internal Covariate Shift), dropout solves lazy neurons and overfitting, GELU solves dead neurons, shortcut connections solve vanishing gradients, and multi-head attention solves the need to attend to multiple representational subspaces. Every design choice traces to a concrete problem. The strict 8-step sequence is not arbitrary — the order determines correctness.

---

## Research Connection

**Vaswani et al. (2017) — Attention Is All You Need** — original transformer block (Post-LN, ReLU, 4× FFN). **Radford et al. (2019) — GPT-2** — Pre-LayerNorm (Section 2.3), GELU, inverted dropout. This `TransformerBlock` matches GPT-2's specification exactly. **He et al. (2016) — Deep Residual Learning** — shortcut connections `y = F(x) + x`. **Srivastava et al. (2014) — Dropout** — inverted dropout in PyTorch. **Ba et al. (2016) — Layer Normalization** — scale (γ), shift (β), epsilon.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — all 8 steps in sequence, LayerNorm security guard, inverted dropout math, shortcut "+1" gradient derivation, full class, shape tracking |
| `LLM_Architecture.ipynb` | Full implementation — all classes, dropout output shows 2.0, TransformerBlock verified `[2, 4, 768]` → `[2, 4, 768]` |
| `Topic17_FullTransformerBlock.docx` | Complete deep dive — 14 sections covering every step in order, LayerNorm explosion example (1.5^40=12,089), inverted dropout signal math (training=10.0=inference), shortcut ×1100 gradient improvement, all paper connections |

---

## Next Topic

**[Topic 18 → GPT-2 124M Parameter Model](../18_gpt2_model/README.md)**
*12 TransformerBlocks assembled into a complete GPTModel. DummyGPTModel from Topic 13 becomes real. Stage 3 closes.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
