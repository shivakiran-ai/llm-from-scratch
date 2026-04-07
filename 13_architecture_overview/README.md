<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2013%20%E2%80%94%20Bird%27s%20Eye%20View%20of%20LLM%20Architecture&fontSize=26&fontColor=ffffff&fontAlignY=55&desc=Opening%20Stage%203%20%7C%20DummyGPTModel%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-13%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%203-LLM%20Architecture-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-DummyGPTModel-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-GPT--2%20End%20to%20End%20Shape%20Verification-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Stage 2 is complete. Topics 1–12 built every component the model needs: tokenization, embeddings, and masked multi-head attention. Topic 13 opens Stage 3 by providing the bird's eye view — how all of those components connect into a single GPT-2 architecture. Rather than implementing real transformer blocks immediately, a `DummyGPTModel` is built first using placeholder layers that return their input unchanged. This verifies the full shape pipeline end-to-end — input `[2, 4]` → output `[2, 4, 50257]` — before any real computation is added.

---

## What Has Been Built vs What Comes Next

```
COVERED — Stages 1 and 2:
   (a) Input tokenization      → BPE tokenizer, vocab size 50,257
   (b) Embeddings              → token embedding (50257×768) + positional embedding (1024×768)
   (c) Masked multi-head attn  → CausalAttention, MultiHeadAttentionWrapper, MultiHeadAttention

YET TO COVER — Stage 3:
   (a) Transformer blocks      → LayerNorm + MultiHeadAttention + FFN + shortcuts
   (b) Scale to 124M params    → full GPT-2 small model
```

---

## The GPT Model — Top-Level Architecture

```
Tokenized text  →  Embedding layers  →  Transformer blocks (×12)  →  Output layers
                                              ↑
                         Masked multi-head attention lives here
                         (built in Topics 10–12)

Goal: generate new text one word at a time.

"Every effort moves you"  →  model predicts: forward
"Every day holds a"       →  model predicts: promise
```

---

## The Transformer Block — Internal Structure

```
Input [batch, tokens, 768]
       ↓
   LayerNorm 1          ← normalize before attention (Pre-LN style — GPT-2)
       ↓
   Masked Multi-Head Attention   ← built in Topics 10–12
       ↓
   Dropout
       ↓
   + ←─────────────────────────── shortcut connection (adds original input)
       ↓
   LayerNorm 2          ← normalize before feed-forward
       ↓
   Feed-Forward Network:  Linear → GELU → Linear   (768 → 3072 → 768)
       ↓
   Dropout
       ↓
   + ←─────────────────────────── shortcut connection (adds input to FFN)
       ↓
Output [batch, tokens, 768]   ← same shape as input
```

> Input and output have the **same form and dimensions**. This is what enables stacking 12 identical transformer blocks without any reshape operations between them.

---

## GPT-2 Model Variants

| Model | Parameters | Layers | d\_model |
|-------|-----------|--------|---------|
| GPT-2 Small | 117M → **124M** | 12 | 768 |
| GPT-2 Medium | 345M | 24 | 1024 |
| GPT-2 Large | 762M | 36 | 1280 |
| GPT-2 XL | 1542M | 48 | 1600 |

This series builds GPT-2 Small. OpenAI made GPT-2 weights public — all four sizes. GPT-3 and GPT-4 weights have not been released.

---

## GPT\_CONFIG\_124M

```python
GPT_CONFIG_124M = {
    "vocab_size":     50257,   # subwords in BPE vocabulary
    "context_length": 1024,    # max tokens the model can process at once
    "emb_dim":        768,     # embedding dimension (d_model) throughout the model
    "n_heads":        12,      # attention heads per block (head_dim = 768/12 = 64)
    "n_layers":       12,      # number of transformer blocks stacked
    "drop_rate":      0.1,     # 10% dropout during training
    "qkv_bias":       False    # no bias in Q/K/V linear layers
}
```

All architectural decisions in GPT-2 small follow from these 7 numbers.

---

## The Embedding Pipeline

```python
# Token embedding — lookup table [50257, 768]
self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
# Every token ID → one 768-dim row. Initialized from Gaussian, trained via backprop.

# "Every effort moves you" → token IDs [6109, 3626, 6100, 345]
# 6109 → [0.2961, ..., 0.4604]   # Every
# 3626 → [0.2238, ..., 0.7598]   # effort
# 6100 → [0.6945, ..., 0.5963]   # moves
#  345 → [0.0890, ..., 0.5833]   # you

# Positional embedding — lookup table [1024, 768]
self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
# Row i = learned position vector for position i.
# Here max we look at 1024 tokens to predict the next word.

# Combined input to first transformer block:
input_embeddings = token_embeddings + pos_embeddings
# shape: [batch, num_tokens, 768]
```

---

## The DummyGPTModel

```python
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg): super().__init__()
    def forward(self, x): return x            # identity — placeholder

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5): super().__init__()
    def forward(self, x): return x            # identity — placeholder

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb    = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb    = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb   = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)                        # [b, seq_len, 768]
        pos_embeds = self.pos_emb(torch.arange(seq_len))         # [seq_len, 768]
        x = self.drop_emb(tok_embeds + pos_embeds)               # [b, seq_len, 768]
        x = self.trf_blocks(x)                                   # [b, seq_len, 768]
        x = self.final_norm(x)                                   # [b, seq_len, 768]
        return self.out_head(x)                                  # [b, seq_len, 50257]

# Test
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)

batch = torch.tensor([[6109, 3626, 6100,  345],    # Every effort moves you
                       [6109, 1110, 6622,  257]])   # Every day holds a

logits = model(batch)
print("Output shape:", logits.shape)
# Output shape: torch.Size([2, 4, 50257])
# 2 text samples × 4 tokens × 50257 vocabulary logits

print(logits)
# tensor([[[-1.2034,  0.3201, -0.7130,  ..., -1.5548, -0.2390, -0.4667],
#          [-0.1192,  0.4539, -0.4432,  ...,  0.2392,  1.3469,  1.2430],
#          [ 0.5307,  1.6720, -0.4695,  ...,  1.1966,  0.0111,  0.5835],
#          [ 0.0139,  1.6755, -0.3388,  ...,  1.1586, -0.0435, -1.0400]],
#         [[-1.0908,  0.1798, -0.9484,  ..., -1.6047,  0.2439, -0.4530],
#          ...]])
# Each row = 50257 logits — highest logit = model's predicted next token
```

---

## Shape Tracking

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input token IDs | `[2, 4]` | 2 samples, 4 tokens each |
| tok\_emb output | `[2, 4, 768]` | each ID → 768-dim vector |
| pos\_emb output | `[4, 768]` | position 0–3, broadcast across batch |
| tok + pos | `[2, 4, 768]` | combined input representation |
| Through 12 trf\_blocks | `[2, 4, 768]` | shape preserved at every block |
| out\_head output | `[2, 4, 50257]` | 768 → 50257 — one logit per vocab token |

---

## Key Insight

> The DummyGPTModel is not a toy — it is an architectural proof. By using placeholders that return their input unchanged, it confirms that every shape is correct end-to-end before any real computation is added. The output `[2, 4, 50257]` is the correct shape regardless of what the transformer blocks do internally. When real blocks replace the placeholders in Topics 14–17, the shape will be identical and the values will be meaningful.

---

## Research Connection

This topic implements the architecture from **Radford et al. (2019) — GPT-2: Language Models are Unsupervised Multitask Learners**. The `GPT_CONFIG_124M` dictionary encodes every key architectural decision from Section 2 of that paper directly in code. The transformer block structure follows **Vaswani et al. (2017)** with one modification: Pre-LayerNorm rather than Post-LayerNorm. Shortcut connections follow **He et al. (2016) — Deep Residual Learning for Image Recognition**.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — pipeline overview, transformer block structure, GPT-2 variants, GPT_CONFIG_124M, full DummyGPTModel with verified output |
| `LLM_Architecture.ipynb` | Full implementation — GPT_CONFIG_124M, DummyTransformerBlock, DummyLayerNorm, DummyGPTModel, tokenization, output shape [2, 4, 50257] verified |
| `Topic13_BirdsEye_LLM_Architecture.docx` | Complete deep dive — 13 sections, 5-stage pipeline table, transformer block component table, GPT-2 variants table, full config parameter table, embedding pipeline, output logits structure, DummyGPTModel implementation, shape tracking, key observations |

---

## Next Topic

**[Topic 14 → Layer Normalization](../14_layer_norm/README.md)**
*The `DummyLayerNorm` placeholder is replaced with a real implementation. GPT-2 uses Pre-LayerNorm — normalize before attention and before the feed-forward network. Topic 14 derives why Pre-LN leads to more stable training for deep networks.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
