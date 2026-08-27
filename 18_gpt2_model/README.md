<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2018%20-%20GPT-2%20124M%20Parameter%20Model&fontSize=26&fontColor=ffffff&fontAlignY=55&desc=Stage%203%20Complete%20%7C%20DummyGPTModel%20Becomes%20GPTModel%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-18%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%203-COMPLETE-22C55E?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-GPTModel-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-163M%20params%20%7C%20124M%20weight%20tying-F59E0B?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topics 14–17 built every real component. Topic 18 assembles them all into `GPTModel` — the complete, functional 124M parameter GPT-2 small architecture. Every `DummyX` placeholder from Topic 13 is replaced. The model produces real output logits of shape `[batch, tokens, 50257]`, where each value represents the probability of that vocabulary token being the next token. **Stage 3 is complete.**

---

## DummyGPTModel → GPTModel

```
TOPIC 13 — DummyGPTModel:
   DummyTransformerBlock(cfg) → returns x unchanged (identity)
   DummyLayerNorm(emb_dim)   → returns x unchanged (identity)
   Output logits: random — no real computation

TOPIC 18 — GPTModel:
   TransformerBlock(cfg) → real 8-step block (Topics 14–17)
   LayerNorm(emb_dim)    → real mean=0, var=1 normalization
   Output logits: meaningful next-token scores

Everything else: identical structure, same GPT_CONFIG_124M, same output shape.
```

---

## The 14-Step Forward Pass — "Every Effort Moves You"

Input: `batch = [[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]]`
("Every effort moves you" / "Every day holds a")

```
Step 1 — Token Embedding:
   Each token ID → 768-dim semantic vector via lookup table [50257, 768]
   Every → [768-dim]   effort → [768-dim]   moves → [768-dim]   you → [768-dim]

Step 2 — Positional Embedding:
   Each position → 768-dim vector via lookup table [1024, 768]
   pos1 [768-dim]  pos2 [768-dim]  pos3 [768-dim]  pos4 [768-dim]
   Context size = max words to predict next word (here = 4)

   IIpembedding = Token embedding + Positional embedding
   x = tok_embeds + pos_embeds   # [2, 4, 768]

Step 3 — Embedding Dropout:
   Randomly turns off some elements to 0 (inverted dropout, active × 1/0.9)

Steps 4–12 — 12 TransformerBlocks (each block runs Steps 5–12):

   Step 5:  LayerNorm 1     → mean=0, variance=1 per token (Pre-LN)
   Step 6:  Masked MHA      → context vectors [768-dim] per token, causal mask
   Step 7:  Dropout         → after attention
   Step 8:  Shortcut 1      → x = x + shortcut (saved before LayerNorm 1)
   Step 9:  LayerNorm 2     → mean=0, variance=1 per token (Pre-LN)
   Step 10: FeedForward     → Linear(768→3072) → GELU → Linear(3072→768)
   Step 11: Dropout         → after FFN
   Step 12: Shortcut 2      → x = x + shortcut (saved before LayerNorm 2)

Step 13 — Final LayerNorm:
   Final normalization after all 12 blocks → mean=0, var=1

Step 14 — Output Head (Linear 768 → 50257):
   Each token's 768-dim vector → 50257 logits
   Each element = probability of being the next token
   "you" row → predicts: "forward"
```

---

## The Complete GPTModel Class

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb    = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])   # 50257×768
        self.pos_emb    = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 1024×768
        self.drop_emb   = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])  # 12 blocks
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)                          # [b, seq_len, 768]
        pos_embeds = self.pos_emb(torch.arange(seq_len,
                                  device=in_idx.device))           # [seq_len, 768]
        x = tok_embeds + pos_embeds  # IIpembedding = tok + pos   # [b, seq_len, 768]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)                                     # [b, seq_len, 768]
        x = self.final_norm(x)
        logits = self.out_head(x)                                  # [b, seq_len, 50257]
        return logits

# ─── Verified test ───────────────────────────────────────────────────────────
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out   = model(batch)

print("Output shape:", out.shape)
# Output shape: torch.Size([2, 4, 50257])

print(out)
# tensor([[[ 0.3613,  0.4222, -0.0711,  ...,  0.3483,  0.4661, -0.2838],
#          [-0.1792, -0.5660, -0.9485,  ...,  0.0477,  0.5181, -0.3168],
#          [ 0.7120,  0.0332,  0.1085,  ...,  0.1018, -0.4327, -0.2553],
#          [-1.0076,  0.3418, -0.1190,  ...,  0.7195,  0.4023,  0.0532]],
#         [[-0.2564,  0.0900,  0.0335,  ...,  0.2659,  0.4454, -0.6806],
#          ...]])
```

---

## Shape Tracking

| Layer / Step | Shape | Description |
|---|---|---|
| Input token IDs | `[2, 4]` | batch=2, seq\_len=4 |
| tok\_embeds | `[2, 4, 768]` | Token embedding — semantic meaning |
| pos\_embeds | `[4, 768]` | Positional embedding — broadcast across batch |
| tok + pos (IIpembedding) | `[2, 4, 768]` | Combined input representation |
| After drop\_emb | `[2, 4, 768]` | Embedding dropout |
| Through 12 trf\_blocks | `[2, 4, 768]` | Each block preserves shape (Steps 5–12) |
| After final\_norm | `[2, 4, 768]` | Final LayerNorm |
| logits = out\_head | `[2, 4, 50257]` | 768→50257 — one logit per vocab token |

---

## Parameter Count — 163M Total, 124M with Weight Tying

```python
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
# Total number of parameters: 163,009,536

# Weight tying: tok_emb and out_head have IDENTICAL shapes [50257, 768]
print("Token embedding layer shape:", model.tok_emb.weight.shape)
# Token embedding layer shape: torch.Size([50257, 768])
print("Output layer shape:", model.out_head.weight.shape)
# Output layer shape: torch.Size([50257, 768])

# Original GPT-2 reuses tok_emb weights for out_head (weight tying)
# → removes 38,597,376 duplicate parameters
total_params_gpt2 = total_params - sum(
    p.numel() for p in model.out_head.parameters())
print(f"Parameters with weight tying: {total_params_gpt2:,}")
# Number of trainable parameters considering weight tying: 124,412,160

# 163,009,536 - 38,597,376 = 124,412,160 ≈ 124M  ← the "124M" figure
```

> **Weight tying reduces memory but our implementation uses separate layers** — in practice this gives better training and model performance. Modern LLMs (LLaMA, GPT-3, etc.) also use separate layers.

---

## Memory Requirements

```python
total_size_bytes = total_params * 4        # float32 = 4 bytes each
total_size_mb    = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")
# Total size of the model: 621.83 MB

# 163,009,536 × 4 bytes = 652,038,144 bytes = 621.83 MB
# At float16: ~311 MB   At int8: ~156 MB
# GPT-3 (175B params) at float32: ~700 GB
```

---

## Understanding the Output [2, 4, 50257]

```
For "Every effort moves you" (batch 0):
   Row 0 ("Every"):  50257 logits — what word follows "Every"?
   Row 1 ("effort"): 50257 logits — what follows "Every effort"?
   Row 2 ("moves"):  50257 logits — what follows "Every effort moves"?
   Row 3 ("you"):    50257 logits — what follows "Every effort moves you"?
                                    ← Prediction: "forward"

Each of the 50257 values = unnormalized score (logit).
→ Apply softmax → actual probabilities
→ argmax → model's next-word prediction

The goal is for these logits to be converted back into text such that
the last row represents the word the model is supposed to generate.
(here, the word "forward")
```

---

## Key Insight

> The `GPTModel` is not a new architecture — it is the assembly of every component built since Topic 1. Token embedding (Topic 4), positional embedding (Topic 5), masked multi-head attention (Topics 10–12), LayerNorm (Topic 14), FeedForward with GELU (Topic 15), shortcut connections (Topic 16), TransformerBlock (Topic 17) — all plugged together in 14 sequential steps. The output `[2, 4, 50257]` is the correct shape and the logit values are real (not random as in Topic 13) — but they are not yet *meaningful* because the weights are randomly initialized. Stage 4 fixes that.

---

## Research Connection

**Radford et al. (2019) — GPT-2** — this `GPTModel` is the direct implementation of GPT-2 small: 12 transformer blocks, 768 embedding dimensions, 12 attention heads, 50257 vocabulary, 1024 context length, Pre-LayerNorm, GELU activation. The "124M" figure comes from weight tying (our implementation is 163M with separate layers). **Vaswani et al. (2017) — Attention Is All You Need** — decoder-only variant implemented here. **Brown et al. (2020) — GPT-3** — 175B parameters, same architecture, only scale changes.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — DummyGPTModel→GPTModel transformation, 14-step forward pass, complete class, shape tracking, weight tying (163M vs 124M), memory requirements (621.83 MB) |
| `LLM_Architecture.ipynb` | Full implementation — GPTModel class, all components assembled, output `[2, 4, 50257]` verified, total_params=163,009,536, weight tying=124,412,160, memory=621.83 MB |
| `Topic18_GPT2_124M.docx` | Complete deep dive — 10 sections, architecture diagram, 14-step forward pass with actual tensor shapes at each step, complete GPTModel class, shape tracking, parameter count table, weight tying derivation, 621.83 MB memory calculation, GPT-2/GPT-3/Vaswani paper connections |

---

## Next Topic

**[Topic 19 → Next Token Prediction](../19_next_token/README.md)**
*Stage 4 opens. The GPTModel produces logits — Topic 19 converts them into generated text using argmax and autoregressive sampling.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
