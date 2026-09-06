<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2019%20-%20Next%20Token%20Prediction&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Stage%204%20Opens%20%7C%20Autoregressive%20Inference%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-19%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%204-Pretraining-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-generate__text__simple-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-Greedy%20Decoding%20%7C%20Autoregressive-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topic 18 built `GPTModel` — producing logits of shape `[batch, tokens, 50257]`. Topic 19 answers: **how do we turn those logits into generated text?** The answer is `generate_text_simple` — a 6-step pipeline: encode → forward pass → extract last vector → softmax → argmax → append and repeat. This is **Stage 4 opening**. The model currently outputs gibberish ("Hello, I am Featureiman Byeswickattribute argue") because weights are randomly initialized. Training fixes this.

---

## What an LLM Does

```
An LLM generates tokens given a certain sequence of input tokens.
It generates one output token at a time.

To generate one output token:
   → model is given context_size tokens as input
   → context_size = maximum number of tokens model looks at before predicting
   → GPT-2: context_size = 1024

Output logits: [batch_size, num_tokens, vocab_size]
Each element = probability of being the next token.

1st batch → 1 × 4 × 50257
2 batches → 2 × 4 × 50257
              ↑   ↑   ↑
           batch  num  vocab
           size  toks  size

We look at the final row (last token position) and find the token ID
that gives the maximum value → that word would hopefully be the next word.
```

---

## Autoregressive Generation — The Growing Context

```
The token generated in the previous round is appended to the input
for the next iteration. The input context grows in each iteration.

1st iteration:  [Hello] [,] [I] [am]                  → predicts: "a"
2nd iteration:  [Hello] [,] [I] [am] [a]              → predicts: "model"
3rd iteration:  [Hello] [,] [I] [am] [a] [model]      → predicts: "ready"
...
6th iteration:  [Hello] [,] [I] [am] [a] [model] [ready] [to] [help] [.]
```

---

## The 6-Step Pipeline — Logits to Text

```
Step 1: Encode text input into token IDs
   "Hello, I am" → [15496, 11, 314, 716]

Step 2: GPT model returns [batch, 4, 50257]
   [[-0.2949, ..., -0.8141],   # "Hello"
    [ 1.2199, ..., -0.3599],   # ","
    [ 1.0446, ...,  0.0020],   # "I"
    [-0.4929, ..., -0.6093]]   # "am" ← next-token prediction row

Step 3: Extract the last vector
   logits = logits[:, -1, :]
   → [-0.4929, ..., 2.4812, ..., -0.6093]
   These logits do NOT add up to 1 — not yet probabilities.

Step 4: Convert logits to probabilities using softmax
   probas = torch.softmax(logits, dim=-1)
   → [0.0001, ..., 0.0200, ..., 0.0001]
   Now values sum to 1.

Step 5: Identify index of the largest value (= token ID)
   idx_next = torch.argmax(probas, dim=-1, keepdim=True)
   → position 257 has highest value (0.0200)
   → token ID = 257 → decoded = "a"

Step 6: Append token ID to previous inputs for the next round
   idx = torch.cat((idx, idx_next), dim=1)
   [15496, 11, 314, 716] + [257] → [15496, 11, 314, 716, 257]
   "Hello , I am" + "a" → next iteration context
```

---

## Complete Implementation

```python
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop context if it exceeds context_size
        # E.g. if LLM supports 5 tokens and context is 10 → only last 5 used
        idx_cond = idx[:, -context_size:]

        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on last time step
        # (batch, n_tokens, vocab_size) → (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get idx of vocab entry with highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

# ─── Verified test ────────────────────────────────────────────────────────────
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
# encoded: [15496, 11, 314, 716]

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)
# encoded_tensor.shape: torch.Size([1, 4])

model.eval()   # disable dropout for deterministic inference
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
# Output: tensor([[15496, 11, 314, 716, 27018, 24086, 47843, 30961, 42348, 7267]])
print("Output Shape:", out.shape)
# Output Shape: torch.Size([1, 10])   # 4 original + 6 new tokens

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
# Hello, I am Featureiman Byeswickattribute argue
```

> **Why gibberish?** The model has not been trained. Randomly initialized weights → random logits → random argmax → random tokens. The architecture and generation pipeline are both correct. Training (Topics 20–26) will produce coherent output.

---

## Iteration Trace

| Iter | Context (input) | Predicted | Output IDs |
|------|----------------|-----------|------------|
| 1 | `[15496, 11, 314, 716]` — Hello, I am | `[257]` — "a" | `[15496,11,314,716,257]` |
| 2 | `[..., 257]` — Hello, I am a | `[2746]` — "model" | `[...,257,2746]` |
| 3 | `[..., 2746]` — Hello, I am a model | `[3492]` — "ready" | `[...,2746,3492]` |
| 6 | `[15496,...,3492,284,1037,13]` — Hello, I am a model ready to help | `[.]` | Full sentence |

After 6 iterations (max\_new\_tokens=6): **"Hello, I am a model ready to help."**

---

## The Context Size Constraint

```python
idx_cond = idx[:, -context_size:]
# Crop: only last context_size tokens used

# Example from notes (context_size = 5):
idx = [[10, 23, 45, 67, 89, 123, 56, 78],   # 8 tokens
       [ 9,  8,  7, 65,  4,  3,  2,  1]]

idx_cond = idx[:, -5:]
# → [[67, 89, 123, 56, 78],
#    [ 6,  5,  4,  3,  2]]

# If context_size = 5, we cannot look at 8 tokens before predicting.
# We can only look at the last 5.
# GPT-2: context_size = 1024 — after 1024+ tokens, earlier tokens are cropped.
```

---

## Greedy Decoding — Softmax is Monotonic

```
Softmax is monotonic: it preserves the ORDER of its inputs.
The position with the highest logit = position with highest probability.

Logits:        [-0.49, ..., 2.48, ..., -0.61]  → argmax at position 257
Probabilities: [0.001, ..., 0.02, ..., 0.001]  → argmax at position 257  SAME

→ argmax(logits) = argmax(softmax(logits))
→ Softmax step is technically redundant for greedy decoding.
→ Included to show the full pipeline and build intuition.

This approach = greedy decoding: always picks the most likely next token.
Limitation: deterministic, can be repetitive.
Topics 23 and 24 introduce temperature scaling and top-k sampling
to add variability and creativity.
```

---

## Key Insight

> The 6-step pipeline is the bridge between the GPT architecture (Stage 3) and language generation (Stage 4). The model itself is unchanged — it still produces `[batch, tokens, 50257]` logits. The pipeline converts those logits into text by: extracting the last row (the next-token prediction), finding the highest-scoring vocabulary entry (argmax), and feeding that prediction back as input for the next step. This autoregressive loop runs once per token generated. After training, the same pipeline produces fluent text.

---

## Research Connection

**Radford et al. (2019) — GPT-2** — the autoregressive generation method implemented here is exactly how GPT-2 generates text. Decoder-only transformer, one token at a time, causal masking. **Bengio et al. (2003) — A Neural Probabilistic Language Model** — established that language models learn probability distributions over next tokens. The logit → softmax → probability distribution directly implements this. **Topics 23–24** introduce temperature scaling and top-k sampling as improvements over greedy decoding.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — what LLMs do, autoregressive loop, 6-step pipeline, generate\_text\_simple complete code, iteration trace, context size constraint, greedy decoding |
| `LLM_Architecture.ipynb` | Full implementation — generate\_text\_simple, encode/decode, model.eval(), torch.no\_grad(), output verified: "Hello, I am Featureiman Byeswickattribute argue" |
| `Topic19_NextTokenPrediction.docx` | Complete deep dive — 10 sections, autoregressive diagram, 6-step pipeline diagram, complete function with 6 annotated steps, iteration trace table, greedy decoding explanation, gibberish explained, paper connections |

---

## Next Topic

**[Topic 20 → LLM Loss Function](../20_loss/README.md)**
*Cross-entropy loss measures how wrong the model's predictions are. This is the training signal that drives all 163M parameters toward producing coherent text.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
