<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2024%20-%20Top-k%20Sampling&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Decoding%20Strategy%202%20%7C%20Top-k%20Filtering%20%7C%20Stage%204%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-24%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%204-Pretraining-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-Decoding%20Strategy%202-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-torch.topk%20then%20-inf%20mask%20then%20softmax-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topic 23 introduced temperature scaling and multinomial sampling — but high temperatures allow semantically wrong low-probability tokens like "pizza" to appear. Topic 24 solves this with **top-k sampling**: restrict the sampled tokens to the top-k most likely tokens and exclude all others by masking their logits with `-inf`. Combined with temperature into the upgraded `generate()` function, the model produces creative text that is no longer a memorized training passage.

---

## The Problem — Temperature Allows "Pizza"

```
Temperature alone (T=5) assigns non-zero probability to ALL tokens.
"pizza" has logit -1.89 — very low, but probability ≈ 4% at T=5.
→ "every effort moves you pizza" becomes possible.

Top-k sampling fixes this:
  Restrict sampled tokens to the top-k most likely tokens.
  Exclude all other tokens by masking their probability scores.
  "pizza" → probability = exactly 0.
```

---

## The -inf Mask Mechanism

```
Setting a logit to -inf before softmax gives exactly zero probability.

softmax: p_i = e^zi / Σ e^zj

If zi = -inf:  e^(-inf) = 0
  → token i gets probability exactly 0
  → remaining k tokens share 100% of probability mass
  → softmax renormalizes automatically

By assigning 0 probabilities to the non-top-k positions, we ensure
that the next token is ALWAYS sampled from a top-k position.
```

---

## Step-by-Step Example — k=3

Using the same 9-token vocabulary: closer(0) every(1) effort(2) forward(3) inches(4) moves(5) pizza(6) toward(7) you(8)

```
Logits:   [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]

Top-k (k=3):
  top_logits, top_pos = torch.topk(next_token_logits, 3)
  # Top logits:     tensor([6.7500, 6.2800, 4.5100])
  # Top positions:  tensor([3, 7, 0])
  #   position 3 = "forward"  (6.75) ← highest
  #   position 7 = "toward"   (6.28)
  #   position 0 = "closer"   (4.51)

-inf mask (set all below min top-k = 4.51 to -inf):
  [4.51, -inf, -inf, 6.75, -inf, -inf, -inf, 6.28, -inf]

After softmax:
  [0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000]
  #   closer:  6.15%
  #   forward: 57.75%  ← most likely
  #   toward:  36.10%
  #   pizza:   exactly 0.0  ← eliminated
```

---

## PyTorch Implementation

```python
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
# Top logits: tensor([6.7500, 6.2800, 4.5100])
print("Top positions:", top_pos)
# Top positions: tensor([3, 7, 0])

# Apply -inf mask
new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],  # below min of top-k
    input=torch.tensor(float("-inf")),             # set to -inf
    other=next_token_logits                        # keep top-k logits
)

# Apply softmax
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)
# tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])
```

---

## The Full Pipeline — Top-k + Temperature Combined

From notes:
```
Logits
   ↓
Top-k → -inf mask    (keep only top k logits, set rest to -inf)
   ↓
÷ temperature        (scale remaining logits)
   ↓
Softmax              (convert to probabilities)
   ↓
Multinomial          (sample from restricted distribution)
   ↓
Sampled next token

Because of temperature scaling and top-k sampling — because of
probabilistic instead of deterministic — model will not memorize
the next token prediction. (Always gives you new output)
```

---

## The Final generate() Function

```python
def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):
        # Step 1: Get logits for last time step (same as before)
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Step 2: NEW — filter with top-k
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        # Step 3: NEW — apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs  = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # probabilistic

        # Step 4: Greedy fallback when temperature=0
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # deterministic

        # Step 5: Stop at end-of-sequence token if specified
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

# ─── Test: top_k=25, temperature=1.4 ────────────────────────────────────────
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Every effort moves you stand to work on surprise, a one of us had gone with random-
# ← NOT a memorized passage — model generating new text

# Compare — generate_text_simple (greedy, no temperature):
# "Every effort moves you know," was one of the axioms he laid down across
#  the Sevres and silver of an exquisitely appointed luncheon-table..."
# ← memorized passage from training set
```

---

## Key Insight

> Top-k sampling and temperature scaling solve complementary problems. Temperature alone allows semantically wrong low-probability tokens (pizza=4% at T=5). Top-k alone without temperature is still deterministic within the k candidates. Together: top-k eliminates the semantic garbage, temperature controls how peaked or flat the remaining distribution is, and multinomial sampling introduces variety. The result is text that is creative ("stand to work on surprise") but never nonsensical ("pizza") — and different on every run because the model no longer follows the memorized greedy path.

---

## Research Connection

**Fan et al. (2018) — Hierarchical Neural Story Generation** — introduced top-k sampling. **Holtzman et al. (2020) — The Curious Case of Neural Text Degeneration** — analyzed top-k limitations and proposed top-p (nucleus) sampling. **Radford et al. (2019) — GPT-2** — used top_k=40 as default; token 50256 is `<|endoftext|>` for eos_id.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — pizza problem, -inf mask mechanism, k=3 example with actual values (forward=0.5775, toward=0.3610, closer=0.0615), complete generate() function, top\_k=25 + temperature=1.4 output |
| `LLM_Decoding_Strategies_Top_K_Sampling.ipynb` | Full implementation — torch.topk output ([6.75,6.28,4.51] positions [3,7,0]), -inf masking, topk_probas verified, generate() with eos\_id, output "stand to work on surprise" |
| `Topic24_TopKSampling.docx` | Complete deep dive — 9 sections, pizza problem analysis, -inf mask math (e^-inf=0), vocabulary table with all 4 rows (logits/top-k/mask/softmax), PyTorch implementation, full pipeline diagram, generate() code, Fan/Holtzman/GPT-2 paper connections |

---

## Next Topic

**[Topic 25 → Saving and Loading Model Weights](../25_checkpointing/README.md)**
*Save the trained model with torch.save so it can be reloaded without retraining. Essential before loading OpenAI's pretrained GPT-2 weights in Topic 26.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
