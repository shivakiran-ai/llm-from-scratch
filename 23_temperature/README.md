<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=0:0D3B6E,100:1A56A0&height=120&text=Topic%2023%20-%20Temperature%20Scaling&fontSize=28&fontColor=ffffff&fontAlignY=55&desc=Decoding%20Strategy%201%20%7C%20Probabilistic%20Sampling%20%7C%20Stage%204%20%7C%20Building%20LLMs%20from%20Scratch%20%7C%20SHIVA%20KIRAN%20DADISHETTY&descSize=13&descAlignY=78"/>

<br/>

[![Topic](https://img.shields.io/badge/Topic-23%20of%2036-0D3B6E?style=for-the-badge)](.)
[![Stage](https://img.shields.io/badge/Stage%204-Pretraining-1A56A0?style=for-the-badge)](.)
[![Also Known As](https://img.shields.io/badge/Also%20Known%20As-Decoding%20Strategy%201-2E75B6?style=for-the-badge)](.)
[![Key Feature](https://img.shields.io/badge/Key%20Feature-logits%20divided%20by%20T%20then%20multinomial-22C55E?style=for-the-badge)](.)

**[← Back to Main Repository](../README.md)**

</div>

---

## Overview

Topic 22 trained a model using `generate_text_simple` — which always picks the highest-probability token (greedy/argmax). Topic 23 introduces the first improvement: **temperature scaling**. Temperature is a fancy term for dividing the logits by a number greater than zero. This changes the shape of the probability distribution and, combined with multinomial sampling, gives the model meaningful variety in text generation.

---

## The Problem — Greedy Decoding Has No Variety

```
Until now: generated token selected corresponding to the largest
probability score among all tokens in the vocabulary.

→ argmax always picks the SAME token given the same logits
→ Deterministic — no variety in generated text
→ 2 techniques to control this randomness:
     Technique 1: Temperature Scaling   ← THIS TOPIC
     Technique 2: Top-k Sampling        ← Topic 24
```

---

## What Is Temperature?

```
Temperature = fancy term for dividing the logits by a number greater than zero.

scaled_logits = logits / temperature
probabilities = torch.softmax(scaled_logits, dim=0)
next_token    = torch.multinomial(probabilities, num_samples=1)

That is the entire mechanism:
  → Replace argmax with multinomial
  → Divide logits by temperature before softmax

Full pipeline (from notes):
logits → ÷ temperature → softmax → multinomial → sampled next token
```

---

## The Softmax Formula — With and Without Temperature

```
Standard softmax (T=1):         e^zi
                          ──────────────
                           Σ e^zj  (j=1..n)

Softmax with temperature:       e^(zi/T)
                          ──────────────────
                           Σ e^(zj/T)  (j=1..n)

T = 1:  same as standard softmax — no change
T < 1:  zi/T LARGER → differences amplified → sharper distribution, less entropy
T > 1:  zi/T SMALLER → differences compressed → flatter distribution, more entropy
```

---

## Multinomial Sampling — The Vocabulary Example

```python
vocab = {
    "closer": 0, "every": 1, "effort": 2, "forward": 3,
    "inches": 4, "moves": 5, "pizza": 6, "toward": 7, "you": 8
}

# LLM given "every effort moves you" → next-token logits:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)
#  forward: 6.75  ← HIGHEST
#  toward:  6.28  ← second
#  closer:  4.51

# Greedy decoding (argmax):
probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
# → "forward"  ← always, deterministic

# Probabilistic sampling (multinomial):
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
# → "forward"  ← same here, but NOT always the same
```

**Sampling 1000 times — verifying the distribution:**

```python
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sampled_tokens(probas)
# 73  x closer
# 0   x every
# 0   x effort
# 582 x forward   ← most likely (highest prob), selected 582/1000 times
# 0   x inches
# 0   x moves
# 0   x pizza
# 343 x toward    ← second (logit 6.28), selected 343/1000 times
# 2   x you
```

---

## Applying Temperature

```python
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
```

| Temperature | Distribution | Behavior | Effect |
|-------------|-------------|----------|--------|
| T = 1 | Same as standard softmax | forward 582/1000 | Baseline — no change |
| T = 0.1 | Sharper / more peaked | forward ≈ 100% | Approaches argmax — conservative |
| T = 5 | Flatter / more uniform | All tokens have similar prob | Creative but unstable — "pizza" 4% |

**What happens at each temperature:**

```
T = 0.1:
  scaled_logits = logits / 0.1  →  [45.1, 8.9, -19.0, 67.5, ...]
  Differences enormously amplified → "forward" (67.5) dominates
  softmax gives "forward" ≈ 1.0, all others ≈ 0.0
  → approaches argmax — "forward" almost every time

T = 5:
  scaled_logits = logits / 5    →  [0.902, 0.178, -0.380, 1.350, ...]
  Differences compressed → all tokens get similar probabilities
  → "pizza" gets ~4% chance → "every effort moves you pizza" becomes possible
  → creative but also unstable/non-sense
```

---

## Entropy and Distribution Shape

```
Less entropy ← ──────────────────────────── → More entropy
 (sharper)      Increase in T with entropy       (flatter)

T < 1: sharper distribution
  → one token dominates
  → conservative, repetitive text

T = 1: standard distribution (baseline)
  → natural probabilistic sampling

T > 1: flatter distribution
  → all tokens have certain probability of being next token
  → creative but unstable (here all tokens have certain value of
    probability of being the next token of prediction)
```

---

## Summary (from notes)

> How temperature scaling can be used to predict token in probabilistic sense. Instead of choosing the next token according to maximum value, we use multinomial probability distribution to sample the next token.

```
BEFORE (Topic 19):  logits → softmax → argmax      → deterministic
AFTER (Topic 23):   logits → ÷T → softmax → multinomial → probabilistic

T → 0:   approaches argmax (conservative)
T = 1:   standard probabilistic sampling (baseline)
T → ∞:   approaches uniform random (pure creativity/chaos)
```

---

## Key Insight

> Temperature is one division operation — `scaled_logits = logits / temperature` — applied before softmax. It does not change which token is most likely; it changes the gap between the most likely and other tokens. At T=0.1 the gap is enormous ("forward" wins almost every time). At T=5 the gap is small ("forward" is still most likely but "pizza" gets 4% of selections). Combined with multinomial sampling (which replaces argmax), this gives direct user control over the creativity-coherence tradeoff at inference time, without changing the model or retraining.

---

## Research Connection

**Ackley, Hinton, and Sejnowski (1985) — Boltzmann Machines** — temperature in probability distributions comes from statistical mechanics. High temperature = more randomness. **Radford et al. (2019) — GPT-2** — temperature=1 as default sampling parameter, now standard in all LLM inference APIs. **Holtzman et al. (2020) — The Curious Case of Neural Text Degeneration** — analyzes both greedy decoding (repetitive) and high-temperature sampling (incoherent); proposes nucleus sampling as improvement.

---

## Files

| File | Description |
|------|-------------|
| `README.md` | This file — greedy decoding problem, temperature formula, multinomial vs argmax, vocab example (forward=582/1000), temperature comparison table, entropy explanation |
| `LLM_Decoding_Strategies_Temperature_Scaling.ipynb` | Full implementation — 9-token vocab, logits [4.51...6.75...], multinomial sampling, softmax_with_temperature, T=[1,0.1,5] comparison, print_sampled_tokens |
| `Topic23_TemperatureScaling.docx` | Complete deep dive — 9 sections, greedy decoding limitation, multinomial mechanism, 1000-sample distribution (582x forward, 343x toward), temperature comparison table, entropy diagrams, Boltzmann/GPT-2/Holtzman paper connections |

---

## Next Topic

**[Topic 24 → Top-k Sampling](../24_topk_sampling/README.md)**
*Temperature allows any token to be sampled. Top-k restricts sampling to only the k most likely tokens — preventing semantically wrong low-probability choices while maintaining diversity.*

---

<div align="center">

*Part of the **Building LLMs from Scratch** series by **SHIVA KIRAN DADISHETTY***

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:1A56A0,100:0D3B6E&height=80&section=footer"/>

</div>
