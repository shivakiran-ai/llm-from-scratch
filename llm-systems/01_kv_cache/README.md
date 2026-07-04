# KV Cache — Complete Interview Reference

> **Topic:** LLM Inference Optimization  
> **Difficulty:** Senior Engineer Level  
> **Papers:** Attention Is All You Need (Vaswani et al., 2017)  
> **Next Topic:** PagedAttention → vLLM  

---

## The One-Line Answer

> KV Cache avoids recomputing Key and Value matrices for past tokens during every decode step — storing them once in memory and reusing them, reducing decode complexity from **O(n²)** to **O(n)**.

---

## 1. Why Does KV Cache Exist? (Start Here)

Inference happens in **2 stages**:

```
PREFILL  →  all input tokens processed in parallel  →  first output token
DECODE   →  one new token produced at a time        →  until EOS
```

**Example:**  
Prompt: `"The sunset is"` → 3 tokens → LLM → `"extremely"` → `"beautiful"` → ...

| Stage   | What happens                                      | Bound         |
|---------|---------------------------------------------------|---------------|
| Prefill | All prompt tokens processed simultaneously        | Compute-bound |
| Decode  | One token generated per step, auto-regressively   | Memory-bound  |

**The problem:** At every decode step, naive attention recomputes K and V for ALL past tokens — wasteful and slow.

---

## 2. The Full Mathematics — Trace Every Dimension

### Prefill Stage: "The sunset is" → 3 tokens

```
Input matrix X:  shape (3, 4)   ← 3 tokens, embedding dim = 4

        Token1 [θ θ θ θ]   ← "The"
X  =    Token2 [θ θ θ θ]   ← "sunset"
        Token3 [θ θ θ θ]   ← "is"
```

Multiply by pretrained weight matrices **(Wq, Wk, Wv are fixed — learned during training)**:

```
Q = X(3,4) × Wq(4,4)  →  Query  matrix Q (3,4)
K = X(3,4) × Wk(4,4)  →  Key    matrix K (3,4)
V = X(3,4) × Wv(4,4)  →  Value  matrix V (3,4)
```

### Compute Attention Scores:

```
Attention scores = Q(3,4) × Kᵀ(4,3)  →  (3,3)

        The  sunset  is
The   [ θ     θ      θ ]   ← how "The" attends to each token
sunset[ θ     θ      θ ]   ← how "sunset" attends to each token
is    [ θ     θ      θ ]   ← how "is" attends to each token
```

### Normalize and Weight Values:

```
Attention weights = softmax( Q·Kᵀ / √dim_keys )   shape: (3,3)

Context matrix   = Attention weights (3,3) × V (3,4)  →  (3,4)

        [θ θ θ θ]   ← enriched "The"
        [θ θ θ θ]   ← enriched "sunset"
        [θ θ θ θ]   ← enriched "is"   ← only THIS row needed for next token
```

### Predict First Token:

```
Last row of context matrix: (1,4)
           ↓
  × Logits/LM-head matrix (4, 50257)   ← vocab size
           ↓
  (1, 50257) → argmax → token ID → "extremely"
```

> **KEY INSIGHT 1:** To predict the next token, we only need the **last row** of the context matrix. The other rows are never used again.

---

## 3. Decode Stage — The Problem Appears

New sequence: `"The sunset is extremely"` → 4 tokens

```
New input X: shape (4,4)

Q = X(4,4) × Wq(4,4)  →  Q (4,4)
K = X(4,4) × Wk(4,4)  →  K (4,4)   ← RECOMPUTING rows 1-3 again!
V = X(4,4) × Wv(4,4)  →  V (4,4)   ← RECOMPUTING rows 1-3 again!
```

**The waste:** Rows 1-3 of K and V were already computed during prefill.  
We are computing the same values over and over for every new token.

As sequence grows to length n:
- Without cache: **O(n²) compute** per token — quadratically increasing cost
- With KV cache: **O(n) compute** per token — linearly increasing cost

---

## 4. The KV Cache Solution — Two Insights

### Insight 1: We only need the last row of context matrix
- To get last row → need attention weights of last token only → shape (1,4)
- To get attention weights → need attention scores of last token → shape (1,4)  
- To get attention scores of last token → need **only Query vector of new token** (1,4) × **entire Key matrix** (n,4)

### Insight 2: Reuse K and V, don't recompute

```
During PREFILL:
  Compute K(3,4) and V(3,4) for "The sunset is"
  CACHE them in GPU HBM memory  ← this is the KV cache

During DECODE (token 4 = "extremely"):
  Compute ONLY:
    Q_new  = x_extremely (1,4) × Wq(4,4)  →  Q_extremely (1,4)
    K_new  = x_extremely (1,4) × Wk(4,4)  →  K_extremely (1,4)
    V_new  = x_extremely (1,4) × Wv(4,4)  →  V_extremely (1,4)

  Append to cache:
    K_cache (3,4) + K_new (1,4)  →  K_full (4,4)
    V_cache (3,4) + V_new (1,4)  →  V_full (4,4)

  Compute attention:
    Q_extremely (1,4) × K_full_transposed (4,4)  →  attention scores (1,4)
    softmax( scores / √dim_keys )                →  attention weights (1,4)
    attention weights (1,4) × V_full (4,4)       →  context vector (1,4)
                                                         ↓
                                               logits → "beautiful"
```

---

## 5. Why NO Query Cache? (Interviewers ALWAYS Ask This)

```
To predict next token I need:
  ① Entire Value   matrix  →  CACHE IT  (reused every decode step)
  ② Entire Key     matrix  →  CACHE IT  (reused every decode step)
  ③ Only Query VECTOR of the NEW token  →  NO CACHE NEEDED
```

**The reason:** Query for the new token is computed fresh each decode step and used **once** — to produce the current attention score. It is never needed again. Caching it would consume memory with zero benefit.

This is why it is called **KV cache** and not **QKV cache**.

---

## 6. HBM Bandwidth Bottleneck (Senior-Level Detail)

| Stage   | Bottleneck              | Why                                                    |
|---------|-------------------------|--------------------------------------------------------|
| Prefill | **Compute-bound**       | All tokens processed in parallel — GPU cores max out   |
| Decode  | **Memory-bound**        | KV cache must transfer from HBM → SRAM every step      |

```
GPU Memory Hierarchy:
  HBM (High Bandwidth Memory)   ← KV cache stored here  ~2-80 TB/s bandwidth
       ↓  transfer every decode step
  SRAM (on-chip compute units)  ← attention computed here
```

**The bottleneck:** GPU compute units sit idle waiting for KV cache to load from HBM. The GPU is not limited by floating-point operations — it is limited by memory bandwidth. This is the fundamental reason why LLM decode is slow and why inference optimization focuses on memory efficiency.

---

## 7. Memory Size Calculation

```
KV Cache memory = 2 × L × H × d × S × B

Where:
  2  = Key and Value (two matrices)
  L  = number of transformer layers
  H  = number of attention heads
  d  = head dimension (embedding_dim / num_heads)
  S  = sequence length (grows during generation)
  B  = bytes per element (2 for FP16, 1 for INT8)
```

**Example — GPT-2 (124M parameters):**
```
L=12, H=12, d=64, S=1024, FP16 (2 bytes)
= 2 × 12 × 12 × 64 × 1024 × 2 = ~37 MB
```

**Example — LLaMA-2 70B:**
```
L=80, H=64, d=128, S=4096, FP16 (2 bytes)
= 2 × 80 × 64 × 128 × 4096 × 2 = ~10.7 GB  ← per request
```

At 100 concurrent requests → **1.07 TB** of KV cache needed. This is why memory is the bottleneck in production LLM serving.

---

## 8. The Dark Side of KV Cache

| Problem                    | Detail                                                                 |
|----------------------------|------------------------------------------------------------------------|
| **Memory consumption**     | Grows linearly with sequence length × batch size                       |
| **HBM bandwidth cost**     | Full KV cache transferred from HBM to SRAM every decode step          |
| **Memory fragmentation**   | Pre-allocating max sequence length wastes memory (solved by PagedAttention) |
| **Multi-request serving**  | Serving 100s of requests simultaneously → KV cache fills GPU VRAM     |

---

## 9. Connection to PagedAttention (Bridge to Round 2)

> **Interviewer will ask:** "So KV cache takes too much memory — how do we solve that?"

**Your answer:**

The naive KV cache allocates one large **contiguous** memory block per request, sized for the maximum possible sequence length. Most of this is wasted — a request using 500 tokens out of a 4096-token allocation wastes 3596 slots. This is **external and internal fragmentation**.

**PagedAttention** (introduced in vLLM) solves this by:
- Dividing KV cache into fixed-size **pages/blocks** (e.g. 16 tokens per block)
- Storing blocks **non-contiguously** in GPU memory
- Maintaining a **block table** mapping logical token positions → physical memory blocks
- Allocating blocks **on demand** — only as tokens are generated

This is analogous to how an OS manages virtual memory with pages.

Result: near-zero memory waste, 2-4× higher throughput for LLM serving.

---

## 10. Quick Reference — Dimension Cheatsheet

```
Sequence length = n,  Embedding dim = d,  Vocab size = V

X input matrix:         (n, d)
Wq, Wk, Wv weights:    (d, d)   ← pretrained, fixed
Q, K, V matrices:       (n, d)
Attention scores:       (n, n)   ← Q(n,d) × Kᵀ(d,n)
Attention weights:      (n, n)   ← after softmax(·/√d)
Context matrix:         (n, d)   ← attention weights × V
Last row (next token):  (1, d)
Logits projection:      (d, V)
Output logits:          (1, V)   ← argmax → next token ID

During DECODE (KV cache active):
  Q_new:              (1, d)   ← only new token
  K_cache + K_new:    (n, d)   ← full history
  V_cache + V_new:    (n, d)   ← full history
  Attention scores:   (1, n)   ← only one query vector
  Context vector:     (1, d)   ← only last row needed
```

---

## 11. Interview Answer Structure (Your Approach)

When asked "Explain KV Cache":

```
1. WHY    → Inference has 2 stages. Decode recomputes K and V wastefully.
2. WHAT   → Cache K and V from prefill. Reuse during decode.
3. HOW    → Trace dimensions. Show append mechanism. Explain no-Q-cache.
4. GAIN   → O(n²) → O(n). Lower inter-token latency.
5. COST   → Memory grows with sequence. HBM bandwidth bottleneck.
6. NEXT   → PagedAttention solves the memory fragmentation problem.
```

---

## 12. Likely Follow-Up Questions

| Question | One-line answer |
|----------|-----------------|
| Why no Query cache? | Q is used once per decode step and discarded — no reuse value |
| What is the memory cost? | 2 × L × H × d × S × B — grows with sequence length |
| Why is decode memory-bound? | KV cache transfers from HBM to SRAM every step — bandwidth limited |
| What is prefill bound by? | Compute — all tokens processed in parallel, GPU cores saturate |
| How does KV cache help latency? | Eliminates redundant matmuls — inter-token latency drops significantly |
| What breaks at scale? | Memory fragmentation — 100s of requests fill VRAM → PagedAttention |
| What does vLLM do differently? | PagedAttention — non-contiguous blocks + block table = near-zero waste |

---

## 13. Your Two Core Insights (Never Forget These)

> **Insight 1:** To predict the next token, I only need the **last row** of the context matrix. The first n-1 rows are computed but never used for token prediction.

> **Insight 2:** To get that last row, I need the **entire K and V matrices** — but I don't need to **recompute** them. I cache K and V during prefill and append one new row per decode step.

These two insights together are the complete logical derivation of KV cache from first principles.

---

*Based on Shiva Kiran Dadishetty's handwritten notes — verified correct against Vaswani et al. (2017)*  
*Next: PagedAttention README → vLLM internals README → Online Softmax README*
