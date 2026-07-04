<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0D3B6E,50:1A56A0,100:2E75B6&height=180&section=header&text=LLM%20Systems&fontSize=48&fontColor=ffffff&fontAlignY=40&desc=Inference%20%C2%B7%20Serving%20%C2%B7%20Hardware%20%C2%B7%20Distributed%20Training%20%E2%80%94%20from%20first%20principles&descAlignY=62&descSize=15"/>

<br/>

[![Author](https://img.shields.io/badge/Author-SHIVA%20KIRAN%20DADISHETTY-0D3B6E?style=for-the-badge)](.)
[![Part Of](https://img.shields.io/badge/Part%20Of-llm--from--scratch-1A56A0?style=for-the-badge)](../README.md)
[![Status](https://img.shields.io/badge/Status-Active%20Development-22C55E?style=for-the-badge)](.)

<br/>

[![Topic 1](https://img.shields.io/badge/KV%20Cache-COMPLETE-22C55E?style=for-the-badge)](./01_kv_cache/README.md)
[![Topic 2](https://img.shields.io/badge/PagedAttention-IN%20PROGRESS-2E75B6?style=for-the-badge)](./02_paged_attention/README.md)
[![Topic 3](https://img.shields.io/badge/Transformer%20Architecture-UPCOMING-6B7280?style=for-the-badge)](.)
[![Topic 4](https://img.shields.io/badge/MHA%20→%20MLA-UPCOMING-6B7280?style=for-the-badge)](.)

</div>

---

## About This Series

This folder documents the **systems layer** of large language models — how the models built in this repository are actually served, optimized, and scaled in production.

Every topic is derived from first principles using the same approach as the main implementation series:

- Every concept **written on paper by hand** before being documented
- Every matrix dimension **traced explicitly** at each step
- Every insight **connected to its source paper**
- Every explanation structured as **WHY → WHAT → HOW → GAIN → COST → NEXT**

This is not a summary of blog posts. Every word here was derived, not copied.

---

## The Chain — How Every Topic Connects

```
Transformer Architecture  (Topic 3)
        │
        │  produces Q, K, V matrices that grow with sequence length
        ▼
KV Cache  (Topic 1)  ✅
        │
        │  stores K and V to avoid recomputation — O(n²) → O(n)
        │  BUT causes memory fragmentation when serving many requests
        ▼
PagedAttention  (Topic 2)  🔄
        │
        │  non-contiguous KV blocks via block table — near-zero VRAM waste
        │  BUT still limited by the number of KV heads per layer
        ▼
MHA → MQA → GQA → MLA  (Topic 4)  📋
        │
        │  reduces KV heads — cuts cache memory per request significantly
        │  BUT attention IO over long contexts still dominates
        ▼
Online Softmax & FlashAttention  (Topic 5)  📋
        │
        │  fuses softmax passes — reduces HBM reads from O(n²) to O(n)
        │  BUT KV cache is still large in raw bytes
        ▼
KV Cache Quantization INT8 / FP8  (Topic 6)  📋
        │
        │  halves KV cache byte size — minimal quality loss
        │  all individual optimizations now need a unified serving framework
        ▼
vLLM Internals  (Topic 7)  📋
        │
        │  PagedAttention + continuous batching + GPU memory scheduler
        │  single model needs to scale across hundreds of GPUs for training
        ▼
Distributed Training  DDP · TP · PP  (Topic 8)  📋
        │
        │  three parallelism strategies — each a different memory/compute tradeoff
        │  need hardware understanding to use parallelism correctly
        ▼
NCCL · NVLink · InfiniBand · DGX  (Topic 9)  📋

        NVLink 600 GB/s within node · InfiniBand RDMA between nodes · DGX cluster
```

> Each topic is the **solution** to the problem created by the previous topic.  
> Understanding this entire chain end-to-end is what separates a systems engineer from a framework user.

---

## How Each Topic Is Built

```
Step 1  →  Topic studied from first principles — source paper read first
Step 2  →  Full explanation written on paper by hand
Step 3  →  Every matrix dimension traced at each step
Step 4  →  Explanation reviewed as a mock technical screen — gaps identified
Step 5  →  Senior-level details added (HBM bottleneck, memory formulas, etc.)
Step 6  →  README generated from notes, cross-checked against source paper
Step 7  →  Pushed here as a permanent reference
```

---

## The Framework — Every Topic Uses This Structure

```
WHY   →  What problem does this solve? State it before anything else.
WHAT  →  Define it precisely in one sentence.
HOW   →  Trace the mathematics. Write every matrix shape.
GAIN  →  What improves? Quantify where possible.
COST  →  What is the tradeoff? Be honest.
NEXT  →  What new problem does this create? Bridge to the next topic.
```

---

## Topics

### ✅ Topic 1 — KV Cache

**One-line answer:**
> KV Cache avoids recomputing Key and Value matrices for past tokens during every decode step — storing them once in memory and reusing them, reducing decode complexity from **O(n²) to O(n)**.

**The two core insights:**

```
Insight 1 → To predict the next token, only the last row of the context
            matrix is needed. The first n-1 rows are never used again.

Insight 2 → To get that last row, the entire K and V matrices are needed
            — but they do not need to be recomputed. Cache K and V during
            prefill. Append one new row per decode step.
```

**Why no Query cache:**
Q is computed fresh each decode step and used once — to produce the current attention score. It is never needed again. Caching it wastes memory with zero benefit. This is why it is called **KV cache** and not QKV cache.

**HBM bottleneck:**

| Stage   | Bound          | Reason                                                    |
|---------|----------------|-----------------------------------------------------------|
| Prefill | Compute-bound  | All tokens processed in parallel — GPU cores saturate     |
| Decode  | Memory-bound   | KV cache transfers from HBM → SRAM every step             |

**Memory formula:**  `2 × L × H × d × S × B`

| Model | Parameters | KV Cache per Request |
|-------|-----------|---------------------|
| GPT-2 | 124M | ~37 MB |
| LLaMA-2 70B | 70B | ~10.7 GB |
| 100 concurrent requests | — | ~1.07 TB |

**Source paper:** Kwon et al. (2023) — *Efficient Memory Management for Large Language Model Serving with PagedAttention*

**→ [01_kv_cache/README.md](./01_kv_cache/README.md) — Full reference with complete mathematics**

---

### 🔄 Topic 2 — PagedAttention

**One-line answer:**
> PagedAttention eliminates KV cache memory fragmentation by storing KV blocks non-contiguously in GPU memory using a block table — analogous to OS virtual memory paging — enabling near-zero VRAM waste and 2-4× higher serving throughput.

**The problem it solves:**
Naive KV cache pre-allocates one large contiguous memory block per request, sized for the maximum sequence length. A request using 500 tokens out of a 4096-token allocation wastes 3596 slots. At 100 concurrent requests this fragmentation wastes gigabytes of VRAM.

**The mechanism:**
- Fixed-size non-contiguous **pages/blocks** (e.g. 16 tokens per block)
- **Block table** maps logical token positions → physical GPU memory blocks
- Blocks allocated **on demand** — only as tokens are generated
- Analogy: OS virtual memory paging for GPU VRAM

**Source paper:** Kwon et al. (2023) — *Efficient Memory Management for Large Language Model Serving with PagedAttention*

**→ [02_paged_attention/README.md](./02_paged_attention/README.md)** *(in progress)*

---

### 📋 Topic 3 — Transformer Architecture

**One-line answer:**
> The Transformer stacks identical blocks of LayerNorm → Multi-Head Attention → Residual → LayerNorm → FeedForward → Residual, converting token embeddings into contextually enriched representations at every layer.

**Full forward pass shape trace:**
```
Input tokens (n,)
    → Token Embeddings          (n, d_model)
    → + Positional Embeddings   (n, d_model)
    → [× N Transformer Blocks]
        → LayerNorm             (n, d_model)
        → Multi-Head Attention  (n, d_model)
        → + Residual            (n, d_model)
        → LayerNorm             (n, d_model)
        → FeedForward d→4d→d    (n, d_model)
        → + Residual            (n, d_model)
    → Final LayerNorm           (n, d_model)
    → LM Head                   (n, vocab_size)
    → argmax last row           → next token
```

**Source paper:** Vaswani et al. (2017) — *Attention Is All You Need*

**→ [03_transformer_architecture/README.md](./03_transformer_architecture/README.md)** *(upcoming)*

---

### 📋 Topic 4 — MHA → MQA → GQA → MLA

**One-line answer:**
> The attention evolution from MHA to MLA is a series of KV cache memory optimizations — each reducing the number of KV heads while preserving as much model quality as possible.

**The evolution:**

| Architecture | KV Heads | KV Cache Memory | Used In |
|-------------|----------|-----------------|---------|
| MHA | H heads | Full — 1 KV pair per head | GPT-2, original Transformer |
| MQA | 1 head | 1/H of MHA | Early efficient models |
| GQA | G groups | G/H of MHA | Llama 2, Llama 3, Mistral |
| MLA | Low-rank | Much less than GQA | DeepSeek-V2, DeepSeek-V3 |

**Source papers:** Ainslie et al. (2023) — *GQA* · DeepSeek-AI (2024) — *DeepSeek-V2*

**→ [04_attention_evolution/README.md](./04_attention_evolution/README.md)** *(upcoming)*

---

### 📋 Topic 5 — Online Softmax & FlashAttention

**One-line answer:**
> FlashAttention implements online softmax in CUDA with block tiling — computing attention without materializing the full attention matrix in HBM, reducing memory IO from O(n²) to O(n).

**The problem:**
Standard softmax over attention scores requires two passes — one to find the max for numerical stability, one to normalize. This means loading the full (n×n) matrix from HBM twice. For long contexts this is the dominant cost.

**The solution:**
Online softmax fuses both passes into one. FlashAttention tiles the computation so blocks fit in SRAM — the full attention matrix is never written to HBM.

**Source paper:** Dao et al. (2022) — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*

**→ [05_flash_attention/README.md](./05_flash_attention/README.md)** *(upcoming)*

---

### 📋 Topic 6 — KV Cache Quantization

**One-line answer:**
> Quantizing the KV cache from FP16 to INT8 or FP8 halves its memory footprint using scale-and-zero-point mapping — enabling longer context or more concurrent requests at the same VRAM budget.

**How it works:**
Find the min and max of K and V tensors → map the range onto integers [-128, 127] → dequantize before attention computation.

**Hardware:** NVIDIA Blackwell Tensor Cores support FP8 natively — purpose-built for this.

**→ [06_kv_quantization/README.md](./06_kv_quantization/README.md)** *(upcoming)*

---

### 📋 Topic 7 — vLLM Internals

**One-line answer:**
> vLLM combines PagedAttention, continuous batching, and iteration-level scheduling to maximize GPU utilization and throughput for production LLM serving — achieving 2-4× higher throughput vs naive KV cache serving.

**What vLLM combines:**
- PagedAttention for memory efficiency
- Continuous batching — new requests join mid-batch without waiting
- Iteration-level scheduling — decisions made per forward pass
- Preemption — evict low-priority sequences to serve high-priority ones

**→ [07_vllm_internals/README.md](./07_vllm_internals/README.md)** *(upcoming)*

---

### 📋 Topic 8 — Distributed Training

**One-line answer:**
> Three parallelism strategies — Data (DDP), Tensor (TP), and Pipeline (PP) — split model training across hundreds of GPUs, each making a different tradeoff between communication overhead and memory efficiency.

```
Data Parallelism / DDP
  Model replicated on all nodes · unique micro-batch per node
  NCCL All-Reduce averages gradients · only gradients cross the network (~90% savings)

Tensor Parallelism
  Weight matrices column/row split across GPUs within one DGX node
  AllReduce in the forward pass · requires NVLink 600 GB/s bandwidth

Pipeline Parallelism
  Transformer layers partitioned across nodes as pipeline stages
  1F1B scheduling minimizes bubble overhead
```

**→ [08_distributed_training/README.md](./08_distributed_training/README.md)** *(upcoming)*

---

### 📋 Topic 9 — NCCL · NVLink · InfiniBand · DGX

**One-line answer:**
> NVLink connects GPUs within a DGX node at 600 GB/s, InfiniBand with RDMA connects DGX nodes across the data center, and NCCL orchestrates collective operations like All-Reduce across this entire hardware hierarchy.

```
Within DGX server
  8× Blackwell B200 GPUs on NVLink SXM baseboard
  NVSwitch — all-to-all GPU communication at 600 GB/s

Between DGX servers
  InfiniBand with RDMA — bypasses OS networking
  Direct memory access across nodes — 400 Gb/s per port

Data center scale
  Thousands of DGX nodes in fat-tree InfiniBand topology
  NCCL library orchestrates All-Reduce across all nodes
```

**→ [09_hardware_architecture/README.md](./09_hardware_architecture/README.md)** *(upcoming)*

---

## Progress

```
Topic 1  KV Cache                         ✅  Complete
Topic 2  PagedAttention                   🔄  In progress
Topic 3  Transformer Architecture         📋  Upcoming
Topic 4  MHA → MQA → GQA → MLA           📋  Upcoming
Topic 5  Online Softmax & FlashAttention  📋  Upcoming
Topic 6  KV Cache Quantization            📋  Upcoming
Topic 7  vLLM Internals                   📋  Upcoming
Topic 8  Distributed Training             📋  Upcoming
Topic 9  NCCL · NVLink · InfiniBand · DGX 📋  Upcoming
```

---

## Source Papers

| Paper | Authors | Year | Topics |
|-------|---------|------|--------|
| Attention Is All You Need | Vaswani et al. | 2017 | 3, 4 |
| Language Models are Unsupervised Multitask Learners | Radford et al. | 2019 | 3 |
| GQA: Training Generalized Multi-Query Transformer Models | Ainslie et al. | 2023 | 4 |
| DeepSeek-V2: A Strong, Economical, and Efficient MoE LM | DeepSeek-AI | 2024 | 4 |
| Efficient Memory Management for LLM Serving with PagedAttention | Kwon et al. | 2023 | 1, 2, 7 |
| FlashAttention: Fast and Memory-Efficient Exact Attention | Dao et al. | 2022 | 5 |
| FlashAttention-2: Faster Attention with Better Parallelism | Dao et al. | 2023 | 5 |
| LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2021 | 6 |

---

← **[Back to Main Repository](../README.md)**

---

<div align="center">

<br/>

**SHIVA KIRAN DADISHETTY**

*Every concept derived from first principles. Every dimension traced by hand. No shortcuts.*

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:2E75B6,50:1A56A0,100:0D3B6E&height=120&section=footer"/>

</div>
