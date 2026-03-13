<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0D3B6E,50:1A56A0,100:2E75B6&height=220&section=header&text=Building%20LLMs%20from%20Scratch&fontSize=48&fontColor=ffffff&fontAlignY=40&desc=Every%20component%20of%20a%20Large%20Language%20Model%20%E2%80%94%20implemented%20from%20first%20principles&descAlignY=60&descSize=16"/>

<br/>

[![Author](https://img.shields.io/badge/Author-SHIVA%20KIRAN%20DADISHETTY-0D3B6E?style=for-the-badge)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](.)
[![Status](https://img.shields.io/badge/Status-Active%20Development-22C55E?style=for-the-badge)](.)
[![License](https://img.shields.io/badge/License-MIT-F59E0B?style=for-the-badge)](./LICENSE)

<br/>

[![Topics Complete](https://img.shields.io/badge/Topics%20Complete-02%20%2F%2036-1A56A0?style=for-the-badge)](.)
[![Stage 1](https://img.shields.io/badge/Stage%201%20%7C%20Data%20Pipeline-2%2F6-2E75B6?style=for-the-badge)](.)
[![Stage 2](https://img.shields.io/badge/Stage%202%20%7C%20Attention-0%2F6-6B7280?style=for-the-badge)](.)
[![Stage 3](https://img.shields.io/badge/Stage%203%20%7C%20Architecture-0%2F6-6B7280?style=for-the-badge)](.)
[![Stage 4](https://img.shields.io/badge/Stage%204%20%7C%20Pretraining-0%2F8-6B7280?style=for-the-badge)](.)
[![Stage 5](https://img.shields.io/badge/Stage%205%20%7C%20Finetuning-0%2F10-6B7280?style=for-the-badge)](.)

</div>

---

## About This Repository

This repository documents a complete, ground-up implementation of a Large Language Model — from raw text processing all the way to a fully pretrained and instruction-finetuned GPT-2 model.

Every single component is built from first principles in Python and PyTorch. No `AutoModel.from_pretrained()`. No black-box abstractions. If it exists in the final model, it was understood, designed, and coded here first.

This is not a collection of reproduced tutorials. It is an active engineering and research series — built as part of preparation for doctoral research in machine learning, with the goal of developing the kind of deep, from-scratch understanding that enables genuine research contributions in LLM training and architecture.

---

## How to Use This Repository

Every topic folder contains exactly **three files**:

| File | Purpose |
|------|---------|
| `README.md` | Sharp summary — key concept, what was built, core insight, paper connection. **Start here.** |
| `TopicN_Title.docx` | Complete deep dive — every iteration, all mathematics, full code reference, every design decision explained. |
| `notebook.ipynb` | Fully annotated, runnable Python implementation — all code built from scratch, extensively commented. |

Read the `README.md` first. Run the `.ipynb` to see it in action. Open the `.docx` to go completely deep.

---

## Progress

```
Overall  ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2 / 36 topics complete

Stage 1  Data Pipeline     ████████░░░░░░░░░░░░  2 / 6
Stage 2  Attention         ░░░░░░░░░░░░░░░░░░░░  0 / 6
Stage 3  Architecture      ░░░░░░░░░░░░░░░░░░░░  0 / 6
Stage 4  Pretraining       ░░░░░░░░░░░░░░░░░░░░  0 / 8
Stage 5  Finetuning        ░░░░░░░░░░░░░░░░░░░░  0 / 10
```

*Updated with every new topic. Star this repository ⭐ to follow the journey.*

---

## The Series Roadmap

### 🏗️ Stage 1 — Data Pipeline & Tokenization

> *How raw text becomes the numerical input an LLM can learn from*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 01 | [LLM Tokenizer from Scratch](./01_tokenizer/README.md) | ✅ | Regex splitting · Vocabulary construction · Encode & Decode |
| 02 | [The GPT Tokenizer — Byte Pair Encoding](./02_bpe/README.md) | ✅ | BPE merges · Subword units · GPT-2 vocabulary |
| 03 | [Input-Target Data Pairs — DataLoader](./03_dataloader/README.md) | 🔲 | Sliding window · Context length · Batch construction |
| 04 | [Token Embeddings](./04_token_embeddings/README.md) | 🔲 | Embedding matrix · Lookup · Dimensionality |
| 05 | [Positional Embeddings](./05_positional_embeddings/README.md) | 🔲 | Absolute positional encoding · Why position matters |
| 06 | [Full Data Preprocessing Pipeline](./06_data_pipeline/README.md) | 🔲 | End-to-end: raw text → model-ready tensors |

---

### 🧠 Stage 2 — The Attention Mechanism

> *The core innovation that makes Transformers work — built from the ground up*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 07 | [Introduction to Attention](./07_attention_intro/README.md) | 🔲 | Context vectors · Why attention exists · Intuition |
| 08 | [Simplified Attention — No Trainable Weights](./08_simplified_attention/README.md) | 🔲 | Dot product attention · Softmax scores · Context vectors |
| 09 | [Self-Attention: Keys, Queries & Values](./09_self_attention/README.md) | 🔲 | Q/K/V matrices · Projection · Scaled dot product |
| 10 | [Causal Self-Attention](./10_causal_attention/README.md) | 🔲 | Masking future tokens · Autoregressive property |
| 11 | [Multi-Head Attention Part 1](./11_multihead_p1/README.md) | 🔲 | Parallel heads · Concatenation · Implementation |
| 12 | [Multi-Head Attention Part 2 — Full Mathematics](./12_multihead_p2/README.md) | 🔲 | Complete mathematical derivation · Projection matrices |

---

### 🏛️ Stage 3 — LLM Architecture

> *Assembling all components into the full GPT-2 Transformer architecture*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 13 | [Bird's Eye View of LLM Architecture](./13_architecture_overview/README.md) | 🔲 | Full transformer block · How all components connect |
| 14 | [Layer Normalization](./14_layer_norm/README.md) | 🔲 | Pre-LN vs Post-LN · Numerical stability · Implementation |
| 15 | [GELU Activation Function](./15_gelu/README.md) | 🔲 | GELU vs ReLU · Smooth gradients · Why GPT uses GELU |
| 16 | [Shortcut Connections](./16_shortcuts/README.md) | 🔲 | Residual streams · Gradient flow · Deep network training |
| 17 | [Full Transformer Block](./17_transformer_block/README.md) | 🔲 | Complete block assembly · All components integrated |
| 18 | [GPT-2 — 124M Parameter Model](./18_gpt2_model/README.md) | 🔲 | Full model · Parameter counting · Architecture details |

---

### 🔥 Stage 4 — Pretraining

> *Training the model on raw text — from loss function to a working language model*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 19 | [Next Token Prediction](./19_next_token/README.md) | 🔲 | Autoregressive generation · Inference pipeline |
| 20 | [LLM Loss Function](./20_loss/README.md) | 🔲 | Cross-entropy loss · Perplexity · Training signal |
| 21 | [Evaluation on Real Dataset](./21_evaluation/README.md) | 🔲 | Book corpus · Train/val split · Evaluation loop |
| 22 | [Full Pretraining Loop](./22_pretraining/README.md) | 🔲 | Optimizer · Scheduler · Gradient clipping · Full loop |
| 23 | [Temperature Scaling](./23_temperature/README.md) | 🔲 | Output distribution · Creativity vs accuracy tradeoff |
| 24 | [Top-k Sampling](./24_topk_sampling/README.md) | 🔲 | Decoding strategies · Nucleus sampling |
| 25 | [Saving & Loading Model Weights](./25_checkpointing/README.md) | 🔲 | PyTorch state dicts · Resumable training |
| 26 | [Loading OpenAI GPT-2 Weights](./26_pretrained_weights/README.md) | 🔲 | Weight mapping · Transfer from OpenAI checkpoint |

---

### 🎯 Stage 5 — Finetuning

> *Specialising the pretrained model for classification and instruction following*

| # | Topic | Status | Core Concept |
|---|-------|:------:|--------------|
| 27 | [Introduction to Finetuning](./27_finetuning_intro/README.md) | 🔲 | Classification vs instruction tuning · Core concepts |
| 28 | [Classification Finetuning DataLoaders](./28_clf_dataloader/README.md) | 🔲 | Label encoding · Batching · Data pipeline |
| 29 | [Classification Model Architecture](./29_clf_architecture/README.md) | 🔲 | Head modification · Output layer design |
| 30 | [Spam Classification — End to End](./30_spam_classifier/README.md) | 🔲 | Complete finetuned classifier from scratch |
| 31 | [Instruction Finetuning — Alpaca Format](./31_instruction_intro/README.md) | 🔲 | Prompt templates · Dataset format · Alpaca |
| 32 | [Data Batching for Instruction Tuning](./32_instruction_batching/README.md) | 🔲 | Variable length sequences · Padding · Loss masking |
| 33 | [Instruction Tuning DataLoaders](./33_instruction_dataloader/README.md) | 🔲 | Full instruction data pipeline |
| 34 | [Loading Pretrained Weights for Finetuning](./34_finetune_weights/README.md) | 🔲 | Transfer learning setup · Weight loading |
| 35 | [Instruction Finetuning Training Loop](./35_finetune_loop/README.md) | 🔲 | Loss masking on prompt tokens · Full loop |
| 36 | [Evaluating Finetuned LLM with Ollama](./36_ollama_eval/README.md) | 🔲 | LLM-as-judge evaluation · Quality assessment |

---

## Design Philosophy

Three principles guide every single implementation in this series:

**1. No magic.**
Every function is written by hand. No `.from_pretrained()` until we have built and understood what it loads. Every matrix multiplication, every normalization, every masking operation — coded explicitly and understood completely.

**2. The why before the what.**
Each component begins with the problem it solves, not just what it does. Understanding *why* multi-head attention exists matters more than knowing *how* to call it.

**3. Everything connects to the literature.**
Every major component maps to a specific section of a research paper. This series is a practical, implementation-level companion to:

| Paper | Relevance | Topics |
|-------|-----------|--------|
| Vaswani et al. (2017) — *Attention Is All You Need* | Original Transformer architecture | 07 – 17 |
| Radford et al. (2019) — *GPT-2* | Language model pretraining at scale | 01 – 26 |
| Brown et al. (2020) — *GPT-3* | Few-shot learning, scaling laws | 19 – 26 |
| Hu et al. (2021) — *LoRA* | Parameter-efficient finetuning | 27 – 36 |

---

## Repository Structure

```
llm-from-scratch/
│
├── README.md                          ← You are here
├── LICENSE                            ← MIT License
├── .gitignore                         ← Python · PyTorch · Jupyter
│
├── 01_tokenizer/                      ✅ Complete
│   ├── README.md
│   ├── LLM_Tokenizer.ipynb
│   └── Topic1_LLM_Tokenizer.docx
│
├── 02_bpe/                            ✅ Complete
│   ├── README.md
│   ├── Byte_Pair_Encoder.ipynb
│   └── Topic2_BPE.docx
│
├── 03_dataloader/                     🔲 Upcoming
├── 04_token_embeddings/               🔲 Upcoming
├── 05_positional_embeddings/          🔲 Upcoming
├── 06_data_pipeline/                  🔲 Upcoming
│
│   ... (Topics 07 – 35 follow same structure)
│
├── 36_ollama_eval/                    🔲 Upcoming
│
└── data/
    └── the-verdict.txt
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/shivakiran-ai/llm-from-scratch.git
cd llm-from-scratch

# Install dependencies
pip install torch numpy jupyter matplotlib tiktoken

# Open Topic 01
cd 01_tokenizer
jupyter notebook LLM_Tokenizer.ipynb
```

> **No GPU required** for Stages 1–3. GPU becomes necessary from Topic 22 (Pretraining Loop) onwards.

---

## Prerequisites

| Requirement | Level |
|------------|-------|
| Python | Comfortable — functions, classes, list comprehensions |
| PyTorch | Basic — tensors, autograd (introduced gradually from Stage 2) |
| Linear Algebra | Helpful — matrix multiplication, dot products |
| Neural Networks | Recommended — this series follows `neural-networks-from-scratch` |

> This series is a direct continuation of [`neural-networks-from-scratch`](https://github.com/shivakiran-ai/neural-networks-from-scratch). If you are new to deep learning, start there first.

---

## Related Repository

| Repository | Description |
|-----------|-------------|
| [`neural-networks-from-scratch`](https://github.com/shivakiran-ai/neural-networks-from-scratch) | The foundation series — neurons, backpropagation, optimizers, and regularization built from pure Python and math, before any framework. Complete this before the LLM series. |

---

## About

This series is part of my preparation for PhD research in machine learning, with a focus on large language model training, architecture, and efficient finetuning.

The researchers who make the biggest contributions to LLMs are not those who know how to call the APIs — they are the ones who understand every layer deeply enough to change them. This series is how I am building that understanding, one component at a time.

Every topic is documented with:
- Handwritten notes capturing the thinking process
- A complete Python implementation
- Connections to the original research papers
- Personal observations and insights from building each component

---

## License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

<br/>

*If this repository helped you understand LLMs more deeply, consider giving it a ⭐*

<br/>

**SHIVA KIRAN DADISHETTY**

*Building one component at a time. No shortcuts.*

<br/>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:2E75B6,50:1A56A0,100:0D3B6E&height=120&section=footer"/>

</div>
