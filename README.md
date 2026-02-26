# 🧪 LLM Playground — Local AI Experiments on Mac M3

A hands-on experimentation lab for learning LLM **Inferencing**, **Fine-Tuning**, and **Weight Manipulation** — running entirely on a MacBook M3 with 16GB unified memory using Apple's MLX framework.

---

## 🖥️ Hardware & Software

| Component | Details |
|---|---|
| **Machine** | MacBook M3, 16GB Unified Memory |
| **Framework** | [Apple MLX](https://github.com/ml-explore/mlx) via `mlx-lm` |
| **Model** | `mlx-community/Meta-Llama-3-8B-Instruct-4bit` (4-bit quantized, ~4.5GB RAM) |
| **Python** | 3.13 (virtual environment `.venv`) |
| **Dataset** | [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) (10K human-expert instructions) |

---

## 📁 Notebook Overview

| # | Notebook | Purpose |
|---|---|---|
| 1 | `1_Local_Inference_MLX.ipynb` | Load and chat with Llama 3 locally using Apple MLX |
| 2 | `2_Finetuning_Data_Prep.ipynb` | Download open-source dataset from Hugging Face and format it for Llama 3 training |
| 3 | `3_Finetuning_LoRA.ipynb` | Run LoRA fine-tuning directly on the Mac using `mlx_lm.lora` |
| 4 | `4_Adapter_Inference.ipynb` | Load a trained LoRA adapter checkpoint and generate text with the custom model |
| 5 | `5_Manual_Weight_Surgery.ipynb` | Inspect raw model tensors, view word embeddings, and perform "brain surgery" |
| 6 | `6_Sensitivity_Analysis.ipynb` | Systematic weight perturbation experiment — the main research notebook |

---

## 🚀 Getting Started

```bash
# Clone/navigate to the project
cd "LLMs Playground"

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install mlx-lm datasets ipykernel jupyterlab

# Register the kernel for Jupyter
python3 -m ipykernel install --user --name=llm_playground

# Launch Jupyter (or open in VS Code)
jupyter lab
```

---

## 📖 Experiment Details

### Part 1: Inferencing (Notebook 1)

Loaded Meta's Llama 3 8B Instruct model (4-bit quantized) using Apple's `mlx-lm` library. The model runs natively on the M3's GPU via Metal, using only **5.4 GB** of the 16 GB unified memory.

**Key metrics:**
- Prompt processing: ~93 tokens/sec
- Text generation: ~7 tokens/sec
- Peak memory: 5.409 GB

### Part 2: Fine-Tuning (Notebooks 2 & 3)

**Dataset:** 500 examples from `HuggingFaceH4/no_robots`, formatted into Llama 3's chat template using `tokenizer.apply_chat_template()`.

**Training Configuration:**
```
Method:         LoRA (Low-Rank Adaptation)
Trainable:      0.131% of parameters (10.49M / 8.03B)
Batch size:     1
Learning rate:  1e-5
Iterations:     200 (completed 130 before OOM)
Peak memory:    10.234 GB
```

**Training Loss Curve:**
```
Iter   1:  Val loss  2.791
Iter  10:  Train loss 2.699
Iter  50:  Train loss 2.041
Iter 100:  Train loss 1.890  ← checkpoint saved
Iter 110:  Train loss 1.793  ← lowest achieved
Iter 130:  OOM crash (Insufficient Memory on Metal GPU)
```

**Lesson learned:** The training crashed at iteration 130 due to a long training example exceeding GPU memory. Fix: add `--max-seq-length 512` and `--grad-checkpoint` flags to cap sequence length and enable gradient checkpointing.

### Part 3: Adapter Loading (Notebook 4)

Successfully loaded the iteration-100 LoRA adapter checkpoint (`adapters/`) on top of the base Llama 3 model and ran inference. The adapter is loaded dynamically at runtime — the original model weights remain untouched, and the tiny adapter file mathematically modifies the model's behavior on the fly.

### Part 4: Manual Weight Inspection (Notebook 5)

Explored the internal structure of the Llama 3 model:

- **Model Architecture:** The MLX model has two top-level components: `model.model` (transformer body) and `model.lm_head` (output projection).
- **Embedding Table:** `model.model.embed_tokens.weight` has shape `[128256, 4096]` — mapping 128,256 tokens to 4,096-dimensional vectors in `float16`.
- **Quantized Layers:** Most layers store weights as packed `uint32` integers (8 × 4-bit values per uint32). Real values are computed as: `real_weight = packed_uint32 × scale + bias`.

**Brain surgery experiment:** Zeroed out the entire embedding table using `mx.zeros_like()`, which completely destroyed the model's ability to understand input — proving that the embedding weights ARE the model's vocabulary knowledge.

### Part 5: Quantized Weight Sensitivity Analysis (Notebook 6)

#### Methodology

Systematically subtracted increasing integer values from ALL `uint32` weights in the `lm_head` layer while keeping `scales` and `biases` untouched. After each perturbation, generated a response to the prompt `"What is 2+2?"` and observed the degradation pattern.

#### Results

| Shift | % of Original | Response | Status |
|---|---|---|---|
| `0` | 0% | `"The answer to 2+2 is 4"` | 🟢 Healthy |
| `1` | 0.00000007% | `"The answer to 2+2 is 4"` | 🟢 Healthy |
| `1,000,000` | 0.074% | `"The answer to 2+2 is 4"` | 🟢 Healthy |
| `1,000,001` | 0.074% | `"The answer is 4."` | 🟡 Slightly degraded |
| `1,000,003` | 0.074% | `"The answer is 4 obceLIKELYHonestlyHonestly..."` | 🟠 Mid-sentence collapse |
| `1,000,004` | 0.074% | `"The answer to -…and…and//\{\{—aigtimet"` | 🔴 Collapsing |
| `1,000,005` | 0.074% | `"The answer permalinkizmetizmet..."` | 💀 Dead after 2 words |
| `1,000,010` | 0.074% | `"eskort uranusypyataires..."` | ☠️ Instant gibberish |
| `10,000,000` | 0.74% | `"监听页面_Parms.Cursors..."` | ☠️ Multilingual chaos |
| `100,000,000` | 7.4% | `"anja_Leanrettankaanka..."` | ☠️ Stuck in loop |

#### Key Findings

1. **The tipping point is between shift 1,000,001 and 1,000,005.** The model tolerates perturbations up to ~0.074% of the uint32 value range, then catastrophically fails within a window of just 4 integers.

2. **Degradation is graceful, then sudden.** The model doesn't break all at once — it progressively loses words:
   - First: drops less-confident words (`"to 2+2"` disappears)
   - Then: correct start + mid-sentence collapse
   - Then: only 2 correct words before gibberish
   - Finally: no coherent output at all

3. **Failure modes vary by severity:**
   - **Small corruption:** Model picks wrong tokens from diverse vocabulary regions (Chinese, code, Spanish)
   - **Large corruption:** Model gets stuck in a loop repeating one broken token

4. **Why it works this way:** Each `uint32` packs 8 separate 4-bit weights. Subtracting from the packed integer doesn't cleanly subtract from each mini-weight — it corrupts bit boundaries. At ~1M shift, the corruption crosses enough 4-bit boundaries to flip token probabilities.

5. **Quantized weights cannot be trivially modified with float math.** Multiplying a `uint32` weight by `0.0` casts it to `float32`, breaking the quantized matrix multiplication engine (which expects `uint32`). Modifications must preserve the `uint32` dtype explicitly.

#### Why ~1,000,000 Is the Magic Number

The tipping point isn't arbitrary — it's a direct consequence of how 4-bit quantization packs weights at the bit level.

**4-bit packing layout:** Each `uint32` (32 bits) stores **8 separate 4-bit mini-weights** packed side by side:

```
One uint32 = 32 bits:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ W7   │ W6   │ W5   │ W4   │ W3   │ W2   │ W1   │ W0   │
│ 4bit │ 4bit │ 4bit │ 4bit │ 4bit │ 4bit │ 4bit │ 4bit │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
 bits    bits    bits   bits   bits   bits   bits   bits
 31-28   27-24   23-20  19-16  15-12  11-8   7-4    3-0
```

Each 4-bit weight can represent values **0–15**. The real weight is: `real_value = (4bit_weight × scale) + bias`.

**Borrow propagation across nibble boundaries:** When we subtract from the packed `uint32`, the subtraction doesn't cleanly affect each 4-bit weight independently — borrows propagate across nibble boundaries (just like `100 - 1 = 99` in decimal).

Using one of our actual values as an example:

```
Original:        1,351,726,743  →  hex: 0x5098B7A7
                                        5 0 9 8 B 7 A 7   (8 packed weights)

Subtract 1M:     0x5098B7A7 - 0x000F4240 = 0x50897567
After -1M:                                 5 0 8 9 7 5 6 7

Change per weight:                         0  0 -1 +1 -4 -2 -4  0
```

Even though we subtracted a single number, **borrow propagation** causes different nibbles to change by different amounts. Some change ±1, others ±4, and some don't change at all.

**Why ±4 nibble units is the tolerance limit:**

```
Real-value change = nibble_change × scale
                  = 4 × 0.004
                  = 0.016
```

At a perturbation of ~0.016 per weight, the softmax function (which converts raw scores into token probabilities) barely notices — the "correct" token still wins the probability race.

**Why +5 more breaks it:** When we go from 1,000,000 to 1,000,005:

```
Subtract 1,000,005:  0x5098B7A7 - 0x000F4245 = 0x50897562
After -1,000,005:                                5 0 8 9 7 5 6 2

W0 changed:  7 → 2  (drop of 5 nibble units!)
Real change: 5 × 0.004 = 0.02
```

That extra shift of 5 causes W0 to flip by **5 units** instead of staying stable. Across millions of weights in the `lm_head`, these "extra" nibble flips accumulate past the softmax decision margin, causing the wrong tokens to win.

**Summary of the threshold mechanics:**

| Factor | Value |
|---|---|
| Bits per packed weight | 4 |
| Max value per weight | 15 |
| Scale factor | ~0.004 |
| Max real perturbation tolerated | ~0.016 (≤4 nibble units × scale) |
| Subtraction that stays within 4 units | ≤ ~1,000,000 |
| Subtraction that pushes past 4 units | ≥ ~1,000,005 |

The magic number is the precise subtraction value where borrow propagation across the 4-bit packing boundaries starts flipping individual mini-weights by more than ~4 units, which when multiplied by the scale factor (~0.004) and summed across millions of weights, finally exceeds the softmax decision margin.

---

## 🧠 Concepts Covered

| Concept | What We Learned |
|---|---|
| **Inferencing** | Loading and running a quantized LLM locally using Apple MLX and Metal GPU |
| **Tokenization** | How text is converted to token IDs and formatted using chat templates |
| **Quantization** | Compressing 16-bit weights to 4-bit packed `uint32` to reduce memory usage by 4x |
| **LoRA** | Training only 0.131% of parameters by adding small adapter matrices on top of frozen weights |
| **Adapter Loading** | Dynamically overlaying trained adapter weights onto a base model at runtime |
| **Weight Surgery** | Directly reading, modifying, and zeroing out tensor matrices to observe effects on output |
| **Sensitivity Analysis** | Measuring the exact perturbation threshold where quantized weights cause catastrophic output failure |

---

## 📚 References

- [Apple MLX](https://github.com/ml-explore/mlx) — Machine learning framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM tools built on MLX (inference, fine-tuning, quantization)
- [Meta Llama 3](https://huggingface.co/meta-llama) — The base model family used in all experiments
- [LoRA Paper](https://arxiv.org/abs/2106.09685) — Low-Rank Adaptation of Large Language Models
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) — Quantized model format for efficient inference
- [HuggingFace no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) — 10K human-written instruction dataset
