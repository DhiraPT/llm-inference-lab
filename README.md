# llm-inference-lab

**A hands-on journey from CPU baseline to optimized CUDA kernels.**

This repository is a **time machine**. Each folder is a snapshot in a learning
journey. We begin with raw, unoptimized C++ and iteratively apply systems
engineering principles to rediscover *why* modern inference engines are built
the way they are.

This project focuses on **single-batch, single-GPU inference of Llama-3.1-8B**
to learn kernel-level fundamentals, distinct from vLLM's serving optimizations.
The base model is used to isolate inference mechanics from chat template complexity.

## 🏗️ System Architecture

This lab uses a split architecture similar to production systems (vLLM/TGI):

1.  **Data Organization**:
    - `data/tokenizers/<model_name>/`: Contains tokenizer vocabulary and configuration files.
    - `data/models/<model_name>_<precision>/`: Contains `config.json` (architecture) and `model.bin` (weights).

2.  **Runtime**:
    - **Python Frontend**: Handles text encoding/decoding using HuggingFace Tokenizers.
    - **C++ Backend**: Performs the heavy matrix multiplication and KV cache management.

## 🗺️ Roadmap

| Phase | The Lesson |
|-------|------------|
| **Phase 1: CPU Baseline** | InferenceState memory architecture, Prefill vs. Decode, Grouped Query Attention (GQA), HuggingFace Tokenizers via Python, pure FP32 forward pass. |
| **Phase 2: CPU Optimized** | OpenMP thread parallelization, AVX2 SIMD via F16C, FP16 weight quantization. Introduce the Roofline Model. |
| **Phase 3: GPU Naive** | Direct CUDA port using only global memory. Experience why naive ports can underperform CPU on latency. |
| **Phase 4: GPU Optimized** | Shared Memory, Warp Shuffles, Coalescing, Kernel Fusion, Softmax reduction. Target peak memory bandwidth. |
| **Phase 5: KV Cache Quantization** | FP16 KV cache conversion, ring buffer management, compiler heuristics, manual loop unrolling. INT8 quantization strategies. |
| **Phase 6: Benchmarks** | Latency vs. throughput, time-to-first-token, perplexity validation. Profile with nsys/ncu. Focus on Bandwidth Utilization % (SOL DRAM) over raw tokens/sec. |
| **Phase 7: Extensions** | FlashAttention (Kernel Fusion), Mini-PagedAttention, Speculative Decoding, RoPE scaling for 128k context. |

## 🚀 Quick Start

**Prerequisites:**

- Linux (WSL2 supported)
- GCC 11+ / Clang 14+
- CMake 3.18+
- CUDA Toolkit 12.0+
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) v0.3.0+
- Hugging Face Account with access to [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)

```bash
# 1. Clone
git clone https://github.com/DhiraPT/llm-inference-lab.git
cd llm-inference-lab

# 2. Setup Environment
uv sync
source .venv/bin/activate

# 3. Log in to Hugging Face (Required for Llama 3.1)
hf auth login

# 4. Download & Setup Data
# This creates:
#   data/tokenizers/meta-llama_Llama-3.1-8B/
#   data/models/meta-llama_Llama-3.1-8B_fp16/config.json
#   data/models/meta-llama_Llama-3.1-8B_fp16/model.bin
python common/download_model.py --model meta-llama/Llama-3.1-8B --precision fp16

# 5. Build
cmake -B build -S .
cmake --build build --parallel $(nproc)

# 6. Run CPU baseline
# The Python wrapper handles tokenization, then calls the C++ binary.
python common/inference.py \
    --tokenizer_dir data/tokenizers/meta-llama_Llama-3.1-8B \
    --model_dir data/models/meta-llama_Llama-3.1-8B_fp16 \
    --engine ./phase-1-cpu/inference \
    --prompt "The meaning of life is"

# 7. Run GPU optimized
# We use the same wrapper, but swap the engine to the GPU binary.
python common/inference.py \
    --tokenizer_dir data/tokenizers/meta-llama_Llama-3.1-8B \
    --model_dir data/models/meta-llama_Llama-3.1-8B_fp16 \
    --engine ./phase-4-gpu-opt/inference \
    --prompt "The meaning of life is" \
    --max_tokens 128
```

## 📁 Project Structure

This is a monorepo style lab. Each phase is self-contained but shares utilities in `common/`.

| Folder              | Focus                 |
| :------------------ | :-------------------- |
| `common/`           | Shared Utilities      |
| `phase-1-cpu/`      | CPU Baseline          |
| `phase-2-cpu-opt/`  | CPU Optimized         |
| `phase-3-gpu-naive/`| GPU Naive             |
| `phase-4-gpu-opt/`  | GPU Optimized         |
| `phase-5-kv-quant/` | KV Cache Quantization |
| `phase-6-bench/`    | Benchmarks            |
| `phase-7-ext/`      | Extensions            |

## 🛠️ Profiling Tools

Systematic Profiling. This lab includes recipes for:

- **Nsight Compute (`ncu`):** To visualize Memory Coalescing, Warp Divergence, and Memory Bandwidth Utilization.
- **Nsight Systems (`nsys`):** To visualize kernel concurrency.

## 🔬 Reproducibility

Reference hardware used for the example tokens/sec numbers:

| Component        | Specification                                               |
| ---------------- | ----------------------------------------------------------- |
| **GPU**          | NVIDIA **H200 NVL** (143 GB HBM3e)                          |
| **CUDA Driver**  | 575.57.08                                                   |
| **CUDA Toolkit** | 12.9 (V12.9.86)                                             |
| **CPU**          | Dual-socket **AMD EPYC 9355**, 64 cores / 128 threads total |
| **System RAM**   | ~1.0 TB                                                     |
| **Kernel**       | Linux 6.8.0-88-generic                                      |
| **GCC**          | 13.3.0                                                      |

## Acknowledgements

- **Andrew Chan**'s *[Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html)* (Dec 2024) provided the foundational methodology and many optimization insights.

## 📜 License

This project is licensed under the [MIT License](LICENSE.md).