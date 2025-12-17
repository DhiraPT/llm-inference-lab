# llm-inference-lab

**A learning-focused LLM inference implementation, from CPU baselines to optimized CUDA kernels.**

This repository is an LLM inference implementation in C++/CUDA, designed as a learning-focused codebase rather than a production engine. We begin with raw, unoptimized C++ and iteratively apply systems engineering principles to rediscover *why* modern inference engines (vLLM, TGI) are built the way they are.

This project focuses on **single-batch, single-GPU inference of Llama-3.1-8B**.

## 🏗️ System Architecture

This lab uses a split architecture similar to production systems (vLLM/TGI):

1.  **Data Organization**:
    - `data/tokenizers/<model_name>/`:HuggingFace tokenizer definitions.
    - `data/models/<model_name>_<precision>/`: Contains `config.json` (architecture) and `model.bin` (weights).

2.  **Runtime**:
    - **Python Frontend**: Handles text encoding/decoding and orchestrates the test harness.
    - **C++ Backend**: Performs the heavy matrix multiplication, attention, and KV cache management.

## 🗺️ Roadmap

| Phase | The Lesson | Implementation Focus | Verification Focus (Test Cases) |
| :--- | :--- | :--- | :--- |
| **Phase 1: CPU Baseline** | **Correctness First.** Implement the Transformer structure in C++. Focus on tensor shapes and the unique Llama-3.1 architecture quirks. | • **Llama-3.1 RoPE** (Scaling Factors) <br>• **GQA** (Broadcasting Logic) <br>• RMSNorm & SwiGLU <br>• BF16 $\to$ FP32 Casting | **Architecture Logic** <br>• `test_layer_0` (End-to-end block) <br>• `test_llama3_rope` (Freq interpolation) <br>• `test_rmsnorm` (Epsilon checks) |
| **Phase 2: CPU Optimized** | **Parallelism.** Understand how threads and SIMD instructions lower latency on general-purpose hardware. | • **OpenMP** Multi-threading <br>• **AVX-512** Intrinsics <br>• BF16 conversion overhead | **Numerical Stability** <br>• `test_odd_shapes` (Tail loop logic) <br>• `test_omp_threading` (Race conditions) <br>• `test_bf16_cast` (Precision limits) |
| **Phase 3: GPU Naive** | **The Memory Wall.** Porting C++ logic to CUDA. Understanding **Arithmetic Intensity**: why moving 16GB of weights for 1 token is the bottleneck. | • **CUDA Mem** (`cudaMalloc`/`memcpy`) <br>• **Global Memory Bandwidth** <br>• Kernel Launch Overhead | **Data Movement** <br>• `test_h2d_d2h` (Pointer arithmetic) <br>• `test_transposition` (Weight layouts) <br>• `test_basic_matmul` (Grid/Block dims) |
| **Phase 4: GPU Optimized** | **Memory Hierarchy.** The core of modern engines. Minimizing global memory access using Shared Memory. | • **Shared Memory Tiling** <br>• **Coalesced Access** <br>• Large Vocab Reductions (128k) | **Kernel Correctness** <br>• `test_tile_boundary` (Vocab edges) <br>• `test_flash_vs_naive` (Shared mem races) <br>• `test_batch_1_vs_32` (Index scaling) |
| **Phase 5: KV Cache** | **Throughput Engineering.** Leveraging GQA (4:1 ratio) and PagedAttention to maximize memory efficiency. | • **PagedAttention** (Virtual Mem) <br>• **Block Tables** <br>• Dynamic Memory Manager | **State Management** <br>• `test_kv_update` (Ring buffer logic) <br>• `test_context_switch` (Memory leaks) <br>• `test_fragmentation` (Block reuse) |
| **Phase 6: Benchmarks** | **Observability.** Moving beyond "feeling" fast to proving it with rigorous metrics. | • Roofline Analysis <br>• Throughput vs Latency <br>• Occupancy Calculations | **Performance Regression** <br>• Ensure P4 > P3 <br>• Validate utilization % matches theory |
| **Phase 7: Extensions** | **State of the Art.** Advanced techniques for production systems. | • Speculative Decoding <br>• FlashAttention <br>• Quantization (W8A16) | **Approximation Quality** <br>• `test_quant_error` (Perplexity drop) <br>• `test_spec_acceptance` (Draft logic) |

## 🚀 Quick Start

**Prerequisites:**

- Linux (WSL2 supported)
- GCC 11+ / Clang 14+
- CMake 3.18+
- CUDA Toolkit 12.0+
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) v0.3.0+
- Hugging Face Account with access to [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
- GPU: NVIDIA Ampere or newer (RTX 30xx+, A100/H100/H200) recommended for native BF16.

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
python common/download_model.py --model meta-llama/Llama-3.1-8B --precision bf16

# 5. Build
cmake -B build -S .
cmake --build build --parallel $(nproc)

# 6. Verify Correctness (Recommended)
# Run the differential test suite. The --engine flag allows testing any phase.
python common/verify.py \
    --engine ./phase-1-cpu/inference \
    --test_suite phase1_cpu \
    --layer 0

# 7. Run CPU baseline
# The Python wrapper handles tokenization, then calls the C++ binary.
python common/inference.py \
    --engine ./phase-1-cpu/inference \
    --tokenizer_dir data/tokenizers/meta-llama_Llama-3.1-8B \
    --model_dir data/models/meta-llama_Llama-3.1-8B_bf16 \
    --prompt "The meaning of life is"

# 8. Run GPU optimized
# We use the same wrapper, but swap the engine to the GPU binary.
python common/inference.py \
    --engine ./phase-4-gpu-opt/inference \
    --tokenizer_dir data/tokenizers/meta-llama_Llama-3.1-8B \
    --model_dir data/models/meta-llama_Llama-3.1-8B_bf16 \
    --prompt "The meaning of life is" \
    --max_tokens 128
```

## 📁 Project Structure

This is a monorepo style lab. Each phase is self-contained but shares utilities in `common/`.

| Folder              | Focus                 |
| :------------------ | :-------------------- |
| `common/`           | Shared Utilities, Verification Harness (Ground Truth) & Profiling Scripts |
| `phase-1-cpu/`      | CPU Baseline (Correctness, GQA Logic, Naive KV)               |
| `phase-2-cpu-opt/`  | CPU Optimized (AVX/OpenMP)                                    |
| `phase-3-gpu-naive/`| GPU Naive (Global Mem Bandwidth & Launch Overhead)            |
| `phase-4-gpu-opt/`  | GPU Optimized (Shared Memory/Kernels)                         |
| `phase-5-kv/`       | Advanced KV Cache (PagedAttention)                            |
| `phase-6-bench/`    | Benchmarks                                                    |
| `phase-7-ext/`      | Extensions (FlashAttn, Speculative Decoding)                  |

## 🛠️ Profiling Tools

Systematic Profiling is integrated into the workflow starting from Phase 3.

- **Nsight Systems (`nsys`):** Used in Phase 3 to visualize kernel concurrency and API overhead.
- **Nsight Compute (`ncu`):** Used in Phase 4 to visualize Memory Coalescing, Warp Divergence, and Memory Bandwidth Utilization.

## 🔬 Reproducibility

The hardware below is provided for reference and reproducibility only.

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

- **Andrew Chan**'s *[Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html)* (Dec 2024)

## 📜 License

This project is licensed under the [MIT License](LICENSE.md).