# llm-inference-lab

**LLM inference from scratch: A hands-on journey from CPU baseline to optimized CUDA kernels.**

This repository is a **time machine**. Each folder is a snapshot in a learning journey. We begin with raw, unoptimized C++ and iteratively apply systems engineering principles to rediscover *why* modern inference engines are built the way they are.

## Roadmap

| Phase | The Lesson |
|-------|------------|
| **Phase 1: CPU Baseline** | The raw mechanics: embedding → transformer → logits. |
| **Phase 2: CPU Optimized** | OpenMP + AVX2. We hit the Memory Wall. |
| **Phase 3: GPU Naive** | Why a direct CUDA port can be *slower* than CPU. |
| **Phase 4: GPU Optimized** | Warp reductions, coalescing, shared memory, kernel fusion. |
| **Phase 5: Quantization** | FP16 KV cache, INT8 weights, fighting compiler heuristics. |
| **Phase 6: Benchmarks** | Writing honest benchmarks (latency vs. throughput). |
| **Phase 7: Extensions** | Implementing Mini-PagedAttention & Speculative Decoding. |

## 🚀 Quick Start

**Prerequisites:**

  * Linux (WSL2 supported)
  * GCC 11+ / Clang 14+
  * CUDA Toolkit 12.0+
  * Python 3.10+

```bash
# 1. Clone
git clone https://github.com/DhiraPT/llm-inference-lab.git
cd llm-inference-lab

# 2. Download + convert Llama-3-8B weights
# (Downloads safetensors + tokenizer, outputs raw binary)
python common/download_model.py

# 3. Build all phases
make -j$(nproc)

# 4. Run Phase 1 (CPU baseline)
./phase-1-cpu/inference \
    --model data/llama-3-8b.bin \
    --prompt "The future of AI is"

# 5. Run Phase 4 (GPU optimized)
./phase-4-gpu-opt/inference \
    --model data/llama-3-8b.bin \
    --prompt "The future of AI is" \
    --max-tokens 128 \
    --sampler greedy
```

## 📁 Project Structure

This is a monorepo style lab. Each phase is self-contained but shares utilities in `common/`.

| Folder | Focus | Key Concepts |
| :--- | :--- | :--- |
| `common/` | Shared Utilities | `SafeTensors` parsing, Llama-3 tokenizer (Tiktoken), Matrix classes. |
| `phase-1-cpu` | **CPU Baseline** | RMSNorm, RoPE, SwiGLU, attention, logits. |
| `phase-2-cpu-opt` | **CPU Optimized** | `OpenMP`, `AVX2`. The Roofline Model. |
| `phase-3-gpu-naive`| **GPU Naive** | 1-to-1 port to CUDA Global Memory. The "Naive Port" trap. |
| `phase-4-gpu-opt` | **GPU Optimized** | Shared Memory, Warp Shuffles, Coalescing. |
| `phase-5-quant` | **Quantization** | FP16 KV Cache, INT8 weights, Manual Loop Unrolling. |
| `phase-6-bench` | **Benchmarks** | Latency vs throughput analysis. |
| `phase-7-ext` | **Extensions** | Mini-PagedAttention (Paged KV Cache), Speculative Decoding. |

-----

## 🛠️ Profiling Tools

We don't guess; we measure. This lab includes recipes for:

  * **Nsight Compute (`ncu`):** To visualize Memory Coalescing and Warp Divergence.
  * **Nsight Systems (`nsys`):** To visualize kernel concurrency.

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

## 📜 License

This project is licensed under the [MIT License](LICENSE.md).