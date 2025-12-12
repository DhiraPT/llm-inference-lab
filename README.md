# llm-inference-lab

**LLM inference from scratch: A hands-on journey from CPU baseline to optimized CUDA kernels.**

This repository is a **time machine**. Each folder is a snapshot in a learning journey. We begin with raw, unoptimized C++ and iteratively apply systems engineering principles to rediscover *why* modern inference engines are built the way they are.

This project focuses on **single-batch, single-GPU inference of Llama-3.1-8B** to learn kernel-level fundamentals, distinct from vLLM's serving optimizations. The base model is used to isolate inference mechanics from chat template complexity.

## Roadmap

| Phase | The Lesson |
|-------|------------|
| **Phase 1: CPU Baseline** | InferenceState memory architecture, Prefill vs. Decode, Grouped Query Attention (GQA), Tiktoken vocabulary handling, pure FP32 forward pass. |
| **Phase 2: CPU Optimized** | OpenMP thread parallelization, AVX2 SIMD via F16C, FP16 weight quantization. Introduce the Roofline Model. |
| **Phase 3: GPU Naive** | Direct CUDA port using only global memory. Experience why naive ports can underperform CPU on latency. |
| **Phase 4: GPU Optimized** | Shared Memory, Warp Shuffles, Coalescing, Kernel Fusion, Softmax reduction. Target peak memory bandwidth. |
| **Phase 5: KV Cache Quantization** | FP16 KV cache conversion, ring buffer management, compiler heuristics, manual loop unrolling. INT8 quantization strategies. |
| **Phase 6: Benchmarks** | Latency vs. throughput, time-to-first-token, perplexity validation. Profile with nsys/ncu. Focus on Bandwidth Utilization % (SOL DRAM) over raw tokens/sec. |
| **Phase 7: Extensions** | FlashAttention (Kernel Fusion), Mini-PagedAttention, Speculative Decoding, RoPE scaling for 128k context. |

## 🚀 Quick Start

**Prerequisites:**

  * Linux (WSL2 supported)
  * GCC 11+ / Clang 14+
  * Make / CMake 3.10+
  * CUDA Toolkit 12.0+
  * Python 3.10+

```bash
# 1. Clone
git clone https://github.com/DhiraPT/llm-inference-lab.git
cd llm-inference-lab

# 2. Download & Convert Model
# Downloads Llama-3.1-8B (native BF16) and converts to IEEE FP16 binary (.bin) for F16C compatibility
python common/download_model.py --model meta-llama/Llama-3.1-8B --precision fp16

# 3. Build
make -j$(nproc)

# 4. Run CPU baseline
./phase-1-cpu/inference \
    --model data/llama-3.1-8b.fp16.bin \
    --prompt "The meaning of life is"

# 5. Run GPU optimized
./phase-4-gpu-opt/inference \
    --model data/llama-3.1-8b.fp16.bin \
    --prompt "The meaning of life is" \
    --max-tokens 128
```

## 📁 Project Structure

This is a monorepo style lab. Each phase is self-contained but shares utilities in `common/`.

| Folder | Focus |
| :--- | :--- |
| `common/` | Shared Utilities |
| `phase-1-cpu/` | CPU Baseline |
| `phase-2-cpu-opt/` | CPU Optimized |
| `phase-3-gpu-naive/`| GPU Naive |
| `phase-4-gpu-opt/` | GPU Optimized |
| `phase-5-kv-quant/` | KV Cache Quantization |
| `phase-6-bench/` | Benchmarks |
| `phase-7-ext/` | Extensions |

-----

## 🛠️ Profiling Tools

Systematic Profiling. This lab includes recipes for:

  * **Nsight Compute (`ncu`):** To visualize Memory Coalescing, Warp Divergence, and Memory Bandwidth Utilization.
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

## Acknowledgements

  - **Andrew Chan**'s *[Fast LLM Inference From Scratch](https://andrewkchan.dev/posts/yalm.html)* (Dec 2024) provided the foundational methodology and many optimization insights.

## 📜 License

This project is licensed under the [MIT License](LICENSE.md).