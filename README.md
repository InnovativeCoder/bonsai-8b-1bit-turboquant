# Bonsai-8B 1-Bit with TurboQuant MLX

This repository demonstrates the integration of **TurboQuant-MLX** into the **Prism Bonsai-8B** 1-bit model, optimized for Apple Silicon (Metal). 

We achieve significant VRAM relief by combining 1-bit weight quantization with a layer-adaptive, 3-bit PolarQuant KV cache.

## 🔬 Architecture & Methodology

### 1-Bit Weight Quantization
The model uses **Prism Bonsai-8B**, an 8-billion parameter model quantized to 1-bit. This reduces the total model weight footprint to approximately **~1.1 GB**, making it highly portable for edge devices.

### TurboQuant: Layer-Adaptive KV Cache
To solve the memory bottleneck in long-context inference, we implement a **Layer-Adaptive KV Cache**:
- **First & Last 4 Layers**: Kept in **FP16** (standard `KVCache`) to preserve attention sensitivity in the critical semantic anchoring layers.
- **Middle 24 Layers**: Compressed to **3-bit** using TurboQuant's PolarQuant (Hadamard rotation + Lloyd-Max quantization).

### Fused Metal Kernels
We utilize fused Metal kernels for the `Q @ K^T` operation. These kernels read bit-packed `uint32` storage directly, performing dequantization and the Walsh-Hadamard Transform (WHT) on-the-fly within the GPU threadgroup, eliminating intermediate buffer overhead.

---

## 📊 Benchmarking Results

Tests were conducted on an **Apple M2 Pro (32GB Unified Memory)** using the following configurations:
- **Baseline**: Standard `mlx-lm` FP16 KV Cache.
- **TurboQuant**: Layer-adaptive 3-bit compression with fused attention patch.

| Context Length | Mode | Decode (TPS) | Prefill (s) | KV Cache Mem (MB) |
| :--- | :--- | :--- | :--- | :--- |
| **512** | Standard | 25.54 | 22.5s | 108.0 MB |
| | **TurboQuant** | **35.80 (+40%)** | **1.92s** | **37.4 MB** |
| **1024** | Standard | 26.78 | 3.82s | 180.0 MB |
| | **TurboQuant** | **28.40 (+6%)** | 3.81s | **65.7 MB** |
| **2048** | **Standard** | **24.22** | **7.56s** | 324.0 MB |
| | TurboQuant | 22.72 (-6%) | 7.68s | **122.2 MB** |
| **4096** | **Standard** | **22.75** | **15.61s** | 612.0 MB |
| | TurboQuant | 20.46 (-10%) | 15.91s | **235.2 MB** |

---

## 🔍 Researcher's Analysis

### 1. Memory-to-Compute Tradeoff
The primary benefit of TurboQuant in this setup is the **~2.7x compression ratio** for the KV cache. 
- At smaller context lengths (512 tokens), the fused kernel efficiency provides a significant **40% speedup** over the standard implementation.
- As context grows (2048+ tokens), the **computational overhead** of dequantizing the 3-bit storage starts to slightly outweigh the memory bandwidth savings on the M2 Pro. 

### 2. VRAM Scaling for Long-Context
Standard FP16 cache scales aggressively: a 32k context would require **~4.8 GB** of memory just for the cache. TurboQuant enables the same window with **~1.8 GB**, allowing 8B models to run long-context tasks on 8GB/16GB consumer MacBooks that would otherwise OOM (Out Of Memory).

### 3. Theoretical Throughput
While single-batch latency (TPS) shows a slight dip at high sequences, the total system throughput potential increases because you can fit **higher batch sizes** or **consecutive long-context sessions** in the same VRAM footprint.

---

## 🚀 Getting Started

### Prerequisites
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX (specifically the `PrismML-Eng/mlx` fork for 1-bit support)
- Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`)

### Installation & Run
1. Clone this repository.
2. Ensure the `turboquant_mlx` local package is in your root directory.
3. Run the optimized generation:
```bash
python3 main.py
```

### Reproducing Benchmarks
To run the comparison suite across different context lengths:
```bash
python3 benchmark_turboquant.py
```

---

## 🙏 Credits
- **TurboQuant**: [GitHub Repository](https://github.com/arozanov/turboquant-mlx)
- **MLX Framework**: [Apple Machine Learning](https://github.com/ml-explore/mlx)
- **Prism Bonsai**: [PrismML](https://huggingface.co/prism-ml/Bonsai-8B-mlx-1bit)
