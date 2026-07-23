# Section 3 Notes

## What I ran

I benchmarked Qwen2.5-0.5B-Instruct in two GGUF formats using llama.cpp on CPU (WSL2, no GPU):
- **Q8_0** — 8-bit quantized, essentially the full-precision baseline
- **Q4_K_M** — 4-bit K-quant, the most common production format

## Results

| Variant         | File Size | RAM   | Avg tok/s | Quality          |
| --------------- | --------- | ----- | --------- | ---------------- |
| Q8_0 (near-FP16)| 644 MB    | 695 MB| 44.4      | accurate, coherent |
| Q4_K_M (4-bit)  | 469 MB    | 497 MB| 35.5      | slightly shorter   |

One thing that surprised me: Q8_0 is actually faster than Q4_K_M on CPU (44 vs 35 tok/s). This makes sense because 4-bit dequantization requires more arithmetic per token on CPU. On a GPU the situation flips — Q4_K_M would win because it reduces memory bandwidth pressure, which is the bottleneck on GPU.

Size-wise, moving from Q8_0 to Q4_K_M saves about 27% on disk and 29% on RAM — meaningful for edge or memory-constrained deployments.

## When to pick each technique

**GGUF via llama.cpp** — my choice here. Works on any CPU, no CUDA needed, tons of pre-quantized models on HuggingFace. Great for edge, local tools, or any server without a GPU.

**bitsandbytes (NF4/INT8)** — best when you have a GPU and are already in the HuggingFace ecosystem. Tight integration with `transformers` and PEFT makes it the go-to for fine-tuning with quantization. Doesn't run on CPU.

**GPTQ / AWQ** — best for GPU production serving where you need INT4 quality close to FP16. Both use calibration data during quantization which preserves accuracy much better than naive post-training quantization. AWQ in particular is very good. I'd use these when deploying on a GPU inference server (e.g. vLLM + AWQ).
