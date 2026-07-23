"""
Benchmark Qwen2.5-0.5B-Instruct: Q8_0 (near-FP16) vs Q4_K_M (4-bit)
Measures: RAM usage, tokens/sec throughput, output quality on 5 fixed prompts.
"""

import os
import time
import psutil
import gc
from llama_cpp import Llama

MODELS = {
    "Q8_0 (near-FP16)": "models/qwen2.5-0.5b-instruct-q8_0.gguf",
    "Q4_K_M (4-bit)":   "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
}

PROMPTS = [
    "What is the capital of Egypt?",
    "Explain what a large language model is in one sentence.",
    "Write a short Python function that reverses a string.",
    "What are the main trade-offs between model size and accuracy?",
    "Summarize the benefits of quantization in two sentences.",
]

MAX_TOKENS = 128


def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_benchmark(label, model_path):
    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"File:  {model_path}")
    print(f"Size:  {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    print(f"{'='*60}")

    ram_before = get_ram_mb()
    llm = Llama(
        model_path=model_path,
        n_ctx=512,
        verbose=False,
    )
    ram_after = get_ram_mb()
    ram_used = ram_after - ram_before
    print(f"RAM used by model: {ram_used:.0f} MB")

    total_tokens = 0
    total_time = 0.0
    outputs = []

    for i, prompt in enumerate(PROMPTS, 1):
        messages = [{"role": "user", "content": prompt}]
        t0 = time.perf_counter()
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - t0

        answer = response["choices"][0]["message"]["content"].strip()
        tokens = response["usage"]["completion_tokens"]
        tps = tokens / elapsed if elapsed > 0 else 0

        total_tokens += tokens
        total_time += elapsed
        outputs.append((prompt, answer, tps))

        print(f"\nQ{i}: {prompt}")
        print(f"A{i}: {answer[:200]}")
        print(f"    tokens: {tokens} | time: {elapsed:.2f}s | tok/s: {tps:.1f}")

    avg_tps = total_tokens / total_time if total_time > 0 else 0
    print(f"\nAverage throughput: {avg_tps:.1f} tokens/sec")

    del llm
    gc.collect()

    return {
        "label": label,
        "file_size_mb": os.path.getsize(model_path) / 1024 / 1024,
        "ram_mb": ram_used,
        "avg_tps": avg_tps,
        "outputs": outputs,
    }


def print_summary(results):
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Variant':<20} {'Size (MB)':>10} {'RAM (MB)':>10} {'Tok/s':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<20} {r['file_size_mb']:>10.0f} {r['ram_mb']:>10.0f} {r['avg_tps']:>8.1f}")
    print("=" * 70)


if __name__ == "__main__":
    results = []
    for label, path in MODELS.items():
        if not os.path.exists(path):
            print(f"SKIP: {path} not found. Run download.py first.")
            continue
        result = run_benchmark(label, path)
        results.append(result)

    if results:
        print_summary(results)
