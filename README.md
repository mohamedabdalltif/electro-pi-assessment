# Electro Pi — AI Engineer Technical Assessment

**Role**: AI Engineer (Mid-Level, 3+ years)

Each section maps to a skill from the job description. I used Python 3.13 in WSL2 (Linux on Windows), with a single shared venv in `section1_livekit/.venv`. All sections are runnable from there.

---

## Quick start

```bash
source section1_livekit/.venv/bin/activate
```

---

## Section 1 — LiveKit Voice Agent

A food delivery support agent built with `livekit-agents`. It exposes two tools the LLM can call: `get_order_status` and `cancel_order`. Tool calls are logged to stdout.

**Task 1.1** — Minimal voice agent with tool calling via `livekit-agents`.

```bash
cd section1_livekit
# Copy .env.example to .env and fill in your keys
cp .env.example .env

# Run in text mode (no microphone required)
python agent.py console --text
```

**Task 1.2 (Bonus)** — Same agent with Deepgram STT swapped in:

```bash
python agent_swapped.py console --text
```

**Gemini Multimodal Live Voice Mode** — Full voice testing via browser for free:

```bash
# Set GOOGLE_API_KEY or GEMINI_API_KEY in .env, then run:
python agent_gemini.py dev
```

**Proof**: `transcript_log.txt` — real live execution showing tool call invocation.
**Write-up**: `NOTES.md`

---

## Section 2 — LangChain RAG Pipeline

RAG over three food delivery domain docs using ONNX embeddings (all-MiniLM-L6-v2), FAISS, and an OpenRouter LLM. Citations are included in answers. Out-of-domain questions trigger a guardrail message instead of a hallucinated answer.

```bash
cd section2_langchain

# First time: download the ONNX embedding model
python download.py

# Run the 3-question evaluation
python run_evaluation.py
```

**Proof**: `evaluation_results.txt`

---

## Section 3 — Quantization

Benchmarks Qwen2.5-0.5B-Instruct as Q8_0 (near full precision) vs Q4_K_M (4-bit), using llama.cpp on CPU. Measures file size, RAM usage, and tokens/sec on 5 fixed prompts.

```bash
cd section3_quantization

# Download both GGUF models (~1.1 GB total)
python download.py

# Run the benchmark
python benchmark.py
```

**Proof**: `benchmark_results.txt`

---

## Section 4 — FastAPI Deployment

FastAPI service wrapping the same Q4_K_M model. Supports full and streaming responses. Inference runs in a thread pool with a semaphore to avoid crashing llama.cpp with concurrent calls.

```bash
cd section4_deployment

# Option A: run directly (reuses section3 model)
uvicorn app:app --host 0.0.0.0 --port 8001

# Option B: Docker (bind-mount the model to skip re-download)
docker build -t quickbite-llm .
docker run -p 8001:8000 \
  -v $(pwd)/../section3_quantization/models:/app/models \
  -e MODEL_PATH=models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  quickbite-llm

# Load test (in a second terminal once server is up)
python load_test.py
```

**Proof**: `load_test_results.txt`

---

## Assumptions and limitations

- **WSL audio**: WSL2 can't see the Windows microphone, so Section 1 uses `console --text`. Full audio works via the LiveKit Playground web UI.
- **No GPU**: Sections 3 and 4 run CPU-only. Q8_0 ends up faster than Q4_K_M on CPU because 4-bit dequantization is more expensive per token — the opposite of what you'd see on GPU.
- **Section 4 concurrency**: Requests queue via semaphore and are processed one at a time. This is a known llama.cpp limitation. For real concurrency you'd need vLLM on GPU or horizontal scaling.
- **OpenRouter**: Sections 1 and 2 use OpenRouter as the API gateway (same interface as OpenAI, different models and free tier).