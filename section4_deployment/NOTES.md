# Section 4 Notes

## What I built

FastAPI service wrapping Qwen2.5-0.5B-Instruct (Q4_K_M GGUF) with three endpoints:
- `GET /health`
- `POST /generate` — full response
- `POST /generate/stream` — token-by-token SSE

The model loads once at startup via `lifespan`. Inference runs in a thread pool (`run_in_executor`) so the async event loop doesn't freeze. I added an `asyncio.Semaphore(1)` because llama.cpp is not thread-safe — without it, concurrent calls crash the process.

## Real numbers (10 concurrent requests, CPU-only)

| Metric              | Value  |
| ------------------- | ------ |
| TTFT (streaming)    | 1.274s |
| Min latency         | 0.94s  |
| Max latency         | 9.16s  |
| Avg latency         | 5.06s  |
| Total wall time     | 9.16s  |

The tok/s drops from 44 to 4.5 across requests because they queue — each one waits for the previous to finish. That's the nature of single-threaded CPU inference.

## Why FastAPI and not vLLM / TGI?

vLLM needs CUDA. TGI is also GPU-first and overkill for a 0.5B model. For CPU-only llama.cpp, FastAPI + llama-cpp-python is the simplest working option with the least dependencies.

## Scaling to 50 concurrent users

At 50 users this setup would be completely saturated. Things I'd add:

- **vLLM on GPU** — the obvious move. PagedAttention handles parallel decoding properly.
- **llama.cpp `--parallel N`** — if staying on CPU, multi-slot batching helps.
- **Multiple replicas + load balancer** — scale horizontally, Nginx in front.
- **Redis cache** — cache responses for repeated identical prompts. Surprisingly effective for support-style apps where many users ask the same thing.
- **Request queue** — something like ARQ to absorb traffic spikes without dropping connections.
