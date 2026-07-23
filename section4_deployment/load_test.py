"""
Load test: 10 concurrent requests to /generate
Measures: time-to-first-token (TTFT) via /generate/stream and total latency.
"""

import asyncio
import time
import httpx

BASE_URL = "http://127.0.0.1:8001"
PROMPT = "Explain what quantization means for machine learning models in two sentences."
N_CONCURRENT = 10


async def measure_full(client: httpx.AsyncClient, idx: int) -> dict:
    """Measures total latency for /generate (non-streaming)."""
    payload = {"prompt": PROMPT, "max_tokens": 64, "temperature": 0.0}
    t0 = time.perf_counter()
    try:
        response = await client.post(f"{BASE_URL}/generate", json=payload, timeout=300)
        elapsed = time.perf_counter() - t0
        if response.status_code == 200:
            data = response.json()
            tps = data.get("tok_per_sec", "?")
            print(f"  [{idx+1:02d}] OK  total={elapsed:.2f}s | tok/s={tps}")
            return {"idx": idx, "total_s": elapsed, "status": 200}
        else:
            print(f"  [{idx+1:02d}] ERR HTTP {response.status_code} | elapsed={elapsed:.2f}s")
            return {"idx": idx, "total_s": elapsed, "status": response.status_code}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  [{idx+1:02d}] ERR {e}")
        return {"idx": idx, "total_s": elapsed, "status": -1}


async def measure_ttft(client: httpx.AsyncClient) -> float:
    """Measures time-to-first-token for /generate/stream."""
    payload = {"prompt": PROMPT, "max_tokens": 64, "temperature": 0.0}
    t0 = time.perf_counter()
    async with client.stream("POST", f"{BASE_URL}/generate/stream", json=payload, timeout=120) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                ttft = time.perf_counter() - t0
                return ttft
    return -1.0


async def main():
    print(f"Load test — {N_CONCURRENT} concurrent requests to {BASE_URL}/generate")
    print(f"Prompt: {PROMPT!r}\n")

    async with httpx.AsyncClient() as client:
        # Check server is up
        try:
            r = await client.get(f"{BASE_URL}/health", timeout=5)
            print(f"Health check: {r.json()}\n")
        except Exception as e:
            print(f"Server not reachable: {e}")
            return

        # Measure TTFT on a single streaming request first
        print("Measuring time-to-first-token (TTFT) via /generate/stream...")
        ttft = await measure_ttft(client)
        print(f"  TTFT: {ttft:.3f}s\n")

        # 10 concurrent full requests
        print(f"Sending {N_CONCURRENT} concurrent requests to /generate ...")
        tasks = [measure_full(client, i) for i in range(N_CONCURRENT)]
        t_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_wall = time.perf_counter() - t_start

    latencies = [r["total_s"] for r in results]
    print(f"\n{'='*50}")
    print("LOAD TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Concurrent requests : {N_CONCURRENT}")
    print(f"  TTFT (stream)       : {ttft:.3f}s")
    print(f"  Min latency         : {min(latencies):.2f}s")
    print(f"  Max latency         : {max(latencies):.2f}s")
    print(f"  Avg latency         : {sum(latencies)/len(latencies):.2f}s")
    print(f"  Total wall time     : {total_wall:.2f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
