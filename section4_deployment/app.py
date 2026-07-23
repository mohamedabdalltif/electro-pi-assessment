import os
import time
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from functools import partial
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "../section3_quantization/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
)
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

llm: Llama | None = None
# llama.cpp is not thread-safe — serialize all inference calls
inference_lock = asyncio.Semaphore(1)


def ensure_model():
    import shutil
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        logger.info("Model not found — downloading from HuggingFace...")
        src = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        shutil.copy2(src, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    ensure_model()
    logger.info("Loading model into memory...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=1024, verbose=False)
    logger.info("Model loaded. Server is ready.")
    yield
    llm = None
    logger.info("Server shutting down.")


app = FastAPI(
    title="QuickBite LLM API",
    description="Serves Qwen2.5-0.5B-Instruct (Q4_K_M GGUF) via FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    async with inference_lock:
        response = await loop.run_in_executor(
            None,
            partial(
                llm.create_chat_completion,
                messages=[{"role": "user", "content": req.prompt}],
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ),
        )
    elapsed = time.perf_counter() - t0

    answer = response["choices"][0]["message"]["content"]
    tokens = response["usage"]["completion_tokens"]
    tps = round(tokens / elapsed, 1) if elapsed > 0 else 0

    logger.info(f"generate | tokens={tokens} | elapsed={elapsed:.2f}s | tok/s={tps}")
    return {
        "answer": answer,
        "tokens": tokens,
        "elapsed_s": round(elapsed, 3),
        "tok_per_sec": tps,
    }


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Token-by-token streaming via Server-Sent Events."""
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def token_generator() -> AsyncIterator[str]:
        stream = llm.create_chat_completion(
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0]["delta"]
            token = delta.get("content", "")
            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")
