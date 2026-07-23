"""
Download Qwen2.5-0.5B-Instruct GGUF files:
  - Q8_0: near full-precision (proxy for FP16 baseline)
  - Q4_K_M: 4-bit quantized (production-friendly)
"""

import os
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODELS_DIR = "models"

files = {
    "Q8_0 (near-FP16 baseline)": "qwen2.5-0.5b-instruct-q8_0.gguf",
    "Q4_K_M (4-bit quantized)":  "qwen2.5-0.5b-instruct-q4_k_m.gguf",
}

os.makedirs(MODELS_DIR, exist_ok=True)

for label, filename in files.items():
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  exists: {dest}")
    else:
        print(f"Downloading {label} ...")
        src = hf_hub_download(repo_id=REPO, filename=filename)
        import shutil
        shutil.copy2(src, dest)
        print(f"  saved: {dest}")

print("\nDone! Models ready in ./models/")
