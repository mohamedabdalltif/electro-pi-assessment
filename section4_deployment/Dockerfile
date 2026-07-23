FROM python:3.11-slim

WORKDIR /app

# Install build tools needed by llama-cpp-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Model is downloaded at startup if not present.
# To pre-supply it, bind-mount: -v ./models:/app/models
ENV MODEL_PATH=models/qwen2.5-0.5b-instruct-q4_k_m.gguf

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
