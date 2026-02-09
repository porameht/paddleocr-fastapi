# PaddleOCR Thai API

FastAPI server wrapping PaddleOCR (PP-OCRv5) for Thai text recognition on GPU.

## Quick Start (RunPod / GPU machine)

```bash
git clone <repo-url> && cd paddleocr-api
uv sync
uv run main.py
```

Server runs at `http://localhost:8000`. Docs at `/docs`.

## API

### POST /ocr/upload

Upload an image file:

```bash
curl -X POST http://localhost:8000/ocr/upload \
  -F "file=@document.png"
```

### POST /ocr/base64

Send base64-encoded image:

```bash
curl -X POST http://localhost:8000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "...", "filename": "doc.png"}'
```

### GET /health

Health check.

## Docker

```bash
docker build -t paddleocr-api .
docker run --gpus all -p 8000:8000 paddleocr-api
```
