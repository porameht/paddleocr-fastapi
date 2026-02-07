"""PaddleOCR API Server - deploy on RunPod GPU pod."""

import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import asyncio
import base64
import binascii
import logging
import os
import tempfile
import time
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from paddleocr import PaddleOCR
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB

app = FastAPI(title="PaddleOCR Thai API")

engine = PaddleOCR(
    lang="th",
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
_engine_lock = Lock()


class OCRResult(BaseModel):
    raw_text: str
    regions: list[dict]
    elapsed_ms: float


class OCRBase64Request(BaseModel):
    image_base64: str
    filename: str = "image.png"


def _validate_extension(filename: str) -> str:
    """Return the file suffix if allowed, raise HTTPException otherwise."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return suffix


def _write_temp_file(data: bytes, suffix: str) -> str:
    """Write bytes to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return tmp.name


def _normalize_box(box: object) -> list[float]:
    """Convert a box (numpy array or list) to a plain list of floats."""
    if hasattr(box, "tolist"):
        return box.tolist()
    return list(box)


def _run_ocr(image_path: str) -> OCRResult:
    """Run PaddleOCR on the image and return regions sorted by vertical position."""
    try:
        t0 = time.perf_counter()
        with _engine_lock:
            results = list(engine.predict(image_path))
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        logger.exception("PaddleOCR inference failed")
        raise HTTPException(status_code=500, detail=f"OCR inference failed: {exc}") from exc
    finally:
        os.unlink(image_path)

    if not results:
        return OCRResult(raw_text="", regions=[], elapsed_ms=elapsed_ms)

    data = results[0].json
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    boxes = data.get("rec_boxes", [])

    regions = []
    for text, score, box in zip(texts, scores, boxes):
        box_list = _normalize_box(box)
        avg_y = (box_list[1] + box_list[3]) / 2
        regions.append({"text": text, "confidence": float(score), "box": box_list, "y": avg_y})

    regions.sort(key=lambda r: r["y"])
    raw_text = "\n".join(r["text"] for r in regions)

    return OCRResult(raw_text=raw_text, regions=regions, elapsed_ms=elapsed_ms)


@app.post("/ocr/upload", response_model=OCRResult)
async def ocr_upload(file: UploadFile = File(...)) -> OCRResult:
    """OCR from uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    suffix = _validate_extension(file.filename)
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Max {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    tmp_path = _write_temp_file(data, suffix)
    return await asyncio.to_thread(_run_ocr, tmp_path)


@app.post("/ocr/base64", response_model=OCRResult)
async def ocr_base64(req: OCRBase64Request) -> OCRResult:
    """OCR from base64-encoded image."""
    suffix = _validate_extension(req.filename)
    if len(req.image_base64) > MAX_IMAGE_BYTES * 4 // 3:
        raise HTTPException(status_code=400, detail=f"Payload too large. Max ~{MAX_IMAGE_BYTES // (1024 * 1024)} MB image")
    try:
        img_bytes = base64.b64decode(req.image_base64)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}") from exc
    tmp_path = _write_temp_file(img_bytes, suffix)
    return await asyncio.to_thread(_run_ocr, tmp_path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": "PP-OCRv5 Thai"}


def _cli() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    _cli()
