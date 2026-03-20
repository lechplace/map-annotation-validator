"""
main.py
-------
FastAPI app serwująca detekcję błędnie oznaczonych drzew na drogach.

Endpoints:
  GET  /          → info
  GET  /health    → health check (wymagany przez Cloud Run)
  POST /detect    → przyjmuje TIFF, zwraca ZIP z heatmap.png + detections.csv
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.predictor import Predictor, build_zip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_TIFF_SIZE_MB = int(100)  # limit uploadu
ALLOWED_CONTENT_TYPES = {"image/tiff", "image/x-tiff", "application/octet-stream"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wczytaj model raz przy starcie
    logger.info("Ładowanie modelu...")
    Predictor.get()
    logger.info("Serwer gotowy.")
    yield


app = FastAPI(
    title="Map Annotation Validator",
    description="Wykrywa błędnie oznaczone drzewa nachodziące na drogi w mapach GeoTIFF.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["info"])
def root():
    return {
        "service": "map-annotation-validator",
        "version": "1.0.0",
        "usage": "POST /detect z plikiem TIFF → ZIP z heatmap.png i detections.csv",
    }


@app.get("/health", tags=["info"])
def health():
    return {"status": "ok"}


@app.post(
    "/detect",
    tags=["detection"],
    response_class=Response,
    responses={
        200: {
            "content": {"application/zip": {}},
            "description": "ZIP zawierający heatmap.png i detections.csv",
        }
    },
    summary="Wykryj błędne oznaczenia drzew w pliku TIFF",
)
async def detect(
    file: UploadFile = File(..., description="Plik GeoTIFF z mapą"),
    threshold: float = Form(0.5, ge=0.0, le=1.0, description="Próg pewności modelu"),
    stride: int = Form(32, ge=8, le=128, description="Krok sliding window (px)"),
):
    # Walidacja rozmiaru
    tiff_bytes = await file.read()
    size_mb = len(tiff_bytes) / 1024 / 1024
    if size_mb > MAX_TIFF_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Plik za duży: {size_mb:.1f} MB (max {MAX_TIFF_SIZE_MB} MB)",
        )

    if size_mb == 0:
        raise HTTPException(status_code=400, detail="Pusty plik")

    logger.info(
        f"Żądanie /detect: plik={file.filename!r} "
        f"rozmiar={size_mb:.1f}MB threshold={threshold} stride={stride}"
    )

    try:
        predictor = Predictor.get()
        heatmap_bytes, csv_bytes = predictor.run(
            tiff_bytes,
            stride=stride,
            threshold=threshold,
        )
    except Exception as e:
        logger.exception("Błąd inferencji")
        raise HTTPException(status_code=500, detail=f"Błąd przetwarzania: {e}")

    zip_bytes = build_zip(heatmap_bytes, csv_bytes)

    logger.info(f"Odpowiedź: ZIP {len(zip_bytes)/1024:.0f} KB")

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="results.zip"'},
    )
