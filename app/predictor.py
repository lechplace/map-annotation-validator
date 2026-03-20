"""
predictor.py
------------
Singleton wrapping model loading i inferencji.
Model ładowany raz przy starcie serwera.

Źródło modelu (w kolejności priorytetu):
  1. ENV MODEL_GCS_URI=gs://bucket/path/model.pt  → pobierz z GCS
  2. ENV MODEL_PATH=/local/path/model.pt           → lokalny plik
  3. Domyślnie: models/best_model.pt obok katalogu app/
"""

import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Katalog projektu (jeden poziom wyżej niż app/)
PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_model_path() -> str:
    gcs_uri = os.getenv("MODEL_GCS_URI")
    if gcs_uri:
        return _download_from_gcs(gcs_uri)

    local = os.getenv("MODEL_PATH")
    if local:
        return local

    default = PROJECT_ROOT / "models" / "best_model.pt"
    return str(default)


def _download_from_gcs(gcs_uri: str) -> str:
    """Pobierz model z GCS do pliku tymczasowego. Zwraca ścieżkę lokalną."""
    from google.cloud import storage  # pip install google-cloud-storage

    # gs://bucket/path/model.pt
    without_prefix = gcs_uri[len("gs://"):]
    bucket_name, blob_path = without_prefix.split("/", 1)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    blob.download_to_filename(tmp.name)
    logger.info(f"Model pobrany z GCS: {gcs_uri} → {tmp.name}")
    return tmp.name


class Predictor:
    """Singleton — inicjalizuj przez Predictor.get()."""

    _instance: "Predictor | None" = None

    def __init__(self):
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.model import load_model

        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        model_path = _resolve_model_path()
        logger.info(f"Ładowanie modelu: {model_path}  (device={self.device})")
        self.model = load_model(model_path, device=self.device)
        logger.info("Model gotowy.")

    @classmethod
    def get(cls) -> "Predictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def run(
        self,
        tiff_bytes: bytes,
        patch_size: int = 128,
        stride: int = 32,
        threshold: float = 0.5,
    ) -> tuple[bytes, bytes]:
        """
        Przyjmuje surowe bajty pliku TIFF.
        Zwraca (heatmap_png_bytes, detections_csv_bytes).
        """
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.inference import run_inference, nms_detections, save_heatmap, save_csv
        from src.patch_extractor import load_tiff

        # Zapisz TIFF do tymczasowego pliku (rasterio wymaga ścieżki)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_tiff:
            tmp_tiff.write(tiff_bytes)
            tmp_tiff_path = tmp_tiff.name

        try:
            img = load_tiff(tmp_tiff_path)
        finally:
            os.unlink(tmp_tiff_path)

        # Inferencja
        prob_map = run_inference(img, self.model, self.device, patch_size, stride)
        detections = nms_detections(prob_map, patch_size, threshold)

        # Heatmapa → bytes
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
            tmp_png_path = tmp_png.name
        try:
            save_heatmap(img, prob_map, tmp_png_path)
            with open(tmp_png_path, "rb") as f:
                heatmap_bytes = f.read()
        finally:
            os.unlink(tmp_png_path)

        # CSV → bytes
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp_csv:
            tmp_csv_path = tmp_csv.name
        try:
            save_csv(detections, tmp_csv_path)
            with open(tmp_csv_path, "rb") as f:
                csv_bytes = f.read()
        finally:
            os.unlink(tmp_csv_path)

        return heatmap_bytes, csv_bytes


def build_zip(heatmap_bytes: bytes, csv_bytes: bytes) -> bytes:
    """Spakuj heatmap.png + detections.csv do archiwum ZIP w pamięci."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("heatmap.png", heatmap_bytes)
        zf.writestr("detections.csv", csv_bytes)
    return buf.getvalue()
