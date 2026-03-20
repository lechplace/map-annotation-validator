"""
patch_extractor.py
------------------
Sliding window po dużym TIFF → auto-labelowanie przez color_detector.
Zapisuje patche do katalogów patches/ok/ i patches/not-ok/.

Użycie:
    python src/patch_extractor.py \
        --tiff dane-mapa/N-34-137-B-d-3.tif \
        --out dane-mapa/patches \
        --patch-size 128 \
        --stride 64 \
        --iou-threshold 0.05 \
        --max-patches 5000
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Dodaj katalog główny projektu do sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.color_detector import classify_patch


def load_tiff(path: str) -> np.ndarray:
    """Wczytaj TIFF jako BGR numpy array (przez OpenCV lub rasterio)."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            # rasterio zwraca (bands, H, W)
            data = src.read()
            if data.shape[0] >= 3:
                # RGB → BGR dla OpenCV
                img = np.stack([data[2], data[1], data[0]], axis=-1)
            elif data.shape[0] == 1:
                img = cv2.cvtColor(data[0], cv2.COLOR_GRAY2BGR)
            else:
                img = np.transpose(data[:3], (1, 2, 0))[:, :, ::-1]
            return img.astype(np.uint8)
    except ImportError:
        # Fallback: OpenCV (może nie obsłużyć GeoTIFF z wieloma pasmami)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Nie można wczytać pliku: {path}")
        return img


def extract_patches(
    tiff_path: str,
    out_dir: str,
    patch_size: int = 128,
    stride: int = 64,
    iou_threshold: float = 0.05,
    max_patches: int = 5000,
) -> dict:
    """
    Wytnij patche z TIFF, sklasyfikuj każdy przez color_detector i zapisz.

    Returns
    -------
    dict z licznikami: {"ok": int, "not_ok": int, "skipped": int}
    """
    img = load_tiff(tiff_path)
    H, W = img.shape[:2]
    print(f"Wczytano TIFF: {W}×{H} px")

    ok_dir = Path(out_dir) / "ok"
    notok_dir = Path(out_dir) / "not-ok"
    ok_dir.mkdir(parents=True, exist_ok=True)
    notok_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "not_ok": 0, "skipped": 0}
    total = 0

    ys = range(0, H - patch_size + 1, stride)
    xs = range(0, W - patch_size + 1, stride)
    pbar = tqdm(total=len(ys) * len(xs), desc="Ekstrakcja patchy")

    for y in ys:
        for x in xs:
            if total >= max_patches:
                pbar.close()
                print(f"Osiągnięto limit {max_patches} patchy.")
                return counts

            patch = img[y : y + patch_size, x : x + patch_size]
            label = classify_patch(patch, iou_threshold=iou_threshold)

            if label == -1:
                counts["skipped"] += 1
            elif label == 0:
                fname = ok_dir / f"patch_{y}_{x}.png"
                cv2.imwrite(str(fname), patch)
                counts["ok"] += 1
                total += 1
            else:
                fname = notok_dir / f"patch_{y}_{x}.png"
                cv2.imwrite(str(fname), patch)
                counts["not_ok"] += 1
                total += 1

            pbar.update(1)

    pbar.close()
    return counts


def main():
    parser = argparse.ArgumentParser(description="Auto-labelowanie patchy z TIFF")
    parser.add_argument("--tiff", default="dane-mapa/N-34-137-B-d-3.tif")
    parser.add_argument("--out", default="dane-mapa/patches")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--iou-threshold", type=float, default=0.05)
    parser.add_argument("--max-patches", type=int, default=5000)
    args = parser.parse_args()

    # Uruchom z katalogu projektu
    project_root = Path(__file__).parent.parent
    tiff_path = project_root / args.tiff
    out_dir = project_root / args.out

    counts = extract_patches(
        str(tiff_path),
        str(out_dir),
        patch_size=args.patch_size,
        stride=args.stride,
        iou_threshold=args.iou_threshold,
        max_patches=args.max_patches,
    )

    print(f"\nWyniki ekstrakcji:")
    print(f"  OK:      {counts['ok']}")
    print(f"  NOT-OK:  {counts['not_ok']}")
    print(f"  Pominięte (brak drzewa): {counts['skipped']}")
    print(f"\nPatche zapisane w: {out_dir}")


if __name__ == "__main__":
    main()
