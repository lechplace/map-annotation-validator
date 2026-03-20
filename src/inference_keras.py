"""
inference_keras.py
------------------
Inferencja modelu Keras na pełnym TIFF.
Sliding window → probability map → heatmapa PNG + CSV z detekcjami.

Użycie:
    uv run python src/inference_keras.py \
        --tiff dane-mapa/N-34-137-B-d-3.tif \
        --model models/best_model_keras.keras \
        --out output \
        --patch-size 128 \
        --stride 32 \
        --threshold 0.5
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import keras

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.patch_extractor import load_tiff
from src.inference import nms_detections, save_heatmap, save_csv  # reużywamy output helpers

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_patch(patch_bgr: np.ndarray, img_size: int) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB))
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def run_inference_keras(
    img_bgr: np.ndarray,
    model: keras.Model,
    patch_size: int = 128,
    stride: int = 32,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Zwraca probability map P(NOT-OK) o tym samym rozmiarze co img_bgr.
    """
    from src.inference import gaussian_kernel

    H, W = img_bgr.shape[:2]
    prob_map   = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    gauss = gaussian_kernel(patch_size)

    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))

    batch_imgs   = []
    batch_coords = []
    pbar = tqdm(total=len(ys) * len(xs), desc="Inferencja (Keras)")

    def flush_batch():
        if not batch_imgs:
            return
        X = np.stack(batch_imgs)            # (B, H, W, 3)
        preds = model.predict(X, verbose=0) # (B, 2)
        probs = preds[:, 1]                 # P(NOT-OK)
        for (y, x), p in zip(batch_coords, probs):
            prob_map[y:y + patch_size, x:x + patch_size]    += p * gauss
            weight_map[y:y + patch_size, x:x + patch_size]  += gauss
        batch_imgs.clear()
        batch_coords.clear()

    for y in ys:
        for x in xs:
            patch = img_bgr[y:y + patch_size, x:x + patch_size]
            batch_imgs.append(preprocess_patch(patch, patch_size))
            batch_coords.append((y, x))
            if len(batch_imgs) >= batch_size:
                flush_batch()
            pbar.update(1)

    flush_batch()
    pbar.close()

    weight_map = np.where(weight_map > 0, weight_map, 1.0)
    prob_map /= weight_map
    return prob_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff",       default="dane-mapa/N-34-137-B-d-3.tif")
    parser.add_argument("--model",      default="models/best_model_keras.keras")
    parser.add_argument("--out",        default="output")
    parser.add_argument("--patch-size", type=int,   default=128)
    parser.add_argument("--stride",     type=int,   default=32)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--batch-size", type=int,   default=64)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    model = keras.saving.load_model(str(project_root / args.model))
    print(f"Model wczytany: {args.model}")

    img = load_tiff(str(project_root / args.tiff))
    print(f"TIFF: {img.shape[1]}×{img.shape[0]} px")

    prob_map = run_inference_keras(img, model, args.patch_size, args.stride, args.batch_size)

    np.save(str(out_dir / "prob_map_keras.npy"), prob_map)
    save_heatmap(img, prob_map, str(out_dir / "heatmap_keras.png"))

    detections = nms_detections(prob_map, args.patch_size, args.threshold)
    save_csv(detections, str(out_dir / "detections_keras.csv"))

    print(f"\nGotowe! Znaleziono {len(detections)} podejrzanych obszarów.")


if __name__ == "__main__":
    main()
