"""
inference.py
------------
Inferencja modelu na pełnym TIFF.
Sliding window → probability map → heatmapa PNG + CSV z detekcjami.

Użycie:
    python src/inference.py \
        --tiff dane-mapa/N-34-137-B-d-3.tif \
        --model models/best_model.pt \
        --out output \
        --patch-size 128 \
        --stride 32 \
        --threshold 0.5
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import load_model
from src.patch_extractor import load_tiff


# ── Transformacja do inferencji ───────────────────────────────────────────────
INFER_TRANSFORM = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def gaussian_kernel(size: int) -> np.ndarray:
    """2D Gaussian do ważonego składania probability map."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (size / 6) ** 2)
    return (kernel / kernel.max()).astype(np.float32)


@torch.no_grad()
def run_inference(
    img_bgr: np.ndarray,
    model: torch.nn.Module,
    device: str,
    patch_size: int = 128,
    stride: int = 32,
) -> np.ndarray:
    """
    Zwraca probability map P(NOT-OK) o tym samym rozmiarze co img_bgr.
    Używa Gaussian blending na zakładkach.
    """
    H, W = img_bgr.shape[:2]
    prob_map = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    gauss = gaussian_kernel(patch_size)

    model.eval()
    batch_imgs = []
    batch_coords = []
    batch_size = 64

    ys = list(range(0, H - patch_size + 1, stride))
    xs = list(range(0, W - patch_size + 1, stride))

    pbar = tqdm(total=len(ys) * len(xs), desc="Inferencja")

    def flush_batch():
        if not batch_imgs:
            return
        tensor = torch.stack(batch_imgs).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(NOT-OK)
        for (y, x), p in zip(batch_coords, probs):
            prob_map[y:y + patch_size, x:x + patch_size] += p * gauss
            weight_map[y:y + patch_size, x:x + patch_size] += gauss
        batch_imgs.clear()
        batch_coords.clear()

    for y in ys:
        for x in xs:
            patch = img_bgr[y:y + patch_size, x:x + patch_size]
            pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            tensor = INFER_TRANSFORM(pil)
            batch_imgs.append(tensor)
            batch_coords.append((y, x))
            if len(batch_imgs) >= batch_size:
                flush_batch()
            pbar.update(1)

    flush_batch()
    pbar.close()

    # Normalizuj przez sumaryczne wagi
    weight_map = np.where(weight_map > 0, weight_map, 1.0)
    prob_map /= weight_map
    return prob_map


def nms_detections(
    prob_map: np.ndarray,
    patch_size: int,
    threshold: float,
    min_distance: int = 64,
) -> list:
    """
    Prosta NMS: znajduje lokalne maksima powyżej progu.
    Zwraca listę dicts: {x, y, w, h, confidence}.
    """
    from scipy.ndimage import maximum_filter, label

    # Binaryzuj
    binary = (prob_map > threshold).astype(np.uint8)
    labeled, n = label(binary)
    detections = []

    for region_id in range(1, n + 1):
        region_mask = labeled == region_id
        region_prob = np.where(region_mask, prob_map, 0)
        idx = np.unravel_index(np.argmax(region_prob), region_prob.shape)
        cy, cx = idx
        conf = float(prob_map[cy, cx])
        half = patch_size // 2
        detections.append({
            "x": max(cx - half, 0),
            "y": max(cy - half, 0),
            "w": patch_size,
            "h": patch_size,
            "confidence": round(conf, 4),
        })

    # Sortuj po confidence malejąco
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def save_heatmap(img_bgr: np.ndarray, prob_map: np.ndarray, out_path: str, alpha: float = 0.55):
    """Nałóż heatmapę na miniaturę mapy i zapisz do PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Skaluj do rozsądnego rozmiaru (~2000px szerokość)
    H, W = img_bgr.shape[:2]
    scale = min(1.0, 2000 / max(H, W))
    new_W, new_H = int(W * scale), int(H * scale)

    thumb = cv2.resize(img_bgr, (new_W, new_H))
    thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
    prob_resized = cv2.resize(prob_map, (new_W, new_H))

    fig, ax = plt.subplots(figsize=(new_W / 100, new_H / 100), dpi=100)
    ax.imshow(thumb_rgb)
    ax.imshow(prob_resized, cmap="hot", alpha=alpha, vmin=0, vmax=1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Heatmapa zapisana: {out_path}")


def save_csv(detections: list, out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x", "y", "w", "h", "confidence"])
        writer.writeheader()
        writer.writerows(detections)
    print(f"CSV zapisany: {out_path}  ({len(detections)} detekcji)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff", default="dane-mapa/N-34-137-B-d-3.tif")
    parser.add_argument("--model", default="models/best_model.pt")
    parser.add_argument("--out", default="output")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    tiff_path = project_root / args.tiff
    model_path = project_root / args.model
    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # Wczytaj model
    model = load_model(str(model_path), device=device)

    # Wczytaj TIFF
    img = load_tiff(str(tiff_path))
    print(f"TIFF: {img.shape[1]}×{img.shape[0]} px")

    # Inferencja
    prob_map = run_inference(img, model, device, args.patch_size, args.stride)

    # Zapisz surową mapę prawdopodobieństwa
    np.save(str(out_dir / "prob_map.npy"), prob_map)

    # Heatmapa
    save_heatmap(img, prob_map, str(out_dir / "heatmap.png"))

    # Detekcje + CSV
    detections = nms_detections(prob_map, args.patch_size, args.threshold)
    save_csv(detections, str(out_dir / "detections.csv"))

    print(f"\nGotowe! Znaleziono {len(detections)} podejrzanych obszarów.")
    print(f"Wyniki w katalogu: {out_dir}")


if __name__ == "__main__":
    main()
