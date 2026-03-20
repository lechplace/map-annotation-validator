"""
train.py
--------
Pętla treningowa EfficientNet-B0 na patchach drzewo/droga.

Użycie:
    python src/train.py \
        --manual-dir dane-mapa/img \
        --auto-dir dane-mapa/patches \
        --epochs 30 \
        --batch-size 32 \
        --lr 1e-4 \
        --out models/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import build_datasets
from src.model import build_model, save_checkpoint


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-dir", default="dane-mapa/img")
    parser.add_argument("--auto-dir", default="dane-mapa/patches")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping: ile epok bez poprawy val_loss")
    parser.add_argument("--out", default="models/best_model.pt")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    manual_dir = project_root / args.manual_dir
    auto_dir = project_root / args.auto_dir
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Urządzenie: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(
        manual_dir=str(manual_dir),
        auto_dir=str(auto_dir),
        val_split=args.val_split,
        img_size=args.img_size,
    )

    n_ok, n_notok = train_ds.class_counts()
    print(f"Train: {len(train_ds)} próbek  (OK={n_ok}, NOT-OK={n_notok})")
    print(f"Val:   {len(val_ds)} próbek")

    if len(train_ds) == 0:
        print("BŁĄD: Brak danych treningowych. Uruchom najpierw patch_extractor.py")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_ds.weighted_sampler(),
        num_workers=0,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    ) if len(val_ds) > 0 else None

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=2, pretrained=True).to(device)

    # Wagi klas w loss (odwrotnie proporcjonalne do liczności)
    total = n_ok + n_notok
    class_weights = torch.tensor(
        [total / max(n_ok, 1), total / max(n_notok, 1)],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Pętla treningowa ──────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(
                f"Epoka {epoch:3d}/{args.epochs} | "
                f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.3f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, val_loss, str(out_path))
                print(f"  ✓ Nowy najlepszy model (val_loss={val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping po {epoch} epokach.")
                    break
        else:
            print(f"Epoka {epoch:3d}/{args.epochs} | train loss={train_loss:.4f} acc={train_acc:.3f}")
            save_checkpoint(model, optimizer, epoch, train_loss, str(out_path))

    print(f"\nModel zapisany: {out_path}")


if __name__ == "__main__":
    main()
