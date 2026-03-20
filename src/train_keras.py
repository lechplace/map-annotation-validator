"""
train_keras.py
--------------
Trening EfficientNetB0 (Keras/keras_hub) na patchach drzewo/droga.

Użycie:
    uv run python src/train_keras.py \
        --manual-dir dane-mapa/img \
        --auto-dir dane-mapa/patches \
        --epochs 30 \
        --batch-size 32 \
        --lr 1e-4 \
        --out models/best_model_keras.keras
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import keras
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_keras import build_model_keras
from src.dataset import build_datasets  # reużywamy logikę podziału danych


# ── Helpers ───────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: np.ndarray) -> np.ndarray:
    """float32 [0,1] → ImageNet normalized."""
    return (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def load_samples_as_arrays(samples: list, img_size: int) -> tuple:
    """
    Wczytaj listę (path, label) → (X: np.ndarray, y: np.ndarray).
    Augmentacja jest obsługiwana przez keras.layers w modelu lub ręcznie.
    """
    X, y = [], []
    for path, label in samples:
        img = Image.open(path).convert("RGB").resize((img_size, img_size))
        X.append(preprocess(np.array(img)))
        y.append(label)
    return np.stack(X), np.array(y, dtype=np.int32)


def build_augmentation_layer(img_size: int) -> keras.Sequential:
    """Warstwa augmentacji — stosowana tylko podczas treningu."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(factor=0.25),
        keras.layers.RandomBrightness(factor=0.2),
        keras.layers.RandomContrast(factor=0.1),
    ], name="augmentation")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-dir", default="dane-mapa/img")
    parser.add_argument("--auto-dir",   default="dane-mapa/patches")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--img-size",   type=int,   default=128)
    parser.add_argument("--val-split",  type=float, default=0.2)
    parser.add_argument("--out",        default="models/best_model_keras.keras")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    out_path = project_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Dane ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(
        manual_dir=str(project_root / args.manual_dir),
        auto_dir=str(project_root / args.auto_dir),
        val_split=args.val_split,
        img_size=args.img_size,
    )

    if len(train_ds) == 0:
        print("BŁĄD: Brak danych treningowych. Uruchom najpierw patch_extractor.py")
        sys.exit(1)

    X_train, y_train = load_samples_as_arrays(train_ds.samples, args.img_size)
    X_val,   y_val   = load_samples_as_arrays(val_ds.samples,   args.img_size) if len(val_ds) > 0 else (None, None)

    n_ok    = int((y_train == 0).sum())
    n_notok = int((y_train == 1).sum())
    print(f"Train: {len(y_train)} próbek  (OK={n_ok}, NOT-OK={n_notok})")
    if X_val is not None:
        print(f"Val:   {len(y_val)} próbek")

    # Wagi klas
    total = n_ok + n_notok
    class_weight = {0: total / max(n_ok, 1), 1: total / max(n_notok, 1)}

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model_keras(num_classes=2, img_size=args.img_size)

    augment = build_augmentation_layer(args.img_size)

    # Budujemy pipeline: augmentacja → model (tylko w trybie train)
    inputs = keras.Input(shape=(args.img_size, args.img_size, 3))
    x = augment(inputs, training=True)
    outputs = model(x)
    train_model = keras.Model(inputs, outputs)

    train_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=1e-2),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_path),
            monitor="val_loss" if X_val is not None else "loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if X_val is not None else "loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if X_val is not None else "loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    # ── Trening ───────────────────────────────────────────────────────────────
    validation_data = (X_val, y_val) if X_val is not None else None

    train_model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=validation_data,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Zapisz finalny model inference (bez warstwy augmentacji)
    model.save(str(out_path))
    print(f"\nModel zapisany: {out_path}")


if __name__ == "__main__":
    main()
