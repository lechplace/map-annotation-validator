"""
dataset.py
----------
PyTorch Dataset łączący:
  - ręcznie oznaczone przykłady (dane-mapa/img/ok, dane-mapa/img/not-ok)
  - auto-labeled patche z TIFF (dane-mapa/patches/ok, dane-mapa/patches/not-ok)

Klasy: 0 = OK, 1 = NOT-OK
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as T


# ── Transformacje ─────────────────────────────────────────────────────────────

def build_transforms(train: bool, img_size: int = 128) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=90),
            T.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class TreeRoadDataset(Dataset):
    """
    Parametry
    ---------
    root_dirs : lista katalogów z podkatalogami ok/ i not-ok/
    train     : czy stosować augmentację
    img_size  : rozmiar wyjściowy patcha (px)
    exclude   : zbiór nazw plików do wykluczenia (np. niejednoznaczny przykład)
    """

    def __init__(
        self,
        root_dirs: list,
        train: bool = True,
        img_size: int = 128,
        exclude: Optional[set] = None,
    ):
        self.transform = build_transforms(train, img_size)
        self.samples: list[Tuple[Path, int]] = []  # (ścieżka, etykieta)
        exclude = exclude or set()

        for root in root_dirs:
            root = Path(root)
            for label, subdir in [(0, "ok"), (1, "not-ok")]:
                d = root / subdir
                if not d.exists():
                    continue
                for f in sorted(d.iterdir()):
                    if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}:
                        if f.name not in exclude:
                            self.samples.append((f, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                "Brak próbek w dataset. "
                "Uruchom najpierw patch_extractor.py lub sprawdź ścieżki."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

    def class_counts(self) -> Tuple[int, int]:
        labels = [s[1] for s in self.samples]
        return labels.count(0), labels.count(1)

    def weighted_sampler(self) -> WeightedRandomSampler:
        """Sampler wyrównujący liczność klas podczas treningu."""
        n_ok, n_notok = self.class_counts()
        total = n_ok + n_notok
        weights_per_class = [total / max(n_ok, 1), total / max(n_notok, 1)]
        sample_weights = [weights_per_class[label] for _, label in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=total,
            replacement=True,
        )


def build_datasets(
    manual_dir: str,
    auto_dir: str,
    val_split: float = 0.2,
    img_size: int = 128,
    exclude: Optional[set] = None,
) -> Tuple[TreeRoadDataset, TreeRoadDataset]:
    """
    Zwraca (train_dataset, val_dataset).
    Ręczne przykłady zawsze trafiają do treningu (jest ich mało).
    Auto-labeled patche są dzielone train/val.
    """
    from sklearn.model_selection import train_test_split

    exclude = exclude or {"dzrzewo_na_drodze_4.PNG"}

    manual_dir = Path(manual_dir)
    auto_dir = Path(auto_dir)

    # Zbierz auto-labeled próbki i podziel
    auto_samples: list[Tuple[Path, int]] = []
    for label, subdir in [(0, "ok"), (1, "not-ok")]:
        d = auto_dir / subdir
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    auto_samples.append((f, label))

    if auto_samples:
        paths, labels = zip(*auto_samples)
        train_p, val_p, train_l, val_l = train_test_split(
            list(paths), list(labels),
            test_size=val_split,
            stratify=list(labels),
            random_state=42,
        )
        train_auto = list(zip(train_p, train_l))
        val_auto = list(zip(val_p, val_l))
    else:
        train_auto = val_auto = []

    # Ręczne przykłady → tylko trening
    manual_samples: list[Tuple[Path, int]] = []
    for label, subdir in [(0, "ok"), (1, "not-ok")]:
        d = manual_dir / subdir
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg"} and f.name not in exclude:
                    manual_samples.append((f, label))

    all_train = train_auto + manual_samples

    # Buduj datasety z gotowymi listami próbek
    train_ds = _SampleListDataset(all_train, train=True, img_size=img_size)
    val_ds = _SampleListDataset(val_auto, train=False, img_size=img_size)

    return train_ds, val_ds


class _SampleListDataset(Dataset):
    """Wewnętrzny Dataset przyjmujący gotową listę (path, label)."""

    def __init__(self, samples: list, train: bool, img_size: int):
        self.samples = samples
        self.transform = build_transforms(train, img_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def class_counts(self):
        labels = [s[1] for s in self.samples]
        return labels.count(0), labels.count(1)

    def weighted_sampler(self):
        n_ok, n_notok = self.class_counts()
        total = max(len(self.samples), 1)
        weights_per_class = [total / max(n_ok, 1), total / max(n_notok, 1)]
        sample_weights = [weights_per_class[label] for _, label in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=total,
            replacement=True,
        )
