"""
model.py
--------
EfficientNet-B0 jako binarny klasyfikator patchy (OK vs NOT-OK).
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def build_model(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    Zwraca EfficientNet-B0 z wymienioną głową klasyfikatora.

    Parameters
    ----------
    num_classes : liczba klas (2 = OK / NOT-OK)
    pretrained  : czy załadować wagi ImageNet
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Wymień ostatnią warstwę (1280 → num_classes)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Wczytaj model z pliku checkpoint."""
    model = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def save_checkpoint(model: nn.Module, optimizer, epoch: int, val_loss: float, path: str):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)
