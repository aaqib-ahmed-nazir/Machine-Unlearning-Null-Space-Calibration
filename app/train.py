"""Supervised training loops (original baseline + selective re-training)."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> tuple[float, int]:
    """Single full pass over batch iterator.

    Args:
        model: Trainable predictor.
        loader: Possibly shuffled training loader.
        optimizer: Torch optimizer referencing ``model`` params.
        criterion: Scalar-valued loss reducer.
        device: Accelerator placement.

    Returns:
        ``(average_loss, num_batches)``.
    """
    model.train()
    running_loss = 0.0
    batches = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        batches += 1

    return running_loss / max(1, batches), batches


@torch.no_grad()
def infer_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Training-set accuracy estimator (cheap smoke metric)."""
    correct = 0
    seen = 0
    model.eval()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum().item())
        seen += int(yb.numel())
    return float(correct / max(1, seen))


def train_model(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    device: torch.device,
) -> dict[str, list[float]]:
    """SGD+C momentum training wrapper with simplistic history dict.

    Args:
        model: Target network mutated in-place.
        train_loader: Batched training iterable.
        val_loader: Hold-out iterable for periodic accuracy bookkeeping.
        epochs: Epoch count FastAPI POC default ~10–20.
        learning_rate: Base LR multiplier.
        momentum: SGD Polyak momentum coefficient.
        weight_decay: Optional L2.
        device: Training placement ``mps``/``cuda``/``cpu``.

    Returns:
        Dictionary ``hist`` populated with ``epoch``, ``train_loss``, ``train_acc``, ``val_acc``.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    hist = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for ep in range(1, epochs + 1):
        loss_mu, _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
        ta = infer_accuracy(model, train_loader, device)
        va = infer_accuracy(model, val_loader, device)

        hist["epoch"].append(float(ep))
        hist["train_loss"].append(loss_mu)
        hist["train_acc"].append(float(ta))
        hist["val_acc"].append(float(va))

    return hist
