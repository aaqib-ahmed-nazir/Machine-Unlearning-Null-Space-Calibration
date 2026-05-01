"""Accuracy helpers for retained / forgotten / overall splits."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def subset_accuracy(model: nn.Module, loader: DataLoader | None, device: torch.device) -> float | None:
    """Compute vanilla accuracy ``correct/total`` on a subset loader.

    Args:
        model: Predictor evaluated in inference mode externally.
        loader: Possibly empty subset.
        device: Batch placement.

    Returns:
        Fraction in ``[0,1]``, or ``None`` if loader empty / undefined.
    """
    if loader is None:
        return None
    n_total = len(loader.dataset)  # type: ignore[arg-type]
    if n_total == 0:
        return None

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

    assert seen == n_total
    return float(correct / max(1, seen))


def evaluate_splits(
    model: nn.Module,
    *,
    test_full_loader: DataLoader | None,
    test_retained_loader: DataLoader | None,
    test_forgotten_loader: DataLoader | None,
    device: torch.device,
) -> tuple[float | None, float | None, float | None]:
    """Return triple ``(overall, retained, forgotten)`` accuracies.

    Args:
        model: Classifier ``FashionCNN``.
        test_full_loader: Entire Fashion-MNIST test split.
        test_retained_loader: Loader excluding forgetting class samples.
        test_forgotten_loader: Loader for forgetting class only.
        device: Accelerator / CPU placement.

    Returns:
        Each entry ``None`` if loader missing (callers coerce to ``nan`` metadata).
    """
    overall = subset_accuracy(model, test_full_loader, device)
    retained = subset_accuracy(model, test_retained_loader, device)
    forgotten = subset_accuracy(model, test_forgotten_loader, device)
    return overall, retained, forgotten


def coerce_metric(x: Optional[float]) -> float:
    """Map ``None`` sentinel to ``nan`` numeric JSON portability."""
    if x is None:
        return float("nan")
    return float(x)


@torch.no_grad()
def classification_accuracy_dataloader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Whole-dataset accuracy ignoring split semantics."""
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



def metrics_triplet(
    model: nn.Module,
    *,
    forget_class_opt: Optional[int],
    test_full_loader: DataLoader | None,
    testers: dict[str, DataLoader | None],
    device: torch.device,
) -> tuple[float, float, float]:
    """Overall / retained / forgotten accuracies handling pre-forget symmetry."""

    if test_full_loader is None:
        raise ValueError("`test_full_loader` required.")

    ov = classification_accuracy_dataloader(model, test_full_loader, device)

    if forget_class_opt is None:
        """Before choosing ``forget_class``, all three coincide."""
        return ov, ov, ov

    rr_acc = subset_accuracy(model, testers.get("test_retained"), device)

    fk_acc = subset_accuracy(model, testers.get("test_forgotten"), device)

    if rr_acc is None or fk_acc is None:
        raise RuntimeError("Retained/forgotten test loaders unavailable after forget-class selection.")

    return float(ov), float(rr_acc), float(fk_acc)

