"""Baselines: retained-only full re-training and random-label fine tuning."""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.evaluate import evaluate_splits
from app.model import FashionCNN, copy_weights, save_checkpoint
from app.train import train_model


def run_full_retrain(
    *,
    forget_class_id: int,
    train_retained_loader: DataLoader,
    val_retained_loader: DataLoader,
    testers: dict[str, DataLoader | None],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float = 0.9,
    checkpoint_path: Path,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, float | None]]:
    """Train ``FashionCNN`` scratch on retained split only."""

    random.seed(seed)
    torch.manual_seed(seed)

    model = FashionCNN(num_classes=10).to(device)
    history = train_model(
        model,
        train_loader=train_retained_loader,
        val_loader=val_retained_loader,
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        momentum=float(momentum),
        weight_decay=0.0,
        device=device,
    )
    model.eval()
    ov, rr, fk = evaluate_splits(
        model,
        test_full_loader=testers.get("test_full"),
        test_retained_loader=testers.get("test_retained"),
        test_forgotten_loader=testers.get("test_forgotten"),
        device=device,
    )
    save_checkpoint(
        str(checkpoint_path),
        model,
        extra={"method": "full_retrain", "forget_class": forget_class_id, "seed": int(seed)},
    )
    meters = {"overall_accuracy": ov, "retained_accuracy": rr, "forgotten_accuracy": fk}
    return history, meters


def run_random_label_finetune(
    *,
    theta_original: FashionCNN,
    theta_work: FashionCNN,
    forgetting_class_id: int,
    train_forgotten_loader: DataLoader,
    testers: dict[str, DataLoader | None],
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    checkpoint_path: Path,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, float | None]]:
    """Hayase-style random labels without projector (gradient baseline).

    Copies θ_o weights then minimizes CE on forgetting mini-batches with labels
    drawn uniformly from remaining classes excluding the forget index.
    """

    random.seed(seed)
    torch.manual_seed(seed)

    allowed = [c for c in range(theta_work.num_classes) if c != forgetting_class_id]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        theta_work.parameters(),
        lr=float(lr),
        momentum=float(momentum),
        nesterov=True,
    )

    copy_weights(theta_original, theta_work)

    histories: dict[str, list[float]] = {"epoch": [], "train_loss": []}
    theta_work.train(True)

    for epoch in range(1, int(epochs) + 1):
        agg = 0.0
        seen_batches = 0
        for imgs, _y in train_forgotten_loader:
            batch_size = imgs.size(0)
            bogus = torch.tensor(
                random.choices(allowed, k=batch_size),
                dtype=torch.long,
                device=device,
                requires_grad=False,
            )
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = theta_work(imgs)
            loss_val = criterion(logits, bogus)
            loss_val.backward()
            optimizer.step()
            agg += float(loss_val.item())
            seen_batches += 1

        histories["epoch"].append(float(epoch))
        histories["train_loss"].append(agg / max(1, seen_batches))

    theta_work.train(False)

    ov, rr, fk = evaluate_splits(
        theta_work,
        test_full_loader=testers.get("test_full"),
        test_retained_loader=testers.get("test_retained"),
        test_forgotten_loader=testers.get("test_forgotten"),
        device=device,
    )

    save_checkpoint(
        str(checkpoint_path),
        theta_work,
        extra={
            "method": "random_label",
            "forget_class": forgetting_class_id,
            "seed": int(seed),
        },
    )

    meters = {"overall_accuracy": ov, "retained_accuracy": rr, "forgotten_accuracy": fk}
    return histories, meters


