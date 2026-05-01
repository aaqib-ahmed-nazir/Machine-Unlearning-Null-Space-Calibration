"""Fashion-MNIST dataloaders: 90/10 train-val split plus retained/forgotten subsets."""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from app.utils import repo_root


FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def default_transform() -> transforms.Compose:
    """Tensor transform for grayscale 28×28."""
    return transforms.Compose([transforms.ToTensor()])


def build_fashion_mnist_loaders(
    *,
    batch_size_train: int = 128,
    batch_size_eval: int = 256,
    val_fraction: float = 0.1,
    seed: int = 42,
    forget_class: Optional[int] = None,
    root: Optional[str] = None,
    num_workers: int = 0,
) -> dict[str, Optional[DataLoader]]:
    """Create train/val/test loaders and optional retained/forgotten splits.

    Args:
        batch_size_train: Batch size for training loops.
        batch_size_eval: Batch size for evaluation.
        val_fraction: Fraction of official training set used for validation.
        seed: RNG seed for split reproducibility.
        forget_class: If set, build ``train_retained``, ``val_retained``,
            ``train_forgotten``, and test subset loaders.
        root: Download root (default ``<repo>/data``).
        num_workers: DataLoader workers (macOS POC often ``0``).

    Returns:
        Dict with keys ``train_full``, ``val_full``, ``test_full``, and optionally
        ``train_retained``, ``val_retained``, ``train_forgotten``, ``test_retained``,
        ``test_forgotten``, or ``None`` for unavailable keys before ``forget_class``.
    """
    if root is None:
        root = str(repo_root() / "data")

    transform = default_transform()

    train_full_ds = datasets.FashionMNIST(
        root,
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.FashionMNIST(
        root,
        train=False,
        download=True,
        transform=transform,
    )

    n_train = len(train_full_ds)
    n_val = int(round(n_train * val_fraction))
    n_tr = n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(
        train_full_ds,
        [n_tr, n_val],
        generator=generator,
    )

    loaders: dict[str, Optional[DataLoader]] = {
        "train_full": DataLoader(
            train_ds,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "val_full": DataLoader(
            val_ds,
            batch_size=batch_size_eval,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "test_full": DataLoader(
            test_ds,
            batch_size=batch_size_eval,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "train_retained": None,
        "val_retained": None,
        "train_forgotten": None,
        "test_retained": None,
        "test_forgotten": None,
    }

    if forget_class is None:
        return loaders

    fc = forget_class

    train_positions = torch.arange(len(train_ds), dtype=torch.long)
    train_global_idxs = torch.tensor(train_ds.indices, dtype=torch.long)
    val_positions = torch.arange(len(val_ds), dtype=torch.long)
    val_global_idxs = torch.tensor(val_ds.indices, dtype=torch.long)

    train_targets = torch.tensor(
        [train_full_ds.targets[i] for i in train_global_idxs.tolist()],
        dtype=torch.long,
    )
    val_targets = torch.tensor(
        [train_full_ds.targets[i] for i in val_global_idxs.tolist()],
        dtype=torch.long,
    )
    test_indices = torch.arange(len(test_ds))

    mask_train_r = train_targets != fc
    mask_train_u = train_targets == fc
    mask_val_r = val_targets != fc

    loaders["train_retained"] = DataLoader(
        Subset(train_ds, train_positions[mask_train_r].tolist()),
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    loaders["train_forgotten"] = DataLoader(
        Subset(train_ds, train_positions[mask_train_u].tolist()),
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    loaders["val_retained"] = DataLoader(
        Subset(val_ds, val_positions[mask_val_r].tolist()),
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    test_targets = torch.tensor(test_ds.targets, dtype=torch.long)
    mask_test_r = test_targets != fc
    mask_test_u = test_targets == fc

    loaders["test_retained"] = DataLoader(
        Subset(test_ds, test_indices[mask_test_r].tolist()),
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    loaders["test_forgotten"] = DataLoader(
        Subset(test_ds, test_indices[mask_test_u].tolist()),
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return loaders
