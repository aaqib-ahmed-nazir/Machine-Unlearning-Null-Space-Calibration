"""In-process mutable application state (models, loaders, hyperparams)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class AppState:
    """Holds tensors and datasets for the lifespan of the server process."""

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    seed: int = 42

    forget_class: Optional[int] = None

    #: ``random_split`` seed for deterministic 90/10 train-val.
    split_seed: int = 42

    batch_size_train: int = 128

    batch_size_eval: int = 256

    loaders: Optional[dict[str, Optional[DataLoader]]] = None
    num_classes: int = 10

    #: Original pretrained model θ_o (frozen for pseudo-labeling).
    model_original: Optional[Any] = None

    #: Active working copy mutated by UNSC/baselines.
    model_current: Optional[Any] = None


state = AppState()
