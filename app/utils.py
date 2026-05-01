"""Shared utilities: paths, device selection, RNG, timers."""

from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch


def repo_root() -> Path:
    """Return repository root directory (parent of ``app``)."""
    return Path(__file__).resolve().parent.parent


def saved_models_dir() -> Path:
    """Directory for persisted checkpoints."""
    path = repo_root() / "saved_models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def outputs_dir() -> Path:
    """Directory for projections, Results JSON, and run logs."""
    path = repo_root() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pick_device() -> torch.device:
    """Prefer MPS on Apple Silicon, then CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.backends.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Timer:
    """Wall-clock elapsed time in seconds."""

    def __init__(self) -> None:
        self._t0: float = time.perf_counter()

    def elapsed_s(self) -> float:
        """Return seconds since creation."""
        return float(time.perf_counter() - self._t0)


def state_dict_digest(model: torch.nn.Module) -> str:
    """Short SHA-256 fingerprint of ``model.state_dict()`` tensors."""

    chunks: list[bytes] = []
    md = sorted(model.state_dict().items(), key=lambda kv: kv[0])
    for name, tens in md:
        chunks.append(name.encode("utf-8"))
        chunks.append(tens.detach().cpu().numpy().tobytes())
    blob = hashlib.sha256(b"".join(chunks)).hexdigest()
    return blob[:16]
