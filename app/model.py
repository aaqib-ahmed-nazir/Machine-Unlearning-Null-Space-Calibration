"""Small Fashion-MNIST CNN with forward hooks exposing layer-wise inputs."""

from __future__ import annotations

from typing import Callable, OrderedDict

import torch
import torch.nn as nn


def _flatten_nchw_to_columns(x: torch.Tensor) -> torch.Tensor:
    """Map ``[N,C,H,W]`` to ``[C*H*W, N]`` (columns = samples).

    Matches UNSC Alg. 1 stacking of flattened layer inputs into matrix columns.

    Args:
        x: Activations arriving at Conv2d (or pooled feature maps).

    Returns:
        2-D tensor suitable for singular value decomposition along columns.
    """
    n = x.shape[0]
    return x.detach().reshape(n, -1).transpose(0, 1)


def _flatten_linear_input(x: torch.Tensor) -> torch.Tensor:
    """Map ``[N,D]`` to ``[D,N]``."""
    return x.detach().transpose(0, 1)


class FashionCNN(nn.Module):
    """Two-conv classifier with explicit hook points for Algorithm 1/2."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

        #: Ordered hooks on modules whose gradients will be projected in UNSC.
        self.projection_modules: OrderedDict[str, nn.Module] = OrderedDict()

        #: Last captured inputs (flattened columns) keyed by projector name (CPU tensors).
        self._layer_flat_inputs_cpu: dict[str, torch.Tensor] = {}

        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for name in ("conv1", "conv2", "fc1", "fc2"):
            module = getattr(self, name)
            self.projection_modules[name] = module
            hook = self._make_pre_hook(name, module)

            handle = module.register_forward_pre_hook(hook)
            self._handles.append(handle)

        self.training_step_device: torch.device = torch.device("cpu")

    def _make_pre_hook(self, layer_name: str, module: nn.Module) -> Callable[..., None]:
        """Build a closure storing flattened inputs keyed by ``layer_name``."""

        def hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x_in = inputs[0]
            if isinstance(module, nn.Conv2d):
                flat_cpu = _flatten_nchw_to_columns(x_in).to("cpu")
            elif isinstance(module, nn.Linear):
                flat_cpu = _flatten_linear_input(x_in).to("cpu")
            else:
                raise TypeError(type(module))

            self._layer_flat_inputs_cpu[layer_name] = flat_cpu

        return hook

    def clear_cached_inputs(self) -> None:
        """Drop references so memory does not grow across unused forwards."""
        self._layer_flat_inputs_cpu.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward classifier path.

        Args:
            x: Batch ``[N,1,28,28]``.

        Returns:
            Class logits ``[N,num_classes]``.
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    def projector_layer_names_ordered(self) -> list[str]:
        """Deterministic projector order aligned with Algorithm 2."""
        return list(self.projection_modules.keys())


def clone_model_structure(source: FashionCNN, device: torch.device) -> FashionCNN:
    """Deep-copy architecture and optionally move to ``device``.

    Args:
        source: Trained blueprint.
        device: Target accelerator.

    Returns:
        Independent module with Xavier-initialized weights.
    """
    m = FashionCNN(num_classes=source.num_classes).to(device)
    return m


def save_checkpoint(
    path: str,
    model: FashionCNN,
    *,
    extra: dict[str, torch.Tensor | float | str | int] | None = None,
) -> None:
    """Save ``state_dict`` plus small metadata blobs.

    Args:
        path: Destination ``.pt`` file.
        model: Snapshot source.
        extra: Optional tensors (projection maps, epsilon, etc.).
    """
    blob: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "projection_layer_names": model.projector_layer_names_ordered(),
        "num_classes": model.num_classes,
    }
    if extra:
        blob["extra"] = extra
    torch.save(blob, path)


def load_checkpoint(path: str, device: torch.device) -> FashionCNN:
    """Load ``FashionCNN`` matching saved architecture.

    Args:
        path: Saved ``save_checkpoint`` file.
        device: Map tensors to this device after load.

    Returns:
        Model in eval-mode by default caller responsibility.
    """
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    nc = int(blob["num_classes"])
    net = FashionCNN(num_classes=nc).to(device)
    net.load_state_dict(blob["model_state_dict"])  # type: ignore[arg-type]
    net.clear_cached_inputs()
    setattr(net, "_checkpoint_extra_", blob.get("extra"))
    return net


def copy_weights(from_model: FashionCNN, to_model: FashionCNN) -> None:
    """Synchronize weights ``from_model`` → ``to_model`` (matched keys)."""
    to_model.load_state_dict(from_model.state_dict())
