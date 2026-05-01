"""Request and response schemas for UNSC Fashion-MNIST FastAPI demos."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    """POST ``/model/train`` body — supervised baseline training knobs."""

    epochs: int = Field(default=15, ge=1, le=500)
    batch_size: int = Field(default=128, ge=1, le=1024)
    learning_rate: float = Field(default=0.05, gt=0, le=1.0)
    dataset: Literal["fashion_mnist"] = "fashion_mnist"
    seed: int = Field(default=42, ge=0)
    force_retrain: bool = False


class TrainResponse(BaseModel):
    """Training completion payload surfaced in Swagger/OpenAPI."""

    status: Literal["training_completed"] = "training_completed"
    dataset: str
    epochs: int
    train_accuracy: float
    test_accuracy: float
    runtime_seconds: float
    model_path: str


class SelectClassRequest(BaseModel):
    """Forget-class selection for downstream UNSC routines."""

    forget_class: int = Field(..., ge=0, le=9)


class SelectClassResponse(BaseModel):
    """Acknowledgement that server state remembers the forget id."""

    status: Literal["forget_class_registered"] = "forget_class_registered"
    forget_class: int


class UnlearnRequest(BaseModel):
    """UNSC stage hyper-parameters (Algorithms 1 & 2)."""

    forget_class: Optional[int] = Field(
        default=None, description="Override server selection when non-null."
    )
    epochs: int = Field(default=3, ge=1, le=200)
    learning_rate: float = Field(default=0.01, gt=0, le=1.0)
    batch_size: Optional[int] = Field(
        default=None, ge=1, le=1024, description="Override training batch during forget splits."
    )
    subspace_samples_per_class: int = Field(
        default=256,
        ge=16,
        le=6000,
        description="Column budget |B_k| for Algorithm 1 (paper≈256).",
    )
    epsilon_trunc_class: float = Field(
        default=0.97,
        gt=0,
        lt=1,
        description="Per-class spectral energy kept before unions.",
    )
    epsilon_union_layer: float = Field(
        default=0.97,
        gt=0,
        lt=1,
        description="Energy threshold after merging retained bases (Eq. 9 analogue).",
    )
    momentum: float = Field(default=0.9, ge=0, lt=1)
    seed: int = Field(default=42, ge=0)


class BaselineRequest(BaseModel):
    """Shared knobs for naive baselines."""

    forget_class: Optional[int] = None
    epochs: int = Field(default=15, ge=1, le=500)
    learning_rate: float = Field(default=0.05, gt=0, le=1.0)
    batch_size: Optional[int] = Field(default=None, ge=1, le=1024)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    seed: int = Field(default=42, ge=0)


class MetricsBlock(BaseModel):
    """Container for split accuracies."""

    overall: float
    retained: float
    forgotten: float


class EvaluateResponse(BaseModel):
    """Returned by ``GET /model/evaluate``."""

    scope: Literal["original", "current"]
    forget_class: Optional[int]
    metrics: MetricsBlock


class UnlearnRunResponse(BaseModel):
    """UNSC forgetting summary metrics."""

    status: Literal["unlearning_completed"] = "unlearning_completed"
    method: Literal["unsc"] = "unsc"
    forget_class: int
    retained_accuracy: float
    forgotten_accuracy: float
    overall_accuracy: float
    runtime_seconds: float
    model_path: str


class BaselineRetrainResponse(BaseModel):
    """Gold-standard retained-only training outcome."""

    status: Literal["baseline_completed"] = "baseline_completed"
    method: Literal["full_retrain"] = "full_retrain"
    forget_class: int
    retained_accuracy: float
    forgotten_accuracy: float
    overall_accuracy: float
    runtime_seconds: float
    model_path: str


class BaselineRandomLabelResponse(BaseModel):
    """Random-label fine-tune diagnostics."""

    status: Literal["baseline_completed"] = "baseline_completed"
    method: Literal["random_label"] = "random_label"
    forget_class: int
    retained_accuracy: float
    forgotten_accuracy: float
    overall_accuracy: float
    runtime_seconds: float
    model_path: str


class ResultRow(BaseModel):
    """Single leaderboard entry for qualitative comparisons."""

    method: str
    forget_class: int
    overall_accuracy: float
    retained_accuracy: float
    forgotten_accuracy: float
    runtime_seconds: float


class CompareResponse(BaseModel):
    """Deduped history rows for dashboards."""

    results: list[ResultRow]


class DatasetInfoResponse(BaseModel):
    """Static Fashion-MNIST metadata."""

    name: str
    num_classes: int
    class_names: list[str]
    input_shape: list[int]


class AnyJsonResponse(BaseModel):
    """Escape hatch payloads for exploratory debugging."""

    data: dict[str, Any]

