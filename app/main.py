"""FastAPI service wiring Fashion-MNIST training, UNSC, and naive baselines."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from torch.utils.data import DataLoader

from app.baselines import run_full_retrain, run_random_label_finetune
from app.data_loader import FASHION_MNIST_CLASSES, build_fashion_mnist_loaders
from app.evaluate import metrics_triplet
from app.model import FashionCNN, load_checkpoint, save_checkpoint
from app.results import append_run, latest_rows_per_key
from app.schemas import (
    BaselineRandomLabelResponse,
    BaselineRequest,
    BaselineRetrainResponse,
    CompareResponse,
    DatasetInfoResponse,
    EvaluateResponse,
    MetricsBlock,
    ResultRow,
    SelectClassRequest,
    SelectClassResponse,
    TrainRequest,
    TrainResponse,
    UnlearnRequest,
    UnlearnRunResponse,
)
from app.state import state
from app.train import infer_accuracy, train_model
from app.unlearn import run_unsc
from app.utils import Timer, outputs_dir, pick_device, saved_models_dir, set_global_seed


def _outputs() -> Path:
    """Artifacts directory shared with notebooks (JSON traces, projector dumps)."""
    return outputs_dir()


def _checkpoint_original() -> Path:
    """Filesystem slot for persisted θ_o."""
    return saved_models_dir() / "original_model.pt"


def _acc_or_zero(x: float | None) -> float:
    """Normalize optional accuracies to finite JSON-friendly floats."""
    return float(0.0 if x is None else x)


def _maybe_load_theta_o_disk() -> None:
    """Reload θ_o from disk after uvicorn supervisor restarts the worker."""
    if state.model_original is not None:
        return
    ckpt = _checkpoint_original()
    if not ckpt.exists():
        return
    try:
        state.model_original = load_checkpoint(str(ckpt), state.device).eval()
    except Exception:  # noqa: BLE001
        state.model_original = None


def _mirror_student_theta() -> None:
    """Ensure ``model_current`` is a disjoint module copy of θ_o."""
    if state.model_original is None:
        raise HTTPException(status_code=400, detail="Execute `/model/train` before destructive routes.")
    twin = FashionCNN(num_classes=10).to(state.device)
    twin.load_state_dict(state.model_original.state_dict())
    twin.train(False)
    state.model_current = twin


def _require_theta_o() -> FashionCNN:
    """Hydrate θ_o from RAM or checkpoint or raise structured HTTP failures."""
    if state.model_original is None:
        _maybe_load_theta_o_disk()
    if state.model_original is None:
        raise HTTPException(status_code=400, detail="θ_o missing — POST `/model/train`.")
    return state.model_original  # type: ignore[return-value]


def _resolved_forget_class(explicit: Optional[int]) -> int:
    """Prefer explicit payloads, fallback to persisted server preference."""
    if explicit is not None:
        return int(explicit)
    if state.forget_class is None:
        raise HTTPException(status_code=400, detail="Forget class unset — `/unlearn/select-class`.")
    return int(state.forget_class)


def _rebuild_loaders(forget_optional: Optional[int]) -> dict[str, Optional[DataLoader]]:
    """Materialize deterministic PyTorch iterators with optional forget masks."""
    bundle = build_fashion_mnist_loaders(
        batch_size_train=state.batch_size_train,
        batch_size_eval=state.batch_size_eval,
        seed=state.split_seed,
        forget_class=forget_optional,
    )
    if bundle.get("train_full") is None or bundle.get("val_full") is None or bundle.get("test_full") is None:
        raise RuntimeError("Fashion-MNIST mandatory splits unresolved")
    state.loaders = bundle
    return bundle


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Select accelerator and opportunistically warm θ_o checkpoints."""
    state.device = pick_device()
    _maybe_load_theta_o_disk()
    yield


app = FastAPI(title="UNSC Fashion-MNIST", lifespan=_lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    """Synthetic heartbeat leveraged by infra smoke tests."""
    return {"service": "unsc-fashion-mnist", "status": "healthy"}


@app.get("/dataset/info", response_model=DatasetInfoResponse)
async def dataset_info() -> DatasetInfoResponse:
    """Advertise canonical Fashion-MNIST metadata for reproducible demos."""
    return DatasetInfoResponse(
        name="Fashion-MNIST",
        num_classes=10,
        class_names=FASHION_MNIST_CLASSES,
        input_shape=[1, 28, 28],
    )


@app.post("/model/train", response_model=TrainResponse)
async def model_train(payload: TrainRequest) -> TrainResponse:
    """Train or warm-load θ_o; baseline before any forgetting."""
    timer = Timer()
    set_global_seed(payload.seed)
    state.seed = payload.seed
    state.split_seed = payload.seed
    state.batch_size_train = payload.batch_size
    state.batch_size_eval = 256

    loaders_nf = build_fashion_mnist_loaders(
        batch_size_train=payload.batch_size,
        batch_size_eval=state.batch_size_eval,
        seed=payload.seed,
        forget_class=None,
    )
    tl, vl = loaders_nf["train_full"], loaders_nf["val_full"]
    te = loaders_nf["test_full"]
    if tl is None or vl is None or te is None:
        raise HTTPException(status_code=500, detail="Fashion-MNIST loaders corrupt")

    ckpt = _checkpoint_original()
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    if ckpt.exists() and not payload.force_retrain:
        theta_o = load_checkpoint(str(ckpt), state.device).eval()
        ta = infer_accuracy(theta_o, tl, state.device)
        oa, rr, fk = metrics_triplet(
            theta_o,
            forget_class_opt=None,
            test_full_loader=te,
            testers={},
            device=state.device,
        )
        append_run(
            _outputs(),
            method="warm_cache_original",
            forget_class=-1,
            overall_accuracy=oa,
            retained_accuracy=rr,
            forgotten_accuracy=fk,
            runtime_seconds=float(timer.elapsed_s()),
            model_path=str(ckpt),
            extras={"train_accuracy_quick": ta},
        )
    else:
        theta_o = FashionCNN(num_classes=10).to(state.device)
        hist_train = train_model(
            theta_o,
            train_loader=tl,
            val_loader=vl,
            epochs=payload.epochs,
            learning_rate=payload.learning_rate,
            device=state.device,
        )
        theta_o.eval()
        save_checkpoint(str(ckpt), theta_o, extra={"epochs": payload.epochs, "history": hist_train, "dataset": payload.dataset})
        ta = float(hist_train["train_acc"][-1]) if hist_train.get("train_acc") else infer_accuracy(theta_o, tl, state.device)
        oa, rr, fk = metrics_triplet(
            theta_o,
            forget_class_opt=None,
            test_full_loader=te,
            testers={},
            device=state.device,
        )
        append_run(
            _outputs(),
            method="train_original",
            forget_class=-1,
            overall_accuracy=oa,
            retained_accuracy=rr,
            forgotten_accuracy=fk,
            runtime_seconds=float(timer.elapsed_s()),
            model_path=str(ckpt),
            extras={"final_train_hist": hist_train},
        )

    state.model_original = theta_o
    _mirror_student_theta()
    state.forget_class = None
    state.loaders = loaders_nf

    oa, _, _ = metrics_triplet(
        theta_o,
        forget_class_opt=None,
        test_full_loader=te,
        testers={},
        device=state.device,
    )
    wall = float(timer.elapsed_s())
    return TrainResponse(
        dataset=payload.dataset,
        epochs=payload.epochs,
        train_accuracy=float(ta),
        test_accuracy=float(oa),
        runtime_seconds=wall,
        model_path=str(ckpt),
    )


@app.post("/unlearn/select-class", response_model=SelectClassResponse)
async def unlearn_select_class(body: SelectClassRequest) -> SelectClassResponse:
    """Bind server-side forget-class id and regenerate masked iterators."""
    state.forget_class = int(body.forget_class)
    _rebuild_loaders(body.forget_class)
    return SelectClassResponse(forget_class=body.forget_class)


@app.post("/unlearn/run", response_model=UnlearnRunResponse)
async def unlearn_run(body: UnlearnRequest) -> UnlearnRunResponse:
    """Apply Algorithm 2 (null-space calibrated pseudo-label forgetting)."""
    theta_o = _require_theta_o()
    fc = _resolved_forget_class(body.forget_class)
    batch_override = body.batch_size or state.batch_size_train
    state.batch_size_train = int(batch_override)
    loaders = _rebuild_loaders(fc)
    train_full_ld = loaders.get("train_full")
    train_forg_ld = loaders.get("train_forgotten")
    test_full_ld = loaders.get("test_full")
    test_ret_ld = loaders.get("test_retained")
    test_forg_ld = loaders.get("test_forgotten")
    if train_full_ld is None or train_forg_ld is None:
        raise HTTPException(status_code=500, detail="Algorithm 1/2 iterators unavailable")

    mon = Timer()
    theta_u = FashionCNN(num_classes=10).to(state.device)
    dest = saved_models_dir() / f"unlearned_unsc_c{fc}.pt"
    summary = run_unsc(
        theta_o=theta_o,
        theta_u=theta_u,
        forget_class=fc,
        train_full_loader_alg1=train_full_ld,
        train_forgotten_loader=train_forg_ld,
        testers={
            "test_full": test_full_ld,
            "test_retained": test_ret_ld,
            "test_forgotten": test_forg_ld,
        },
        device=state.device,
        outputs_folder=_outputs(),
        epochs_ul=int(body.epochs),
        lr_ul=float(body.learning_rate),
        momentum_ul=float(body.momentum),
        samples_algo1_cls=int(body.subspace_samples_per_class),
        epsilon_trunc_class=float(body.epsilon_trunc_class),
        epsilon_union_layer=float(body.epsilon_union_layer),
        seed=int(body.seed),
        unlearn_ckpt_fp=dest,
    )
    state.model_current = theta_u
    wall = float(mon.elapsed_s())
    append_run(
        _outputs(),
        method="unsc",
        forget_class=fc,
        overall_accuracy=_acc_or_zero(summary.get("overall_accuracy")),  # type: ignore[arg-type]
        retained_accuracy=_acc_or_zero(summary.get("retained_accuracy")),  # type: ignore[arg-type]
        forgotten_accuracy=_acc_or_zero(summary.get("forgotten_accuracy")),  # type: ignore[arg-type]
        runtime_seconds=wall,
        model_path=str(dest),
        extras={"unsc_history": summary.get("training_history_ul")},
    )
    return UnlearnRunResponse(
        forget_class=fc,
        retained_accuracy=_acc_or_zero(summary.get("retained_accuracy")),
        forgotten_accuracy=_acc_or_zero(summary.get("forgotten_accuracy")),
        overall_accuracy=_acc_or_zero(summary.get("overall_accuracy")),
        runtime_seconds=wall,
        model_path=str(dest),
    )


@app.post("/baseline/retrain", response_model=BaselineRetrainResponse)
async def baseline_retrain(payload: BaselineRequest) -> BaselineRetrainResponse:
    """Gold-standard retraining using only retained samples."""
    fc = _resolved_forget_class(payload.forget_class)
    batch_override = payload.batch_size or state.batch_size_train
    state.batch_size_train = int(batch_override)
    loaders = _rebuild_loaders(fc)
    train_r = loaders.get("train_retained")
    val_r = loaders.get("val_retained")
    if train_r is None or val_r is None:
        raise HTTPException(status_code=500, detail="Retained splits missing")
    mon = Timer()
    dest = saved_models_dir() / f"retrained_c{fc}.pt"
    _hist, meters = run_full_retrain(
        forget_class_id=fc,
        train_retained_loader=train_r,
        val_retained_loader=val_r,
        testers={
            "test_full": loaders.get("test_full"),
            "test_retained": loaders.get("test_retained"),
            "test_forgotten": loaders.get("test_forgotten"),
        },
        device=state.device,
        epochs=int(payload.epochs),
        learning_rate=float(payload.learning_rate),
        momentum=float(payload.momentum),
        checkpoint_path=dest,
        seed=int(payload.seed),
    )
    wall = float(mon.elapsed_s())
    append_run(
        _outputs(),
        method="full_retrain",
        forget_class=fc,
        overall_accuracy=_acc_or_zero(meters.get("overall_accuracy")),  # type: ignore[arg-type]
        retained_accuracy=_acc_or_zero(meters.get("retained_accuracy")),  # type: ignore[arg-type]
        forgotten_accuracy=_acc_or_zero(meters.get("forgotten_accuracy")),  # type: ignore[arg-type]
        runtime_seconds=wall,
        model_path=str(dest),
        extras={"history": _hist},
    )
    return BaselineRetrainResponse(
        forget_class=fc,
        retained_accuracy=_acc_or_zero(meters.get("retained_accuracy")),
        forgotten_accuracy=_acc_or_zero(meters.get("forgotten_accuracy")),
        overall_accuracy=_acc_or_zero(meters.get("overall_accuracy")),
        runtime_seconds=wall,
        model_path=str(dest),
    )


@app.post("/baseline/random-label", response_model=BaselineRandomLabelResponse)
async def baseline_random(payload: BaselineRequest) -> BaselineRandomLabelResponse:
    """Random-label poisoning baseline omitting projector safeguards."""
    theta_o = _require_theta_o()
    fc = _resolved_forget_class(payload.forget_class)
    batch_override = payload.batch_size or state.batch_size_train
    state.batch_size_train = int(batch_override)
    loaders = _rebuild_loaders(fc)
    forgotten_ld = loaders.get("train_forgotten")
    if forgotten_ld is None:
        raise HTTPException(status_code=500, detail="Forgotten-stream loader unresolved")
    mon = Timer()
    dest = saved_models_dir() / f"random_label_c{fc}.pt"
    rl_model = FashionCNN(num_classes=10).to(state.device)
    histories, meters = run_random_label_finetune(
        theta_original=theta_o,
        theta_work=rl_model,
        forgetting_class_id=fc,
        train_forgotten_loader=forgotten_ld,
        testers={
            "test_full": loaders.get("test_full"),
            "test_retained": loaders.get("test_retained"),
            "test_forgotten": loaders.get("test_forgotten"),
        },
        device=state.device,
        epochs=int(payload.epochs),
        lr=float(payload.learning_rate),
        momentum=float(payload.momentum),
        checkpoint_path=dest,
        seed=int(payload.seed),
    )
    state.model_current = rl_model
    wall = float(mon.elapsed_s())
    append_run(
        _outputs(),
        method="random_label",
        forget_class=fc,
        overall_accuracy=_acc_or_zero(meters.get("overall_accuracy")),  # type: ignore[arg-type]
        retained_accuracy=_acc_or_zero(meters.get("retained_accuracy")),  # type: ignore[arg-type]
        forgotten_accuracy=_acc_or_zero(meters.get("forgotten_accuracy")),  # type: ignore[arg-type]
        runtime_seconds=wall,
        model_path=str(dest),
        extras={"random_label_hist": histories},
    )
    return BaselineRandomLabelResponse(
        forget_class=fc,
        retained_accuracy=_acc_or_zero(meters.get("retained_accuracy")),
        forgotten_accuracy=_acc_or_zero(meters.get("forgotten_accuracy")),
        overall_accuracy=_acc_or_zero(meters.get("overall_accuracy")),
        runtime_seconds=wall,
        model_path=str(dest),
    )


@app.get("/model/evaluate", response_model=EvaluateResponse)
async def model_evaluate(
    scope: Literal["original", "current"] = Query("original"),
) -> EvaluateResponse:
    """Report retained/forgotten/utility metrics on ``original`` θ_o vs ``current`` student."""
    if scope == "original":
        model = _require_theta_o()
    else:
        _require_theta_o()
        if state.model_current is None:
            _mirror_student_theta()
        model = state.model_current  # type: ignore[assignment]

    fc = state.forget_class
    if fc is None:
        loaders_local = _rebuild_loaders(None)
        te = loaders_local.get("test_full")
        if te is None:
            raise HTTPException(status_code=500, detail="Test loader missing")
        ov, rr, fk = metrics_triplet(
            model,
            forget_class_opt=None,
            test_full_loader=te,
            testers={},
            device=state.device,
        )
    else:
        loaders_local = _rebuild_loaders(fc)
        te = loaders_local.get("test_full")
        if te is None:
            raise HTTPException(status_code=500, detail="Test iterator missing during forget eval")
        ov, rr, fk = metrics_triplet(
            model,
            forget_class_opt=fc,
            test_full_loader=te,
            testers={
                "test_retained": loaders_local.get("test_retained"),
                "test_forgotten": loaders_local.get("test_forgotten"),
            },
            device=state.device,
        )

    metrics = MetricsBlock(overall=float(ov), retained=float(rr), forgotten=float(fk))
    return EvaluateResponse(scope=scope, forget_class=fc, metrics=metrics)


@app.get("/results/compare", response_model=CompareResponse)
async def results_compare() -> CompareResponse:
    """Return latest deduplicated leaderboard rows keyed by `(method, forget_class)`."""
    rows_raw = latest_rows_per_key(_outputs())
    curated: list[ResultRow] = []
    for r in rows_raw:
        curated.append(
            ResultRow(
                method=str(r["method"]),
                forget_class=int(r["forget_class"]),
                overall_accuracy=float(r["overall_accuracy"]),
                retained_accuracy=float(r["retained_accuracy"]),
                forgotten_accuracy=float(r["forgotten_accuracy"]),
                runtime_seconds=float(r["runtime_seconds"]),
            )
        )
    return CompareResponse(results=curated)
