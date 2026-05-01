"""Append-only persistence of experiment runs into ``outputs/results.json``."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def results_json_path(outputs: Path) -> Path:
    """Path to the aggregate results file."""
    return outputs / "results.json"


def run_json_path(outputs: Path, run_id: str) -> Path:
    """Per-run detailed log (epochs, projections metadata, etc.)."""
    return outputs / f"run_{run_id}.json"


def _load_or_empty(path: Path) -> dict[str, Any]:
    """Load JSON object or start empty aggregate."""
    if not path.exists():
        return {"runs": []}
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "runs" not in obj:
        return {"runs": []}
    return obj


def append_run(
    outputs: Path,
    *,
    method: str,
    forget_class: int,
    overall_accuracy: float,
    retained_accuracy: float,
    forgotten_accuracy: float,
    runtime_seconds: float,
    model_path: Optional[str],
    extras: Optional[dict[str, Any]] = None,
) -> tuple[str, Path]:
    """Append one run to ``results.json`` and write ``run_<id>.json``.

    Args:
        outputs: Output directory.
        method: Logical name ``original``, ``unsc``, ``full_retrain``, ``random_label``.
        forget_class: Forgetting index (still stored for completeness on train runs).
        overall_accuracy: Test overall accuracy [0,1].
        retained_accuracy: Test accuracy excluding forget class [0,1].
        forgotten_accuracy: Test accuracy on forget class only [0,1].
        runtime_seconds: Wall time for operation.
        model_path: Checkpoint path saved on disk if any.
        extras: Nested detail for per-run JSON (epoch curves).

    Returns:
        Tuple ``(run_id, run_file_path)``.
    """
    run_id = uuid.uuid4().hex[:12]
    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "forget_class": forget_class,
        "overall_accuracy": overall_accuracy,
        "retained_accuracy": retained_accuracy,
        "forgotten_accuracy": forgotten_accuracy,
        "runtime_seconds": runtime_seconds,
        "model_path": model_path,
    }
    if extras:

        row["extras_summary_keys"] = list(extras.keys())

    agg = _load_or_empty(results_json_path(outputs))
    agg["runs"].append(row)
    agg_path = results_json_path(outputs)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    detail: dict[str, Any] = {**row, "detail": extras or {}}
    rpath = run_json_path(outputs, run_id)
    with rpath.open("w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)

    return run_id, rpath


def load_all_runs(outputs: Path) -> list[dict[str, Any]]:
    """Deserialize ``runs`` entries from persistent JSON aggregate."""

    path = results_json_path(outputs)

    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": []}

    rows = payload.get("runs")

    assert isinstance(rows, list)

    return rows


def latest_rows_per_key(outputs: Path) -> list[dict[str, Any]]:
    """Dedup runs by `(method, forget_class)` preferring freshest timestamp."""

    rows = sorted(load_all_runs(outputs), key=lambda r: str(r.get("timestamp_utc", "")), reverse=True)
    uniq: dict[tuple[str, int], dict[str, Any]] = {}
    for item in rows:
        key = (str(item["method"]), int(item["forget_class"]))
        if key not in uniq:
            uniq[key] = item
    return list(uniq.values())
