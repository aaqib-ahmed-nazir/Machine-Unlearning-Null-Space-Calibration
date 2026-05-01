# UNSC Fashion-MNIST (FastAPI + PyTorch)

Proof-of-concept for **machine unlearning via null space calibration (UNSC)** on Fashion-MNIST, runnable on Apple Silicon via **PyTorch MPS**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API

```bash
python3 run_api.py
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger UI.

Suggested flow:

1. `POST /model/train` — train or warm-load θ_o (`saved_models/original_model.pt`)
2. `POST /unlearn/select-class` — choose forget class id `0–9`
3. `POST /unlearn/run` — UNSC (Algorithms 1 & 2)
4. `POST /baseline/retrain` — gold retained-only baseline
5. `POST /baseline/random-label` — random-label baseline
6. `GET /results/compare` — latest deduplicated leaderboard rows  
7. `GET /model/evaluate?scope=original|current`

Artifacts: **`saved_models/`** (weights), **`outputs/`** (`results.json`, projector dumps, Algorithm 1 cache keyed by θ_o digest).

## Notes

- Async routes call blocking PyTorch work in-process (POC simplicity).
- For analysis notebooks later, consume `outputs/results.json` + `outputs/run_*.json` and regenerate plots offline.
