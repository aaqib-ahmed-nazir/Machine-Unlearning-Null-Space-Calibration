# UNSC Unlearning POC — Fashion-MNIST (FastAPI + PyTorch MPS)

FastAPI-based proof-of-concept implementing **Machine Unlearning via Null Space Calibration (UNSC)** on **Fashion-MNIST**. Designed to run well on Apple Silicon using **PyTorch MPS**.

## Objective

Demonstrate **class-level unlearning**: after training an image classifier, “forget” one selected class while preserving performance on the remaining classes.

This repo provides:

- **Training** an original model on full data
- **UNSC unlearning** (paper-faithful Algorithms 1 & 2)
- **Baselines**: full retrain (gold standard) and random-label fine-tuning
- **Persistence**: all models + results + UNSC artifacts saved to disk for later plotting/analysis in a notebook

## UNSC in this repo (concise)

- **Algorithm 1 (Subspace discovery + projector build)**:
  - For each class and each projection-enabled layer, collect **layer input features** (conv layers use patch-columns via `unfold`).
  - Compute SVD and keep a truncated orthonormal basis using an energy threshold \( \epsilon \).
  - Merge retained-class bases and build per-layer null-space projectors: \( P_\ell = I - Q_\ell Q_\ell^\top \).
  - **Saved to** `outputs/unsc_subspaces_*.pt` and `outputs/projection_*.pt`.

- **Algorithm 2 (Projected unlearning with pseudo-labels)**:
  - For forgotten samples, generate pseudo-labels from the frozen original model excluding the forget class:
    \( \tilde{y} = \arg\max_{c \neq f} \theta_o(x)_c \)
  - Train a student model on forgotten samples while projecting gradients at each layer using \( P_\ell \) to protect retained behavior.
  - **Saved model** to `saved_models/unlearned_unsc_c<f>.pt` and run logs to `outputs/`.

## Repository structure

```text
.
├── app/                       # Core modules
│   ├── main.py                # FastAPI app + endpoints (all async def)
│   ├── schemas.py             # Pydantic request/response models
│   ├── state.py               # In-process state (models/loaders/forget_class)
│   ├── data_loader.py         # Fashion-MNIST loaders + retained/forgotten splits
│   ├── model.py               # Small CNN + hooks for UNSC layer inputs
│   ├── train.py               # Supervised training utilities
│   ├── evaluate.py            # overall/retained/forgotten accuracy metrics
│   ├── unlearn.py             # UNSC Algorithm 1 + Algorithm 2
│   ├── baselines.py           # Full retrain + random-label baselines
│   └── results.py             # Append-only JSON persistence (`outputs/results.json`)
├── outputs/                   # Persisted run results + UNSC caches (generated)
├── saved_models/              # Persisted checkpoints (generated)
├── data/                      # Fashion-MNIST download cache (gitignored)
├── UNSC_FashionMNIST_Report.ipynb  # Notebook: explanations + plots + inference demo
├── app.py                     # Uvicorn launcher for FastAPI
├── requirements.txt
└── PRD.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
python3 app.py
```

Then open Swagger UI at `http://127.0.0.1:8000/docs`.

## Suggested API flow (demo)

1. `POST /model/train` — train or warm-load the original model \( \theta_o \)
2. `POST /unlearn/select-class` — choose forget class id `0–9`
3. `POST /unlearn/run` — run UNSC (Algorithms 1 & 2) and persist artifacts
4. `POST /baseline/retrain` — full retrain baseline on retained data only
5. `POST /baseline/random-label` — random-label fine-tune baseline
6. `GET /results/compare` — show latest metrics per `(method, forget_class)`
7. `GET /model/evaluate?scope=original|current` — evaluate the original vs current model

## Outputs and persistence

- **Aggregate runs**: `outputs/results.json`
- **Per-run detail** (curves/histories): `outputs/run_<run_id>.json`
- **UNSC Algorithm 1 caches**: `outputs/unsc_subspaces_*.pt` + `.meta.json`
- **UNSC projectors**: `outputs/projection_*.pt`
- **Checkpoints**:
  - `saved_models/original_model.pt`
  - `saved_models/unlearned_unsc_c<f>.pt`
  - `saved_models/retrained_c<f>.pt`
  - `saved_models/random_label_c<f>.pt`

## Notebook (plots + inference)

Open and run `UNSC_FashionMNIST_Report.ipynb` to:

- Read the math + explanation of Algorithms 1 & 2
- Generate bar plots and learning/unlearning curves from saved results
- Run a small inference demo using any saved checkpoint

## Notes

- FastAPI endpoints are `async def`, but the PyTorch work runs synchronously in-process (POC simplicity).
- SVD/projector computations run on CPU for stability; model training/inference can use MPS when available.

## References

- **Paper (IJCAI 2024)**: `https://www.ijcai.org/proceedings/2024/0040.pdf`
