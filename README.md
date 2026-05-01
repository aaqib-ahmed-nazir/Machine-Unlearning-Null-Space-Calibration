# Machine-Unlearning-Null-Space-Calibration 

FastAPI-based proof-of-concept implementing **Machine Unlearning via Null Space Calibration (UNSC)** on **Fashion-MNIST**, designed for Apple Silicon with **PyTorch MPS**.

- **Paper(used to built the POC)**: [Machine Unlearning via Null Space Calibration (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/0040.pdf) 
- **Report notebook**: `[UNSC_FashionMNIST_Report.ipynb](UNSC_FashionMNIST_Report.ipynb)`

## Objective

Demonstrate **class-level unlearning**: after training an image classifier, forget one selected class while preserving behavior on the retained classes.

The repository provides:

- Training of an original model on full Fashion-MNIST
- UNSC class-forgetting using Algorithm 1 + Algorithm 2
- Baselines: full retrain and random-label fine-tuning
- Disk persistence for models, metrics, and UNSC artifacts

## UNSC in this repo

### Algorithm 1 - Subspaces + projectors (cached)

- Collect layer input features per class (conv layers use patch-columns via `unfold`).
- Compute SVD and keep a truncated orthonormal basis using an energy threshold `ε`.
- Merge retained-class bases and build each per-layer null-space projector:

`P_l = I - Q_l Q_l^T`

- Saved artifacts:
  - `outputs/unsc_subspaces_*.pt` (+ `.meta.json`)
  - `outputs/projection_*.pt`

### Algorithm 2 - Projected unlearning with pseudo-labels

- For each forgotten sample, generate a pseudo-label from the frozen original model excluding the forget class `f`:

`y_hat = argmax_{c != f} theta_o(x)_c`

- Train a student model on forgotten data while projecting each layer gradient by `P_l`.
- Saved artifacts:
  - `saved_models/unlearned_unsc_c<f>.pt`
  - `outputs/results.json`
  - `outputs/run_<run_id>.json`

## Repository structure

```text
.
├── app/
│   ├── baselines.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── main.py
│   ├── model.py
│   ├── results.py
│   ├── schemas.py
│   ├── state.py
│   ├── train.py
│   └── unlearn.py
├── UNSC_FashionMNIST_Report.ipynb  # Analysis + plotting + inference demo
├── app.py                         # Uvicorn entry point
├── outputs/                       # Generated artifacts (created at run time)
├── saved_models/                  # Generated checkpoints
├── data/                          # Fashion-MNIST download cache (gitignored)
└── requirements.txt
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

Open Swagger UI at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Suggested API flow

1. `POST /model/train` — train or warm-load original model `θ_o`
2. `POST /unlearn/select-class` — choose forget class (`0..9`)
3. `POST /unlearn/run` — run UNSC unlearning
4. `POST /baseline/retrain` — full retrain baseline
5. `POST /baseline/random-label` — random-label baseline
6. `GET /results/compare` — latest metrics table
7. `GET /model/evaluate?scope=original|current` — evaluate saved models

## Persistence outputs

- `outputs/results.json` (aggregate runs)
- `outputs/run_<run_id>.json` (per-run curves and metadata)
- `outputs/unsc_subspaces_*.pt` + `.meta.json` (Algorithm 1 caches)
- `outputs/projection_*.pt` (per-layer projectors)
- `saved_models/original_model.pt`
- `saved_models/unlearned_unsc_c<f>.pt`
- `saved_models/retrained_c<f>.pt`
- `saved_models/random_label_c<f>.pt`

## References

- [Machine Unlearning via Null Space Calibration (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/0040.pdf)

