# Part 3 Repository — Churn Prediction Model & Model Card

**Independent & runnable.** Uses **only** pre-snapshot data (≤ `2025-09-30`).  
**Target:** `churn_next_60d` — `1` if the customer made no purchase from `2025-10-01` to `2025-11-29`.

## Overview

This repository builds a 60-day churn classifier for a D2C personal-care brand:

- **Baseline:** Logistic Regression (balanced, scaled features)
- **Production model:** LightGBM with early stopping
- **Threshold:** Tuned on validation assuming FN cost ₹2,000 vs FP cost ₹200 (~10:1)

Data source: `rfm_modeling_snapshot.csv` (included in this repository; download from the capstone Google Drive if missing).

## Setup

```bash
cd part3_churn_model
pip install -r requirements.txt
```

## Run

**Option A — Notebook (recommended for grading):**

```bash
jupyter notebook churn_model.ipynb
```

Run all cells top-to-bottom. The final cell runs `train_pipeline.py` to export artifacts.

**Option B — Script only:**

```bash
python train_pipeline.py
```

## File structure

| File | Description |
|------|-------------|
| `churn_model.ipynb` | Full workflow: EDA checks, training, threshold, validation |
| `train_pipeline.py` | Reproducible training & export script |
| `model.pkl` | Saved artifact: preprocessor, LightGBM, threshold, baseline |
| `metrics.json` | Validation/test metrics and model comparison |
| `error_analysis.md` | ≥10 customer-level FP/FN cases |
| `model_card.md` | Intended use, limits, ethics, monitoring |
| `charts/feature_importance.png` | Top-10 feature importance chart |

## Load model & predict

```python
import joblib
import pandas as pd
from pathlib import Path

artifact = joblib.load("model.pkl")
preprocessor = artifact["preprocessor"]
model = artifact["model"]
threshold = artifact["threshold"]

# Example: score one row from the snapshot (drop target columns)
df = pd.read_csv("rfm_modeling_snapshot.csv")
row = df.iloc[[0]].drop(columns=["customer_id", "snapshot_date", "churn_next_60d", "split"], errors="ignore")
X = preprocessor.transform(row)
proba = model.predict_proba(X)[0, 1]
label = int(proba >= threshold)
print({"churn_probability": round(proba, 4), "predicted_churn": label, "threshold": threshold})
```

## Performance summary (validation)

After running `train_pipeline.py`, see `metrics.json` for exact numbers. Typical results:

| Model | ROC-AUC | F1 @ 0.5 |
|-------|---------|----------|
| Logistic baseline | ~0.88 | ~0.78 |
| LightGBM @ 0.5 | ~0.88 | ~0.76 |
| LightGBM @ business threshold | ~0.88 | ~0.74 (higher recall ~0.95) |

**Threshold rationale:** A lower threshold (~0.20) minimizes expected retention cost when missing a churner is ~10× more expensive than a wasted offer. This prioritizes **recall** for high-LTV protection at the cost of more false positives.

## Leakage guardrails

- Features come from the provided snapshot (pre-`2025-09-30` engineering).
- `churn_next_60d` and `split` are never used as model inputs.
- Train/validation/test splits use the dataset’s `split` column with no customer overlap.

## Submission checklist

- [x] `churn_model.ipynb`
- [x] `model.pkl`
- [x] `metrics.json`
- [x] `error_analysis.md` (≥10 customer IDs)
- [x] `model_card.md`
- [x] Two models + comparison
- [x] Business threshold justification

Push this folder as a **public GitHub repository** for Part 3 submission.
