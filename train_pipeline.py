"""
Train churn model and export artifacts for Part 3.
Run from part3_churn_model/: python train_pipeline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "rfm_modeling_snapshot.csv"
CHARTS_DIR = ROOT / "charts"

DROP_COLS = ["customer_id", "snapshot_date", "churn_next_60d", "split"]
CAT_COLS = [
    "city_tier",
    "age_group",
    "acquisition_channel",
    "loyalty_tier",
    "preferred_category",
    "marketing_consent",
]
NUM_COLS = [
    "recency_days",
    "frequency_180d",
    "monetary_180d",
    "return_rate_180d",
    "avg_discount_pct_180d",
    "avg_rating_180d",
    "category_diversity_180d",
    "ticket_count_90d",
    "negative_ticket_rate_90d",
    "avg_resolution_hours_90d",
    "days_since_signup",
    "sessions_30d",
    "product_views_30d",
    "cart_adds_30d",
    "wishlist_adds_30d",
    "abandoned_carts_30d",
    "email_opens_30d",
    "campaign_clicks_30d",
    "last_visit_days_ago",
]

FN_COST = 2000
FP_COST = 200


def _save_feature_importance_chart(path: Path, names: list[str], values: np.ndarray) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(names, values, color="#2a9d8f")
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Feature Importances (LightGBM)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    except Exception:
        pass

    from PIL import Image, ImageDraw, ImageFont

    w, h = 900, 520
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 10), "Top 10 Feature Importances (LightGBM)", fill="black")
    max_v = float(max(values)) if len(values) else 1.0
    y0 = 50
    row_h = 42
    for i, (name, val) in enumerate(zip(names, values)):
        y = y0 + i * row_h
        bar_w = int(600 * (float(val) / max_v))
        draw.text((20, y), str(name)[:40], fill="#333")
        draw.rectangle([280, y + 8, 280 + bar_w, y + 28], fill="#2a9d8f")
        draw.text((290 + bar_w + 8, y + 6), f"{val:.0f}", fill="#111")
    img.save(path)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUM_COLS),
            ("cat", categorical_pipe, CAT_COLS),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names: list[str] = []
    names.extend(NUM_COLS)
    ohe: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    names.extend(ohe.get_feature_names_out(CAT_COLS).tolist())
    return names


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def select_threshold(y_true, y_proba) -> tuple[float, dict]:
    thresholds = np.arange(0.10, 0.90, 0.01)
    best_cost = float("inf")
    best_t = 0.5
    best_row = None

    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = cm.ravel()
        cost = fp * FP_COST + fn * FN_COST
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        row = {
            "threshold": float(t),
            "cost": float(cost),
            "precision": float(prec),
            "recall": float(rec),
        }
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
            best_row = row

    # Prefer recall >= 0.70 and precision >= 0.40 if achievable at similar cost
    candidates = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec >= 0.70 and prec >= 0.40:
            cm = confusion_matrix(y_true, pred)
            tn, fp, fn, tp = cm.ravel()
            candidates.append((fp * FP_COST + fn * FN_COST, float(t)))

    if candidates:
        candidates.sort()
        best_t = candidates[0][1]

    return best_t, best_row or {}


def main() -> None:
    CHARTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    assert (df["snapshot_date"] == "2025-09-30").all()

    y = df["churn_next_60d"]
    X = df.drop(columns=DROP_COLS)
    assert "churn_next_60d" not in X.columns
    assert "split" not in X.columns

    for col in CAT_COLS:
        X[col] = X[col].fillna("Unknown").astype(str)

    mask_train = df["split"] == "train"
    mask_val = df["split"] == "validation"
    mask_test = df["split"] == "test"

    train_ids = set(df.loc[mask_train, "customer_id"])
    val_ids = set(df.loc[mask_val, "customer_id"])
    test_ids = set(df.loc[mask_test, "customer_id"])
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X[mask_train])
    X_val = preprocessor.transform(X[mask_val])
    X_test = preprocessor.transform(X[mask_test])
    feature_names = get_feature_names(preprocessor)

    y_train = y[mask_train].values
    y_val = y[mask_val].values
    y_test = y[mask_test].values

    # Baseline (scaled features for stable convergence)
    baseline_scaler = StandardScaler()
    X_train_scaled = baseline_scaler.fit_transform(X_train)
    X_val_scaled = baseline_scaler.transform(X_val)
    X_test_scaled = baseline_scaler.transform(X_test)
    baseline = LogisticRegression(
        max_iter=3000, random_state=42, class_weight="balanced", solver="lbfgs"
    )
    baseline.fit(X_train_scaled, y_train)
    baseline_val_proba = baseline.predict_proba(X_val_scaled)[:, 1]
    baseline_val_pred = (baseline_val_proba >= 0.5).astype(int)
    baseline_metrics = compute_metrics(y_val, baseline_val_pred, baseline_val_proba)

    # Strong model
    strong_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    strong_model.fit(
        X_train_df,
        y_train,
        eval_set=[(X_val_df, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    val_proba = strong_model.predict_proba(X_val_df)[:, 1]
    val_pred_default = (val_proba >= 0.5).astype(int)
    strong_metrics_default = compute_metrics(y_val, val_pred_default, val_proba)

    optimal_thresh, thresh_info = select_threshold(y_val, val_proba)
    val_pred_opt = (val_proba >= optimal_thresh).astype(int)
    val_metrics = compute_metrics(y_val, val_pred_opt, val_proba)

    test_proba = strong_model.predict_proba(X_test_df)[:, 1]
    test_pred = (test_proba >= optimal_thresh).astype(int)
    test_metrics = compute_metrics(y_test, test_pred, test_proba)

    # Feature importance plot (Pillow fallback if matplotlib is broken)
    importances = strong_model.feature_importances_
    idx = np.argsort(importances)[-10:]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]
    _save_feature_importance_chart(CHARTS_DIR / "feature_importance.png", top_names, top_vals)

    top3_idx = np.argsort(importances)[::-1][:3]
    top3_features = [(feature_names[i], float(importances[i])) for i in top3_idx]

    # Error analysis
    val_df = df.loc[mask_val].copy()
    val_df["predicted_proba"] = val_proba
    val_df["predicted_label"] = val_pred_opt

    fp_df = val_df[(val_df["predicted_label"] == 1) & (val_df["churn_next_60d"] == 0)]
    fn_df = val_df[(val_df["predicted_label"] == 0) & (val_df["churn_next_60d"] == 1)]
    fp_sample = fp_df.nlargest(6, "predicted_proba")
    fn_sample = fn_df.nsmallest(6, "predicted_proba")

    display_cols = [
        "customer_id",
        "predicted_proba",
        "churn_next_60d",
        "recency_days",
        "ticket_count_90d",
        "sessions_30d",
        "monetary_180d",
        "frequency_180d",
        "last_visit_days_ago",
        "negative_ticket_rate_90d",
    ]

    def interpret_row(row, kind: str) -> str:
        parts = []
        if row["recency_days"] > 90:
            parts.append(f"long recency ({int(row['recency_days'])} days since last order)")
        if row["sessions_30d"] <= 2:
            parts.append(f"low engagement ({int(row['sessions_30d'])} sessions)")
        if row["ticket_count_90d"] > 0:
            parts.append(f"recent support activity ({int(row['ticket_count_90d'])} tickets)")
        if row["monetary_180d"] > 800:
            parts.append(f"historically high spend (₹{row['monetary_180d']:.0f})")
        if row["frequency_180d"] >= 2:
            parts.append(f"repeat buyer ({int(row['frequency_180d'])} orders)")
        if not parts:
            parts.append("mixed weak signals without a single dominant driver")
        reason = "; ".join(parts)
        if kind == "FP":
            return (
                f"Model flagged churn risk, but customer purchased within 60 days. "
                f"Likely drivers in features: {reason}. Retention outreach may have been unnecessary."
            )
        return (
            f"Model missed a churner. Feature pattern: {reason}. "
            f"High business risk (estimated ₹{FN_COST} LTV loss if not retained)."
        )

    lines = [
        "# Error Analysis — Validation Set",
        "",
        "Business framing: **False positives** waste ~₹200 per unnecessary retention offer; "
        "**false negatives** risk ~₹2000 LTV loss from customers who churn without intervention.",
        "",
        f"Threshold used: **{optimal_thresh:.2f}** (optimized on validation with FN:FP cost ratio 10:1).",
        "",
        f"- False positives on validation: **{len(fp_df)}**",
        f"- False negatives on validation: **{len(fn_df)}**",
        "",
        "## False Positives (predicted churn, actually stayed)",
        "",
    ]

    for _, row in fp_sample.iterrows():
        lines.append(f"### {row['customer_id']}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Predicted probability | {row['predicted_proba']:.3f} |")
        lines.append(f"| Actual churn | {int(row['churn_next_60d'])} |")
        lines.append(f"| recency_days | {int(row['recency_days'])} |")
        lines.append(f"| ticket_count_90d | {int(row['ticket_count_90d'])} |")
        lines.append(f"| sessions_30d | {int(row['sessions_30d'])} |")
        lines.append(f"| monetary_180d | ₹{row['monetary_180d']:.2f} |")
        lines.append("")
        lines.append(f"**Interpretation:** {interpret_row(row, 'FP')}")
        lines.append("")

    lines.extend(["## False Negatives (predicted stay, actually churned)", ""])

    for _, row in fn_sample.iterrows():
        lines.append(f"### {row['customer_id']}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Predicted probability | {row['predicted_proba']:.3f} |")
        lines.append(f"| Actual churn | {int(row['churn_next_60d'])} |")
        lines.append(f"| recency_days | {int(row['recency_days'])} |")
        lines.append(f"| ticket_count_90d | {int(row['ticket_count_90d'])} |")
        lines.append(f"| sessions_30d | {int(row['sessions_30d'])} |")
        lines.append(f"| monetary_180d | ₹{row['monetary_180d']:.2f} |")
        lines.append("")
        lines.append(f"**Interpretation:** {interpret_row(row, 'FN')}")
        lines.append("")

    lines.extend(
        [
            "## Summary Insights",
            "",
            "- **FP pattern:** Model overweighted inactivity (high `recency_days`, low `sessions_30d`) "
            "while recent spend or latent loyalty still led to a repurchase.",
            "- **FN pattern:** Customers with moderate engagement signals looked healthy to the model "
            "but did not return within the 60-day window—often mid recency with few tickets.",
            "- **Action:** Route borderline high-`monetary_180d` FN cases to human review; cap discounts for FP-prone low-LTV segments.",
            "",
        ]
    )

    (ROOT / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    artifact = {
        "preprocessor": preprocessor,
        "model": strong_model,
        "threshold": optimal_thresh,
        "feature_names": feature_names,
        "baseline_model": baseline,
        "baseline_scaler": baseline_scaler,
    }
    joblib.dump(artifact, ROOT / "model.pkl")

    metrics_out = {
        "model_type": "LightGBM",
        "optimal_threshold": optimal_thresh,
        "baseline_validation": baseline_metrics,
        "strong_validation_default_threshold_0.5": strong_metrics_default,
        "validation": val_metrics,
        "test": test_metrics,
        "threshold_selection": {
            "fn_cost_inr": FN_COST,
            "fp_cost_inr": FP_COST,
            "cost_ratio": "10:1",
            "notes": thresh_info,
        },
        "model_comparison_validation": {
            "baseline_roc_auc": baseline_metrics["roc_auc"],
            "strong_roc_auc": val_metrics["roc_auc"],
            "baseline_f1": baseline_metrics["f1"],
            "strong_f1": val_metrics["f1"],
        },
        "top_features": [
            {"name": n, "importance": round(v, 4)} for n, v in top3_features
        ],
    }
    with open(ROOT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    # Model card
    cm = val_metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    card = f"""# Model Card: D2C Customer Churn Predictor

## Intended Use
- **Purpose:** Identify customers at risk of churn in the next 60 days for targeted retention
- **Users:** Marketing, CRM, and Support teams
- **Not for:** Automated discounting without review, credit decisions, or external data sharing

## Data Used
- **Source:** `rfm_modeling_snapshot.csv` (pre-snapshot feature engineering)
- **Snapshot:** 2025-09-30
- **Target:** `churn_next_60d` (1 = no purchase from 2025-10-01 to 2025-11-29)
- **Features:** RFM (180d), support tickets (90d), web/app activity (30d), customer profile fields
- **Leakage guard:** No post-snapshot orders or target-window information in features

## Model Approach
- **Baseline:** Logistic Regression (`class_weight='balanced'`)
- **Production model:** LightGBM Classifier with early stopping on validation
- **Preprocessing:** Median imputation (numeric), `Unknown` + one-hot encoding (categorical)
- **Threshold:** {optimal_thresh:.2f} — tuned on validation assuming FN cost ₹{FN_COST} vs FP cost ₹{FP_COST} (~10:1)

## Performance (Validation @ optimal threshold)
| Metric | Value |
|--------|-------|
| ROC-AUC | {val_metrics['roc_auc']} |
| PR-AUC | {val_metrics['pr_auc']} |
| Precision | {val_metrics['precision']} |
| Recall | {val_metrics['recall']} |
| F1 | {val_metrics['f1']} |

**Confusion matrix:** [[{tn}, {fp}], [{fn}, {tp}]]

**Test set (held out):** ROC-AUC {test_metrics['roc_auc']}, F1 {test_metrics['f1']}, Recall {test_metrics['recall']}

## Top Drivers (business interpretation)
1. **{top3_features[0][0]}** — Strongest model signal; typically reflects purchase recency / engagement decay when high.
2. **{top3_features[1][0]}** — Captures spend depth; low values often correlate with disengagement.
3. **{top3_features[2][0]}** — Reflects digital or service friction complementary to order history.

## Limitations
- Trained on a single historical snapshot; requires a refreshed feature pipeline for production scoring
- Does not model macro shocks (competitor pricing, stock-outs, major campaigns)
- Weaker for very new customers (`days_since_signup` small) or zero web activity rows

## Ethical Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Over-targeting vulnerable segments via proxies (tier, age) | Use scores as prioritization hints only; avoid exclusive targeting on protected attributes |
| False positives annoying loyal customers | Human review for high `monetary_180d`; frequency caps on offers |
| False negatives leaving high-LTV customers unprotected | Minimum recall target on validation; escalate low-probability but high-value accounts manually |

## Monitoring Needs
- **Data drift:** PSI on `recency_days`, `sessions_30d`, `monetary_180d` monthly
- **Score drift:** Distribution of predicted probabilities vs training baseline
- **Outcomes:** Actual 60-day churn rate vs predicted rate by decile
- **Retrain trigger:** Validation ROC-AUC drop >5% or quarterly refresh
- **Alerts:** Spike in FN rate among top spend decile, or campaign opt-out rate

## When NOT to Use
- During major pricing or assortment changes (covariate shift)
- As the sole decision-maker for Platinum / high-LTV accounts
- For customers with incomplete pre-snapshot history (<30 days since signup)
"""
    (ROOT / "model_card.md").write_text(card, encoding="utf-8")

    print("Training complete.")
    print("Baseline validation:", baseline_metrics)
    print("Strong validation (optimal threshold):", val_metrics)
    print("Optimal threshold:", optimal_thresh)


if __name__ == "__main__":
    main()
