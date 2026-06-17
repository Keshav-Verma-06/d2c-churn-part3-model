# Model Card: D2C Customer Churn Predictor

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
- **Threshold:** 0.20 — tuned on validation assuming FN cost ₹2000 vs FP cost ₹200 (~10:1)

## Performance (Validation @ optimal threshold)
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8752 |
| PR-AUC | 0.8675 |
| Precision | 0.6087 |
| Recall | 0.9524 |
| F1 | 0.7427 |

**Confusion matrix:** [[99, 90], [7, 140]]

**Test set (held out):** ROC-AUC 0.8766, F1 0.7651, Recall 0.9405

## Top Drivers (business interpretation)
1. **recency_days** — Strongest model signal; typically reflects purchase recency / engagement decay when high.
2. **monetary_180d** — Captures spend depth; low values often correlate with disengagement.
3. **days_since_signup** — Reflects digital or service friction complementary to order history.

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
