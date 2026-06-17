# Error Analysis — Validation Set

Business framing: **False positives** waste ~₹200 per unnecessary retention offer; **false negatives** risk ~₹2000 LTV loss from customers who churn without intervention.

Threshold used: **0.20** (optimized on validation with FN:FP cost ratio 10:1).

- False positives on validation: **90**
- False negatives on validation: **7**

## False Positives (predicted churn, actually stayed)

### CUST00961

| Metric | Value |
|--------|-------|
| Predicted probability | 0.898 |
| Actual churn | 0 |
| recency_days | 151 |
| ticket_count_90d | 0 |
| sessions_30d | 2 |
| monetary_180d | ₹1378.71 |

**Interpretation:** Model flagged churn risk, but customer purchased within 60 days. Likely drivers in features: long recency (151 days since last order); low engagement (2 sessions); historically high spend (₹1379); repeat buyer (2 orders). Retention outreach may have been unnecessary.

### CUST01744

| Metric | Value |
|--------|-------|
| Predicted probability | 0.887 |
| Actual churn | 0 |
| recency_days | 111 |
| ticket_count_90d | 0 |
| sessions_30d | 3 |
| monetary_180d | ₹1021.08 |

**Interpretation:** Model flagged churn risk, but customer purchased within 60 days. Likely drivers in features: long recency (111 days since last order); historically high spend (₹1021). Retention outreach may have been unnecessary.

### CUST02364

| Metric | Value |
|--------|-------|
| Predicted probability | 0.886 |
| Actual churn | 0 |
| recency_days | 153 |
| ticket_count_90d | 0 |
| sessions_30d | 1 |
| monetary_180d | ₹1091.09 |

**Interpretation:** Model flagged churn risk, but customer purchased within 60 days. Likely drivers in features: long recency (153 days since last order); low engagement (1 sessions); historically high spend (₹1091); repeat buyer (2 orders). Retention outreach may have been unnecessary.

### CUST02313

| Metric | Value |
|--------|-------|
| Predicted probability | 0.863 |
| Actual churn | 0 |
| recency_days | 171 |
| ticket_count_90d | 0 |
| sessions_30d | 3 |
| monetary_180d | ₹583.08 |

**Interpretation:** Model flagged churn risk, but customer purchased within 60 days. Likely drivers in features: long recency (171 days since last order). Retention outreach may have been unnecessary.

### CUST01864

| Metric | Value |
|--------|-------|
| Predicted probability | 0.861 |
| Actual churn | 0 |
| recency_days | 221 |
| ticket_count_90d | 0 |
| sessions_30d | 3 |
| monetary_180d | ₹0.00 |

**Interpretation:** Model flagged churn risk, but customer purchased within 60 days. Likely drivers in features: long recency (221 days since last order). Retention outreach may have been unnecessary.

### CUST00165

| Metric | Value |
|--------|-------|
| Predicted probability | 0.829 |
| Actual churn | 0 |
| recency_days | 103 |
| ticket_count_90d | 0 |
| sessions_30d | 2 |
| monetary_180d | ₹1825.77 |

**Interpretation:** Model flagged churn risk, but customer purchased within 60 days. Likely drivers in features: long recency (103 days since last order); low engagement (2 sessions); historically high spend (₹1826); repeat buyer (3 orders). Retention outreach may have been unnecessary.

## False Negatives (predicted stay, actually churned)

### CUST00727

| Metric | Value |
|--------|-------|
| Predicted probability | 0.029 |
| Actual churn | 1 |
| recency_days | 8 |
| ticket_count_90d | 0 |
| sessions_30d | 10 |
| monetary_180d | ₹2696.01 |

**Interpretation:** Model missed a churner. Feature pattern: historically high spend (₹2696); repeat buyer (4 orders). High business risk (estimated ₹2000 LTV loss if not retained).

### CUST01700

| Metric | Value |
|--------|-------|
| Predicted probability | 0.053 |
| Actual churn | 1 |
| recency_days | 6 |
| ticket_count_90d | 0 |
| sessions_30d | 14 |
| monetary_180d | ₹2044.90 |

**Interpretation:** Model missed a churner. Feature pattern: historically high spend (₹2045); repeat buyer (2 orders). High business risk (estimated ₹2000 LTV loss if not retained).

### CUST02096

| Metric | Value |
|--------|-------|
| Predicted probability | 0.053 |
| Actual churn | 1 |
| recency_days | 5 |
| ticket_count_90d | 0 |
| sessions_30d | 9 |
| monetary_180d | ₹1424.87 |

**Interpretation:** Model missed a churner. Feature pattern: historically high spend (₹1425); repeat buyer (2 orders). High business risk (estimated ₹2000 LTV loss if not retained).

### CUST00850

| Metric | Value |
|--------|-------|
| Predicted probability | 0.064 |
| Actual churn | 1 |
| recency_days | 7 |
| ticket_count_90d | 1 |
| sessions_30d | 5 |
| monetary_180d | ₹2278.03 |

**Interpretation:** Model missed a churner. Feature pattern: recent support activity (1 tickets); historically high spend (₹2278); repeat buyer (2 orders). High business risk (estimated ₹2000 LTV loss if not retained).

### CUST00188

| Metric | Value |
|--------|-------|
| Predicted probability | 0.074 |
| Actual churn | 1 |
| recency_days | 29 |
| ticket_count_90d | 1 |
| sessions_30d | 11 |
| monetary_180d | ₹1880.31 |

**Interpretation:** Model missed a churner. Feature pattern: recent support activity (1 tickets); historically high spend (₹1880); repeat buyer (2 orders). High business risk (estimated ₹2000 LTV loss if not retained).

### CUST01626

| Metric | Value |
|--------|-------|
| Predicted probability | 0.086 |
| Actual churn | 1 |
| recency_days | 28 |
| ticket_count_90d | 0 |
| sessions_30d | 6 |
| monetary_180d | ₹2085.26 |

**Interpretation:** Model missed a churner. Feature pattern: historically high spend (₹2085); repeat buyer (3 orders). High business risk (estimated ₹2000 LTV loss if not retained).

## Summary Insights

- **FP pattern:** Model overweighted inactivity (high `recency_days`, low `sessions_30d`) while recent spend or latent loyalty still led to a repurchase.
- **FN pattern:** Customers with moderate engagement signals looked healthy to the model but did not return within the 60-day window—often mid recency with few tickets.
- **Action:** Route borderline high-`monetary_180d` FN cases to human review; cap discounts for FP-prone low-LTV segments.
