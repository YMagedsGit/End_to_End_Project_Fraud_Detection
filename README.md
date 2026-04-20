# Credit Card Fraud Detection — Imbalanced Classification with Threshold Tuning

An end-to-end fraud detection pipeline on 284,807 credit card transactions, covering class imbalance handling, threshold optimization, Precision-Recall analysis, and SHAP interpretability — built with a time-based split and a trial in production thinking.

---

## Project Overview

Fraud detection is one of the most demanding classification problems in practice: the positive class (fraud) represents less than 0.2% of all transactions, standard metrics like accuracy are misleading, and the cost of different error types is asymmetric — missing a fraud is not the same as flagging a legitimate transaction.

This project addresses all three dimensions: handling extreme class imbalance through two distinct strategies, evaluating with Precision-Recall AUC instead of ROC-AUC, and tuning the decision threshold explicitly rather than accepting the default 0.5 — including a tiered deployment strategy that mirrors how fraud systems work in production.

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
**Size:** 284,807 transactions, 31 features
**Target:** `Class` (0 = Legitimate, 1 = Fraud)
**Imbalance ratio:** ~580:1 (492 fraud cases out of 284,807)

**Feature notes:**
- V1–V28 are PCA-transformed components from the original transaction data (anonymized for confidentiality) and i am suggesting that they are orthogonal 
- `Time` and `Amount` are the only raw features
- Because V1–V28 are already PCA-scaled, only `Time` and `Amount` required additional scaling

---

## Approach

### 1. Time-Based Split
Row order in this dataset proxies transaction time. The first 80% of rows were used for training and the last 20% for testing — preserving the temporal structure of the data. A random split was avoided because:
- Fraud patterns evolve over time; a random split leaks future fraud signatures into training
- Real deployment always predicts on future transactions, never on randomly sampled historical ones

### 2. Class Imbalance — Two Strategies Compared

**Strategy A — SMOTE (Synthetic Minority Oversampling)**
Generates synthetic fraud samples by interpolating between real minority class examples in feature space. Applied only to the training set — never the test set, since the test set must reflect the real-world class distribution for evaluation to be meaningful.

**Strategy B — Class Weighting (`scale_pos_weight`)**
Tells XGBoost to penalize misclassification of the minority class more heavily during training. No synthetic data is generated — the original distribution is preserved.

| Strategy | PR-AUC | F1 (Fraud) | Precision | Recall |
|---|---|---|---|---|
| XGBoost + SMOTE | 0.775 | 0.60 | 0.49 | 0.77 |
| **XGBoost + scale_pos_weight=200** | **0.803** | **0.80** | **0.85** | **0.76** |

**Why class weighting outperformed SMOTE here:**
SMOTE assumes the space between two minority class examples is also minority space. In fraud detection, fraudsters deliberately make transactions resemble legitimate ones — meaning minority class examples are often surrounded by majority class points in feature space. Interpolating between fraud examples in that region generates synthetic points that overlap heavily with legitimate transactions, blurring the decision boundary rather than sharpening it.

### 3. Modeling

Three models were trained and evaluated with Precision-Recall AUC as the primary metric:

| Model | PR-AUC | F1 (Fraud) | ROC-AUC |
|---|---|---|---|
| Logistic Regression (SMOTE) | 0.779 | 0.06 | — |
| LightGBM (SMOTE) | 0.776 | 0.43 | — |
| **XGBoost (scale_pos_weight=200)** | **0.803** | **0.80** | **0.976** |

**Why PR-AUC over ROC-AUC:**
ROC-AUC measures performance across all thresholds using True Positive Rate vs False Positive Rate. With a 580:1 imbalance, even a model with thousands of false positives has a low FPR because the denominator (total negatives) is enormous. PR-AUC uses Precision vs Recall — both computed only from the minority class — making it a far more honest evaluation metric when the positive class is rare.

### 4. Isolation Forest as a Feature Generator
Isolation Forest was initially tested as a standalone anomaly detector but achieved PR-AUC of only 0.04 — unsurprising, since it has no access to labels during training. Rather than discarding it, the `decision_function` scores (anomaly scores) were used as an engineered feature fed into XGBoost, allowing the supervised model to incorporate unsupervised anomaly signal.

### 5. Threshold Tuning
The default classification threshold of 0.5 is rarely optimal for imbalanced problems. Two threshold strategies were explored:

**Strategy 1 — Maximize F1**
The threshold that maximizes F1 score on the test set, found by scanning the full Precision-Recall curve:
- **Optimal threshold: 0.76** (validated with TimeSeriesSplit cross-validation, std = 0.006 across folds — indicating a stable decision boundary)

**Strategy 2 — 90% Recall floor**
The highest-precision threshold that still catches at least 90% of fraud:
- At this threshold: for every 1 fraud caught, approximately 40 legitimate transactions are flagged — generating ~2,800 false alarms per 10,000 transactions

**Tiered deployment strategy:**

| Zone | Threshold | Action |
|---|---|---|
| High confidence fraud | > 0.97 | Auto-decline |
| Suspicious | 0.50 – 0.97 | Trigger MFA / step-up authentication |
| Low risk | < 0.50 | Approve automatically |

This mirrors how fraud systems actually operate — hard declines are reserved for near-certain fraud, while the middle zone triggers friction rather than blocking legitimate customers.

### 6. SHAP Interpretability
SHAP values were computed on a 500-row sample of the test set:

**Top 3 features driving fraud predictions:**
1. `V14` — strongest single predictor; large negative values strongly associated with fraud
2. `V4` — positive values push predictions toward fraud
3. `V12` — contributes across a wide range of transaction types

A waterfall plot for a correctly detected fraud transaction shows the exact SHAP contribution of each feature — which PCA components pushed the prediction toward fraud and by how much.

**On interpreting PCA features:**
V1–V28 cannot be mapped back to original transaction attributes due to confidentiality. SHAP values still provide value: they identify which anonymized components the model relies on most heavily, enabling monitoring for feature drift in production even without knowing what those components represent.

---

## Key Takeaways

- PR-AUC is the correct primary metric for extreme class imbalance — ROC-AUC is misleadingly optimistic when the negative class dominates the FPR denominator
- SMOTE can actively hurt performance when minority class examples are surrounded by majority class points in feature space — class weighting is a safer default for fraud data
- The default threshold of 0.5 should never be assumed optimal; threshold tuning against the Precision-Recall curve is a required step before deployment
- Cross-validating the optimal threshold (not just the model) gives confidence that the decision boundary is stable across time windows
- A tiered threshold strategy (auto-decline / step-up auth / approve) is more practical than a single binary cutoff and directly maps model output to business actions
- Catching 95% of fraud means nothing without knowing the dollar value of the 5% that slips through — Recall alone does not measure business impact

---

## Results Summary

| Metric | Value |
|---|---|
| Best Model | XGBoost (scale_pos_weight=200) |
| PR-AUC | 0.803 |
| F1 on Fraud Class | 0.80 |
| Optimal Threshold (max F1) | 0.76 |
| Threshold Stability (CV std) | 0.006 |
| Dataset Size | 284,807 transactions |
| Fraud Rate | 0.172% (492 / 284,807) |
