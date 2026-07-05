from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, proba) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def classification_cost(
    y_true,
    pred,
    fn_cost: float,
    fp_cost: float,
) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    total = fn * fn_cost + fp * fp_cost
    n = len(y_true)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": float(total),
        "cost_per_applicant": float(total / n),
    }


def optimize_business_threshold(
    y_true,
    proba,
    fn_cost: float,
    fp_cost: float,
    thresholds: np.ndarray | None = None,
):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 199)

    rows = []
    for threshold in thresholds:
        pred = (np.asarray(proba) >= threshold).astype(int)
        costs = classification_cost(y_true, pred, fn_cost, fp_cost)
        rows.append({"threshold": threshold, **costs})

    table = pd.DataFrame(rows)
    best_row = table.loc[table["total_cost"].idxmin()]
    return float(best_row["threshold"]), table


def evaluate_predictions(
    y_true,
    proba,
    threshold: float,
    fn_cost: float,
    fp_cost: float,
) -> dict:
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, proba)
    costs = classification_cost(y_true, pred, fn_cost, fp_cost)

    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, proba),
        "average_precision": float(average_precision_score(y_true, proba)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        **costs,
    }
