from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .metrics import classification_cost


def bootstrap_auc_difference(
    y_true,
    proba_a,
    proba_b,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Paired bootstrap for AUC(model A) - AUC(model B)."""
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    differences = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_sample = y_true[idx]
        if np.unique(y_sample).size < 2:
            continue
        differences.append(
            roc_auc_score(y_sample, proba_a[idx])
            - roc_auc_score(y_sample, proba_b[idx])
        )

    differences = np.asarray(differences)
    alpha = 1.0 - confidence_level
    observed = roc_auc_score(y_true, proba_a) - roc_auc_score(y_true, proba_b)

    return {
        "auc_difference": float(observed),
        "ci_lower": float(np.quantile(differences, alpha / 2)),
        "ci_upper": float(np.quantile(differences, 1 - alpha / 2)),
        "p_value_two_sided": float(
            min(
                1.0,
                2 * min(
                    np.mean(differences <= 0),
                    np.mean(differences >= 0),
                ),
            )
        ),
        "probability_a_better": float(np.mean(differences > 0)),
        "n_valid_bootstrap": int(len(differences)),
    }


def bootstrap_cost_difference(
    y_true,
    pred_a,
    pred_b,
    fn_cost: float,
    fp_cost: float,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Paired bootstrap for cost(model A) - cost(model B), per applicant."""
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    differences = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        cost_a = classification_cost(
            y_true[idx], pred_a[idx], fn_cost, fp_cost
        )["cost_per_applicant"]
        cost_b = classification_cost(
            y_true[idx], pred_b[idx], fn_cost, fp_cost
        )["cost_per_applicant"]
        differences.append(cost_a - cost_b)

    differences = np.asarray(differences)
    alpha = 1.0 - confidence_level

    return {
        "mean_cost_difference_per_applicant": float(differences.mean()),
        "ci_lower": float(np.quantile(differences, alpha / 2)),
        "ci_upper": float(np.quantile(differences, 1 - alpha / 2)),
        "probability_a_cheaper": float(np.mean(differences < 0)),
    }
