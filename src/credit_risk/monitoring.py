from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_edges(reference: pd.Series, bins: int) -> np.ndarray:
    clean = pd.to_numeric(reference, errors="coerce").dropna()
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(clean.quantile(quantiles).to_numpy())
    if len(edges) < 3:
        lo, hi = clean.min(), clean.max()
        if lo == hi:
            return np.array([-np.inf, np.inf])
        edges = np.linspace(lo, hi, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def population_stability_index(
    reference,
    current,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    ref = pd.Series(reference)
    cur = pd.Series(current)
    edges = _safe_edges(ref, bins)

    ref_bins = pd.cut(ref, bins=edges, include_lowest=True)
    cur_bins = pd.cut(cur, bins=edges, include_lowest=True)

    ref_dist = ref_bins.value_counts(normalize=True, sort=False).clip(lower=epsilon)
    cur_dist = cur_bins.value_counts(normalize=True, sort=False).clip(lower=epsilon)

    ref_dist, cur_dist = ref_dist.align(cur_dist, fill_value=epsilon)
    psi = ((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum()
    return float(psi)


def feature_psi_table(
    X_reference: pd.DataFrame,
    X_current: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    rows = []
    common = [
        c for c in X_reference.columns
        if c in X_current.columns
        and pd.api.types.is_numeric_dtype(X_reference[c])
    ]

    for col in common:
        try:
            value = population_stability_index(
                X_reference[col],
                X_current[col],
                bins=bins,
            )
        except Exception:
            continue
        rows.append({"feature": col, "psi": value})

    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def create_stress_scenario(X: pd.DataFrame) -> pd.DataFrame:
    """
    Artificially worsen payment behavior.
    This is a methodological stress test, not a real future period.
    """
    stressed = X.copy()

    pay_cols = [c for c in ["PAY_0", "PAY_2", "PAY_3"] if c in stressed.columns]
    for col in pay_cols:
        stressed[col] = stressed[col] + 1

    if "BILL_AMT1" in stressed.columns:
        stressed["BILL_AMT1"] = stressed["BILL_AMT1"] * 1.20

    for col in ["PAY_AMT1", "PAY_AMT2"]:
        if col in stressed.columns:
            stressed[col] = stressed[col] * 0.70

    return stressed
