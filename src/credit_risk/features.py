from __future__ import annotations

import numpy as np
import pandas as pd


PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
]
PAYMENT_COLS = [
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def _has_columns(df: pd.DataFrame, cols: list[str]) -> bool:
    return all(col in df.columns for col in cols)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stable, interpretable aggregates used in both baseline and extended sets."""
    out = df.copy()

    if _has_columns(out, PAY_STATUS_COLS):
        pay = out[PAY_STATUS_COLS].clip(lower=0)
        out["avg_delay"] = pay.mean(axis=1)
        out["max_delay"] = pay.max(axis=1)
        out["months_with_delay"] = (pay > 0).sum(axis=1)

    if _has_columns(out, BILL_COLS):
        bills = out[BILL_COLS]
        out["avg_bill"] = bills.mean(axis=1)
        out["max_bill"] = bills.max(axis=1)
        out["bill_trend"] = bills["BILL_AMT1"] - bills["BILL_AMT6"]

    if _has_columns(out, PAYMENT_COLS):
        payments = out[PAYMENT_COLS]
        out["avg_payment"] = payments.mean(axis=1)
        out["max_payment"] = payments.max(axis=1)

    if _has_columns(out, BILL_COLS + PAYMENT_COLS):
        total_bill = out[BILL_COLS].clip(lower=0).sum(axis=1)
        total_payment = out[PAYMENT_COLS].clip(lower=0).sum(axis=1)
        out["payment_to_bill_ratio"] = total_payment / (total_bill + 1.0)

    if "LIMIT_BAL" in out.columns and _has_columns(out, BILL_COLS):
        latest_bill = out["BILL_AMT1"].clip(lower=0)
        out["utilization"] = latest_bill / (out["LIMIT_BAL"].clip(lower=0) + 1.0)

    return out


def add_innovative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add behavioral features whose incremental value will be tested separately.
    """
    out = add_base_features(df)

    if _has_columns(out, PAY_STATUS_COLS):
        historical = out[["PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].mean(axis=1)
        out["delinquency_momentum"] = out["PAY_0"] - historical

    if _has_columns(out, BILL_COLS + PAYMENT_COLS):
        bills = out[BILL_COLS].clip(lower=0).to_numpy(dtype=float)
        payments = out[PAYMENT_COLS].clip(lower=0).to_numpy(dtype=float)

        out["payment_coverage_ratio"] = (
            payments.sum(axis=1) / (bills.sum(axis=1) + 1.0)
        )

        # Most recent month receives the largest weight.
        weights = np.array([6, 5, 4, 3, 2, 1], dtype=float)
        weights /= weights.sum()
        shortfall = np.maximum(bills - payments, 0.0)

        denom = (
            out["LIMIT_BAL"].clip(lower=0).to_numpy(dtype=float) + 1.0
            if "LIMIT_BAL" in out.columns
            else np.ones(len(out))
        )
        out["weighted_payment_shortfall"] = (shortfall * weights).sum(axis=1) / denom

        recent_bills = out[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].mean(axis=1)
        older_bills = out[["BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)
        out["recent_bill_growth"] = (recent_bills - older_bills) / (older_bills.abs() + 1.0)

    return out
