from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_CANDIDATES = [
    "default payment next month",
    "default_payment_next_month",
    "default",
    "target",
]


def load_credit_data(path: str | Path) -> pd.DataFrame:
    """Load the UCI Taiwan credit-card default dataset from xls/xlsx/csv."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        # Original UCI file usually has a descriptive first row.
        try:
            df = pd.read_excel(path, header=1)
        except Exception:
            df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    df.columns = [str(c).strip() for c in df.columns]

    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Drop duplicate identifier if present.
    for candidate in ["ID", "id"]:
        if candidate in df.columns:
            df = df.drop(columns=[candidate])

    target = find_target_column(df)
    df = df.rename(columns={target: "target"})
    df["target"] = pd.to_numeric(df["target"], errors="raise").astype(int)

    if set(df["target"].unique()) - {0, 1}:
        raise ValueError("Target must be binary with values 0/1.")

    return df


def find_target_column(df: pd.DataFrame) -> str:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    fuzzy = [c for c in df.columns if "default" in str(c).lower()]
    if len(fuzzy) == 1:
        return fuzzy[0]

    raise KeyError(
        "Could not identify target column. "
        f"Expected one of {TARGET_CANDIDATES}; got {list(df.columns)}"
    )


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
):
    """
    Return train, validation, test splits with stratification.

    validation_size is the fraction of the full dataset.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    relative_val_size = validation_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
