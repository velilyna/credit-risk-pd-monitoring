from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance


def save_permutation_importance(
    model,
    X: pd.DataFrame,
    y,
    output_csv: str | Path,
    scoring: str = "roc_auc",
    n_repeats: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    table.to_csv(output_csv, index=False)
    return table


def _extract_tree_parts(model, X: pd.DataFrame):
    """
    Extract transformer/model from supported pipelines.
    Calibrated wrappers are intentionally not used for SHAP.
    """
    if not hasattr(model, "named_steps"):
        raise TypeError("SHAP helper expects an uncalibrated sklearn Pipeline.")

    transformer = model.named_steps.get("imputer")
    estimator = model.named_steps.get("model")
    transformed = transformer.transform(X) if transformer is not None else X.to_numpy()
    transformed_df = pd.DataFrame(
        transformed,
        columns=X.columns,
        index=X.index,
    )
    return estimator, transformed_df


def generate_shap_outputs(
    model,
    X: pd.DataFrame,
    output_dir: str | Path,
    max_samples: int = 1000,
    random_state: int = 42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = X.sample(
        n=min(max_samples, len(X)),
        random_state=random_state,
    )

    estimator, transformed_df = _extract_tree_parts(model, sample)
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(transformed_df)

    # Binary classifiers sometimes return an extra class axis.
    if getattr(explanation.values, "ndim", 0) == 3:
        explanation = explanation[:, :, 1]

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    shap.plots.bar(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=160, bbox_inches="tight")
    plt.close()

    risk_index = int(np.argmax(model.predict_proba(sample)[:, 1]))
    shap.plots.waterfall(explanation[risk_index], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_high_risk_waterfall.png", dpi=160, bbox_inches="tight")
    plt.close()
