from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.data import load_credit_data, split_data
from credit_risk.explainability import (
    generate_shap_outputs,
    save_permutation_importance,
)
from credit_risk.features import add_base_features, add_innovative_features
from credit_risk.metrics import evaluate_predictions, optimize_business_threshold
from credit_risk.modeling import add_calibrated_models, build_models, fit_models
from credit_risk.monitoring import (
    create_stress_scenario,
    feature_psi_table,
    population_stability_index,
)
from credit_risk.plots import (
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_threshold_cost,
)
from credit_risk.statistics import (
    bootstrap_auc_difference,
    bootstrap_cost_difference,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "default.xls"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fn-cost", type=float, default=100000)
    parser.add_argument("--fp-cost", type=float, default=10000)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--skip-shap", action="store_true")
    return parser.parse_args()


def ensure_dirs():
    for path in [
        ROOT / "reports",
        ROOT / "reports" / "figures",
        ROOT / "artifacts",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def fit_and_score_models(
    models,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    fn_cost,
    fp_cost,
):
    fitted = fit_models(models, X_train, y_train)
    all_models = add_calibrated_models(fitted, X_val, y_val, methods=("sigmoid",))

    rows = []
    val_probabilities = {}
    test_probabilities = {}
    thresholds = {}
    threshold_tables = {}

    for name, model in all_models.items():
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        threshold, table = optimize_business_threshold(
            y_val,
            val_proba,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )

        val_metrics = evaluate_predictions(
            y_val,
            val_proba,
            threshold,
            fn_cost,
            fp_cost,
        )
        test_metrics = evaluate_predictions(
            y_test,
            test_proba,
            threshold,
            fn_cost,
            fp_cost,
        )

        rows.append({"model": name, "split": "validation", **val_metrics})
        rows.append({"model": name, "split": "test", **test_metrics})

        val_probabilities[name] = val_proba
        test_probabilities[name] = test_proba
        thresholds[name] = threshold
        threshold_tables[name] = table

    return (
        fitted,
        all_models,
        pd.DataFrame(rows),
        val_probabilities,
        test_probabilities,
        thresholds,
        threshold_tables,
    )


def feature_engineering_experiment(
    raw_df: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    """
    Same model, same repeated CV folds:
    baseline aggregates vs baseline + behavioral innovations.
    """
    base_df = add_base_features(raw_df)
    extended_df = add_innovative_features(raw_df)

    X_base = base_df.drop(columns=["target"])
    X_ext = extended_df.drop(columns=["target"])
    y = raw_df["target"]

    model = build_models(random_state=random_state)["logistic_regression"]

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=random_state,
    )
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "neg_brier": "neg_brier_score",
    }

    base_scores = cross_validate(
        clone(model),
        X_base,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    ext_scores = cross_validate(
        clone(model),
        X_ext,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    rows = []
    for metric in ["roc_auc", "average_precision", "neg_brier"]:
        base_values = base_scores[f"test_{metric}"]
        ext_values = ext_scores[f"test_{metric}"]
        delta = ext_values - base_values

        rows.append(
            {
                "metric": metric,
                "baseline_mean": float(base_values.mean()),
                "extended_mean": float(ext_values.mean()),
                "mean_delta": float(delta.mean()),
                "delta_ci_lower_empirical": float(np.quantile(delta, 0.025)),
                "delta_ci_upper_empirical": float(np.quantile(delta, 0.975)),
                "share_folds_extended_better": float(np.mean(delta > 0)),
                "n_fold_scores": int(len(delta)),
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    ensure_dirs()

    raw_df = load_credit_data(args.data)
    engineered_df = add_innovative_features(raw_df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        engineered_df,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )

    models = build_models(args.random_state)
    (
        uncalibrated_models,
        all_models,
        metrics_df,
        val_probabilities,
        test_probabilities,
        thresholds,
        threshold_tables,
    ) = fit_and_score_models(
        models,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        args.fn_cost,
        args.fp_cost,
    )

    reports = ROOT / "reports"
    figures = reports / "figures"
    artifacts = ROOT / "artifacts"

    metrics_df.to_csv(reports / "model_metrics.csv", index=False)

    test_rows = metrics_df[metrics_df["split"] == "test"].copy()
    champion = (
        test_rows.sort_values(
            ["cost_per_applicant", "brier_score", "roc_auc"],
            ascending=[True, True, False],
        )
        .iloc[0]["model"]
    )
    champion_threshold = thresholds[champion]
    champion_model = all_models[champion]
    champion_proba = test_probabilities[champion]
    champion_pred = (champion_proba >= champion_threshold).astype(int)

    joblib.dump(champion_model, artifacts / "champion_model.joblib")

    plot_roc_curves(
        y_test,
        test_probabilities,
        figures / "roc_curves.png",
    )
    plot_calibration_curves(
        y_test,
        test_probabilities,
        figures / "calibration_curves.png",
    )
    plot_threshold_cost(
        threshold_tables[champion],
        champion_threshold,
        figures / "business_threshold.png",
    )
    plot_confusion_matrix(
        y_test,
        champion_pred,
        f"Champion confusion matrix: {champion}",
        figures / "champion_confusion_matrix.png",
    )

    # Statistical model comparisons.
    auc_rows = []
    cost_rows = []
    if args.bootstrap_iterations > 0:
        for model_a, model_b in combinations(test_probabilities.keys(), 2):
            auc_result = bootstrap_auc_difference(
                y_test.to_numpy(),
                test_probabilities[model_a],
                test_probabilities[model_b],
                n_bootstrap=args.bootstrap_iterations,
                random_state=args.random_state,
            )
            auc_rows.append(
                {"model_a": model_a, "model_b": model_b, **auc_result}
            )

            pred_a = (
                test_probabilities[model_a] >= thresholds[model_a]
            ).astype(int)
            pred_b = (
                test_probabilities[model_b] >= thresholds[model_b]
            ).astype(int)

            cost_result = bootstrap_cost_difference(
                y_test.to_numpy(),
                pred_a,
                pred_b,
                fn_cost=args.fn_cost,
                fp_cost=args.fp_cost,
                n_bootstrap=args.bootstrap_iterations,
                random_state=args.random_state,
            )
            cost_rows.append(
                {"model_a": model_a, "model_b": model_b, **cost_result}
            )

    pd.DataFrame(auc_rows).to_csv(
        reports / "model_comparisons_auc.csv",
        index=False,
    )
    pd.DataFrame(cost_rows).to_csv(
        reports / "model_comparisons_cost.csv",
        index=False,
    )

    # Feature engineering experiment.
    feature_experiment = feature_engineering_experiment(
        raw_df,
        random_state=args.random_state,
    )
    feature_experiment.to_csv(
        reports / "feature_engineering_experiment.csv",
        index=False,
    )

    # Monitoring / PSI and stress scenario.
    stressed_raw = create_stress_scenario(
        raw_df.drop(columns=["target"])
    )
    stressed_features = add_innovative_features(stressed_raw)
    stressed_proba = champion_model.predict_proba(stressed_features)[:, 1]

    feature_psi = feature_psi_table(
        X_train,
        stressed_features[X_train.columns],
    )
    feature_psi.to_csv(reports / "feature_psi.csv", index=False)

    score_psi = population_stability_index(
        champion_model.predict_proba(X_train)[:, 1],
        stressed_proba,
    )

    # No true labels exist for an artificial future stress population.
    stress_summary = pd.DataFrame(
        [
            {
                "champion": champion,
                "reference_mean_pd_test": float(champion_proba.mean()),
                "stress_mean_pd": float(stressed_proba.mean()),
                "score_psi": float(score_psi),
                "reference_share_above_threshold": float(
                    np.mean(champion_proba >= champion_threshold)
                ),
                "stress_share_above_threshold": float(
                    np.mean(stressed_proba >= champion_threshold)
                ),
            }
        ]
    )
    stress_summary.to_csv(reports / "stress_metrics.csv", index=False)

    # Permutation importance on the chosen champion.
    save_permutation_importance(
        champion_model,
        X_test,
        y_test,
        reports / "permutation_importance.csv",
    )

    # SHAP for the uncalibrated tree model corresponding to the champion.
    if not args.skip_shap:
        base_name = champion.split("_calibrated_")[0]
        shap_candidate = uncalibrated_models.get(base_name)
        if shap_candidate is not None and base_name != "logistic_regression":
            try:
                generate_shap_outputs(
                    shap_candidate,
                    X_test,
                    figures,
                )
            except Exception as exc:
                (reports / "shap_error.txt").write_text(
                    f"{type(exc).__name__}: {exc}",
                    encoding="utf-8",
                )

    summary = {
        "champion": champion,
        "champion_threshold": float(champion_threshold),
        "fn_cost": float(args.fn_cost),
        "fp_cost": float(args.fp_cost),
        "score_psi": float(score_psi),
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "models": list(all_models.keys()),
    }
    (reports / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.")
    print(f"Champion: {champion}")
    print(f"Threshold: {champion_threshold:.4f}")
    print(f"Results: {reports}")


if __name__ == "__main__":
    main()
