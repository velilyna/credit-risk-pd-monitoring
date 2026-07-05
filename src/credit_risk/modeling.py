from __future__ import annotations

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def build_models(random_state: int = 42) -> dict:
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=20,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=15,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    if LGBMClassifier is not None:
        models["lightgbm"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=15,
                        min_child_samples=40,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.5,
                        reg_lambda=1.0,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )

    return models


def fit_models(models: dict, X_train, y_train) -> dict:
    fitted = {}
    for name, model in models.items():
        fitted[name] = clone(model).fit(X_train, y_train)
    return fitted


def add_calibrated_models(
    fitted_models: dict,
    X_val,
    y_val,
    methods: tuple[str, ...] = ("sigmoid",),
) -> dict:
    """
    Calibrate already fitted estimators on a separate validation split.
    """
    result = dict(fitted_models)

    for name, model in fitted_models.items():
        # Logistic regression is usually already reasonably calibrated.
        if name == "logistic_regression":
            continue
        for method in methods:
            calibrated = CalibratedClassifierCV(
                estimator=model,
                method=method,
                cv="prefit",
            )
            calibrated.fit(X_val, y_val)
            result[f"{name}_calibrated_{method}"] = calibrated

    return result
