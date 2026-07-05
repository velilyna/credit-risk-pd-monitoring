from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_roc_curves(y_true, probabilities: dict, output_path: str | Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in probabilities.items():
        RocCurveDisplay.from_predictions(
            y_true,
            proba,
            name=name,
            ax=ax,
        )
    ax.set_title("ROC curves — test")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_calibration_curves(y_true, probabilities: dict, output_path: str | Path):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, proba in probabilities.items():
        mean_pred, observed = calibration_curve(
            y_true,
            proba,
            n_bins=10,
            strategy="quantile",
        )
        ax.plot(mean_pred, observed, marker="o", label=name)

    ax.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_threshold_cost(table, best_threshold: float, output_path: str | Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["threshold"], table["cost_per_applicant"])
    ax.axvline(best_threshold, linestyle="--")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost per applicant")
    ax.set_title("Business threshold optimization")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(y_true, pred, title: str, output_path: str | Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, pred, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
