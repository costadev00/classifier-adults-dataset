#!/usr/bin/env python3
"""Build the final holdout comparison table and plot for all classifiers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_CONFIGS = [
    {
        "model": "kNN",
        "selection_hyperparameter": "k, weights, p",
        "metrics_filename": "knn_holdout_metrics.json",
    },
    {
        "model": "Linear SVM",
        "selection_hyperparameter": "C",
        "metrics_filename": "svm_holdout_metrics.json",
    },
    {
        "model": "Logistic Regression",
        "selection_hyperparameter": "C",
        "metrics_filename": "logreg_holdout_metrics.json",
    },
]

METRIC_COLUMNS = [
    "accuracy",
    "error_rate",
    "balanced_accuracy",
    "precision_positive",
    "recall_positive",
    "sensitivity",
    "specificity",
    "f1_positive",
    "roc_auc",
]

OUTPUT_CSV = "knn_vs_svm_vs_logreg_holdout_comparison.csv"
OUTPUT_PNG = "knn_vs_svm_vs_logreg_holdout_metrics.png"


def resolve_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent]
    for candidate in candidates:
        if (candidate / "submissions").exists() and (candidate / "data" / "prepared" / "adult_income").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the project root. Run this script from the repository root or from a direct child directory."
    )


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def validate_required_files(submissions_dir: Path) -> None:
    missing_paths = [
        submissions_dir / config["metrics_filename"]
        for config in MODEL_CONFIGS
        if not (submissions_dir / config["metrics_filename"]).exists()
    ]

    if missing_paths:
        missing_list = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing required holdout metrics JSON file(s):\n"
            f"{missing_list}\n\n"
            "Run the corresponding classifier notebook(s) first. "
            "The Logistic Regression comparison requires submissions/logreg_holdout_metrics.json."
        )


def validate_payload(payload: dict, path: Path) -> None:
    required_top_level_keys = ["selected_params", "holdout_metrics"]
    missing_top_level = [key for key in required_top_level_keys if key not in payload]
    if missing_top_level:
        raise KeyError(f"{path} is missing top-level key(s): {missing_top_level}")

    holdout_metrics = payload["holdout_metrics"]
    missing_metrics = [metric for metric in METRIC_COLUMNS if metric not in holdout_metrics]
    if missing_metrics:
        raise KeyError(f"{path} is missing holdout metric(s): {missing_metrics}")

    invalid_metrics = [
        metric
        for metric in METRIC_COLUMNS
        if not isinstance(holdout_metrics[metric], (int, float)) or not 0 <= float(holdout_metrics[metric]) <= 1
    ]
    if invalid_metrics:
        raise ValueError(f"{path} has metric value(s) outside the expected [0, 1] range: {invalid_metrics}")


def build_comparison_df(submissions_dir: Path) -> pd.DataFrame:
    records = []

    for config in MODEL_CONFIGS:
        metrics_path = submissions_dir / config["metrics_filename"]
        payload = load_json(metrics_path)
        validate_payload(payload, metrics_path)

        records.append(
            {
                "model": config["model"],
                "selection_hyperparameter": config["selection_hyperparameter"],
                "selected_value": json.dumps(payload["selected_params"], sort_keys=True),
                **{metric: float(payload["holdout_metrics"][metric]) for metric in METRIC_COLUMNS},
            }
        )

    return pd.DataFrame(records, columns=["model", "selection_hyperparameter", "selected_value", *METRIC_COLUMNS])


def save_metric_plot(comparison_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(METRIC_COLUMNS))
    bar_width = 0.24
    offsets = np.linspace(-bar_width, bar_width, len(comparison_df))
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for offset, color, (_, row) in zip(offsets, colors, comparison_df.iterrows()):
        values = [row[metric] for metric in METRIC_COLUMNS]
        bars = ax.bar(x + offset, values, width=bar_width, label=row["model"], color=color)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=8, rotation=90)

    ax.set_title("Holdout Metrics Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_COLUMNS, rotation=35, ha="right")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the final holdout comparison CSV and grouped metric plot for kNN, SVM, and Logistic Regression."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repository root. Defaults to auto-detection from the current working directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve() if args.project_root else resolve_project_root()
    submissions_dir = project_root / "submissions"

    validate_required_files(submissions_dir)

    comparison_df = build_comparison_df(submissions_dir)

    csv_path = submissions_dir / OUTPUT_CSV
    png_path = submissions_dir / OUTPUT_PNG

    comparison_df.to_csv(csv_path, index=False)
    save_metric_plot(comparison_df, png_path)

    if not csv_path.exists() or not png_path.exists():
        raise RuntimeError("Expected comparison artifacts were not created.")

    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison plot: {png_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
