"""Train and evaluate the robustness logistic-regression model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from robustness_data import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    GROUP_COLUMN,
    TARGET_COLUMN,
    assert_no_leakage,
    summarize_frame,
)
from robustness_modeling import (
    RANDOM_SEED,
    coefficient_frame,
    evaluate_estimator,
    group_holdout_split,
    majority_baseline,
    make_group_cv,
    make_pipeline,
    roc_points,
    tune_pipeline,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_pyplot() -> Any:
    """Import the optional plotting dependency only for analysis commands."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aimo-matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/aimo-cache")
    import matplotlib.pyplot as pyplot

    return pyplot


def _plot_confusion(matrix: list[list[int]], path: Path) -> None:
    plt = _load_pyplot()
    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(values, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(values[row, column]), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["not robust", "robust"],
        yticklabels=["not robust", "robust"],
        xlabel="Predicted label",
        ylabel="True label",
        title="Logistic regression confusion matrix",
    )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_roc(points: pd.DataFrame, auc: float, path: Path) -> None:
    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(
        points["false_positive_rate"],
        points["true_positive_rate"],
        label=f"logistic regression (AUC={auc:.3f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="Holdout ROC curve",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_coefficients(coefficients: pd.DataFrame, path: Path) -> None:
    plt = _load_pyplot()
    ordered = coefficients.sort_values("coefficient")
    colors = np.where(ordered["coefficient"] >= 0, "#2878B5", "#D1495B")
    height = max(5, 0.32 * len(ordered))
    figure, axis = plt.subplots(figsize=(8, height))
    axis.barh(ordered["feature"], ordered["coefficient"], color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Coefficient on standardized feature",
        title="Logistic-regression feature associations",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_baseline(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Run Experiment 1 on a fixed, unseen-problem holdout."""

    feature_names = list(FEATURE_COLUMNS)
    assert_no_leakage(feature_names)
    y = frame[TARGET_COLUMN]
    groups = frame[GROUP_COLUMN]
    split = group_holdout_split(y, groups)
    X_train = frame.iloc[split.train][feature_names]
    y_train = y.iloc[split.train]
    groups_train = groups.iloc[split.train]
    X_test = frame.iloc[split.test][feature_names]
    y_test = y.iloc[split.test]

    search = tune_pipeline(X_train, y_train, groups_train)
    metrics, prediction, probability = evaluate_estimator(search.best_estimator_, X_test, y_test)
    baseline = majority_baseline(y_train, y_test)
    coefficients = coefficient_frame(search.best_estimator_, feature_names)
    joblib.dump(search.best_estimator_, output_dir / "baseline_model.joblib")

    predictions = frame.iloc[split.test][
        ["row_id", GROUP_COLUMN, "model_id", "reasoning_effort", TARGET_COLUMN]
    ].copy()
    predictions["predicted_robust"] = prediction
    predictions["probability_robust"] = probability
    predictions.to_csv(output_dir / "baseline_predictions.csv", index=False)
    coefficients.to_csv(output_dir / "coefficients.csv", index=False)

    result = {
        "experiment": "baseline_classification",
        "random_seed": RANDOM_SEED,
        "features": feature_names,
        "train_rows": int(len(split.train)),
        "test_rows": int(len(split.test)),
        "train_problem_groups": int(groups.iloc[split.train].nunique()),
        "test_problem_groups": int(groups.iloc[split.test].nunique()),
        "group_overlap": sorted(
            set(groups.iloc[split.train]) & set(groups.iloc[split.test])
        ),
        "best_parameters": search.best_params_,
        "inner_cv_best_f1": float(search.best_score_),
        "logistic_regression": metrics,
        "majority_class_baseline": baseline,
    }
    _write_json(output_dir / "baseline_metrics.json", result)
    _plot_confusion(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    _plot_coefficients(coefficients, output_dir / "feature_coefficients.png")
    if metrics["roc_auc"] is not None:
        points = roc_points(y_test, probability)
        points.to_csv(output_dir / "roc_curve.csv", index=False)
        _plot_roc(points, float(metrics["roc_auc"]), output_dir / "roc_curve.png")
    return result


def run_ablation(
    frame: pd.DataFrame, output_dir: Path, baseline_result: dict[str, Any]
) -> pd.DataFrame:
    """Run controlled group and leave-one-feature-out ablations."""

    y = frame[TARGET_COLUMN]
    groups = frame[GROUP_COLUMN]
    split = group_holdout_split(y, groups)
    full_features = list(FEATURE_COLUMNS)
    best = baseline_result["best_parameters"]
    c_value = float(best["classifier__C"])
    class_weight = best["classifier__class_weight"]
    reference_f1 = float(baseline_result["logistic_regression"]["f1"])
    reference_auc = baseline_result["logistic_regression"]["roc_auc"]

    specifications: list[tuple[str, str, list[str]]] = []
    for group_name, removed in FEATURE_GROUPS.items():
        specifications.append(
            ("feature_group", group_name, [f for f in full_features if f not in removed])
        )
    for feature in full_features:
        specifications.append(
            ("single_feature", feature, [f for f in full_features if f != feature])
        )

    rows: list[dict[str, Any]] = []
    for ablation_type, removed_name, kept_features in specifications:
        assert_no_leakage(kept_features)
        estimator = make_pipeline(C=c_value, class_weight=class_weight)
        estimator.fit(
            frame.iloc[split.train][kept_features],
            y.iloc[split.train],
        )
        metrics, _, _ = evaluate_estimator(
            estimator,
            frame.iloc[split.test][kept_features],
            y.iloc[split.test],
        )
        auc = metrics["roc_auc"]
        rows.append(
            {
                "ablation_type": ablation_type,
                "removed": removed_name,
                "features_remaining": len(kept_features),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": auc,
                "delta_f1_vs_full": float(metrics["f1"]) - reference_f1,
                "delta_roc_auc_vs_full": (
                    float(auc) - float(reference_auc)
                    if auc is not None and reference_auc is not None
                    else np.nan
                ),
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["ablation_type", "delta_f1_vs_full", "removed"]
    )
    results.to_csv(output_dir / "ablation_results.csv", index=False)
    return results


def run_generalization(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Run nested grouped CV so each outer fold contains unseen problems."""

    features = list(FEATURE_COLUMNS)
    assert_no_leakage(features)
    y = frame[TARGET_COLUMN]
    groups = frame[GROUP_COLUMN]
    outer_cv = make_group_cv(y, groups, max_splits=5)
    outer_rows: list[dict[str, Any]] = []

    for fold, (train, test) in enumerate(
        outer_cv.split(frame[features], y, groups), start=1
    ):
        if y.iloc[train].nunique() < 2:
            raise ValueError(f"Outer fold {fold} training data has only one class")
        search = tune_pipeline(
            frame.iloc[train][features], y.iloc[train], groups.iloc[train]
        )
        metrics, _, _ = evaluate_estimator(
            search.best_estimator_, frame.iloc[test][features], y.iloc[test]
        )
        overlap = set(groups.iloc[train]) & set(groups.iloc[test])
        outer_rows.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_problem_groups": groups.iloc[train].nunique(),
                "test_problem_groups": groups.iloc[test].nunique(),
                "group_overlap_count": len(overlap),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "best_C": search.best_params_["classifier__C"],
                "best_class_weight": str(
                    search.best_params_["classifier__class_weight"]
                ),
            }
        )

    folds = pd.DataFrame(outer_rows)
    folds.to_csv(output_dir / "generalization_folds.csv", index=False)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary: dict[str, Any] = {
        "experiment": "unseen_original_problem_generalization",
        "outer_folds": int(len(folds)),
        "random_seed": RANDOM_SEED,
        "all_group_overlaps_zero": bool((folds["group_overlap_count"] == 0).all()),
        "metrics": {},
        "topic_generalization": {
            "status": "not_run",
            "reason": "No mathematical topic column is present in the generated trace data.",
        },
        "difficulty_generalization": {
            "status": "not_run",
            "reason": "No problem difficulty column is present in the generated trace data.",
        },
    }
    for metric in metric_names:
        valid = pd.to_numeric(folds[metric], errors="coerce").dropna()
        summary["metrics"][metric] = {
            "mean": float(valid.mean()) if len(valid) else None,
            "standard_deviation": float(valid.std(ddof=0)) if len(valid) else None,
            "valid_folds": int(len(valid)),
        }
    _write_json(output_dir / "generalization_summary.json", summary)
    return summary


def write_report(
    frame: pd.DataFrame,
    output_dir: Path,
    baseline: dict[str, Any],
    ablation: pd.DataFrame,
    generalization: dict[str, Any],
) -> None:
    """Write a concise, data-backed report without overstating evidence."""

    summary = summarize_frame(frame)
    coefficients = pd.read_csv(output_dir / "coefficients.csv")
    top_positive = coefficients.sort_values("coefficient", ascending=False).head(3)
    top_negative = coefficients.sort_values("coefficient", ascending=True).head(3)
    group_ablations = ablation[ablation["ablation_type"] == "feature_group"].sort_values(
        "delta_f1_vs_full"
    )
    logistic = baseline["logistic_regression"]
    majority = baseline["majority_class_baseline"]
    general = generalization["metrics"]

    def feature_lines(table: pd.DataFrame) -> str:
        return "\n".join(
            f"- `{row.feature}`: {row.coefficient:+.3f}"
            for row in table.itertuples()
        )

    ablation_lines = "\n".join(
        f"- Remove `{row.removed}`: F1 {row.f1:.3f} "
        f"(change {row.delta_f1_vs_full:+.3f})"
        for row in group_ablations.itertuples()
    )
    roc_text = (
        f"{float(logistic['roc_auc']):.3f}"
        if logistic["roc_auc"] is not None
        else "not defined"
    )
    report = f"""# Logistic-regression experiment report

## Data and prediction task

The analysis used {summary['rows']} real generated response traces covering
{summary['original_problem_groups']} unique original problems and {summary['models']}
models. The unit of analysis is one response from a model/problem/reasoning-effort
configuration. There are {summary['class_counts']['robust']} robust and
{summary['class_counts']['not_robust']} non-robust labels.

The classifier predicts `model_is_robust` using {summary['feature_count']} numeric
features derived only from the original response: aggregate token uncertainty,
temporal uncertainty, and response length. `max_drop` is intentionally excluded
because it is used to derive the target. Using it would make evaluation circular.

## Methodology

All splits group rows by a stable hash of `original_problem`, preventing responses
to the same problem from appearing in both train and test data. Missing-value
imputation, standardization, and L2-regularized logistic regression are one
scikit-learn pipeline. Regularization strength and class weighting are selected
using grouped cross-validation on training data only. The fixed random seed is
{RANDOM_SEED}.

## Experiment 1: baseline classification

The fixed unseen-problem holdout contains {baseline['test_rows']} rows from
{baseline['test_problem_groups']} problem groups. Logistic-regression performance:

- Accuracy: {float(logistic['accuracy']):.3f}
- Precision: {float(logistic['precision']):.3f}
- Recall: {float(logistic['recall']):.3f}
- F1: {float(logistic['f1']):.3f}
- ROC-AUC: {roc_text}

The training-majority baseline achieved accuracy {float(majority['accuracy']):.3f}
and F1 {float(majority['f1']):.3f}. These are descriptive results from a very small
development set, not evidence that the classifier will generalize to private data.

## Experiment 2: feature ablation and coefficients

Feature-group ablations on the same holdout:

{ablation_lines}

Largest positive standardized coefficients:

{feature_lines(top_positive)}

Largest negative standardized coefficients:

{feature_lines(top_negative)}

Coefficient signs and ablation changes describe associations in this fitted sample.
They do not establish that a feature causes robust reasoning. Correlated uncertainty
features can also redistribute coefficient magnitude, so individual ranks are not
stable causal importance scores.

## Experiment 3: unseen-problem generalization

Nested grouped cross-validation produced:

- Accuracy: {general['accuracy']['mean']:.3f} ± {general['accuracy']['standard_deviation']:.3f}
- F1: {general['f1']['mean']:.3f} ± {general['f1']['standard_deviation']:.3f}
- ROC-AUC: {general['roc_auc']['mean']:.3f} ± {general['roc_auc']['standard_deviation']:.3f}

Every outer fold has zero original-problem overlap. Topic- and difficulty-held-out
experiments could not be run because those metadata fields are absent.

## Limitations and conclusions

The sample has only {summary['rows']} rows and {summary['original_problem_groups']}
independent problem groups. Several responses are truncated at the generation limit,
and the available fields do not include original-answer correctness, verified reasoning
steps, mathematical topic, difficulty, or an independently adjudicated robustness
label. The label is derived from perturbation performance; the pipeline avoids direct
leakage from those outcomes, but the label still operationalizes behavior rather than
proving genuine reasoning.

Logistic regression is useful here as an interpretable diagnostic: it estimates how
trace statistics are associated with the operational robustness label and provides a
transparent baseline. Classification performance alone cannot establish that a model
reasons mathematically. A credible research conclusion requires more labeled problem
groups, an explicit labeling protocol, independent reasoning annotations, and ideally
external validation on models and problem families not used for development.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def run_all(frame: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Run all experiments and write all required outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    data_summary = summarize_frame(frame)
    _write_json(output_dir / "data_summary.json", data_summary)
    output_columns = [
        "row_id",
        GROUP_COLUMN,
        "model_id",
        "reasoning_effort",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    ]
    frame[output_columns].to_csv(output_dir / "modeling_table.csv", index=False)
    baseline = run_baseline(frame, output_dir)
    ablation = run_ablation(frame, output_dir, baseline)
    generalization = run_generalization(frame, output_dir)
    write_report(frame, output_dir, baseline, ablation, generalization)
    return {
        "data": data_summary,
        "baseline": baseline,
        "generalization": generalization,
    }


def export_submission_model(frame: pd.DataFrame, artifact_path: Path) -> dict[str, Any]:
    """Tune on grouped CV, refit on all trace rows, and export the runtime model."""

    features = list(FEATURE_COLUMNS)
    assert_no_leakage(features)
    search = tune_pipeline(
        frame[features],
        frame[TARGET_COLUMN],
        frame[GROUP_COLUMN],
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, artifact_path)
    return {
        "artifact": str(artifact_path),
        "training_rows": int(len(frame)),
        "problem_groups": int(frame[GROUP_COLUMN].nunique()),
        "feature_count": len(features),
        "best_parameters": search.best_params_,
        "grouped_cv_f1": float(search.best_score_),
    }
