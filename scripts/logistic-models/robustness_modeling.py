"""Splitting, tuning, fitting, and evaluation for the robustness model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 464
DEFAULT_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class Split:
    """Row indices for a leakage-safe group holdout."""

    train: np.ndarray
    test: np.ndarray


def make_pipeline(
    *, C: float = 1.0, class_weight: str | None = "balanced"
) -> Pipeline:
    """Create a train-fitted imputation, scaling, and logistic model pipeline."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=5_000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _valid_split(y: pd.Series, train: np.ndarray, test: np.ndarray) -> bool:
    return y.iloc[train].nunique() == 2 and y.iloc[test].nunique() == 2


def group_holdout_split(
    y: pd.Series,
    groups: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Split:
    """Choose a deterministic stratified fold closest to the requested holdout."""

    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    n_groups = int(groups.nunique())
    n_splits = min(round(1.0 / test_size), n_groups)
    if n_splits < 2:
        raise ValueError("At least two problem groups are required for a holdout")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    overall_rate = float(y.mean())
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    placeholder = np.zeros((len(y), 1))
    for train, test in splitter.split(placeholder, y, groups):
        if not _valid_split(y, train, test):
            continue
        score = abs(len(test) / len(y) - test_size) + abs(
            float(y.iloc[test].mean()) - overall_rate
        )
        candidates.append((score, train, test))
    if not candidates:
        raise ValueError(
            "Could not create a group-disjoint holdout containing both classes. "
            "Collect more labeled original-problem groups."
        )
    _, train, test = min(candidates, key=lambda item: item[0])
    return Split(train=train, test=test)


def make_group_cv(
    y: pd.Series, groups: pd.Series, *, max_splits: int = 5, random_state: int = RANDOM_SEED
) -> StratifiedGroupKFold:
    """Construct the largest grouped CV whose folds contain both classes."""

    class_count = int(y.value_counts().min())
    upper_bound = min(max_splits, int(groups.nunique()), class_count)
    if upper_bound < 2:
        raise ValueError("At least two examples in each class and two groups are required")

    placeholder = np.zeros((len(y), 1))
    for n_splits in range(upper_bound, 1, -1):
        candidate = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        folds = list(candidate.split(placeholder, y, groups))
        if all(_valid_split(y, train, test) for train, test in folds):
            return candidate
    raise ValueError(
        "Could not construct grouped cross-validation folds containing both classes. "
        "Collect more labeled original-problem groups."
    )


def tune_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
) -> GridSearchCV:
    """Tune regularization and imbalance handling inside group-aware CV."""

    cv = make_group_cv(y, groups, max_splits=4)
    search = GridSearchCV(
        make_pipeline(),
        param_grid={
            "classifier__C": list(c_grid),
            "classifier__class_weight": [None, "balanced"],
        },
        scoring="f1",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    return search


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray | None,
) -> dict[str, float | int | None | list[list[int]]]:
    """Compute classification metrics without failing on a one-class fold."""

    y_array = np.asarray(y_true, dtype=int)
    result: dict[str, float | int | None | list[list[int]]] = {
        "n": int(len(y_array)),
        "accuracy": float(accuracy_score(y_array, y_pred)),
        "precision": float(precision_score(y_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_array, y_pred, zero_division=0)),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(y_array, y_pred, labels=[0, 1]).tolist(),
    }
    if y_probability is not None and np.unique(y_array).size == 2:
        result["roc_auc"] = float(roc_auc_score(y_array, y_probability))
    return result


def evaluate_estimator(
    estimator: Pipeline, X: pd.DataFrame, y: pd.Series
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Predict labels/probabilities and return metrics plus raw predictions."""

    prediction = np.asarray(estimator.predict(X), dtype=int)
    probability = np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
    return evaluate_predictions(y, prediction, probability), prediction, probability


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate a majority label chosen using the training partition only."""

    majority = int(y_train.value_counts().idxmax())
    prediction = np.full(len(y_test), majority, dtype=int)
    metrics = evaluate_predictions(y_test, prediction, None)
    metrics["predicted_class"] = majority
    return metrics


def coefficient_frame(estimator: Pipeline, input_features: list[str]) -> pd.DataFrame:
    """Extract standardized coefficients, including any missingness indicators."""

    imputer = estimator.named_steps["imputer"]
    output_features = list(imputer.get_feature_names_out(input_features))
    coefficients = estimator.named_steps["classifier"].coef_[0]
    if len(output_features) != len(coefficients):
        raise RuntimeError("Could not align transformed features with coefficients")
    frame = pd.DataFrame(
        {
            "feature": output_features,
            "coefficient": coefficients.astype(float),
        }
    )
    frame["absolute_coefficient"] = frame["coefficient"].abs()
    frame["association"] = np.where(
        frame["coefficient"] >= 0, "positive", "negative"
    )
    return frame.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def roc_points(y_true: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    """Return ROC coordinates for plotting and reproducibility."""

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, probability)
    return pd.DataFrame(
        {
            "false_positive_rate": false_positive_rate,
            "true_positive_rate": true_positive_rate,
            "threshold": thresholds,
        }
    )
