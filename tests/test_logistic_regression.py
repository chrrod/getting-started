"""Tests for preprocessing, leakage prevention, and group-aware splitting."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGISTIC_MODEL_DIR = ROOT / "scripts" / "logistic-models"
if str(LOGISTIC_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(LOGISTIC_MODEL_DIR))
LOGISTIC_SOLUTION_DIR = ROOT / "solutions" / "logistic-regression"
if str(LOGISTIC_SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(LOGISTIC_SOLUTION_DIR))

from aleatoric_features import extract_aleatoric_features  # noqa: E402
from logistic_features import FEATURE_NAMES, extract_features  # noqa: E402
from robustness_data import assert_no_leakage  # noqa: E402
from robustness_modeling import (  # noqa: E402
    group_holdout_split,
    make_group_cv,
    make_pipeline,
)


class LogisticRegressionExperimentTests(unittest.TestCase):
    def test_submission_features_match_training_features(self) -> None:
        log_probabilities = [-0.1, -0.4, -1.2, -0.2, -0.8, -0.05]
        entropy = [0.2, 0.5, 0.8, 0.3, 0.7, 0.1]
        top1 = np.exp(log_probabilities).tolist()
        margins = [0.7, 0.4, 0.2, 0.6, 0.3, 0.8]
        trace = "A short generated reasoning trace."
        training = extract_aleatoric_features(
            {
                "selected_logprob_trace": log_probabilities,
                "entropy_trace": entropy,
                "top1_prob_trace": top1,
                "top2_margin_trace": margins,
            }
        )
        training.update(
            {
                "generation_num_tokens": float(len(log_probabilities)),
                "reasoning_character_count": float(len(trace)),
            }
        )
        submission = extract_features(
            selected_log_probs=log_probabilities,
            entropy=entropy,
            top1_probabilities=top1,
            top2_margins=margins,
            reasoning_trace=trace,
        )
        self.assertEqual(tuple(submission), FEATURE_NAMES)
        for feature in FEATURE_NAMES:
            self.assertAlmostEqual(training[feature], submission[feature], places=12)

    def test_forbids_label_and_label_derived_predictors(self) -> None:
        for leaking in ("model_is_robust", "max_drop", "permutation_accuracy"):
            with self.subTest(leaking=leaking), self.assertRaisesRegex(
                ValueError, "Target leakage"
            ):
                assert_no_leakage(["mean_entropy", leaking])

    def test_problem_groups_never_cross_holdout_boundary(self) -> None:
        groups = pd.Series(np.repeat([f"problem-{i}" for i in range(12)], 2))
        # Each group has one row of each class, making a stratified group split feasible.
        y = pd.Series(np.tile([0, 1], 12))
        split = group_holdout_split(y, groups, test_size=0.25)
        train_groups = set(groups.iloc[split.train])
        test_groups = set(groups.iloc[split.test])
        self.assertTrue(train_groups)
        self.assertTrue(test_groups)
        self.assertTrue(train_groups.isdisjoint(test_groups))

    def test_group_cv_folds_contain_both_classes_without_group_overlap(self) -> None:
        groups = pd.Series(np.repeat([f"problem-{i}" for i in range(12)], 2))
        y = pd.Series(np.tile([0, 1], 12))
        cv = make_group_cv(y, groups, max_splits=5)
        for train, test in cv.split(np.zeros((len(y), 1)), y, groups):
            self.assertEqual(set(y.iloc[train]), {0, 1})
            self.assertEqual(set(y.iloc[test]), {0, 1})
            self.assertTrue(set(groups.iloc[train]).isdisjoint(groups.iloc[test]))

    def test_imputation_and_scaling_are_fit_from_training_data_only(self) -> None:
        X_train = pd.DataFrame({"feature": [0.0, 2.0, np.nan, 4.0]})
        y_train = pd.Series([0, 0, 1, 1])
        pipeline = make_pipeline(C=1.0, class_weight=None)
        pipeline.fit(X_train, y_train)

        # Training median is 2.0. A held-out extreme value must not affect it.
        self.assertAlmostEqual(float(pipeline.named_steps["imputer"].statistics_[0]), 2.0)
        pipeline.predict(pd.DataFrame({"feature": [10_000.0, np.nan]}))
        self.assertAlmostEqual(float(pipeline.named_steps["imputer"].statistics_[0]), 2.0)

    def test_pipeline_handles_missing_values_and_returns_binary_predictions(self) -> None:
        X = pd.DataFrame(
            {
                "a": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0],
                "b": [1.0, np.nan, 1.0, 2.0, 3.0, 3.0],
            }
        )
        y = pd.Series([0, 0, 0, 1, 1, 1])
        model = make_pipeline()
        model.fit(X, y)
        prediction = model.predict(X)
        self.assertEqual(len(prediction), len(y))
        self.assertTrue(set(prediction).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
