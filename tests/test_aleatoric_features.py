"""Unit tests for Data-Based (Aleatoric) Uncertainty Feature Extraction (Issue #2)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from aleatoric_features import (
    ALEATORIC_AGGREGATE_FEATURES,
    ALEATORIC_TEMPORAL_FEATURES,
    ALL_ALEATORIC_FEATURES,
    compute_ols_trend,
    compute_stage_means,
    extract_aleatoric_aggregate_features,
    extract_aleatoric_features,
    extract_aleatoric_features_dataset,
    extract_aleatoric_temporal_features,
)


class AleatoricUnitTests(unittest.TestCase):
    def test_aggregate_features_deterministic_values(self) -> None:
        log_probs = np.array([-1.0, -2.0, -3.0])
        probs = np.exp(log_probs)
        entropy = np.array([0.2, 0.4, 0.6])
        top1 = np.array([0.7, 0.8, 0.9])
        margin = np.array([0.4, 0.5, 0.6])
        is_top1 = np.array([1.0, 0.0, 1.0])

        feats = extract_aleatoric_aggregate_features(
            log_probs=log_probs,
            probs=probs,
            entropy=entropy,
            top1_probs=top1,
            top2_margins=margin,
            selected_is_top1=is_top1,
            min_k_fraction=0.5,
            high_conf_threshold=0.3,
            low_conf_threshold=0.1,
        )

        self.assertEqual(len(feats), 14)
        for name in ALEATORIC_AGGREGATE_FEATURES:
            self.assertIn(name, feats)

        self.assertAlmostEqual(feats["mean_entropy"], 0.4)
        self.assertAlmostEqual(feats["mean_log_probability"], -2.0)
        self.assertAlmostEqual(feats["mean_negative_log_likelihood"], 2.0)
        self.assertAlmostEqual(feats["perplexity"], math.exp(2.0))
        self.assertAlmostEqual(feats["min_log_probability"], -3.0)
        self.assertAlmostEqual(feats["max_log_probability"], -1.0)
        self.assertAlmostEqual(feats["min_k_log_probability"], -2.5)
        self.assertAlmostEqual(feats["mean_top1_probability"], 0.8)
        self.assertAlmostEqual(feats["mean_top2_probability_margin"], 0.5)
        self.assertAlmostEqual(feats["fraction_selected_is_top1"], 2.0 / 3.0)

    def test_temporal_features_linear_progression(self) -> None:
        u_t = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        feats = extract_aleatoric_temporal_features(u_t)

        self.assertEqual(len(feats), 5)
        for name in ALEATORIC_TEMPORAL_FEATURES:
            self.assertIn(name, feats)

        self.assertAlmostEqual(feats["early_stage_aleatoric_mean"], 3.0)
        self.assertAlmostEqual(feats["middle_stage_aleatoric_mean"], 7.0)
        self.assertAlmostEqual(feats["late_stage_aleatoric_mean"], 11.0)
        self.assertAlmostEqual(feats["aleatoric_slope"], 12.0)
        self.assertAlmostEqual(feats["aleatoric_r2_linearity"], 1.0)

    def test_empty_aleatoric_inputs_produce_nans(self) -> None:
        agg = extract_aleatoric_aggregate_features(log_probs=np.array([]))
        self.assertEqual(len(agg), 14)
        for name, val in agg.items():
            self.assertTrue(math.isnan(val), f"{name} should be NaN")

        temp = extract_aleatoric_temporal_features(np.array([]))
        self.assertEqual(len(temp), 5)
        for name, val in temp.items():
            self.assertTrue(math.isnan(val), f"{name} should be NaN")

    def test_constant_uncertainty_zero_variance(self) -> None:
        constant_u = np.full(10, 0.42)
        slope, r2 = compute_ols_trend(constant_u)
        self.assertEqual(slope, 0.0)
        self.assertEqual(r2, 1.0)

        early, mid, late = compute_stage_means(constant_u)
        self.assertAlmostEqual(early, 0.42)
        self.assertAlmostEqual(mid, 0.42)
        self.assertAlmostEqual(late, 0.42)

    def test_short_traces_t_less_than_3(self) -> None:
        t1 = np.array([0.75])
        slope, r2 = compute_ols_trend(t1)
        self.assertEqual(slope, 0.0)
        self.assertEqual(r2, 1.0)
        e, m, l = compute_stage_means(t1)
        self.assertEqual(e, 0.75)
        self.assertEqual(m, 0.75)
        self.assertEqual(l, 0.75)

        t2 = np.array([0.2, 0.8])
        slope, r2 = compute_ols_trend(t2)
        self.assertGreater(slope, 0.0)
        self.assertAlmostEqual(r2, 1.0)
        e, m, l = compute_stage_means(t2)
        self.assertAlmostEqual(e, 0.2)
        self.assertAlmostEqual(m, 0.5)
        self.assertAlmostEqual(l, 0.8)

    def test_extract_aleatoric_dataset(self) -> None:
        records = [
            {
                "problem_id": f"p_{i}",
                "model_id": "model_1",
                "generation_num_tokens": 4,
                "original_problem": f"Problem {i}",
                "reasoning_trace": "Trace",
                "model_is_robust": True,
                "selected_logprob_trace": [-0.2, -0.3, -0.4, -0.5],
                "entropy_trace": [0.1, 0.2, 0.3, 0.4],
            }
            for i in range(3)
        ]
        df_all = extract_aleatoric_features_dataset(records)
        self.assertEqual(len(df_all), 3)
        self.assertEqual(len(df_all.columns), 6 + 19)

        df_agg = extract_aleatoric_features_dataset(records, aggregate_only=True)
        self.assertEqual(len(df_agg.columns), 6 + 14)

        df_temp = extract_aleatoric_features_dataset(records, temporal_only=True)
        self.assertEqual(len(df_temp.columns), 6 + 5)


if __name__ == "__main__":
    unittest.main()
