"""Unit tests for Aleatoric and Epistemic Uncertainty Feature Extraction (Issues #2 & #3)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from uncertainty_features import (
    ALEATORIC_AGGREGATE_FEATURES,
    ALEATORIC_TEMPORAL_FEATURES,
    ALL_29_UNCERTAINTY_FEATURES,
    ALL_ALEATORIC_FEATURES,
    ALL_EPISTEMIC_FEATURES,
    EPISTEMIC_AGGREGATE_FEATURES,
    EPISTEMIC_TEMPORAL_FEATURES,
    compute_ols_trend,
    compute_stage_means,
    estimate_epistemic_layer_dispersion,
    estimate_epistemic_mc_dropout,
    estimate_epistemic_volatility_proxy,
    extract_aleatoric_aggregate_features,
    extract_aleatoric_temporal_features,
    extract_epistemic_aggregate_features,
    extract_epistemic_temporal_features,
    extract_features_dataset,
    extract_trace_features,
)


class AleatoricFeatureTests(unittest.TestCase):
    def test_aggregate_features_match_hand_calculated_values(self) -> None:
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
            min_k_fraction=0.5,  # k = ceil(3 * 0.5) = 2 -> [-3.0, -2.0]
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
        # Perfectly linear sequence: u_t = 2.0 * t for t in 1..6
        u_t = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        feats = extract_aleatoric_temporal_features(u_t)

        self.assertEqual(len(feats), 5)
        for name in ALEATORIC_TEMPORAL_FEATURES:
            self.assertIn(name, feats)

        # Early (items 0, 1): (2 + 4) / 2 = 3.0
        self.assertAlmostEqual(feats["early_stage_aleatoric_mean"], 3.0)
        # Middle (items 2, 3): (6 + 8) / 2 = 7.0
        self.assertAlmostEqual(feats["middle_stage_aleatoric_mean"], 7.0)
        # Late (items 4, 5): (10 + 12) / 2 = 11.0
        self.assertAlmostEqual(feats["late_stage_aleatoric_mean"], 11.0)

        # Slope over t/6: Delta u = 12 - 2 = 10 over Delta t_norm = (6/6 - 1/6) = 5/6 -> slope = 12.0
        self.assertAlmostEqual(feats["aleatoric_slope"], 12.0)
        # Perfectly linear fit -> R^2 = 1.0
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
        # Constant uncertainty across 10 tokens
        constant_u = np.full(10, 0.42)
        slope, r2 = compute_ols_trend(constant_u)
        self.assertEqual(slope, 0.0)
        self.assertEqual(r2, 1.0)

        early, mid, late = compute_stage_means(constant_u)
        self.assertAlmostEqual(early, 0.42)
        self.assertAlmostEqual(mid, 0.42)
        self.assertAlmostEqual(late, 0.42)

    def test_short_traces_t_less_than_3(self) -> None:
        # T = 1
        t1 = np.array([0.75])
        slope, r2 = compute_ols_trend(t1)
        self.assertEqual(slope, 0.0)
        self.assertEqual(r2, 1.0)
        e, m, l = compute_stage_means(t1)
        self.assertEqual(e, 0.75)
        self.assertEqual(m, 0.75)
        self.assertEqual(l, 0.75)

        # T = 2
        t2 = np.array([0.2, 0.8])
        slope, r2 = compute_ols_trend(t2)
        self.assertGreater(slope, 0.0)
        self.assertAlmostEqual(r2, 1.0)
        e, m, l = compute_stage_means(t2)
        self.assertAlmostEqual(e, 0.2)
        self.assertAlmostEqual(m, 0.5)
        self.assertAlmostEqual(l, 0.8)


class EpistemicFeatureTests(unittest.TestCase):
    def test_epistemic_aggregate_features_match_expected_values(self) -> None:
        e_t = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        feats = extract_epistemic_aggregate_features(e_t)

        self.assertEqual(len(feats), 5)
        for name in EPISTEMIC_AGGREGATE_FEATURES:
            self.assertIn(name, feats)

        self.assertAlmostEqual(feats["mean_epistemic_uncertainty"], 0.3)
        self.assertAlmostEqual(feats["min_epistemic_uncertainty"], 0.1)
        self.assertAlmostEqual(feats["max_epistemic_uncertainty"], 0.5)
        expected_var = float(np.var(e_t))
        self.assertAlmostEqual(feats["variance_epistemic_uncertainty"], expected_var)
        self.assertAlmostEqual(feats["std_epistemic_uncertainty"], math.sqrt(expected_var))

    def test_epistemic_temporal_features_linear_trend(self) -> None:
        # e_t increasing over 9 tokens
        e_t = np.linspace(0.1, 0.9, 9)
        feats = extract_epistemic_temporal_features(e_t)

        self.assertEqual(len(feats), 5)
        for name in EPISTEMIC_TEMPORAL_FEATURES:
            self.assertIn(name, feats)

        self.assertLess(feats["early_stage_epistemic_mean"], feats["middle_stage_epistemic_mean"])
        self.assertLess(feats["middle_stage_epistemic_mean"], feats["late_stage_epistemic_mean"])
        self.assertGreater(feats["epistemic_slope"], 0.0)
        self.assertAlmostEqual(feats["epistemic_r2_linearity"], 1.0)

    def test_empty_epistemic_inputs(self) -> None:
        agg = extract_epistemic_aggregate_features(np.array([]))
        self.assertEqual(len(agg), 5)
        for val in agg.values():
            self.assertTrue(math.isnan(val))

        temp = extract_epistemic_temporal_features(np.array([]))
        self.assertEqual(len(temp), 5)
        for val in temp.values():
            self.assertTrue(math.isnan(val))

    def test_estimate_epistemic_layer_dispersion(self) -> None:
        # Simulate 6 layers, 10 tokens, hidden dimension 16
        np.random.seed(42)
        layers = [np.random.randn(10, 16) for _ in range(6)]
        e_t = estimate_epistemic_layer_dispersion(layers)

        self.assertEqual(e_t.shape, (10,))
        self.assertTrue(np.all(e_t >= 0.0))

    def test_estimate_epistemic_mc_dropout(self) -> None:
        # Simulate 4 MC dropout passes, 8 tokens, vocabulary size 20
        np.random.seed(42)
        samples = [np.random.randn(8, 20) for _ in range(4)]
        e_t = estimate_epistemic_mc_dropout(samples)

        self.assertEqual(e_t.shape, (8,))
        self.assertTrue(np.all(e_t >= 0.0))

    def test_estimate_epistemic_volatility_proxy(self) -> None:
        entropy = np.array([0.1, 0.2, 0.8, 0.2, 0.1])
        log_probs = np.array([-0.1, -0.3, -2.5, -0.4, -0.2])
        e_t = estimate_epistemic_volatility_proxy(entropy, log_probs, window=3)

        self.assertEqual(e_t.shape, (5,))
        self.assertTrue(np.all(e_t >= 0.0))
        # The peak in uncertainty at index 2 should have higher epistemic hesitation
        self.assertGreater(e_t[2], e_t[0])


class UnifiedFeatureSystemTests(unittest.TestCase):
    def test_extract_trace_features_computes_all_29_features(self) -> None:
        record = {
            "problem_id": "prob_test",
            "model_id": "test_model",
            "generation_num_tokens": 5,
            "original_problem": "2 + 2?",
            "reasoning_trace": "4",
            "model_is_robust": True,
            "selected_logprob_trace": [-0.1, -0.2, -0.3, -0.4, -0.5],
            "entropy_trace": [0.05, 0.1, 0.15, 0.2, 0.25],
            "top1_prob_trace": [0.95, 0.9, 0.85, 0.8, 0.75],
            "top2_margin_trace": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
        features = extract_trace_features(record)

        self.assertEqual(len(features), 29)
        for name in ALL_29_UNCERTAINTY_FEATURES:
            self.assertIn(name, features)
            self.assertFalse(math.isnan(features[name]), f"{name} should not be NaN")

    def test_extract_features_dataset_feature_sets(self) -> None:
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
            for i in range(5)
        ]

        df_all = extract_features_dataset(records, feature_set="all")
        self.assertEqual(len(df_all), 5)
        # 6 core metadata + 29 features = 35 columns
        self.assertEqual(len(df_all.columns), 35)

        df_aleatoric_agg = extract_features_dataset(records, feature_set="aleatoric_agg")
        self.assertEqual(len(df_aleatoric_agg.columns), 6 + 14)

        df_aleatoric_temp = extract_features_dataset(records, feature_set="aleatoric_temp")
        self.assertEqual(len(df_aleatoric_temp.columns), 6 + 5)

        df_epistemic_agg = extract_features_dataset(records, feature_set="epistemic_agg")
        self.assertEqual(len(df_epistemic_agg.columns), 6 + 5)

        df_epistemic_temp = extract_features_dataset(records, feature_set="epistemic_temp")
        self.assertEqual(len(df_epistemic_temp.columns), 6 + 5)


if __name__ == "__main__":
    unittest.main()

