"""Feature schema and extraction shared by submission-side inference."""

from __future__ import annotations

import math

import numpy as np

AGGREGATE_FEATURES: tuple[str, ...] = (
    "mean_entropy",
    "mean_token_probability",
    "mean_log_probability",
    "mean_negative_log_likelihood",
    "perplexity",
    "min_log_probability",
    "max_log_probability",
    "std_log_probability",
    "min_k_log_probability",
    "mean_top1_probability",
    "mean_top2_probability_margin",
    "fraction_selected_is_top1",
    "fraction_high_confidence",
    "fraction_low_confidence",
)

TEMPORAL_FEATURES: tuple[str, ...] = (
    "early_stage_aleatoric_mean",
    "middle_stage_aleatoric_mean",
    "late_stage_aleatoric_mean",
    "aleatoric_slope",
    "aleatoric_r2_linearity",
)

FEATURE_NAMES: tuple[str, ...] = (
    *AGGREGATE_FEATURES,
    *TEMPORAL_FEATURES,
    "generation_num_tokens",
    "reasoning_character_count",
)


def _stage_means(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    if values.size == 2:
        return float(values[0]), float(values.mean()), float(values[1])
    first = max(1, values.size // 3)
    second = max(first + 1, (2 * values.size) // 3)
    return (
        float(values[:first].mean()),
        float(values[first:second].mean()),
        float(values[second:].mean()),
    )


def _trend(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1 or math.isclose(float(np.var(values)), 0.0, abs_tol=1e-12):
        return 0.0, 1.0
    timeline = np.linspace(1.0 / values.size, 1.0, values.size, dtype=float)
    timeline_centered = timeline - timeline.mean()
    values_centered = values - values.mean()
    denominator = float(np.sum(timeline_centered**2))
    slope = float(np.sum(timeline_centered * values_centered) / denominator)
    fitted = slope * timeline + (float(values.mean()) - slope * float(timeline.mean()))
    residual = float(np.sum((values - fitted) ** 2))
    total = float(np.sum(values_centered**2))
    r_squared = max(0.0, min(1.0, 1.0 - residual / total))
    return slope, r_squared


def extract_features(
    *,
    selected_log_probs: list[float],
    entropy: list[float],
    top1_probabilities: list[float],
    top2_margins: list[float],
    reasoning_trace: str,
) -> dict[str, float]:
    """Calculate features in the exact ordering used by the fitted pipeline."""

    log_prob = np.asarray(selected_log_probs, dtype=float)
    entropy_array = np.asarray(entropy, dtype=float)
    top1 = np.asarray(top1_probabilities, dtype=float)
    margins = np.asarray(top2_margins, dtype=float)
    token_count = log_prob.size
    if token_count == 0:
        return {name: float("nan") for name in FEATURE_NAMES}

    probability = np.exp(log_prob)
    mean_log_probability = float(log_prob.mean())
    mean_nll = -mean_log_probability
    min_k_items = max(1, int(np.ceil(token_count * 0.20)))
    early, middle, late = _stage_means(entropy_array)
    slope, r_squared = _trend(entropy_array)

    return {
        "mean_entropy": float(entropy_array.mean()),
        "mean_token_probability": float(probability.mean()),
        "mean_log_probability": mean_log_probability,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": float(np.exp(mean_nll)),
        "min_log_probability": float(log_prob.min()),
        "max_log_probability": float(log_prob.max()),
        "std_log_probability": float(log_prob.std()),
        "min_k_log_probability": float(np.sort(log_prob)[:min_k_items].mean()),
        "mean_top1_probability": float(top1.mean()),
        "mean_top2_probability_margin": float(margins.mean()),
        "fraction_selected_is_top1": 1.0,
        "fraction_high_confidence": float(np.mean(probability >= 0.90)),
        "fraction_low_confidence": float(np.mean(probability <= 0.10)),
        "early_stage_aleatoric_mean": early,
        "middle_stage_aleatoric_mean": middle,
        "late_stage_aleatoric_mean": late,
        "aleatoric_slope": slope,
        "aleatoric_r2_linearity": r_squared,
        "generation_num_tokens": float(token_count),
        "reasoning_character_count": float(len(reasoning_trace)),
    }
