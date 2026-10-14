"""Modular extraction for Aleatoric and Epistemic Uncertainty Features (Aggregate & Temporal).

This module implements:
- Issue #2: 14 Aggregate & 5 Temporal Aleatoric Uncertainty Features
- Issue #3: Epistemic Uncertainty Estimation, 5 Aggregate & 5 Temporal Epistemic Features
- Full 29-feature Multimodal Uncertainty System
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Feature Name Constants (Matching Issues #2, #3, and #4)
# -----------------------------------------------------------------------------

ALEATORIC_AGGREGATE_FEATURES: tuple[str, ...] = (
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

ALEATORIC_TEMPORAL_FEATURES: tuple[str, ...] = (
    "early_stage_aleatoric_mean",
    "middle_stage_aleatoric_mean",
    "late_stage_aleatoric_mean",
    "aleatoric_slope",
    "aleatoric_r2_linearity",
)

EPISTEMIC_AGGREGATE_FEATURES: tuple[str, ...] = (
    "mean_epistemic_uncertainty",
    "min_epistemic_uncertainty",
    "max_epistemic_uncertainty",
    "std_epistemic_uncertainty",
    "variance_epistemic_uncertainty",
)

EPISTEMIC_TEMPORAL_FEATURES: tuple[str, ...] = (
    "early_stage_epistemic_mean",
    "middle_stage_epistemic_mean",
    "late_stage_epistemic_mean",
    "epistemic_slope",
    "epistemic_r2_linearity",
)

ALL_ALEATORIC_FEATURES: tuple[str, ...] = (
    ALEATORIC_AGGREGATE_FEATURES + ALEATORIC_TEMPORAL_FEATURES
)

ALL_EPISTEMIC_FEATURES: tuple[str, ...] = (
    EPISTEMIC_AGGREGATE_FEATURES + EPISTEMIC_TEMPORAL_FEATURES
)

ALL_29_UNCERTAINTY_FEATURES: tuple[str, ...] = (
    ALL_ALEATORIC_FEATURES + ALL_EPISTEMIC_FEATURES
)


# -----------------------------------------------------------------------------
# Core Temporal Dynamics Engine: Splitting and OLS Regression
# -----------------------------------------------------------------------------

def compute_ols_trend(series: np.ndarray) -> tuple[float, float]:
    """Compute OLS linear regression slope and R^2 over normalized step index t/T.

    Normalized step index t_tilde = t / T where t in 1, ..., T.
    Model: u_t ~ m * t_tilde + c

    Args:
        series: 1D array of values over time.

    Returns:
        tuple (slope, r2_linearity):
            - slope: Rate of change across the normalized reasoning timeline.
            - r2_linearity: Coefficient of determination in [0.0, 1.0].
    """
    arr = np.asarray(series, dtype=float)
    t_len = arr.size

    if t_len == 0:
        return float("nan"), float("nan")

    if t_len == 1:
        # A single point has 0 slope and is trivially collinear
        return 0.0, 1.0

    t_norm = np.linspace(1.0 / t_len, 1.0, t_len, dtype=float)

    # Check for zero variance
    var_u = float(np.var(arr))
    if var_u == 0.0 or math.isclose(var_u, 0.0, abs_tol=1e-12):
        return 0.0, 1.0

    # OLS closed-form calculation
    t_mean = float(np.mean(t_norm))
    u_mean = float(np.mean(arr))

    cov_tu = float(np.sum((t_norm - t_mean) * (arr - u_mean)))
    var_t = float(np.sum((t_norm - t_mean) ** 2))

    if var_t == 0.0 or math.isclose(var_t, 0.0, abs_tol=1e-12):
        return 0.0, 1.0

    slope = cov_tu / var_t
    intercept = u_mean - slope * t_mean

    predicted = slope * t_norm + intercept
    ss_res = float(np.sum((arr - predicted) ** 2))
    ss_tot = float(np.sum((arr - u_mean) ** 2))

    if ss_tot == 0.0 or math.isclose(ss_tot, 0.0, abs_tol=1e-12):
        r2 = 1.0
    else:
        r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))

    return float(slope), float(r2)


def compute_stage_means(series: np.ndarray) -> tuple[float, float, float]:
    """Compute average uncertainty across early (0-33%), middle (33-66%), and late (66-100%) stages.

    Handles boundary conditions and short traces (T < 3) gracefully without crashing.

    Args:
        series: 1D array of uncertainty metrics.

    Returns:
        tuple (early_mean, middle_mean, late_mean).
    """
    arr = np.asarray(series, dtype=float)
    t_len = arr.size

    if t_len == 0:
        return float("nan"), float("nan"), float("nan")

    if t_len == 1:
        val = float(arr[0])
        return val, val, val

    if t_len == 2:
        # Split 2 items: item 0 as early, item 1 as late, mean of both as middle
        return float(arr[0]), float(np.mean(arr)), float(arr[1])

    # Standard stage splitting: t in 1..floor(T/3), floor(T/3)+1..floor(2T/3), floor(2T/3)+1..T
    i1 = max(1, t_len // 3)
    i2 = max(i1 + 1, (2 * t_len) // 3)

    early_slice = arr[:i1]
    mid_slice = arr[i1:i2]
    late_slice = arr[i2:]

    early_mean = float(np.mean(early_slice)) if early_slice.size else float(arr[0])
    mid_mean = float(np.mean(mid_slice)) if mid_slice.size else float(np.mean(arr))
    late_mean = float(np.mean(late_slice)) if late_slice.size else float(arr[-1])

    return early_mean, mid_mean, late_mean


# -----------------------------------------------------------------------------
# Issue #2: Data-Based (Aleatoric) Feature Extractors
# -----------------------------------------------------------------------------

def extract_aleatoric_aggregate_features(
    *,
    log_probs: np.ndarray,
    probs: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
    top1_probs: np.ndarray | None = None,
    top2_margins: np.ndarray | None = None,
    selected_is_top1: np.ndarray | None = None,
    min_k_fraction: float = 0.20,
    high_conf_threshold: float = 0.90,
    low_conf_threshold: float = 0.10,
) -> dict[str, float]:
    """Compute the 14 aggregate aleatoric uncertainty features across generated tokens.

    Args:
        log_probs: Token log-probabilities log P_t(y_t).
        probs: Token probabilities P_t(y_t). If None, exp(log_probs) is used.
        entropy: Predictive Shannon entropy H(P_t).
        top1_probs: Top-1 candidate probabilities.
        top2_margins: Probability margin between top-1 and top-2 candidates.
        selected_is_top1: Boolean or 0/1 array indicating if selected token was top-1.
        min_k_fraction: Fraction k for Min-K least likely token log-prob (default 0.20).
        high_conf_threshold: High confidence token threshold (default 0.90).
        low_conf_threshold: Low confidence token threshold (default 0.10).

    Returns:
        Dictionary mapping all 14 aggregate feature names to float values.
    """
    log_p = np.asarray(log_probs, dtype=float)
    t_len = log_p.size

    if t_len == 0:
        return {name: float("nan") for name in ALEATORIC_AGGREGATE_FEATURES}

    p = np.exp(log_p) if probs is None else np.asarray(probs, dtype=float)
    ent = (
        -log_p * p
        if entropy is None
        else np.asarray(entropy, dtype=float)
    )
    t1_p = p if top1_probs is None else np.asarray(top1_probs, dtype=float)
    t2_m = (
        np.zeros_like(log_p)
        if top2_margins is None
        else np.asarray(top2_margins, dtype=float)
    )
    is_top1 = (
        np.ones_like(log_p)
        if selected_is_top1 is None
        else np.asarray(selected_is_top1, dtype=float)
    )

    mean_log_p = float(np.mean(log_p))
    mean_nll = -mean_log_p if np.isfinite(mean_log_p) else float("nan")
    perplexity = float(np.exp(mean_nll)) if np.isfinite(mean_nll) else float("nan")

    k_items = max(1, int(np.ceil(t_len * min_k_fraction)))
    min_k_log_p = float(np.mean(np.sort(log_p)[:k_items]))

    return {
        "mean_entropy": float(np.mean(ent)),
        "mean_token_probability": float(np.mean(p)),
        "mean_log_probability": mean_log_p,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": perplexity,
        "min_log_probability": float(np.min(log_p)),
        "max_log_probability": float(np.max(log_p)),
        "std_log_probability": float(np.std(log_p)),
        "min_k_log_probability": min_k_log_p,
        "mean_top1_probability": float(np.mean(t1_p)),
        "mean_top2_probability_margin": float(np.mean(t2_m)),
        "fraction_selected_is_top1": float(np.mean(is_top1.astype(float))),
        "fraction_high_confidence": float(np.mean((p >= high_conf_threshold).astype(float))),
        "fraction_low_confidence": float(np.mean((p <= low_conf_threshold).astype(float))),
    }


def extract_aleatoric_temporal_features(
    uncertainty_series: np.ndarray,
) -> dict[str, float]:
    """Compute the 5 temporal aleatoric features over the uncertainty sequence u_t.

    Features:
    1. early_stage_aleatoric_mean (first third)
    2. middle_stage_aleatoric_mean (middle third)
    3. late_stage_aleatoric_mean (final third)
    4. aleatoric_slope (OLS linear slope over t/T)
    5. aleatoric_r2_linearity (R^2 linearity of regression fit)

    Args:
        uncertainty_series: 1D array of token uncertainty metrics (e.g. entropy or NLL).

    Returns:
        Dictionary mapping all 5 temporal feature names to float values.
    """
    arr = np.asarray(uncertainty_series, dtype=float)

    if arr.size == 0:
        return {name: float("nan") for name in ALEATORIC_TEMPORAL_FEATURES}

    early, mid, late = compute_stage_means(arr)
    slope, r2 = compute_ols_trend(arr)

    return {
        "early_stage_aleatoric_mean": early,
        "middle_stage_aleatoric_mean": mid,
        "late_stage_aleatoric_mean": late,
        "aleatoric_slope": slope,
        "aleatoric_r2_linearity": r2,
    }


# -----------------------------------------------------------------------------
# Issue #3: Model-Based (Epistemic) Uncertainty Estimation & Features
# -----------------------------------------------------------------------------

def estimate_epistemic_layer_dispersion(
    hidden_states_by_layer: Sequence[np.ndarray],
) -> np.ndarray:
    """Estimate epistemic uncertainty per token via transformer layer hidden state dispersion.

    Measures the representation drift and divergence across intermediate and upper
    layers of the model. When a model possesses high epistemic certainty, higher layer
    representations align stably; divergence signals internal model doubt.

    Args:
        hidden_states_by_layer: List/array of shape (L, T, D) or list of L arrays of shape (T, D).

    Returns:
        1D numpy array e_t of length T representing estimated epistemic uncertainty.
    """
    layers = [np.asarray(layer, dtype=float) for layer in hidden_states_by_layer]
    num_layers = len(layers)

    if num_layers < 2:
        # Cannot measure dispersion with fewer than 2 layers
        return np.zeros(layers[0].shape[0], dtype=float)

    # Use the upper half of layers where semantics and internal belief consolidate
    start_layer = num_layers // 2
    eval_layers = np.stack(layers[start_layer:], axis=0)  # Shape: (sub_L, T, D)

    # Compute variance across layers for each token dimension, then average over dimension D
    layer_var = np.var(eval_layers, axis=0)  # Shape: (T, D)
    e_t = np.mean(layer_var, axis=-1)  # Shape: (T,)

    return e_t


def estimate_epistemic_mc_dropout(
    logits_samples: Sequence[np.ndarray],
) -> np.ndarray:
    """Estimate epistemic uncertainty per token from stochastic (MC Dropout) forward passes.

    Measures predictive distribution variance across multiple stochastic realizations:
    e_t = mean_v( Var_sample( P_sample(v) ) )

    Args:
        logits_samples: Sequence of S sample logits arrays, each of shape (T, V).

    Returns:
        1D numpy array e_t of length T.
    """
    samples = [np.asarray(s, dtype=float) for s in logits_samples]
    num_samples = len(samples)

    if num_samples < 2:
        return np.zeros(samples[0].shape[0], dtype=float)

    # Convert logits to probabilities via softmax
    probs_stack = []
    for s in samples:
        exp_s = np.exp(s - np.max(s, axis=-1, keepdims=True))
        p = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        probs_stack.append(p)

    probs = np.stack(probs_stack, axis=0)  # Shape: (S, T, V)
    # Variance of token probabilities across passes
    sample_var = np.var(probs, axis=0)  # Shape: (T, V)
    e_t = np.sum(sample_var, axis=-1)  # Sum of variances over vocabulary, Shape: (T,)

    return e_t


def estimate_epistemic_volatility_proxy(
    entropy_series: np.ndarray,
    log_probs: np.ndarray,
    *,
    window: int = 5,
) -> np.ndarray:
    """High-speed epistemic proxy from trace sequences for offline datasets.

    Quantifies local confidence volatility and non-stationary reasoning hesitation:
    combining local entropy acceleration and unexpected probability dips.

    Args:
        entropy_series: Token entropy sequence H(P_t).
        log_probs: Token log probabilities log P_t(y_t).
        window: Rolling window size.

    Returns:
        1D array e_t of length T.
    """
    ent = np.asarray(entropy_series, dtype=float)
    lp = np.asarray(log_probs, dtype=float)
    t_len = ent.size

    if t_len == 0:
        return np.array([], dtype=float)

    if t_len == 1:
        return np.array([float(ent[0])], dtype=float)

    # Ensure rolling window does not exceed sequence length
    eff_window = max(1, min(window, t_len))
    rolling_ent = np.convolve(ent, np.ones(eff_window) / eff_window, mode="same")

    # Epistemic hesitation: deviation from local expectation + negative logprob acceleration
    diff_ent = np.abs(ent - rolling_ent)
    nll = -lp
    rolling_nll = np.convolve(nll, np.ones(eff_window) / eff_window, mode="same")
    diff_nll = np.abs(nll - rolling_nll)

    e_t = 0.5 * (diff_ent + diff_nll)
    return e_t


def extract_epistemic_aggregate_features(
    epistemic_series: np.ndarray,
) -> dict[str, float]:
    """Compute the 5 aggregate epistemic uncertainty features across reasoning steps.

    Features:
    1. mean_epistemic_uncertainty
    2. min_epistemic_uncertainty
    3. max_epistemic_uncertainty
    4. std_epistemic_uncertainty
    5. variance_epistemic_uncertainty

    Args:
        epistemic_series: Estimated epistemic uncertainty e_t for t = 1, ..., T.

    Returns:
        Dictionary mapping all 5 aggregate epistemic feature names to float values.
    """
    arr = np.asarray(epistemic_series, dtype=float)

    if arr.size == 0:
        return {name: float("nan") for name in EPISTEMIC_AGGREGATE_FEATURES}

    var_val = float(np.var(arr))
    return {
        "mean_epistemic_uncertainty": float(np.mean(arr)),
        "min_epistemic_uncertainty": float(np.min(arr)),
        "max_epistemic_uncertainty": float(np.max(arr)),
        "std_epistemic_uncertainty": float(np.std(arr)),
        "variance_epistemic_uncertainty": var_val,
    }


def extract_epistemic_temporal_features(
    epistemic_series: np.ndarray,
) -> dict[str, float]:
    """Compute the 5 temporal epistemic features over the reasoning progression.

    Features:
    1. early_stage_epistemic_mean (0 - 33%)
    2. middle_stage_epistemic_mean (33 - 66%)
    3. late_stage_epistemic_mean (66 - 100%)
    4. epistemic_slope (OLS linear slope over t/T)
    5. epistemic_r2_linearity (R^2 linearity of epistemic progression)

    Args:
        epistemic_series: Estimated epistemic uncertainty e_t for t = 1, ..., T.

    Returns:
        Dictionary mapping all 5 temporal epistemic feature names to float values.
    """
    arr = np.asarray(epistemic_series, dtype=float)

    if arr.size == 0:
        return {name: float("nan") for name in EPISTEMIC_TEMPORAL_FEATURES}

    early, mid, late = compute_stage_means(arr)
    slope, r2 = compute_ols_trend(arr)

    return {
        "early_stage_epistemic_mean": early,
        "middle_stage_epistemic_mean": mid,
        "late_stage_epistemic_mean": late,
        "epistemic_slope": slope,
        "epistemic_r2_linearity": r2,
    }


# -----------------------------------------------------------------------------
# Unified Feature Extraction & Dataset Batching
# -----------------------------------------------------------------------------

def extract_trace_features(
    record: Mapping[str, Any],
    *,
    epistemic_series: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract all 29 aleatoric and epistemic features from a trace record.

    Args:
        record: Mapping containing token trace arrays (logprobs, entropy, etc.).
        epistemic_series: Optional precomputed epistemic series e_t. If None,
            it is estimated using the trace volatility proxy.

    Returns:
        Dictionary containing all 29 features.
    """
    log_probs = (
        np.asarray(record["selected_logprob_trace"], dtype=float)
        if "selected_logprob_trace" in record
        else np.array([], dtype=float)
    )
    entropy = (
        np.asarray(record["entropy_trace"], dtype=float)
        if "entropy_trace" in record
        else np.array([], dtype=float)
    )
    top1 = (
        np.asarray(record["top1_prob_trace"], dtype=float)
        if "top1_prob_trace" in record
        else None
    )
    margin = (
        np.asarray(record["top2_margin_trace"], dtype=float)
        if "top2_margin_trace" in record
        else None
    )

    # 1. Aleatoric aggregate (14)
    aleatoric_agg = extract_aleatoric_aggregate_features(
        log_probs=log_probs,
        entropy=entropy if entropy.size else None,
        top1_probs=top1,
        top2_margins=margin,
    )

    # 2. Aleatoric temporal (5)
    temporal_base = entropy if entropy.size else (-log_probs if log_probs.size else np.array([]))
    aleatoric_temp = extract_aleatoric_temporal_features(temporal_base)

    # 3. Epistemic uncertainty estimation
    if epistemic_series is not None:
        e_t = np.asarray(epistemic_series, dtype=float)
    elif "epistemic_trace" in record and record["epistemic_trace"] is not None:
        e_t = np.asarray(record["epistemic_trace"], dtype=float)
    elif entropy.size and log_probs.size:
        e_t = estimate_epistemic_volatility_proxy(entropy, log_probs)
    else:
        e_t = np.array([], dtype=float)

    # 4. Epistemic aggregate (5)
    epistemic_agg = extract_epistemic_aggregate_features(e_t)

    # 5. Epistemic temporal (5)
    epistemic_temp = extract_epistemic_temporal_features(e_t)

    # Combine all 29 features
    features: dict[str, float] = {}
    features.update(aleatoric_agg)
    features.update(aleatoric_temp)
    features.update(epistemic_agg)
    features.update(epistemic_temp)

    return features


def extract_features_dataset(
    records_or_df: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    feature_set: str = "all",
) -> pd.DataFrame:
    """Extract features over an entire dataset of reasoning trace records.

    Args:
        records_or_df: Sequence of trace records or DataFrame.
        feature_set: One of 'all' (all 29), 'aleatoric_agg' (14), 'aleatoric_temp' (5),
            'aleatoric_full' (19), 'epistemic_agg' (5), 'epistemic_temp' (5),
            'epistemic_full' (10).

    Returns:
        DataFrame containing core metadata columns alongside requested features.
    """
    if isinstance(records_or_df, pd.DataFrame):
        records = records_or_df.to_dict(orient="records")
    else:
        records = list(records_or_df)

    feature_col_map = {
        "all": ALL_29_UNCERTAINTY_FEATURES,
        "aleatoric_agg": ALEATORIC_AGGREGATE_FEATURES,
        "aleatoric_temp": ALEATORIC_TEMPORAL_FEATURES,
        "aleatoric_full": ALL_ALEATORIC_FEATURES,
        "epistemic_agg": EPISTEMIC_AGGREGATE_FEATURES,
        "epistemic_temp": EPISTEMIC_TEMPORAL_FEATURES,
        "epistemic_full": ALL_EPISTEMIC_FEATURES,
    }

    if feature_set not in feature_col_map:
        raise ValueError(
            f"unknown feature_set '{feature_set}', choose from: {list(feature_col_map.keys())}"
        )

    selected_cols = feature_col_map[feature_set]
    extracted_rows: list[dict[str, Any]] = []

    for r in records:
        features = extract_trace_features(r)
        row = {
            "problem_id": r.get("problem_id", ""),
            "model_id": r.get("model_id", ""),
            "generation_num_tokens": int(r.get("generation_num_tokens", 0)),
            "original_problem": r.get("original_problem", ""),
            "reasoning_trace": r.get("reasoning_trace", ""),
            "model_is_robust": bool(r.get("model_is_robust", False)),
        }
        for col in selected_cols:
            row[col] = features[col]
        extracted_rows.append(row)

    return pd.DataFrame.from_records(extracted_rows)
