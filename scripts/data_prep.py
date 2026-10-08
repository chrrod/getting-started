"""Data preparation, ingestion pipeline, and standard schema utilities for AIMO traces.

This module fulfills Issue #1:
- Standard 6-column dataset schema definition and strict validation.
- Ingestion of public/local AIMO development datasets.
- Resumable checkpointing and sharded execution support.
- Leakage-free StratifiedGroupKFold problem-level splitting and verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CORE_SCHEMA_COLUMNS: tuple[str, ...] = (
    "problem_id",
    "model_id",
    "generation_num_tokens",
    "original_problem",
    "reasoning_trace",
    "model_is_robust",
)


def generate_problem_id(problem_text: str) -> str:
    """Generate a deterministic, stable problem identifier from prompt text.

    Args:
        problem_text: The original problem statement.

    Returns:
        A unique problem id string prefixed with 'prob_'.
    """
    clean_text = problem_text.strip().encode("utf-8")
    digest = hashlib.sha256(clean_text).hexdigest()[:12]
    return f"prob_{digest}"


def extract_problem_text(row: Mapping[str, Any]) -> str:
    """Extract original problem text from varied column naming conventions.

    Args:
        row: Record containing problem prompt.

    Returns:
        The string prompt.

    Raises:
        KeyError: If neither 'original_problem' nor 'problem' is present.
    """
    if "original_problem" in row and row["original_problem"] is not None:
        return str(row["original_problem"])
    if "problem" in row and row["problem"] is not None:
        return str(row["problem"])
    raise KeyError("row must contain either 'original_problem' or 'problem'")


def extract_robustness_label(row: Mapping[str, Any]) -> bool:
    """Extract or compute the ground-truth target boolean model_is_robust.

    Args:
        row: Record containing robustness outcome indicators.

    Returns:
        True if the solution is robust, False otherwise.

    Raises:
        ValueError: If no valid boolean robustness indicator can be extracted.
    """
    for key in ("model_is_robust", "is_robust"):
        if key in row and row[key] is not None:
            val = row[key]
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            if isinstance(val, str):
                if val.strip().lower() in ("true", "1"):
                    return True
                if val.strip().lower() in ("false", "0"):
                    return False
            if isinstance(val, (int, float)) and not np.isnan(val):
                return bool(val)

    if "absolute_accuracy_decay" in row and row["absolute_accuracy_decay"] is not None:
        decay = float(row["absolute_accuracy_decay"])
        return decay <= 0.0

    if "max_drop" in row and row["max_drop"] is not None:
        drop = float(row["max_drop"])
        return drop <= 0.0

    raise ValueError(f"cannot compute robustness label from row keys: {list(row.keys())}")


def validate_standard_schema(
    frame: pd.DataFrame,
    *,
    allow_extra_columns: bool = True,
) -> pd.DataFrame:
    """Validate that a DataFrame conforms strictly to the Issue #1 schema contract.

    Required columns:
    - problem_id (str): Non-empty unique problem identifier.
    - model_id (str): Non-empty evaluated model identifier.
    - generation_num_tokens (int): Non-negative number of generated tokens.
    - original_problem (str): Non-empty prompt text.
    - reasoning_trace (str): Model reasoning chain-of-thought text.
    - model_is_robust (bool): Native boolean target label.

    Args:
        frame: The DataFrame to validate.
        allow_extra_columns: Whether non-core columns (e.g. token traces) are permitted.

    Returns:
        A validated copy of the DataFrame with normalized types.

    Raises:
        ValueError: If required columns are missing, contain nulls, or have invalid types.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected pd.DataFrame, got {type(frame)}")

    missing = [col for col in CORE_SCHEMA_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required schema columns: {missing}")

    if not allow_extra_columns:
        extra = [col for col in frame.columns if col not in CORE_SCHEMA_COLUMNS]
        if extra:
            raise ValueError(f"disallowed extra columns found: {extra}")

    validated = frame.copy()

    # Check nulls in core columns
    for col in CORE_SCHEMA_COLUMNS:
        if validated[col].isna().any():
            null_count = int(validated[col].isna().sum())
            raise ValueError(f"schema column '{col}' contains {null_count} null/NaN values")

    # Validate problem_id
    validated["problem_id"] = validated["problem_id"].astype(str)
    if (validated["problem_id"].str.strip().str.len() == 0).any():
        raise ValueError("problem_id column contains empty strings")

    # Validate model_id
    validated["model_id"] = validated["model_id"].astype(str)
    if (validated["model_id"].str.strip().str.len() == 0).any():
        raise ValueError("model_id column contains empty strings")

    # Validate generation_num_tokens
    token_counts = pd.to_numeric(validated["generation_num_tokens"], errors="coerce")
    if token_counts.isna().any() or (token_counts < 0).any():
        raise ValueError("generation_num_tokens must contain non-negative integers")
    validated["generation_num_tokens"] = token_counts.astype(int)

    # Validate original_problem
    validated["original_problem"] = validated["original_problem"].astype(str)
    if (validated["original_problem"].str.strip().str.len() == 0).any():
        raise ValueError("original_problem column contains empty strings")

    # Validate reasoning_trace
    validated["reasoning_trace"] = validated["reasoning_trace"].astype(str)

    # Validate model_is_robust
    robust_vals = validated["model_is_robust"].tolist()
    if not all(isinstance(v, (bool, np.bool_)) for v in robust_vals):
        raise ValueError("model_is_robust must contain native boolean values (True/False)")
    validated["model_is_robust"] = [bool(v) for v in robust_vals]

    return validated


def ingest_aimo_dataset(
    source: str | Path | pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Ingest a public/local AIMO labeled dataset and normalize input columns.

    Args:
        source: File path (parquet or jsonl), DataFrame, or list of record dicts.

    Returns:
        Normalized DataFrame with 'problem_id', 'model_id', 'original_problem',
        'model_is_robust', and any available metadata like 'reasoning_effort'.

    Raises:
        ValueError: If the source contains no rows or missing essential data.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"dataset source not found: {path}")
        if path.suffix.lower() == ".parquet":
            raw_df = pd.read_parquet(path)
        elif path.suffix.lower() in (".jsonl", ".json"):
            raw_df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
        else:
            raise ValueError(f"unsupported file extension: {path.suffix}")
    elif isinstance(source, pd.DataFrame):
        raw_df = source.copy()
    elif isinstance(source, Sequence):
        raw_df = pd.DataFrame.from_records(source)
    else:
        raise TypeError(f"unsupported source type: {type(source)}")

    if len(raw_df) == 0:
        raise ValueError("source dataset is empty")

    records: list[dict[str, Any]] = []
    for _, row_dict in raw_df.iterrows():
        row_map = dict(row_dict)
        problem_text = extract_problem_text(row_map)
        problem_id = (
            str(row_map["problem_id"]).strip()
            if "problem_id" in row_map and pd.notna(row_map["problem_id"]) and str(row_map["problem_id"]).strip()
            else generate_problem_id(problem_text)
        )
        model_id = str(row_map["model_id"]).strip() if "model_id" in row_map else "unknown"
        robust = extract_robustness_label(row_map)

        normalized: dict[str, Any] = {
            "problem_id": problem_id,
            "model_id": model_id,
            "original_problem": problem_text,
            "model_is_robust": robust,
        }
        if "reasoning_effort" in row_map and pd.notna(row_map["reasoning_effort"]):
            normalized["reasoning_effort"] = str(row_map["reasoning_effort"])
        if "max_drop" in row_map and pd.notna(row_map["max_drop"]):
            normalized["max_drop"] = float(row_map["max_drop"])

        records.append(normalized)

    df = pd.DataFrame.from_records(records)
    return df


def partition_shards(
    df: pd.DataFrame,
    num_shards: int,
    shard_index: int,
) -> pd.DataFrame:
    """Deterministically partition a DataFrame for parallel/sharded execution.

    Args:
        df: Input DataFrame.
        num_shards: Total number of shards (>= 1).
        shard_index: 0-indexed index of this shard (0 <= shard_index < num_shards).

    Returns:
        Subset DataFrame belonging to this shard.

    Raises:
        ValueError: If shard arguments are invalid.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in range [0, {num_shards - 1}], got {shard_index}"
        )

    # Interleaved round-robin partitioning ensures even distribution
    indices = np.arange(shard_index, len(df), num_shards)
    return df.iloc[indices].copy().reset_index(drop=True)


def load_completed_keys(output_path: Path | str) -> set[tuple[str, str]]:
    """Inspect an existing JSONL or Parquet file for already generated keys.

    Key is (problem_id, model_id).

    Args:
        output_path: Path to target or partial output file.

    Returns:
        Set of (problem_id, model_id) tuples that have already been generated.
    """
    path = Path(output_path)
    if not path.is_file() or path.stat().st_size == 0:
        return set()

    completed: set[tuple[str, str]] = set()
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path, columns=["problem_id", "model_id"])
            for _, row in df.iterrows():
                completed.add((str(row["problem_id"]), str(row["model_id"])))
        except Exception:
            pass
    elif path.suffix.lower() in (".jsonl", ".partial"):
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    pid = record.get("problem_id")
                    if not pid and "original_problem" in record:
                        pid = generate_problem_id(record["original_problem"])
                    mid = record.get("model_id")
                    if pid and mid:
                        completed.add((str(pid), str(mid)))
        except Exception:
            pass

    return completed


def make_stratified_group_splits(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "problem_id",
    target_col: str = "model_is_robust",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create StratifiedGroupKFold splits strictly grouped to eliminate data leakage.

    Args:
        df: Input DataFrame with group and target columns.
        n_splits: Number of cross-validation folds.
        seed: Random seed for shuffling.
        group_col: Column containing group identifiers (e.g. 'problem_id').
        target_col: Column containing binary target labels (e.g. 'model_is_robust').

    Returns:
        List of (train_indices, val_indices) numpy array tuples.

    Raises:
        ValueError: If data cannot be partitioned or has conflicting labels.
    """
    if len(df) == 0:
        raise ValueError("cannot split an empty DataFrame")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}")
    if group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not found in DataFrame")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in DataFrame")

    unique_groups = df[group_col].nunique()
    if unique_groups < n_splits:
        raise ValueError(
            f"cannot create {n_splits} folds with only {unique_groups} unique groups"
        )

    # Check for conflicting group labels (same problem with different robustness targets)
    label_counts = df.groupby(group_col, sort=False)[target_col].nunique()
    conflicting = label_counts[label_counts > 1]
    if not conflicting.empty:
        raise ValueError(
            f"conflicting labels found for groups: {list(conflicting.index[:5])}"
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = np.zeros(len(df))
    y = df[target_col].to_numpy().astype(int)
    groups = df[group_col].to_numpy()

    splits = list(splitter.split(X, y, groups))
    return [(np.asarray(train_idx), np.asarray(val_idx)) for train_idx, val_idx in splits]


def verify_grouped_splits(
    df: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    group_col: str = "problem_id",
    target_col: str = "model_is_robust",
) -> dict[str, Any]:
    """Verify that grouped splits have zero data leakage across folds.

    Args:
        df: Evaluated DataFrame.
        splits: List of (train_idx, val_idx) tuples.
        group_col: Column identifier for problem grouping.
        target_col: Target column identifier.

    Returns:
        Summary dict containing validation metrics and fold stats.

    Raises:
        AssertionError: If any problem group appears in both train and validation splits.
    """
    all_val_indices: list[int] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, start=1):
        train_groups = set(df.iloc[train_idx][group_col].tolist())
        val_groups = set(df.iloc[val_idx][group_col].tolist())

        leakage = train_groups & val_groups
        if leakage:
            raise AssertionError(
                f"Fold {fold_num} has data leakage! {len(leakage)} groups cross split boundary: {list(leakage)[:3]}"
            )

        train_targets = df.iloc[train_idx][target_col].astype(bool)
        val_targets = df.iloc[val_idx][target_col].astype(bool)

        fold_summaries.append(
            {
                "fold": fold_num,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_groups": len(train_groups),
                "val_groups": len(val_groups),
                "train_positive_ratio": float(train_targets.mean()),
                "val_positive_ratio": float(val_targets.mean()),
            }
        )
        all_val_indices.extend(val_idx.tolist())

    # Verify every row appears in validation exactly once
    index_set = set(all_val_indices)
    expected_set = set(range(len(df)))
    if index_set != expected_set or len(all_val_indices) != len(df):
        raise AssertionError("Validation indices across folds do not partition the DataFrame exactly once")

    return {
        "num_folds": len(splits),
        "total_samples": len(df),
        "unique_groups": df[group_col].nunique(),
        "has_leakage": False,
        "folds": fold_summaries,
    }


def atomic_write_dataframe(
    df: pd.DataFrame,
    path: Path | str,
    *,
    format: str = "auto",
) -> Path:
    """Atomically write a DataFrame to Parquet or JSONL using a temporary file.

    Args:
        df: DataFrame to serialize.
        path: Destination file path.
        format: 'parquet', 'jsonl', or 'auto' (inferred from suffix).

    Returns:
        Final destination Path.
    """
    dest_path = Path(path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "auto":
        suffix = dest_path.suffix.lower()
        if suffix == ".parquet":
            format = "parquet"
        elif suffix in (".jsonl", ".json"):
            format = "jsonl"
        else:
            format = "parquet"

    temp_fd, temp_name = tempfile.mkstemp(
        dir=dest_path.parent,
        prefix=f".{dest_path.name}.",
        suffix=f".{format}",
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)

    try:
        if format == "parquet":
            df.to_parquet(temp_path, index=False)
        else:
            df.to_json(temp_path, orient="records", lines=True, force_ascii=False)
        os.replace(temp_path, dest_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

    return dest_path


def standardize_trace_records(
    records: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Convert raw trace records into a validated standard schema DataFrame.

    Ensures all 6 core columns are present and correctly typed, while preserving
    any token-level distribution traces.

    Args:
        records: Iterable of record dictionaries.

    Returns:
        Validated DataFrame conforming to CORE_SCHEMA_COLUMNS.
    """
    normalized_list: list[dict[str, Any]] = []

    for row in records:
        row_dict = dict(row)
        problem_text = extract_problem_text(row_dict)
        problem_id = (
            str(row_dict["problem_id"]).strip()
            if "problem_id" in row_dict and pd.notna(row_dict["problem_id"]) and str(row_dict["problem_id"]).strip()
            else generate_problem_id(problem_text)
        )
        model_id = str(row_dict.get("model_id", "unknown")).strip()
        robust = extract_robustness_label(row_dict)
        trace_text = str(row_dict.get("reasoning_trace", ""))
        num_tokens = int(row_dict.get("generation_num_tokens", len(row_dict.get("generated_token_ids", []))))

        record = {
            "problem_id": problem_id,
            "model_id": model_id,
            "generation_num_tokens": num_tokens,
            "original_problem": problem_text,
            "reasoning_trace": trace_text,
            "model_is_robust": robust,
        }

        # Preserve any optional distribution traces if present
        for trace_key in (
            "generated_token_ids",
            "entropy_trace",
            "top1_prob_trace",
            "top2_margin_trace",
            "selected_logprob_trace",
            "reasoning_effort",
            "max_drop",
        ):
            if trace_key in row_dict:
                record[trace_key] = row_dict[trace_key]

        normalized_list.append(record)

    df = pd.DataFrame.from_records(normalized_list)
    return validate_standard_schema(df, allow_extra_columns=True)

