"""Load and validate traces for the robustness logistic-regression model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aleatoric_features import (
    ALEATORIC_AGGREGATE_FEATURES,
    ALEATORIC_TEMPORAL_FEATURES,
    extract_aleatoric_features,
)

TARGET_COLUMN = "model_is_robust"
GROUP_COLUMN = "problem_group_id"

# max_drop is used to construct model_is_robust in the source dataset. It must
# never enter X. The other names are common aliases worth rejecting explicitly.
LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        TARGET_COLUMN,
        "is_robust",
        "max_drop",
        "accuracy_drop",
        "robustness_score",
        "permutation_accuracy",
        "modified_problem_accuracy",
    }
)

LENGTH_FEATURES: tuple[str, ...] = (
    "generation_num_tokens",
    "reasoning_character_count",
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "aggregate_uncertainty": tuple(ALEATORIC_AGGREGATE_FEATURES),
    "temporal_uncertainty": tuple(ALEATORIC_TEMPORAL_FEATURES),
    "response_length": LENGTH_FEATURES,
}
FEATURE_COLUMNS: tuple[str, ...] = tuple(
    feature for group in FEATURE_GROUPS.values() for feature in group
)

REQUIRED_TRACE_COLUMNS: frozenset[str] = frozenset(
    {"model_id", "original_problem", TARGET_COLUMN}
)


def _problem_group_id(problem: str) -> str:
    """Return a stable privacy-friendly ID for an original problem string."""

    normalized = " ".join(problem.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _as_binary_label(value: Any, *, source: Path, line_number: int) -> int:
    """Validate a JSON label without accepting truthy strings or numbers."""

    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    raise ValueError(
        f"{source}:{line_number}: {TARGET_COLUMN!r} must be true/false or 0/1"
    )


def read_jsonl_files(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Read trace records and provide line-specific validation errors."""

    if not paths:
        raise ValueError("No input JSONL files were supplied")

    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Trace file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                missing = REQUIRED_TRACE_COLUMNS - record.keys()
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: missing required fields {sorted(missing)}"
                    )
                if not isinstance(record["original_problem"], str) or not record[
                    "original_problem"
                ].strip():
                    raise ValueError(
                        f"{path}:{line_number}: 'original_problem' must be non-empty text"
                    )
                record[TARGET_COLUMN] = _as_binary_label(
                    record[TARGET_COLUMN], source=path, line_number=line_number
                )
                record["source_file"] = path.name
                records.append(record)

    if not records:
        raise ValueError("The supplied JSONL files contained no records")
    return records


def discover_trace_files(data_dir: Path) -> list[Path]:
    """Find generated trace files deterministically."""

    paths = sorted(data_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No .jsonl trace files found in {data_dir}")
    return paths


def assert_no_leakage(feature_columns: Iterable[str]) -> None:
    """Fail fast if target-derived measurements are requested as predictors."""

    selected = set(feature_columns)
    leaking = sorted(selected & LEAKAGE_COLUMNS)
    if leaking:
        raise ValueError(
            "Target leakage: these columns cannot be predictors because they are "
            f"labels or label-derived measurements: {leaking}"
        )


def build_feature_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Build one numeric feature row per model response trace.

    The label describes robustness across problem perturbations, while every
    predictor is computed from the model's original response trace. No
    perturbation outcome or label-construction measurement is included.
    """

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        features = extract_aleatoric_features(record)
        trace = record.get("reasoning_trace", "")
        generated_tokens = record.get("generation_num_tokens")
        if generated_tokens is None:
            generated_tokens = len(record.get("selected_logprob_trace", []))

        row: dict[str, Any] = {
            "row_id": index,
            "model_id": str(record["model_id"]),
            "reasoning_effort": str(record.get("reasoning_effort", "default")),
            GROUP_COLUMN: _problem_group_id(record["original_problem"]),
            "source_file": str(record.get("source_file", "")),
            TARGET_COLUMN: int(record[TARGET_COLUMN]),
            "generation_num_tokens": float(generated_tokens),
            "reasoning_character_count": float(len(trace)) if isinstance(trace, str) else np.nan,
        }
        row.update({name: features[name] for name in ALEATORIC_AGGREGATE_FEATURES})
        row.update({name: features[name] for name in ALEATORIC_TEMPORAL_FEATURES})
        rows.append(row)

    frame = pd.DataFrame(rows)
    assert_no_leakage(FEATURE_COLUMNS)
    _validate_feature_frame(frame)
    return frame


def _validate_feature_frame(frame: pd.DataFrame) -> None:
    """Validate class, group, and numeric-feature requirements."""

    required = {TARGET_COLUMN, GROUP_COLUMN, *FEATURE_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Feature frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Feature frame is empty")
    labels = set(frame[TARGET_COLUMN].dropna().astype(int).unique())
    if labels != {0, 1}:
        raise ValueError(
            f"The experiment requires both target classes 0 and 1; found {sorted(labels)}"
        )
    if frame[GROUP_COLUMN].nunique() < 2:
        raise ValueError("At least two distinct original problems are required")
    non_numeric = [
        name for name in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(frame[name])
    ]
    if non_numeric:
        raise ValueError(f"Predictors must be numeric; found {non_numeric}")


def load_feature_frame(data_dir: Path) -> pd.DataFrame:
    """Discover JSONL files, load them, and compute the modeling table."""

    return build_feature_frame(read_jsonl_files(discover_trace_files(data_dir)))


def summarize_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable data provenance and missingness details."""

    counts = frame[TARGET_COLUMN].value_counts().sort_index()
    return {
        "unit_of_analysis": "one generated response for one model/problem/configuration",
        "rows": int(len(frame)),
        "original_problem_groups": int(frame[GROUP_COLUMN].nunique()),
        "models": int(frame["model_id"].nunique()),
        "model_ids": sorted(frame["model_id"].unique().tolist()),
        "class_counts": {
            "not_robust": int(counts.get(0, 0)),
            "robust": int(counts.get(1, 0)),
        },
        "reasoning_efforts": sorted(frame["reasoning_effort"].unique().tolist()),
        "source_files": sorted(frame["source_file"].unique().tolist()),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "missing_values_by_feature": {
            name: int(frame[name].isna().sum()) for name in FEATURE_COLUMNS
        },
        "excluded_leakage_columns": sorted(LEAKAGE_COLUMNS),
    }
