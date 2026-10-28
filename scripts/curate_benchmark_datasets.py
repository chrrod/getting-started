"""Curate, export, and verify standardized benchmark uncertainty datasets for AIMO.

This module fulfills Issue #4:
1. Compiles and exports the 4 core benchmark datasets + combined multimodal dataset:
   - dataset_1_aleatoric_aggregate.parquet (6 core cols + 14 aleatoric agg)
   - dataset_2_aleatoric_temporal.parquet (6 core cols + 5 aleatoric temp)
   - dataset_3_epistemic_aggregate.parquet (6 core cols + 5 epistemic agg)
   - dataset_4_epistemic_temporal.parquet (6 core cols + 5 epistemic temp)
   - dataset_5_multimodal_combined.parquet (6 core cols + 29 all uncertainty features)
2. Preserves strictly identical row ordering, indexing, and problem-model pairs across all datasets.
3. Implements automated verification suite checking schema, row parity, NaN/Inf absence, and class balance.
4. Generates a comprehensive metadata manifest.json with hashes, column schemas, and summary statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from data_prep import CORE_SCHEMA_COLUMNS, atomic_write_dataframe
from uncertainty_features import (
    ALEATORIC_AGGREGATE_FEATURES,
    ALEATORIC_TEMPORAL_FEATURES,
    ALL_29_UNCERTAINTY_FEATURES,
    ALL_ALEATORIC_FEATURES,
    ALL_EPISTEMIC_FEATURES,
    EPISTEMIC_AGGREGATE_FEATURES,
    EPISTEMIC_TEMPORAL_FEATURES,
    extract_features_dataset,
)

# -----------------------------------------------------------------------------
# Dataset Specification Mapping
# -----------------------------------------------------------------------------

BENCHMARK_DATASET_SPECS: dict[str, dict[str, Any]] = {
    "dataset_1_aleatoric_aggregate.parquet": {
        "dataset_id": "dataset_1",
        "name": "Data-Based (Aleatoric) Aggregate",
        "description": "Baseline assessing overall next-token confidence without temporal awareness.",
        "features": ALEATORIC_AGGREGATE_FEATURES,
        "feature_count": 14,
        "uncertainty_type": "aleatoric",
        "granularity": "aggregate",
    },
    "dataset_2_aleatoric_temporal.parquet": {
        "dataset_id": "dataset_2",
        "name": "Data-Based (Aleatoric) Temporal",
        "description": "Evaluates whether the trajectory of aleatoric uncertainty over time predicts robustness.",
        "features": ALEATORIC_TEMPORAL_FEATURES,
        "feature_count": 5,
        "uncertainty_type": "aleatoric",
        "granularity": "temporal",
    },
    "dataset_3_epistemic_aggregate.parquet": {
        "dataset_id": "dataset_3",
        "name": "Model-Based (Epistemic) Aggregate",
        "description": "Evaluates static model-level uncertainty (dispersion across weights/layers).",
        "features": EPISTEMIC_AGGREGATE_FEATURES,
        "feature_count": 5,
        "uncertainty_type": "epistemic",
        "granularity": "aggregate",
    },
    "dataset_4_epistemic_temporal.parquet": {
        "dataset_id": "dataset_4",
        "name": "Model-Based (Epistemic) Temporal",
        "description": "Evaluates whether epistemic uncertainty shifts over reasoning steps provide key predictive signals.",
        "features": EPISTEMIC_TEMPORAL_FEATURES,
        "feature_count": 5,
        "uncertainty_type": "epistemic",
        "granularity": "temporal",
    },
    "dataset_5_multimodal_combined.parquet": {
        "dataset_id": "dataset_5",
        "name": "Multimodal Uncertainty Combined",
        "description": "Full multimodal system combining all aleatoric and epistemic features across aggregate and temporal dynamics.",
        "features": ALL_29_UNCERTAINTY_FEATURES,
        "feature_count": 29,
        "uncertainty_type": "multimodal",
        "granularity": "combined",
    },
}

# Optional convenience groupings for Issue #5 ablation experiments
OPTIONAL_ABLATION_SPECS: dict[str, dict[str, Any]] = {
    "dataset_aleatoric_full.parquet": {
        "dataset_id": "dataset_aleatoric_full",
        "name": "Data-Based (Aleatoric) Full",
        "description": "All aleatoric aggregate and temporal features (19 features total).",
        "features": ALL_ALEATORIC_FEATURES,
        "feature_count": 19,
        "uncertainty_type": "aleatoric",
        "granularity": "full",
    },
    "dataset_epistemic_full.parquet": {
        "dataset_id": "dataset_epistemic_full",
        "name": "Model-Based (Epistemic) Full",
        "description": "All epistemic aggregate and temporal features (10 features total).",
        "features": ALL_EPISTEMIC_FEATURES,
        "feature_count": 10,
        "uncertainty_type": "epistemic",
        "granularity": "full",
    },
}


# -----------------------------------------------------------------------------
# Dataset Curation & Export Engine
# -----------------------------------------------------------------------------

def curate_benchmark_datasets(
    source_traces: pd.DataFrame | Sequence[Mapping[str, Any]] | Path | str,
    output_dir: Path | str,
    *,
    include_optional_ablations: bool = True,
    sort_canonical: bool = True,
) -> dict[str, pd.DataFrame]:
    """Curate and export all standardized benchmark datasets from raw reasoning traces.

    Args:
        source_traces: Input standardized traces DataFrame, list of records, or Parquet file path.
        output_dir: Destination folder where Parquet datasets will be written.
        include_optional_ablations: Whether to also export full aleatoric and full epistemic slices.
        sort_canonical: Whether to sort rows deterministically by (problem_id, model_id).

    Returns:
        Dictionary mapping filename to the exported DataFrame.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load source DataFrame
    if isinstance(source_traces, (str, Path)):
        p = Path(source_traces)
        if not p.exists():
            raise FileNotFoundError(f"Source traces file does not exist: {p}")
        df_raw = pd.read_parquet(p)
    elif isinstance(source_traces, pd.DataFrame):
        df_raw = source_traces.copy()
    else:
        df_raw = pd.DataFrame.from_records(list(source_traces))

    if len(df_raw) == 0:
        raise ValueError("Source traces dataset is empty; cannot curate benchmark datasets.")

    # 2. Extract full 29-feature multimodal dataset in a single deterministic pass
    df_features = extract_features_dataset(df_raw, feature_set="all")

    # 3. Canonical deterministic sorting
    if sort_canonical:
        sort_cols = [c for c in ["problem_id", "model_id"] if c in df_features.columns]
        if sort_cols:
            df_features = df_features.sort_values(by=sort_cols, ignore_index=True)
        else:
            df_features = df_features.reset_index(drop=True)
    else:
        df_features = df_features.reset_index(drop=True)

    # 4. Compile specs
    all_specs = dict(BENCHMARK_DATASET_SPECS)
    if include_optional_ablations:
        all_specs.update(OPTIONAL_ABLATION_SPECS)

    exported_datasets: dict[str, pd.DataFrame] = {}

    # 5. Export each dataset atomically
    for filename, spec in all_specs.items():
        feature_cols = list(spec["features"])
        target_cols = list(CORE_SCHEMA_COLUMNS) + feature_cols

        # Ensure all columns exist
        missing_cols = [c for c in target_cols if c not in df_features.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns for {filename}: {missing_cols}")

        sub_df = df_features[target_cols].copy()

        # Enforce clean typing
        sub_df["problem_id"] = sub_df["problem_id"].astype(str)
        sub_df["model_id"] = sub_df["model_id"].astype(str)
        sub_df["generation_num_tokens"] = sub_df["generation_num_tokens"].astype(int)
        sub_df["original_problem"] = sub_df["original_problem"].astype(str)
        sub_df["reasoning_trace"] = sub_df["reasoning_trace"].astype(str)
        sub_df["model_is_robust"] = sub_df["model_is_robust"].astype(bool)

        for fc in feature_cols:
            sub_df[fc] = sub_df[fc].astype(float)

        file_dest = out_path / filename
        atomic_write_dataframe(sub_df, file_dest)
        exported_datasets[filename] = sub_df

    return exported_datasets


# -----------------------------------------------------------------------------
# Automated Verification & Integrity Suite
# -----------------------------------------------------------------------------

def compute_file_sha256(path: Path | str) -> str:
    """Compute sha256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_benchmark_datasets(
    dataset_dir: Path | str,
    *,
    expected_filenames: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Rigorously verify schema, alignment, row count, null absence, and class balance across benchmark datasets.

    Checks:
    1. Existence of all expected Parquet files.
    2. Identical row count parity across all datasets.
    3. Identical (problem_id, model_id) row ordering across all datasets.
    4. Identical model_is_robust target labels row-by-row.
    5. Absence of NaN, Inf, or null values in all feature columns.
    6. Correct column counts matching 6 core + spec feature counts.
    7. Consistent class balance reporting.

    Args:
        dataset_dir: Directory containing the exported benchmark parquet files.
        expected_filenames: Optional list of files to check. Defaults to the 5 core datasets.

    Returns:
        Verification report dictionary with metrics and status.

    Raises:
        FileNotFoundError: If any expected dataset is missing.
        ValueError: If any integrity or parity check fails.
    """
    target_dir = Path(dataset_dir)
    if not target_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {target_dir}")

    if expected_filenames is None:
        expected_files = list(BENCHMARK_DATASET_SPECS.keys())
    else:
        expected_files = list(expected_filenames)

    # 1. Existence check
    missing_files = [f for f in expected_files if not (target_dir / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Verification failed: missing benchmark dataset files in {target_dir}: {missing_files}"
        )

    # 2. Load all datasets
    loaded: dict[str, pd.DataFrame] = {}
    for f in expected_files:
        loaded[f] = pd.read_parquet(target_dir / f)

    reference_file = expected_files[0]
    ref_df = loaded[reference_file]
    ref_n_rows = len(ref_df)

    if ref_n_rows == 0:
        raise ValueError(f"Reference dataset {reference_file} is empty (0 rows).")

    # 3. Check row counts and alignments against reference
    ref_keys = list(zip(ref_df["problem_id"].astype(str), ref_df["model_id"].astype(str)))
    ref_targets = ref_df["model_is_robust"].astype(bool).tolist()

    report: dict[str, Any] = {
        "status": "PASSED",
        "dataset_count": len(expected_files),
        "row_count": ref_n_rows,
        "datasets": {},
    }

    for filename in expected_files:
        df = loaded[filename]
        n_rows = len(df)

        # Check row count
        if n_rows != ref_n_rows:
            raise ValueError(
                f"Row count mismatch in {filename}: expected {ref_n_rows}, found {n_rows}"
            )

        # Check 6 core columns present
        for col in CORE_SCHEMA_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing core schema column '{col}' in {filename}")

        # Check key alignment
        cur_keys = list(zip(df["problem_id"].astype(str), df["model_id"].astype(str)))
        if cur_keys != ref_keys:
            raise ValueError(
                f"Row order / key misalignment in {filename} compared to reference {reference_file}"
            )

        # Check target alignment
        cur_targets = df["model_is_robust"].astype(bool).tolist()
        if cur_targets != ref_targets:
            raise ValueError(
                f"Target 'model_is_robust' misalignment in {filename} compared to reference"
            )

        # Check expected features according to spec if defined
        spec = BENCHMARK_DATASET_SPECS.get(filename) or OPTIONAL_ABLATION_SPECS.get(filename)
        if spec:
            expected_feats = list(spec["features"])
            for feat in expected_feats:
                if feat not in df.columns:
                    raise ValueError(f"Missing expected feature '{feat}' in {filename}")

            # Total columns = 6 core + len(features)
            expected_total_cols = len(CORE_SCHEMA_COLUMNS) + len(expected_feats)
            if len(df.columns) != expected_total_cols:
                raise ValueError(
                    f"Column count mismatch in {filename}: expected {expected_total_cols}, found {len(df.columns)}"
                )

            # Check NaN and Inf in features
            feat_df = df[expected_feats]
            nan_counts = feat_df.isna().sum().to_dict()
            total_nans = sum(nan_counts.values())
            if total_nans > 0:
                bad_cols = {k: v for k, v in nan_counts.items() if v > 0}
                raise ValueError(f"NaN values detected in feature columns of {filename}: {bad_cols}")

            # Check for Infs
            is_inf = np.isinf(feat_df.to_numpy(dtype=float))
            if np.any(is_inf):
                raise ValueError(f"Infinite (Inf) values detected in feature columns of {filename}")

        # Target class balance
        robust_count = int(sum(cur_targets))
        spurious_count = ref_n_rows - robust_count
        robust_ratio = robust_count / ref_n_rows

        report["datasets"][filename] = {
            "rows": n_rows,
            "columns": len(df.columns),
            "features_count": len(df.columns) - len(CORE_SCHEMA_COLUMNS),
            "robust_count": robust_count,
            "spurious_count": spurious_count,
            "robust_ratio": round(robust_ratio, 4),
            "sha256": compute_file_sha256(target_dir / filename),
            "file_size_bytes": os.path.getsize(target_dir / filename),
        }

    # Summary class balance
    report["class_balance"] = {
        "robust_count": int(sum(ref_targets)),
        "spurious_count": ref_n_rows - int(sum(ref_targets)),
        "robust_ratio": round(int(sum(ref_targets)) / ref_n_rows, 4),
    }

    # Model diversity count
    report["model_breakdown"] = ref_df["model_id"].value_counts().to_dict()

    return report


# -----------------------------------------------------------------------------
# Metadata Manifest Generator
# -----------------------------------------------------------------------------

def generate_dataset_manifest(
    dataset_dir: Path | str,
    output_manifest_path: Path | str | None = None,
) -> dict[str, Any]:
    """Generate a comprehensive JSON manifest documenting the curated benchmark datasets.

    Args:
        dataset_dir: Path to directory containing curated parquet files.
        output_manifest_path: Optional destination path for manifest.json.
            Defaults to `<dataset_dir>/manifest.json`.

    Returns:
        Manifest dictionary.
    """
    target_dir = Path(dataset_dir)
    if output_manifest_path is None:
        manifest_file = target_dir / "manifest.json"
    else:
        manifest_file = Path(output_manifest_path)

    # Run verification to get base stats
    verification = verify_benchmark_datasets(target_dir)

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "description": "AIMO Uncertainty Benchmark Curated Datasets for Mathematical Robustness Prediction",
        "total_rows": verification["row_count"],
        "class_balance": verification["class_balance"],
        "model_distribution": verification["model_breakdown"],
        "core_schema_columns": list(CORE_SCHEMA_COLUMNS),
        "datasets": {},
    }

    # Collect per-dataset metadata and summary feature statistics
    all_specs = {**BENCHMARK_DATASET_SPECS, **OPTIONAL_ABLATION_SPECS}

    for filename, base_meta in verification["datasets"].items():
        spec = all_specs.get(filename, {})
        df = pd.read_parquet(target_dir / filename)
        feature_cols = [c for c in df.columns if c not in CORE_SCHEMA_COLUMNS]

        # Compute summary stats per feature
        feature_stats: dict[str, dict[str, float]] = {}
        for col in feature_cols:
            series = df[col].astype(float)
            feature_stats[col] = {
                "mean": round(float(series.mean()), 6),
                "std": round(float(series.std()), 6),
                "min": round(float(series.min()), 6),
                "max": round(float(series.max()), 6),
            }

        manifest["datasets"][filename] = {
            "dataset_id": spec.get("dataset_id", filename),
            "name": spec.get("name", filename),
            "description": spec.get("description", ""),
            "uncertainty_type": spec.get("uncertainty_type", "unknown"),
            "granularity": spec.get("granularity", "unknown"),
            "rows": base_meta["rows"],
            "columns": base_meta["columns"],
            "feature_columns": feature_cols,
            "feature_count": len(feature_cols),
            "file_size_bytes": base_meta["file_size_bytes"],
            "sha256": base_meta["sha256"],
            "feature_statistics": feature_stats,
        }

    # Write manifest to disk
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate, export, and verify standardized AIMO uncertainty benchmark datasets."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/generated/standardized_traces.parquet",
        help="Path to source standardized traces Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/uncertainty_datasets",
        help="Destination directory for benchmark Parquet datasets.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip dataset curation and run verification checks on existing files in output-dir.",
    )
    parser.add_argument(
        "--no-optional-ablations",
        action="store_true",
        help="Do not export full aleatoric and full epistemic optional ablation datasets.",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if not args.verify_only:
        print(f"Curating benchmark datasets from {args.input} -> {out_dir} ...")
        curate_benchmark_datasets(
            source_traces=args.input,
            output_dir=out_dir,
            include_optional_ablations=not args.no_optional_ablations,
        )
        print("Curated datasets exported successfully.")

    print(f"Running automated verification checks in {out_dir} ...")
    report = verify_benchmark_datasets(out_dir)
    print(f"Verification status: {report['status']}")
    print(f"Total rows: {report['row_count']}")
    print(f"Class balance: {report['class_balance']}")
    print("Exporting manifest.json ...")
    generate_dataset_manifest(out_dir)
    print(f"Manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
