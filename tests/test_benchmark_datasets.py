"""Unit tests for the curated uncertainty benchmark datasets and verification suite.

Covers Issue #4:
- Parquet export of all 4 benchmark datasets + combined multimodal dataset.
- Exact row count parity and identical row-by-row indexation across datasets.
- Schema conformity (6 core metadata columns + specified feature sets).
- 0 missing/NaN/Inf values across all features.
- Verification engine catches tampering, misalignments, and NaNs.
- Manifest generation and checksums.
- Leakage-free StratifiedGroupKFold CV partitioning.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from curate_benchmark_datasets import (
    BENCHMARK_DATASET_SPECS,
    OPTIONAL_ABLATION_SPECS,
    compute_file_sha256,
    curate_benchmark_datasets,
    generate_dataset_manifest,
    verify_benchmark_datasets,
)
from data_prep import CORE_SCHEMA_COLUMNS
from uncertainty_features import (
    ALEATORIC_AGGREGATE_FEATURES,
    ALEATORIC_TEMPORAL_FEATURES,
    ALL_29_UNCERTAINTY_FEATURES,
    EPISTEMIC_AGGREGATE_FEATURES,
    EPISTEMIC_TEMPORAL_FEATURES,
)

DATA_DIR = Path("data/uncertainty_datasets")
TRACES_PATH = Path("data/generated/standardized_traces.parquet")


class TestBenchmarkDatasetIntegrity:
    """Tests verifying the actual curated benchmark datasets in data/uncertainty_datasets."""

    def test_all_expected_files_exist(self) -> None:
        """Verify all 5 primary benchmark datasets, 2 ablation datasets, and manifest exist."""
        assert DATA_DIR.exists()
        for filename in BENCHMARK_DATASET_SPECS:
            file_path = DATA_DIR / filename
            assert file_path.exists(), f"Missing primary benchmark dataset: {file_path}"
            assert file_path.stat().st_size > 0

        for filename in OPTIONAL_ABLATION_SPECS:
            file_path = DATA_DIR / filename
            assert file_path.exists(), f"Missing ablation dataset: {file_path}"
            assert file_path.stat().st_size > 0

        manifest_path = DATA_DIR / "manifest.json"
        assert manifest_path.exists()
        assert manifest_path.stat().st_size > 0

    def test_schema_and_column_counts(self) -> None:
        """Verify exact column counts and column names for each dataset."""
        expected_counts = {
            "dataset_1_aleatoric_aggregate.parquet": (20, ALEATORIC_AGGREGATE_FEATURES),
            "dataset_2_aleatoric_temporal.parquet": (11, ALEATORIC_TEMPORAL_FEATURES),
            "dataset_3_epistemic_aggregate.parquet": (11, EPISTEMIC_AGGREGATE_FEATURES),
            "dataset_4_epistemic_temporal.parquet": (11, EPISTEMIC_TEMPORAL_FEATURES),
            "dataset_5_multimodal_combined.parquet": (35, ALL_29_UNCERTAINTY_FEATURES),
        }

        for filename, (expected_total_cols, expected_feats) in expected_counts.items():
            df = pd.read_parquet(DATA_DIR / filename)
            assert len(df.columns) == expected_total_cols, (
                f"{filename} has {len(df.columns)} cols, expected {expected_total_cols}"
            )

            # Check 6 core columns present
            for core_col in CORE_SCHEMA_COLUMNS:
                assert core_col in df.columns, f"Missing core column '{core_col}' in {filename}"

            # Check specific feature columns present
            for feat in expected_feats:
                assert feat in df.columns, f"Missing feature column '{feat}' in {filename}"

            # Check core column types
            assert df["model_is_robust"].dtype == bool
            assert pd.api.types.is_integer_dtype(df["generation_num_tokens"])
            assert pd.api.types.is_string_dtype(df["problem_id"])
            assert pd.api.types.is_string_dtype(df["model_id"])

    def test_row_parity_and_alignment(self) -> None:
        """Verify exact row count and identical row-for-row ordering across all datasets."""
        reference_df = pd.read_parquet(DATA_DIR / "dataset_1_aleatoric_aggregate.parquet")
        ref_n_rows = len(reference_df)
        assert ref_n_rows > 0

        ref_problem_ids = reference_df["problem_id"].tolist()
        ref_model_ids = reference_df["model_id"].tolist()
        ref_targets = reference_df["model_is_robust"].tolist()

        all_files = list(BENCHMARK_DATASET_SPECS.keys()) + list(OPTIONAL_ABLATION_SPECS.keys())

        for filename in all_files:
            df = pd.read_parquet(DATA_DIR / filename)
            assert len(df) == ref_n_rows, f"Row count mismatch in {filename}"
            assert df["problem_id"].tolist() == ref_problem_ids, f"Problem ID mismatch in {filename}"
            assert df["model_id"].tolist() == ref_model_ids, f"Model ID mismatch in {filename}"
            assert df["model_is_robust"].tolist() == ref_targets, f"Target label mismatch in {filename}"

    def test_zero_nan_or_inf_in_features(self) -> None:
        """Verify that absolutely zero NaN, null, or Inf values exist in any feature column."""
        for filename in BENCHMARK_DATASET_SPECS:
            df = pd.read_parquet(DATA_DIR / filename)
            feature_cols = [c for c in df.columns if c not in CORE_SCHEMA_COLUMNS]

            # Null / NaN check
            nans_per_col = df[feature_cols].isna().sum()
            total_nans = nans_per_col.sum()
            assert total_nans == 0, f"Found NaN values in {filename}: {nans_per_col[nans_per_col > 0]}"

            # Inf check
            arr = df[feature_cols].to_numpy(dtype=float)
            assert not np.any(np.isinf(arr)), f"Found Inf values in feature columns of {filename}"

    def test_verify_benchmark_datasets_passes(self) -> None:
        """Verify that the automated integrity verification function passes on the datasets."""
        report = verify_benchmark_datasets(DATA_DIR)
        assert report["status"] == "PASSED"
        assert report["dataset_count"] == 5
        assert report["row_count"] == 34
        assert report["class_balance"]["robust_count"] == 24
        assert report["class_balance"]["spurious_count"] == 10
        assert report["class_balance"]["robust_ratio"] == pytest.approx(0.7059, abs=1e-4)

    def test_manifest_schema_and_checksums(self) -> None:
        """Verify the JSON manifest contains valid structure, statistics, and matches file hashes."""
        manifest_path = DATA_DIR / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["schema_version"] == "1.0.0"
        assert manifest["total_rows"] == 34
        assert len(manifest["core_schema_columns"]) == 6
        assert len(manifest["datasets"]) >= 5

        for filename, meta in manifest["datasets"].items():
            file_path = DATA_DIR / filename
            actual_sha = compute_file_sha256(file_path)
            assert meta["sha256"] == actual_sha, f"Checksum mismatch for {filename}"
            assert meta["rows"] == 34
            assert "feature_statistics" in meta
            # Check every feature in the dataset has stats in manifest
            for feat in meta["feature_columns"]:
                assert feat in meta["feature_statistics"]
                assert "mean" in meta["feature_statistics"][feat]
                assert "std" in meta["feature_statistics"][feat]

    def test_leakage_free_stratified_group_kfold(self) -> None:
        """Verify that 5-fold StratifiedGroupKFold partitions data with 0 problem leakage."""
        df = pd.read_parquet(DATA_DIR / "dataset_5_multimodal_combined.parquet")
        sgkf = StratifiedGroupKFold(n_splits=5)

        folds_tested = 0
        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(df, df["model_is_robust"], groups=df["problem_id"])
        ):
            train_problems = set(df.iloc[train_idx]["problem_id"])
            val_problems = set(df.iloc[val_idx]["problem_id"])

            # Must have zero problem overlap
            overlap = train_problems.intersection(val_problems)
            assert len(overlap) == 0, f"Fold {fold} has leaking problem IDs: {overlap}"

            # Both robust and non-robust instances present in train
            train_targets = df.iloc[train_idx]["model_is_robust"]
            assert train_targets.sum() > 0
            assert (~train_targets).sum() > 0

            folds_tested += 1

        assert folds_tested == 5


class TestBenchmarkCurationPipeline:
    """Tests evaluating the dataset curation engine on synthetic / edge-case traces."""

    def test_curate_synthetic_traces_in_tempdir(self) -> None:
        """Test curation end-to-end on synthetic data written to a temporary folder."""
        synthetic_records = [
            {
                "problem_id": "prob_alpha",
                "model_id": "model_test_1",
                "generation_num_tokens": 5,
                "original_problem": "Compute 2 + 2",
                "reasoning_trace": "2 plus 2 is 4",
                "model_is_robust": True,
                "selected_logprob_trace": [-0.1, -0.2, -0.15, -0.05, -0.12],
                "entropy_trace": [0.3, 0.4, 0.35, 0.2, 0.25],
                "top1_prob_trace": [0.9, 0.85, 0.88, 0.95, 0.92],
                "top2_margin_trace": [0.8, 0.7, 0.75, 0.9, 0.85],
            },
            {
                "problem_id": "prob_beta",
                "model_id": "model_test_2",
                "generation_num_tokens": 4,
                "original_problem": "Compute 5 * 5",
                "reasoning_trace": "5 times 5 is 25",
                "model_is_robust": False,
                "selected_logprob_trace": [-0.5, -0.6, -0.4, -0.8],
                "entropy_trace": [0.8, 0.9, 0.7, 1.1],
                "top1_prob_trace": [0.6, 0.55, 0.65, 0.45],
                "top2_margin_trace": [0.2, 0.1, 0.3, 0.05],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            exported = curate_benchmark_datasets(
                source_traces=synthetic_records,
                output_dir=tmp_path,
                include_optional_ablations=True,
            )

            assert len(exported) == 7
            report = verify_benchmark_datasets(tmp_path)
            assert report["status"] == "PASSED"
            assert report["row_count"] == 2
            assert report["class_balance"]["robust_count"] == 1
            assert report["class_balance"]["spurious_count"] == 1

            manifest = generate_dataset_manifest(tmp_path)
            assert manifest["total_rows"] == 2

    def test_verify_detects_missing_file(self) -> None:
        """Test that verification raises FileNotFoundError when an expected file is absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with pytest.raises(FileNotFoundError, match="missing benchmark dataset"):
                verify_benchmark_datasets(tmp_path)

    def test_verify_detects_row_mismatch(self) -> None:
        """Test that verification raises ValueError when one dataset has a different row count."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create a 2-row dataset 1, and 1-row dataset 2
            df2 = pd.DataFrame(
                {
                    "problem_id": ["p1", "p2"],
                    "model_id": ["m1", "m2"],
                    "generation_num_tokens": [10, 20],
                    "original_problem": ["prob 1", "prob 2"],
                    "reasoning_trace": ["t1", "t2"],
                    "model_is_robust": [True, False],
                    **{f: [0.1, 0.2] for f in ALEATORIC_AGGREGATE_FEATURES},
                }
            )
            df2.to_parquet(tmp_path / "dataset_1_aleatoric_aggregate.parquet")

            df1 = df2.iloc[:1].copy()
            df1.to_parquet(tmp_path / "dataset_2_aleatoric_temporal.parquet")

            with pytest.raises(ValueError, match="Row count mismatch"):
                verify_benchmark_datasets(
                    tmp_path,
                    expected_filenames=[
                        "dataset_1_aleatoric_aggregate.parquet",
                        "dataset_2_aleatoric_temporal.parquet",
                    ],
                )

    def test_verify_detects_nan_in_features(self) -> None:
        """Test that verification raises ValueError if a NaN exists in a feature column."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df = pd.DataFrame(
                {
                    "problem_id": ["p1", "p2"],
                    "model_id": ["m1", "m2"],
                    "generation_num_tokens": [10, 20],
                    "original_problem": ["prob 1", "prob 2"],
                    "reasoning_trace": ["t1", "t2"],
                    "model_is_robust": [True, False],
                    **{f: [0.1, np.nan] for f in ALEATORIC_AGGREGATE_FEATURES},
                }
            )
            df.to_parquet(tmp_path / "dataset_1_aleatoric_aggregate.parquet")

            with pytest.raises(ValueError, match="NaN values detected"):
                verify_benchmark_datasets(
                    tmp_path,
                    expected_filenames=["dataset_1_aleatoric_aggregate.parquet"],
                )

    def test_verify_detects_key_misalignment(self) -> None:
        """Test that verification raises ValueError if problem_id/model_id row order differs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            df1 = pd.DataFrame(
                {
                    "problem_id": ["p1", "p2"],
                    "model_id": ["m1", "m2"],
                    "generation_num_tokens": [10, 20],
                    "original_problem": ["prob 1", "prob 2"],
                    "reasoning_trace": ["t1", "t2"],
                    "model_is_robust": [True, False],
                    **{f: [0.1, 0.2] for f in ALEATORIC_AGGREGATE_FEATURES},
                }
            )
            df1.to_parquet(tmp_path / "dataset_1_aleatoric_aggregate.parquet")

            # Scrambled order
            df2 = pd.DataFrame(
                {
                    "problem_id": ["p2", "p1"],
                    "model_id": ["m2", "m1"],
                    "generation_num_tokens": [20, 10],
                    "original_problem": ["prob 2", "prob 1"],
                    "reasoning_trace": ["t2", "t1"],
                    "model_is_robust": [False, True],
                    **{f: [0.2, 0.1] for f in ALEATORIC_TEMPORAL_FEATURES},
                }
            )
            df2.to_parquet(tmp_path / "dataset_2_aleatoric_temporal.parquet")

            with pytest.raises(ValueError, match="misalignment"):
                verify_benchmark_datasets(
                    tmp_path,
                    expected_filenames=[
                        "dataset_1_aleatoric_aggregate.parquet",
                        "dataset_2_aleatoric_temporal.parquet",
                    ],
                )
