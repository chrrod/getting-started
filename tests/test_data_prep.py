"""Unit tests for the data preparation pipeline, standard schema, and leakage-free splits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from data_prep import (
    CORE_SCHEMA_COLUMNS,
    atomic_write_dataframe,
    extract_problem_text,
    extract_robustness_label,
    generate_problem_id,
    ingest_aimo_dataset,
    load_completed_keys,
    partition_shards,
    standardize_trace_records,
    make_stratified_group_splits,
    validate_standard_schema,
    verify_grouped_splits,
)


class DataPrepSchemaTests(unittest.TestCase):
    def test_generate_problem_id_is_deterministic(self) -> None:
        p1 = "Find the sum of all integers x such that 1 <= x <= 100."
        p2 = "Find the sum of all integers x such that 1 <= x <= 100.  \n"
        p3 = "Find the product of all integers x."

        id1 = generate_problem_id(p1)
        id2 = generate_problem_id(p2)
        id3 = generate_problem_id(p3)

        self.assertTrue(id1.startswith("prob_"))
        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)

    def test_extract_problem_text(self) -> None:
        self.assertEqual(extract_problem_text({"original_problem": "Hello"}), "Hello")
        self.assertEqual(extract_problem_text({"problem": "World"}), "World")
        with self.assertRaises(KeyError):
            extract_problem_text({"unrelated": 123})

    def test_extract_robustness_label(self) -> None:
        self.assertTrue(extract_robustness_label({"model_is_robust": True}))
        self.assertFalse(extract_robustness_label({"model_is_robust": False}))
        self.assertTrue(extract_robustness_label({"is_robust": True}))
        self.assertFalse(extract_robustness_label({"is_robust": "false"}))
        self.assertTrue(extract_robustness_label({"absolute_accuracy_decay": 0.0}))
        self.assertFalse(extract_robustness_label({"absolute_accuracy_decay": 0.25}))
        self.assertTrue(extract_robustness_label({"max_drop": -0.05}))
        self.assertFalse(extract_robustness_label({"max_drop": 0.15}))
        with self.assertRaises(ValueError):
            extract_robustness_label({"unrelated": "unknown"})

    def test_validate_standard_schema_success(self) -> None:
        records = [
            {
                "problem_id": "prob_1",
                "model_id": "model_a",
                "generation_num_tokens": 128,
                "original_problem": "Problem 1",
                "reasoning_trace": "<think>Step 1</think>",
                "model_is_robust": True,
            },
            {
                "problem_id": "prob_2",
                "model_id": "model_b",
                "generation_num_tokens": 0,
                "original_problem": "Problem 2",
                "reasoning_trace": "",
                "model_is_robust": False,
            },
        ]
        df = pd.DataFrame.from_records(records)
        validated = validate_standard_schema(df)
        self.assertEqual(list(validated.columns), list(CORE_SCHEMA_COLUMNS))
        self.assertEqual(validated["generation_num_tokens"].dtype, np.int64 if pd.Series([1]).dtype == np.int64 else int)
        self.assertEqual(validated["model_is_robust"].tolist(), [True, False])

    def test_validate_standard_schema_rejects_missing_columns(self) -> None:
        df = pd.DataFrame(
            [{"problem_id": "prob_1", "model_id": "model_a", "generation_num_tokens": 10}]
        )
        with self.assertRaisesRegex(ValueError, "missing required schema columns"):
            validate_standard_schema(df)

    def test_validate_standard_schema_rejects_nulls_and_empty_strings(self) -> None:
        valid_row = {
            "problem_id": "prob_1",
            "model_id": "model_a",
            "generation_num_tokens": 128,
            "original_problem": "Problem 1",
            "reasoning_trace": "Trace",
            "model_is_robust": True,
        }

        # Null in problem_id
        df_null = pd.DataFrame([dict(valid_row, problem_id=None)])
        with self.assertRaisesRegex(ValueError, "contains 1 null"):
            validate_standard_schema(df_null)

        # Empty string in original_problem
        df_empty = pd.DataFrame([dict(valid_row, original_problem="   ")])
        with self.assertRaisesRegex(ValueError, "empty strings"):
            validate_standard_schema(df_empty)

        # Negative token count
        df_neg = pd.DataFrame([dict(valid_row, generation_num_tokens=-5)])
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            validate_standard_schema(df_neg)


class IngestionAndSplittingTests(unittest.TestCase):
    def test_ingest_aimo_dataset_from_existing_parquet(self) -> None:
        parquet_path = ROOT / "data" / "train-main-v2.parquet"
        if not parquet_path.exists():
            self.skipTest("train-main-v2.parquet not present")

        df = ingest_aimo_dataset(parquet_path)
        self.assertGreater(len(df), 0)
        self.assertIn("problem_id", df.columns)
        self.assertIn("model_id", df.columns)
        self.assertIn("original_problem", df.columns)
        self.assertIn("model_is_robust", df.columns)
        self.assertTrue(all(p.startswith("prob_") for p in df["problem_id"]))
        self.assertTrue(all(type(v) is bool for v in df["model_is_robust"]))

    def test_partition_shards_completeness_and_disjointness(self) -> None:
        records = [{"id": i} for i in range(25)]
        df = pd.DataFrame(records)

        num_shards = 4
        shards = [partition_shards(df, num_shards, idx) for idx in range(num_shards)]

        # Check total count
        total_items = sum(len(s) for s in shards)
        self.assertEqual(total_items, 25)

        # Check disjointness
        seen_indices = set()
        for s in shards:
            shard_ids = set(s["id"].tolist())
            self.assertEqual(len(seen_indices & shard_ids), 0)
            seen_indices.update(shard_ids)
        self.assertEqual(seen_indices, set(range(25)))

        # Check invalid shard arguments
        with self.assertRaises(ValueError):
            partition_shards(df, 0, 0)
        with self.assertRaises(ValueError):
            partition_shards(df, 4, 4)

    def test_resumable_checkpointing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "test_traces.jsonl"

            # Empty file has 0 completed keys
            self.assertEqual(len(load_completed_keys(temp_file)), 0)

            # Write 2 lines
            records = [
                {"problem_id": "prob_001", "model_id": "model_x", "trace": "foo"},
                {"problem_id": "prob_002", "model_id": "model_x", "trace": "bar"},
            ]
            with temp_file.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            completed = load_completed_keys(temp_file)
            self.assertEqual(completed, {("prob_001", "model_x"), ("prob_002", "model_x")})

    def test_make_stratified_group_splits_zero_leakage(self) -> None:
        # Create dataset with multiple perturbations per problem
        data = []
        for prob_num in range(30):
            prob_id = f"prob_{prob_num:03d}"
            # Half robust, half not
            is_robust = (prob_num % 2 == 0)
            # 3 perturbed variants per problem
            for var_idx in range(3):
                data.append(
                    {
                        "problem_id": prob_id,
                        "model_id": "test_model",
                        "original_problem": f"Question {prob_num} variant {var_idx}",
                        "model_is_robust": is_robust,
                    }
                )
        df = pd.DataFrame(data)

        splits = make_stratified_group_splits(df, n_splits=5, seed=42)
        summary = verify_grouped_splits(df, splits)

        self.assertFalse(summary["has_leakage"])
        self.assertEqual(summary["num_folds"], 5)
        self.assertEqual(summary["total_samples"], 90)
        self.assertEqual(summary["unique_groups"], 30)

        # Conflicting group robustness label error
        conflicting_df = df.copy()
        conflicting_df.loc[0, "model_is_robust"] = not conflicting_df.loc[0, "model_is_robust"]
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            make_stratified_group_splits(conflicting_df, n_splits=5)

    def test_standardize_and_export_existing_generated_traces(self) -> None:
        deepseek_file = ROOT / "data" / "generated" / "deepseek_test.jsonl"
        if not deepseek_file.exists():
            self.skipTest("deepseek_test.jsonl not found")

        records = []
        with deepseek_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        standard_df = standardize_trace_records(records)
        self.assertEqual(len(standard_df), len(records))
        for col in CORE_SCHEMA_COLUMNS:
            self.assertIn(col, standard_df.columns)
            self.assertFalse(standard_df[col].isna().any())

        # Test atomic write to parquet and reload
        with tempfile.TemporaryDirectory() as temp_dir:
            out_parquet = Path(temp_dir) / "standard_traces.parquet"
            atomic_write_dataframe(standard_df, out_parquet)
            self.assertTrue(out_parquet.exists())

            reloaded = pd.read_parquet(out_parquet)
            self.assertEqual(len(reloaded), len(standard_df))
            self.assertEqual(list(reloaded.columns[:6]), list(CORE_SCHEMA_COLUMNS))


if __name__ == "__main__":
    unittest.main()

