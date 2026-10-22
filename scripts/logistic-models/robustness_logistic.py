"""Command-line entry point for the robustness logistic-regression model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustness_data import load_feature_frame, summarize_frame
from robustness_experiments import (
    export_submission_model,
    run_ablation,
    run_all,
    run_baseline,
    run_generalization,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "generated"
DEFAULT_RESULTS = ROOT / "results" / "logistic_regression"
DEFAULT_ARTIFACT = (
    ROOT
    / "solutions"
    / "logistic-regression"
    / "model_artifacts"
    / "logistic_model.joblib"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run leakage-aware logistic regression on AIMO trace data."
    )
    parser.add_argument(
        "experiment",
        choices=("inspect", "baseline", "ablation", "generalization", "export", "all"),
        help="Experiment or inspection step to run.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA, help="Directory of trace JSONL files."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_RESULTS, help="Directory for generated results."
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT,
        help="Destination for the submission-ready fitted pipeline.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frame = load_feature_frame(args.data_dir)

    if args.experiment == "inspect":
        result = summarize_frame(frame)
    elif args.experiment == "baseline":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = run_baseline(frame, args.output_dir)
    elif args.experiment == "ablation":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        baseline = run_baseline(frame, args.output_dir)
        result = {
            "rows": len(run_ablation(frame, args.output_dir, baseline)),
            "output": str(args.output_dir / "ablation_results.csv"),
        }
    elif args.experiment == "generalization":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = run_generalization(frame, args.output_dir)
    elif args.experiment == "export":
        result = export_submission_model(frame, args.artifact_path)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = run_all(frame, args.output_dir)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
