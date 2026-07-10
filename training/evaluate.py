"""Evaluate a trained exit-head bundle on a JSONL benchmark."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from config.loader import load_manifest
from evaluation.benchmarks import Benchmark, BenchmarkSuite
from evaluation.metrics import MetricTracker
from evaluation.runner import EvaluationConfig, EvaluationRunner
from evaluation.scorers import build_default_scorers
from runtime.builder import build_segmented_pipeline
from runtime.factories import RuntimeFactories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exit-head artifacts on a benchmark dataset")
    parser.add_argument("--manifest", default="config/manifest.json", help="Path to manifest.json")
    parser.add_argument("--metadata", required=True, help="Metadata JSON produced by training exporter")
    parser.add_argument("--thresholds", help="Threshold JSON for exit heads")
    parser.add_argument("--dataset", required=True, help="Benchmark dataset JSONL")
    parser.add_argument("--benchmark-name", default="benchmark", help="Friendly name for the dataset")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens per run (unused placeholder)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Log progress every N samples when verbose is enabled",
    )
    return parser.parse_args()


def apply_metadata(manifest: dict, metadata_path: Path) -> dict:
    manifest_copy = deepcopy(manifest)
    metadata = json.loads(metadata_path.read_text("utf-8"))
    exit_map = metadata.get("exit_heads", {})
    for segment in manifest_copy.get("model", {}).get("segments", []):
        forwarder = segment.get("forwarder")
        if forwarder is not None:
            exit_id = forwarder.get("id")
            if exit_id and exit_id in exit_map:
                forwarder["exit_head_weights"] = exit_map[exit_id]
        segment.setdefault("artifacts", {})
    return manifest_copy


def load_thresholds(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def main() -> None:
    args = parse_args()
    manifest_data = load_manifest(Path(args.manifest))
    manifest_with_artifacts = apply_metadata(manifest_data, Path(args.metadata))
    thresholds = load_thresholds(Path(args.thresholds)) if args.thresholds else {}

    factories = RuntimeFactories()
    pipeline = build_segmented_pipeline(manifest_with_artifacts, thresholds, factories)
    segment_order = [segment["id"] for segment in manifest_with_artifacts["model"]["segments"]]

    if args.verbose:
        print("=== Configuration ===")
        print(f"manifest   : {args.manifest}")
        print(f"metadata   : {args.metadata}")
        print(f"thresholds : {args.thresholds if args.thresholds else '(none)'}")
        print(f"dataset    : {args.dataset}")
        print(f"segment_order : {segment_order}")
        if thresholds:
            print(f"threshold keys: {sorted(thresholds.keys())}")

    dataset_path = Path(args.dataset)
    if dataset_path.is_dir():
        benchmark_paths = list(dataset_path.glob("*.jsonl"))
        if not benchmark_paths:
            raise FileNotFoundError(f"No JSONL files found under benchmark directory: {dataset_path}")
        if args.verbose:
            print("benchmarks discovered:")
            for path in benchmark_paths:
                print(f" - {path.stem}: {path}")
        benchmarks = BenchmarkSuite(Benchmark(path.stem, path) for path in benchmark_paths)
    else:
        benchmarks = BenchmarkSuite([Benchmark(args.benchmark_name, dataset_path)])
    tracker = MetricTracker()
    scorers = build_default_scorers()
    runner = EvaluationRunner(
        EvaluationConfig(max_tokens=args.max_tokens, progress_interval=args.progress_interval, verbose=args.verbose),
        benchmarks,
        tracker,
        pipeline,
        segment_order,
        scorers,
    )
    runner.run_all()

    print("\n=== Evaluation Summary ===")
    for metric, values in tracker.metrics.items():
        avg = sum(values) / len(values) if values else float("nan")
        print(f"{metric}: {avg:.4f} (n={len(values)})")


if __name__ == "__main__":
    main()
