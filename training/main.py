"""CLI entrypoint for running KD training on HPC."""

from __future__ import annotations

import argparse

from training.config import TrainingJobConfig
from training.job import run_training_job


def build_arg_parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Train exit heads with knowledge distillation")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--dataset", required=True, help="Tokenized training dataset (JSONL)")
    parser.add_argument("--output-dir", required=True, help="Directory for exported artifacts")
    parser.add_argument("--checkpoints", required=True, help="Directory for checkpoints")
    parser.add_argument("--calibration-path", help="Optional calibration dataset (JSONL)")
    parser.add_argument("--thresholds-output", help="Where to save calibrated thresholds JSON")
    parser.add_argument("--device", default="cuda", help="Torch device string (default: cuda)")
    parser.add_argument("--device-map", help="Optional HuggingFace device map (e.g. auto, balanced) for multi-GPU")
    parser.add_argument("--shard-exits", action="store_true", help="Place exit heads on the same devices as their source layers")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples per optimizer step")
    parser.add_argument("--save-interval", type=int, default=100, help="Checkpoint interval in optimizer steps")
    parser.add_argument("--max-steps", type=int, help="Optional maximum optimizer steps")
    parser.add_argument("--epochs", type=int, help="Optional epoch override")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    # DeepSpeed integration (optional)
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed ZeRO for training")
    parser.add_argument("--deepspeed-config", help="Path to DeepSpeed JSON config file")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling between epochs")
    parser.add_argument(
        "--dataset-chunk-bytes",
        type=int,
        default=0,
        help="Approximate byte cap per streaming chunk (0 disables chunking)",
    )
    parser.add_argument(
        "--dataset-chunk-samples",
        type=int,
        default=0,
        help="Optional maximum number of samples per streaming chunk (0 disables)",
    )
    return parser


def main() -> None:  # pragma: no cover - CLI glue
    parser = build_arg_parser()
    args = parser.parse_args()
    config = TrainingJobConfig.from_args(args)
    run_training_job(config)


if __name__ == "__main__":  # pragma: no cover - CLI glue
    main()
