"""Configuration structures for training jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.loader import load_manifest


@dataclass
class TrainingJobConfig:
    """Top-level configuration for an exit-head KD job."""

    manifest_path: Path
    dataset_path: Path
    output_dir: Path
    checkpoint_dir: Path
    calibration_path: Optional[Path]
    thresholds_output: Optional[Path]
    resume: bool
    save_interval: int
    max_steps: Optional[int]
    epochs: Optional[int]
    device: str
    device_map: Optional[str]
    shard_exits: bool
    batch_size: int
    shuffle: bool
    dataset_chunk_bytes: Optional[int]
    dataset_chunk_samples: Optional[int]
    # Optional DeepSpeed toggles
    deepspeed: bool = False
    deepspeed_config: Optional[Path] = None

    @property
    def manifest(self) -> dict:
        return load_manifest(self.manifest_path)

    @property
    def training_manifest(self) -> dict:
        return self.manifest["training"]

    @classmethod
    def from_args(cls, args: "argparse.Namespace") -> "TrainingJobConfig":  # pragma: no cover - CLI glue
        calibration = Path(args.calibration_path) if args.calibration_path else None
        thresholds = Path(args.thresholds_output) if args.thresholds_output else None
        return cls(
            manifest_path=Path(args.manifest),
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output_dir),
            checkpoint_dir=Path(args.checkpoints),
            calibration_path=calibration,
            thresholds_output=thresholds,
            resume=args.resume,
            save_interval=args.save_interval,
            max_steps=args.max_steps,
            epochs=args.epochs,
            device=args.device,
            device_map=args.device_map,
            shard_exits=args.shard_exits,
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle,
            dataset_chunk_bytes=args.dataset_chunk_bytes or None,
            dataset_chunk_samples=args.dataset_chunk_samples or None,
            deepspeed=getattr(args, "deepspeed", False),
            deepspeed_config=Path(args.deepspeed_config) if getattr(args, "deepspeed_config", None) else None,
        )
