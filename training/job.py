"""High-level training orchestration for exit-head KD."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist

from config.loader import load_manifest

from training.calibration import CalibrationConfig, CalibrationSample, ThresholdCalibrator
from training.checkpoint import CheckpointManager
from training.config import TrainingJobConfig
from training.data_pipeline import (
    SampleRecord,
    TrainingDatasetConfig,
    build_calibration_dataset,
    build_training_dataset,
    stream_training_records,
)
from training.exporter import ArtifactExporter
from training.kd_trainer import KDConfig, KnowledgeDistillationTrainer
from training.model_factory import ModelBundle, load_model_bundle
from training.utils import batch_samples


class StreamingBatchIterable:
    """Re-iterable view that materialises training batches on demand."""

    def __init__(
        self,
        record_source: Iterable[SampleRecord],
        batch_size: int,
        *,
        chunk_samples: int | None,
        chunk_bytes: int | None,
        shuffle: bool,
        rank: int,
        world_size: int,
    ) -> None:
        self._record_source = record_source
        self._batch_size = batch_size
        self._chunk_samples = max(chunk_samples or 0, 0) or None
        self._chunk_bytes = max(chunk_bytes or 0, 0) or None
        self._shuffle = shuffle
        self._rank = rank
        self._world_size = world_size

    def __iter__(self):
        chunk: list[SampleRecord] = []
        chunk_bytes = 0
        for idx, record in enumerate(self._record_source):
            if self._world_size > 1 and idx % self._world_size != self._rank:
                continue
            chunk.append(record)
            chunk_bytes += record.size_bytes
            if self._should_emit(chunk, chunk_bytes):
                yield from self._emit_batches(chunk)
                chunk = []
                chunk_bytes = 0
        if chunk:
            yield from self._emit_batches(chunk)

    def _should_emit(self, chunk: list[SampleRecord], chunk_bytes: int) -> bool:
        if not chunk:
            return False
        if self._chunk_samples is not None and len(chunk) >= self._chunk_samples:
            return True
        if self._chunk_bytes is not None and chunk_bytes >= self._chunk_bytes:
            return True
        return False

    def _emit_batches(self, records: list[SampleRecord]):
        if not records:
            return
        samples = [record.to_dict() for record in records]
        if self._shuffle:
            random.shuffle(samples)
        for batch in batch_samples(samples, self._batch_size):
            yield batch


def _exit_points_from_manifest(manifest: dict) -> list[tuple[str, int]]:
    layers: list[tuple[str, int]] = []
    for segment in manifest["model"]["segments"]:
        forwarder = segment.get("forwarder")
        if forwarder is None:
            continue
        layer_stop = segment["layers"]["end"]
        layers.append((forwarder["id"], layer_stop))
    return layers


def _collect_calibration_samples(
    bundle: ModelBundle,
    samples: list[dict[str, list[int]]],
) -> dict[str, dict[str, list[float]]]:
    device = bundle.input_device if hasattr(bundle, "input_device") else next(bundle.backbone.parameters()).device
    bundle.backbone.eval()
    collected: dict[str, dict[str, list[float]]] = {spec.identifier: {"logits": []} for spec in bundle.exit_specs}
    with torch.no_grad():
        for sample in samples:
            input_ids = torch.tensor(sample["input_ids"], device=device, dtype=torch.long)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            outputs = bundle.backbone.forward(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            for spec in bundle.exit_specs:
                hidden_state = hidden_states[spec.layer_index]
                logits = spec.head(hidden_state)
                if logits.dim() == 3:
                    logits = logits[:, -1, :]
                logits = logits.detach().cpu().view(-1).tolist()
                collected[spec.identifier]["logits"].append(logits)
    return collected

def run_training_job(config: TrainingJobConfig) -> None:
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    if rank == 0:
        print("[job] Loading manifest…", flush=True)
    manifest = load_manifest(config.manifest_path)
    training_manifest = manifest["training"]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(f"[job] Output dir: {config.output_dir}")
        print(f"[job] Checkpoints dir: {config.checkpoint_dir}")

    exit_points = _exit_points_from_manifest(manifest)
    if rank == 0:
        print(f"[job] Loading model bundle: {manifest['model']['base_model']} on {config.device}", flush=True)
    precision = manifest["model"].get("precision")
    dtype = None
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision in {"fp16", "float16"}:
        dtype = torch.float16

    bundle = load_model_bundle(
        manifest["model"]["base_model"],
        device=config.device,
        exit_points=exit_points,
        device_map=config.device_map,
        shard_exits=config.shard_exits,
        torch_dtype=dtype,
    )
    has_exit_heads = len(bundle.exit_specs) > 0
    if not has_exit_heads:
        print(f"[job][rank {rank}] No exit heads instantiated; continuing without updates.", flush=True)
    if config.device_map and rank == 0:
        hf_map = getattr(bundle.backbone, "hf_device_map", None)
        if hf_map:
            print(f"[job] Backbone device map: {hf_map}", flush=True)
    training_device = str(bundle.input_device)
    if config.shard_exits:
        exit_device_map: dict[str, str] = {}
        for spec in bundle.exit_specs:
            param_device = None
            try:
                param = next(spec.head.parameters())
                param_device = param.device
            except StopIteration:
                param_device = bundle.input_device
            exit_device_map[spec.identifier] = str(param_device)
        print(f"[job][rank {rank}] Exit head device map: {exit_device_map}", flush=True)
    if rank == 0:
        print(f"[job] Trainer primary device: {training_device}", flush=True)
    kd_config = KDConfig(
        temperature=training_manifest["knowledge_distillation"]["temperature"],
        exit_loss_weights=training_manifest["knowledge_distillation"]["loss_weights"],
        ce_loss_weight=training_manifest["knowledge_distillation"].get("ce_loss_weight", 1.0),
        learning_rate=training_manifest.get("learning_rate", 1e-4),
        gradient_accumulation_steps=training_manifest.get("gradient_accumulation_steps", 1),
        max_steps=config.max_steps,
        epochs=config.epochs or training_manifest.get("epochs", 1),
        frozen_backbone=training_manifest.get("frozen_backbone", True),
        optimizer=training_manifest.get("optimizer", "adamw"),
        device=training_device,
        deepspeed=config.deepspeed,
        deepspeed_config_path=str(config.deepspeed_config) if config.deepspeed_config else None,
    )
    if rank == 0:
        print("[job] Model loaded; preparing dataset…", flush=True)

    dataset_cfg = TrainingDatasetConfig(
        source_path=config.dataset_path,
        token_budget=training_manifest["token_budget"],
        shuffle=config.shuffle,
    )
    chunk_samples = config.dataset_chunk_samples if config.dataset_chunk_samples and config.dataset_chunk_samples > 0 else None
    chunk_bytes = config.dataset_chunk_bytes if config.dataset_chunk_bytes and config.dataset_chunk_bytes > 0 else None
    streaming = chunk_samples is not None or chunk_bytes is not None

    if streaming:
        record_source = stream_training_records(dataset_cfg)
        samples_iterable = StreamingBatchIterable(
            record_source,
            config.batch_size,
            chunk_samples=chunk_samples,
            chunk_bytes=chunk_bytes,
            shuffle=config.shuffle,
            rank=rank,
            world_size=world_size,
        )
        if rank == 0:
            parts: list[str] = []
            if chunk_bytes is not None:
                parts.append(f"{chunk_bytes / (1024**3):.2f} GiB")
            if chunk_samples is not None:
                parts.append(f"{chunk_samples} samples")
            descriptor = ", ".join(parts) if parts else "automatic"
            print(f"[job] Streaming dataset in chunks ({descriptor}); batches prepared lazily.", flush=True)
        has_batches = True  # trainer will early-exit if no batches materialise
        effective_shuffle = False
    else:
        samples = build_training_dataset(dataset_cfg)
        if distributed:
            samples = samples[rank::world_size]
            print(f"[job][rank {rank}] Training samples loaded: {len(samples)}", flush=True)
        else:
            print(f"[job] Training samples loaded: {len(samples)} records (pre-batch)", flush=True)
        samples = batch_samples(samples, config.batch_size)
        if distributed:
            print(f"[job][rank {rank}] Batches prepared (batch_size={config.batch_size}): {len(samples)}", flush=True)
        else:
            print(f"[job] Batches prepared (batch_size={config.batch_size}): {len(samples)}", flush=True)
        samples_iterable = samples
        has_batches = len(samples) > 0
        effective_shuffle = config.shuffle
    if rank == 0:
        print("[job] Starting training loop…", flush=True)
    if has_exit_heads and has_batches:
        trainer = KnowledgeDistillationTrainer(
            bundle.backbone,
            bundle.exit_specs,
            kd_config,
        )
        checkpoint_manager = None if config.deepspeed else CheckpointManager(config.checkpoint_dir)
        trainer.train(
            samples_iterable,
            checkpoint_manager=checkpoint_manager,
            checkpoint_dir=config.checkpoint_dir,
            checkpoint_interval=config.save_interval,
            resume=config.resume,
            shuffle=effective_shuffle,
            epochs_override=config.epochs,
            max_steps_override=config.max_steps,
        )
    else:
        print(f"[job][rank {rank}] Skipping training (heads={has_exit_heads}, batches={has_batches}).", flush=True)
    if distributed:
        dist.barrier()
    if rank == 0:
        print("[job] Training finished; exporting artifacts…", flush=True)

    if rank == 0:
        exporter = ArtifactExporter(config.output_dir)
        exporter.export_backbone(bundle.backbone)
        exported_exit_heads = exporter.export_exit_heads(bundle.exit_specs)
        exporter.write_metadata(exported_exit_heads)
        print("[job] Export complete.", flush=True)

    if config.calibration_path and config.thresholds_output:
        if rank == 0:
            print("[job] Running calibration to estimate thresholds…", flush=True)
        calibration_samples = build_calibration_dataset(
            config.calibration_path,
            token_budget=training_manifest["token_budget"],
            limit=training_manifest.get("calibration_sample_limit", 512),
        )
        collected = _collect_calibration_samples(bundle, calibration_samples)
        thresholds: dict[str, dict[str, float]] = {}
        for spec in bundle.exit_specs:
            logits_list = collected[spec.identifier]["logits"]
            flat_samples = [CalibrationSample(logits=logits, label=0) for logits in logits_list]
            calibrator = ThresholdCalibrator(CalibrationConfig(temperature=kd_config.temperature))
            thresholds[spec.identifier] = calibrator.calibrate(flat_samples)
        if rank == 0:
            config.thresholds_output.parent.mkdir(parents=True, exist_ok=True)
            with config.thresholds_output.open("w", encoding="utf-8") as handle:
                json.dump(thresholds, handle, indent=2)
            print(f"[job] Calibration complete. Thresholds saved to {config.thresholds_output}", flush=True)
