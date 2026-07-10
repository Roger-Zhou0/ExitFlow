"""Factory utilities for constructing the segmented inference runtime."""

from __future__ import annotations

from typing import Dict, Iterable

from config.loader import Manifest

from .factories import RuntimeFactories
from .forwarder import Forwarder, ForwarderConfig
from .pipeline import PipelineConfig, SegmentedPipeline
from .router import RequestRouter, RouterConfig
from .queue_executor import SegmentedQueueExecutor
from .segment_worker import SegmentContext, SegmentWorker


def build_segmented_pipeline(
    manifest: Manifest,
    thresholds: dict[str, dict[str, float]],
    factories: RuntimeFactories,
) -> SegmentedPipeline:
    """Create a segmented pipeline using manifest definitions and factories."""
    model_cfg = manifest["model"]
    runtime_cfg = manifest["runtime"]
    segments = model_cfg["segments"]
    router_cfg = RouterConfig(**runtime_cfg["router"])
    pipeline_cfg = PipelineConfig(
        entry_segment_id=segments[0]["id"],
        max_segments=len(segments),
    )
    entry_workers = _build_entry_workers(segments[0], runtime_cfg, thresholds, factories, router_cfg, manifest)
    other_workers = _build_other_workers(segments[1:], runtime_cfg, thresholds, factories, manifest)
    router = RequestRouter(router_cfg, entry_workers)
    unique_workers = [entry_workers[0], *other_workers]
    return SegmentedPipeline(pipeline_cfg, router, unique_workers)


def build_segment_worker_pools(
    manifest: Manifest,
    thresholds: dict[str, dict[str, float]],
    factories: RuntimeFactories,
) -> dict[str, list[SegmentWorker]]:
    """Instantiate workers for each segment, possibly with multiple replicas per segment."""
    model_cfg = manifest["model"]
    runtime_cfg = manifest["runtime"]
    segments = model_cfg["segments"]
    pool_cfg = runtime_cfg.get("segment_pools", {})
    pools: dict[str, list[SegmentWorker]] = {}

    for segment_cfg in segments:
        seg_id = segment_cfg["id"]
        worker_key = seg_id.replace("-", "_")
        worker_settings = runtime_cfg["workers"].get(worker_key, {})
        devices: list[str] = []

        # Preferred: explicit device list per segment in segment_pools.
        if seg_id in pool_cfg:
            declared = pool_cfg[seg_id]
            if isinstance(declared, int):
                devices = [worker_settings.get("device", "cpu")] * int(declared)
            elif isinstance(declared, (list, tuple)):
                devices = list(declared)
        # Fallback: single worker (or router pool size for segment-1).
        if not devices:
            if seg_id == segments[0]["id"]:
                router_cfg = runtime_cfg.get("router", {})
                replicas = int(router_cfg.get("segment_1_pool_size", 1))
                devices = [worker_settings.get("device", "cpu")] * replicas
            else:
                devices = [worker_settings.get("device", "cpu")]

        workers: list[SegmentWorker] = []
        for replica_id, device in enumerate(devices):
            settings = dict(worker_settings)
            settings["device"] = device
            workers.append(
                _instantiate_worker(
                    segment_cfg,
                    settings,
                    thresholds,
                    factories,
                    replica_id=replica_id,
                    manifest=manifest,
                )
            )
        pools[seg_id] = workers
    return pools


def build_queue_executor(
    manifest: Manifest,
    thresholds: dict[str, dict[str, float]],
    factories: RuntimeFactories,
    queue_capacity: int | None = None,
) -> SegmentedQueueExecutor:
    """Create a queue-based executor with competitive pull scheduling."""
    pools = build_segment_worker_pools(manifest, thresholds, factories)
    entry_segment_id = manifest["model"]["segments"][0]["id"]
    runtime_cfg = manifest.get("runtime", {})
    capacity = queue_capacity or runtime_cfg.get("queue_capacity", 32)
    max_segments = len(manifest["model"]["segments"])
    return SegmentedQueueExecutor(
        pools,
        entry_segment=entry_segment_id,
        queue_capacity=int(capacity),
        max_hops=max_segments,
    )


def _build_entry_workers(
    segment_cfg: dict,
    runtime_cfg: dict,
    thresholds: dict[str, dict[str, float]],
    factories: RuntimeFactories,
    router_cfg: RouterConfig,
    manifest: Manifest | None,
) -> list[SegmentWorker]:
    worker_key = segment_cfg["id"].replace("-", "_")
    worker_settings = runtime_cfg["workers"][worker_key]
    workers: list[SegmentWorker] = []
    for replica in range(router_cfg.segment_1_pool_size):
        worker = _instantiate_worker(
            segment_cfg,
            worker_settings,
            thresholds,
            factories,
            replica,
            manifest=manifest,
        )
        workers.append(worker)
    return workers


def _build_other_workers(
    segment_cfgs: Iterable[dict],
    runtime_cfg: dict,
    thresholds: dict[str, dict[str, float]],
    factories: RuntimeFactories,
    manifest: Manifest | None,
) -> list[SegmentWorker]:
    workers: list[SegmentWorker] = []
    for segment_cfg in segment_cfgs:
        worker_key = segment_cfg["id"].replace("-", "_")
        worker_settings = runtime_cfg["workers"].get(worker_key, {})
        workers.append(
            _instantiate_worker(segment_cfg, worker_settings, thresholds, factories, replica_id=0, manifest=manifest)
        )
    return workers


def _instantiate_worker(
    segment_cfg: dict,
    worker_settings: dict,
    thresholds: dict[str, dict[str, float]],
    factories: RuntimeFactories,
    replica_id: int,
    manifest: Manifest | None,
) -> SegmentWorker:
    runtime_cfg = manifest.get("runtime", {}) if manifest is not None else {}
    native_backbone_dir = runtime_cfg.get("native_backbone_dir")
    native_dtype = runtime_cfg.get("native_dtype", "float16")
    use_native = native_backbone_dir is not None
    # Determine if this is the last segment to attach final norm.
    include_norm = False
    if manifest is not None:
        all_segments = manifest["model"]["segments"]
        max_end = max(seg["layers"]["end"] for seg in all_segments)
        include_norm = segment_cfg["layers"]["end"] == max_end

    context = SegmentContext(
        segment_id=segment_cfg["id"],
        device=worker_settings.get("device", "cpu"),
    )
    if use_native:
        segment = factories.native_segment_factory(
            {
                "segment_id": segment_cfg["id"],
                "layers": segment_cfg["layers"],
                "components": segment_cfg.get("components", []),
                "device": context.device,
                "replica_id": replica_id,
                "artifacts": segment_cfg.get("artifacts", {}),
                "backbone_dir": native_backbone_dir,
                "dtype": native_dtype,
                "include_embed": segment_cfg["layers"]["start"] == 1,
                "include_norm": include_norm,
            }
        )
    else:
        segment = factories.segment_factory(
            {
                "segment_id": segment_cfg["id"],
                "layers": segment_cfg["layers"],
                "components": segment_cfg.get("components", []),
                "device": context.device,
                "replica_id": replica_id,
                "artifacts": segment_cfg.get("artifacts", {}),
            }
        )
    forwarder_cfg = segment_cfg.get("forwarder")
    forwarder = None
    if forwarder_cfg is not None:
        threshold_values = thresholds.get(forwarder_cfg["id"], {})
        forwarder = _build_forwarder(forwarder_cfg, threshold_values, factories, context.device)
    return SegmentWorker(context, segment, forwarder)


def _build_forwarder(
    forwarder_cfg: dict,
    threshold_values: Dict[str, float],
    factories: RuntimeFactories,
    device: str,
) -> Forwarder:
    exit_head = factories.exit_head_factory(
        {
            "forwarder_id": forwarder_cfg["id"],
            "exit_head_weights": forwarder_cfg.get("exit_head_weights"),
            "forwarder_weights": forwarder_cfg.get("forwarder_weights"),
            "device": device,
        }
    )
    config = ForwarderConfig(
        logit_threshold=threshold_values.get("logit_threshold", 0.0),
        margin_gate=threshold_values.get("margin_gate", 0.0),
        entropy_gate=threshold_values.get("entropy_gate", float("inf")),
        temperature_scale=threshold_values.get("temperature_scale", 1.0),
        next_segment_id=forwarder_cfg.get("next_segment"),
    )
    return Forwarder(exit_head, config)
