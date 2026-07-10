"""Calibrate forwarder thresholds from exported exit-head artifacts.

This utility loads:
- base model from the manifest (HF repo id)
- exported TorchScript exit heads from an artifacts directory (metadata.json)
- a calibration dataset (JSONL or directory of JSONL files)

It then runs the calibration set through the backbone, collects per-head logits,
and computes thresholds (logit_threshold, margin_gate, entropy_gate,
temperature_scale) using the same logic as training.job.

Usage example:

  python calibrate_thresholds_from_artifacts.py \
    --manifest config/manifest.json \
    --metadata artifacts/trained-exit-heads_kd_round_final/metadata.json \
    --dataset data/prepared_dataset/calibration/dolly15k_llama31.jsonl \
    --output artifacts/trained-exit-heads_kd_round_final/tr_0.6/thresholds.json \
    --device cuda \
    --limit 2000

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.loader import load_manifest
from training.calibration import CalibrationConfig, CalibrationSample, ThresholdCalibrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate thresholds from exported exit-head artifacts")
    parser.add_argument("--manifest", default="config/manifest.json", help="Path to manifest.json")
    parser.add_argument("--metadata", required=True, help="Path to exporter metadata.json under artifacts")
    parser.add_argument("--dataset", required=True, help="Calibration dataset JSONL or directory of JSONL files")
    parser.add_argument("--output", required=True, help="Where to write thresholds.json")
    parser.add_argument("--device", default="cuda", help="Device for model inference (default: cuda)")
    parser.add_argument("--limit", type=int, help="Optional limit on number of samples")
    return parser.parse_args()


def load_exit_specs(manifest: dict) -> list[dict]:
    specs: list[dict] = []
    for segment in manifest["model"]["segments"]:
        forwarder = segment.get("forwarder")
        if forwarder is None:
            continue
        specs.append(
            {
                "id": forwarder["id"],
                "layer_index": segment["layers"]["end"],
                "segment_id": segment["id"],
            }
        )
    return specs


def load_exit_heads(metadata_path: Path, device: torch.device) -> tuple[dict[str, torch.nn.Module], dict]:
    metadata = json.loads(metadata_path.read_text("utf-8"))
    exit_map = metadata.get("exit_heads", {})
    modules: dict[str, torch.nn.Module] = {}
    for exit_id, path in exit_map.items():
        module = torch.jit.load(path, map_location=device)
        module.eval()
        modules[exit_id] = module
    return modules, metadata


def iter_dataset(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.is_dir():
        sources = sorted(path.glob("*.jsonl"))
        if not sources:
            raise FileNotFoundError(f"No JSONL files found under directory: {path}")
    else:
        sources = [path]

    for source in sources:
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def ensure_token_ids(sample: dict, tokenizer, device: torch.device) -> torch.Tensor:
    if "input_ids" in sample:
        ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device)
        return ids
    prompt = sample.get("prompt")
    if prompt is None:
        raise ValueError("Sample missing both input_ids and prompt")
    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded.input_ids[0].to(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    manifest = load_manifest(Path(args.manifest))
    base_model = manifest["model"]["base_model"]
    exit_specs = load_exit_specs(manifest)
    exit_heads, metadata = load_exit_heads(Path(args.metadata), device)

    missing_heads = {spec["id"] for spec in exit_specs} - exit_heads.keys()
    if missing_heads:
        raise RuntimeError(f"Exit heads missing in metadata for ids: {sorted(missing_heads)}")

    # Load base model/tokenizer
    precision = manifest["model"].get("precision")
    dtype = torch.bfloat16 if (device.type == "cuda" and precision == "bf16") else (
        torch.float16 if (device.type == "cuda" and precision in {"fp16", "float16"}) else torch.float32
    )
    backbone_dir = (metadata.get("backbone_dir") or "").strip()
    if backbone_dir:
        backbone_path = Path(backbone_dir)
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone directory from metadata not found: {backbone_path}")
        print(f"[calib] Loading backbone from metadata.backbone_dir: {backbone_path}")
        model = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(backbone_path)
    else:
        print(f"[calib] metadata.backbone_dir 为空，回退使用 HF 仓库: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    model.to(device)
    model.eval()

    # Collect logits per-exit across the calibration set
    collected: dict[str, list[list[float]]] = {spec["id"]: [] for spec in exit_specs}
    total = 0
    for idx, sample in enumerate(iter_dataset(Path(args.dataset))):
        if args.limit and idx >= args.limit:
            break
        input_ids = ensure_token_ids(sample, tokenizer, device)
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
        hidden_states = outputs.hidden_states
        for spec in exit_specs:
            hidden_state = hidden_states[spec["layer_index"]]
            head_module = exit_heads[spec["id"]]
            # Align dtype to the head module parameters if available
            sample_param = next(head_module.parameters(), None)
            target_dtype = sample_param.dtype if sample_param is not None else hidden_state.dtype
            hidden_state = hidden_state.to(device=device, dtype=target_dtype)
            logits = head_module(hidden_state)
            # If time dimension present, take last token
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            # Flatten to 1D per-sample
            logits = logits.detach().cpu().view(-1).tolist()
            collected[spec["id"]].append(logits)
        total += 1
        if total % 50 == 0:
            print(f"[calib] Processed {total} samples…", flush=True)

    if total == 0:
        raise RuntimeError("No samples processed from calibration dataset")

    # Temperature from training manifest
    kd_cfg = manifest.get("training", {}).get("knowledge_distillation", {})
    temperature = float(kd_cfg.get("temperature", 1.0))
    calibrator = ThresholdCalibrator(CalibrationConfig(temperature=temperature))

    thresholds: dict[str, dict[str, float]] = {}
    for spec in exit_specs:
        exit_id = spec["id"]
        logits_list = collected[exit_id]
        flat_samples = [CalibrationSample(logits=logits, label=0) for logits in logits_list]
        thresholds[exit_id] = calibrator.calibrate(flat_samples)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2)
    print(f"[calib] Wrote thresholds -> {output_path}")


if __name__ == "__main__":
    main()
