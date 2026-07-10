"""Evaluate exit-head artifacts on a dataset and report exit depth distribution."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.loader import load_manifest
from runtime.forwarder import Forwarder, ForwarderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exit-head artifacts on a dataset")
    parser.add_argument("--manifest", default="config/manifest.json", help="Path to manifest.json")
    parser.add_argument("--metadata", required=True, help="Path to exporter metadata.json")
    parser.add_argument("--thresholds", help="Threshold JSON produced during calibration")
    parser.add_argument("--dataset", required=True, help="JSONL dataset (tokenised or raw prompts)")
    parser.add_argument("--device", default="cuda", help="Device for model inference (default: cuda)")
    parser.add_argument("--limit", type=int, help="Optional limit on number of samples")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="Log progress every N samples when verbose is enabled",
    )
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
                "next_segment": forwarder.get("next_segment"),
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


def load_thresholds(threshold_path: Path | None) -> dict[str, dict[str, float]]:
    if threshold_path is None:
        return {}
    return json.loads(threshold_path.read_text("utf-8"))


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
    thresholds = load_thresholds(Path(args.thresholds) if args.thresholds else None)
    final_segment_id = manifest["model"]["segments"][-1]["id"]
    backbone_dir = (metadata.get("backbone_dir") or "").strip()

    if args.verbose:
        print("=== Configuration ===")
        print(f"manifest   : {args.manifest}")
        print(f"metadata   : {args.metadata}")
        print(f"thresholds : {args.thresholds if args.thresholds else '(none)'}")
        print(f"dataset    : {args.dataset}")
        print(f"device     : {device}")
        if args.limit:
            print(f"limit      : {args.limit}")
        print("=== Manifest Summary ===")
        print(f"base_model : {base_model}")
        print(
            "exit specs : "
            + ", ".join(
                f"{s['id']}@{s['layer_index']}->{s.get('next_segment','?')}" for s in exit_specs
            )
        )

    missing_heads = {spec["id"] for spec in exit_specs} - exit_heads.keys()
    if missing_heads:
        raise RuntimeError(f"Exit heads missing in metadata for ids: {sorted(missing_heads)}")
    if args.verbose:
        print(f"loaded exit heads: {sorted(exit_heads.keys())}")
        if thresholds:
            print("=== Thresholds ===")
            for spec in exit_specs:
                cfg = thresholds.get(spec["id"], {})
                print(
                    f"{spec['id']}: logit>={cfg.get('logit_threshold','-')}, "
                    f"margin>={cfg.get('margin_gate','-')}, entropy<={cfg.get('entropy_gate','-')}, "
                    f"temp={cfg.get('temperature_scale','-')}"
                )
    precision = manifest["model"].get("precision")
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and precision == "bf16")
        else (torch.float16 if (device.type == "cuda" and precision in {"fp16", "float16"}) else torch.float32)
    )
    if backbone_dir:
        backbone_path = Path(backbone_dir)
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone directory from metadata not found: {backbone_path}")
        if args.verbose:
            print(f"[eval] Loading backbone from metadata.backbone_dir: {backbone_path}")
        model = AutoModelForCausalLM.from_pretrained(
            backbone_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(backbone_path)
    else:
        if args.verbose:
            print(f"[eval] metadata.backbone_dir 为空，回退使用 HF 仓库: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    model.to(device)
    model.eval()

    forwarders: dict[str, Forwarder] = {}
    for spec in exit_specs:
        head_module = exit_heads[spec["id"]]
        head_module.to(device)
        config = thresholds.get(spec["id"], {})
        fwd_cfg = ForwarderConfig(
            logit_threshold=config.get("logit_threshold", 0.0),
            margin_gate=config.get("margin_gate", 0.0),
            entropy_gate=config.get("entropy_gate", float("inf")),
            temperature_scale=config.get("temperature_scale", 1.0),
            next_segment_id=spec.get("next_segment"),
        )
        forwarders[spec["id"]] = Forwarder(head_module, fwd_cfg)

    exit_counts: Counter[str] = Counter()
    total_tokens = 0
    total_samples = 0

    last_exit = None
    for idx, sample in enumerate(iter_dataset(Path(args.dataset))):
        if args.limit and idx >= args.limit:
            break
        input_ids = ensure_token_ids(sample, tokenizer, device)
        total_tokens += input_ids.shape[-1]
        with torch.no_grad():
            outputs = model(input_ids.unsqueeze(0), output_hidden_states=True)
        hidden_states = outputs.hidden_states

        exit_segment = final_segment_id
        for spec in exit_specs:
            hidden_state = hidden_states[spec["layer_index"]]
            head_module = exit_heads[spec["id"]]
            sample_param = next(head_module.parameters(), None)
            target_dtype = sample_param.dtype if sample_param is not None else hidden_state.dtype
            hidden_state = hidden_state.to(device=device, dtype=target_dtype)
            decision = forwarders[spec["id"]].evaluate(hidden_state)
            if decision.exit_now:
                exit_segment = spec["segment_id"]
                break
            if spec.get("next_segment"):
                exit_segment = spec["next_segment"]

        exit_counts[exit_segment] += 1
        total_samples += 1
        last_exit = exit_segment
        if args.verbose and (total_samples % max(1, args.progress_interval) == 0):
            print(
                f"[progress] processed={total_samples} tokens≈{total_tokens} last_exit={last_exit}"
            )

    print("\n=== Exit Head Evaluation Summary ===")
    print(f"Samples evaluated: {total_samples}")
    if total_samples:
        print(f"Average tokens per sample: {total_tokens / total_samples:.2f}")
    for exit_id, count in exit_counts.most_common():
        ratio = count / total_samples if total_samples else 0.0
        print(f"{exit_id}: {count} ({ratio:.2%})")


if __name__ == "__main__":
    main()
