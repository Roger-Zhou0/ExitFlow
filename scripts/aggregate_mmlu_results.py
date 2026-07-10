#!/usr/bin/env python3
"""Aggregate MMLU benchmark results from early-exit logs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


def load_gold_answers(path: Path) -> list[str]:
    answers = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            response = payload.get("response") or payload.get("answer")
            if response is None:
                raise ValueError(f"Missing response field in {path}")
            answers.append(response.strip())
    return answers


def iter_log_files(log_dir: Path) -> Iterable[Path]:
    for path in sorted(log_dir.glob("question_*.json")):
        yield path


def normalize_prediction(token: str) -> str:
    token = token.strip()
    if not token:
        return token
    for char in token:
        if char.isalpha():
            return char.upper()
    return token.upper()


def parse_choice_logits(choice_logits: Sequence[float] | dict | None) -> dict:
    labels = ("A", "B", "C", "D")
    if not choice_logits:
        return {label: None for label in labels}
    if isinstance(choice_logits, Sequence) and not isinstance(choice_logits, (str, bytes, bytearray, dict)):
        mapped = {}
        for idx, label in enumerate(labels):
            mapped[label] = choice_logits[idx] if idx < len(choice_logits) else None
        return mapped
    return {label: choice_logits.get(label) for label in labels}


def select_highest(choice_logits: dict) -> str | None:
    best_label = None
    best_value = float("-inf")
    for label, value in choice_logits.items():
        if value is None:
            continue
        if value > best_value:
            best_label = label
            best_value = value
    return best_label


def parse_result(data: dict) -> dict:
    per_step = data.get("per_step", [])
    first_step = per_step[0] if per_step else {}
    token_text = first_step.get("token", "").strip()
    first_token_logit = first_step.get("scores", {}).get("logit")
    choice_logits_source = data.get("choice_logits")
    if choice_logits_source is None:
        choice_logits_source = first_step.get("choice_logits")
    choice_logits = parse_choice_logits(choice_logits_source)
    predicted_choice = data.get("predicted_choice") or select_highest(choice_logits)
    result = {
        "prompt": data.get("prompt") or "",
        "generated_text": data.get("generated_text") or "",
        "predicted_letter": normalize_prediction(token_text),
        "choice_pred": predicted_choice,
        "logit": first_token_logit,
        "base_exit_id": first_step.get("exit_forwarder_id"),
        "base_layer": first_step.get("exit_layer_index"),
        "total_latency": data.get("total_latency_ms"),
    }
    result.update({f"logit_{label}": choice_logits[label] for label in ("A", "B", "C", "D")})
    result["choice_logits"] = choice_logits
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise MMLU benchmark outputs")
    parser.add_argument("--logs", required=True, help="Path to logs/benchmark/mmlu directory")
    parser.add_argument("--dataset", required=True, help="Path to data/raw_benchmarks/mmlu_en.jsonl")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    log_dir = Path(args.logs)
    dataset_path = Path(args.dataset)
    out_path = Path(args.output)

    gold_answers = load_gold_answers(dataset_path)

    rows: list[dict] = []
    total = 0
    correct = 0

    # Aggregates for additional summary statistics
    sample_exit_layers: Counter[int] = Counter()
    sample_exit_ids: Counter[str] = Counter()
    token_exit_layers: Counter[int] = Counter()
    token_exit_ids: Counter[str] = Counter()
    latencies: list[float] = []
    tokens_per_sample: list[int] = []

    for idx, log_file in enumerate(iter_log_files(log_dir)):
        if idx >= len(gold_answers):
            break
        gold = gold_answers[idx]
        data = json.loads(log_file.read_text("utf-8"))
        result = parse_result(data)
        pred = result.get("choice_pred") or result["predicted_letter"]
        is_correct = pred == gold.upper()
        choice_logits = result.get("choice_logits", {})
        rows.append(
            {
                "index": idx + 1,
                "prompt": result["prompt"],
                "gold": gold,
                "pred": pred,
                "choice_pred": result.get("choice_pred"),
                "logit": result["logit"],
                "logit_A": choice_logits.get("A"),
                "logit_B": choice_logits.get("B"),
                "logit_C": choice_logits.get("C"),
                "logit_D": choice_logits.get("D"),
                "exit": result["base_exit_id"],
                "layer": result["base_layer"],
                "total_latency": result["total_latency"],
                "correct": is_correct,
            }
        )
        total += 1
        if is_correct:
            correct += 1

        # Collect per-sample latency (if present)
        if result["total_latency"] is not None:
            try:
                latencies.append(float(result["total_latency"]))
            except (TypeError, ValueError):
                pass

        # Collect per-sample final-exit statistics
        final_layer = data.get("final_exit_layer_index")
        final_exit_id = data.get("final_exit_forwarder_id")
        # Fallback to first-step exit if final_* not present (older logs)
        if final_layer is None:
            final_layer = result.get("base_layer")
        if final_exit_id is None:
            final_exit_id = result.get("base_exit_id")
        if final_layer is not None:
            try:
                sample_exit_layers[int(final_layer)] += 1
            except (TypeError, ValueError):
                pass
        if final_exit_id:
            sample_exit_ids[str(final_exit_id)] += 1

        # Collect per-token exit statistics from per_step
        per_step = data.get("per_step", [])
        tokens_per_sample.append(len(per_step))
        for step in per_step:
            layer = step.get("exit_layer_index")
            exit_id = step.get("exit_forwarder_id")
            if layer is not None:
                try:
                    token_exit_layers[int(layer)] += 1
                except (TypeError, ValueError):
                    pass
            if exit_id:
                token_exit_ids[str(exit_id)] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    if total:
        accuracy = correct / total
        print(f"Processed {total} samples, accuracy {accuracy:.4f} ({correct}/{total})")

        # Latency statistics (per-sample total_latency_ms)
        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            mean_lat = sum(sorted_lat) / n
            median_lat = sorted_lat[n // 2]
            p95_lat = sorted_lat[int(0.95 * (n - 1))]
            print("\nLatency (total_latency_ms per sample):")
            print(f"  mean   = {mean_lat:.2f} ms")
            print(f"  median = {median_lat:.2f} ms")
            print(f"  p95    = {p95_lat:.2f} ms")

        # Final-exit distribution (per-sample)
        if sample_exit_layers:
            print("\nFinal-exit distribution (per-sample, by layer):")
            for layer in sorted(sample_exit_layers):
                count = sample_exit_layers[layer]
                print(f"  layer {layer:2d}: {count:4d} ({count / total:.2%})")

        # Exit distribution (per-token across generated steps)
        total_tokens = sum(token_exit_layers.values())
        if total_tokens:
            print("\nExit distribution (per-token across generated steps, by layer):")
            for layer in sorted(token_exit_layers):
                count = token_exit_layers[layer]
                print(f"  layer {layer:2d}: {count:6d} ({count / total_tokens:.2%})")

        # Tokens per sample (from per_step length)
        if tokens_per_sample:
            total_tok = sum(tokens_per_sample)
            mean_tok = total_tok / total
            print("\nTokens per sample (from per_step):")
            print(f"  mean tokens/sample = {mean_tok:.2f}")

    else:
        print("No log files processed.")


if __name__ == "__main__":
    main()
