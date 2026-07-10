#!/usr/bin/env python3
"""Download benchmark datasets and emit prompt/response JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import gzip
from typing import Callable, Iterable, Iterator, Optional

from datasets import Dataset, DatasetDict, load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi, hf_hub_download


DatasetConverter = Callable[[dict], Optional[dict[str, str]]]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch benchmark datasets and export prompt/response JSONL")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where processed JSONL files will be written",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["ifeval", "alpacaeval2", "gsm8k", "mmlu_en"],
        help="Subset of datasets to download (default: all)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Optional cap on number of records per dataset (useful for smoke tests)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Split to download when applicable (default: train)",
    )
    return parser


def _first_key(record: dict, candidates: Iterable[str]) -> Optional[str]:
    for key in candidates:
        if key in record and record[key]:
            return key
    return None


def convert_generic(record: dict, prompt_keys: Iterable[str], response_keys: Iterable[str]) -> Optional[dict[str, str]]:
    prompt_key = _first_key(record, prompt_keys)
    response_key = _first_key(record, response_keys)
    if prompt_key is None and response_key is None:
        return None
    prompt = record.get(prompt_key) if prompt_key else ""
    response = record.get(response_key) if response_key else ""
    prompt = prompt.strip() if isinstance(prompt, str) else str(prompt)
    response = response.strip() if isinstance(response, str) else str(response)
    if not prompt and not response:
        return None
    return {"prompt": prompt, "response": response}


def convert_gsm8k(record: dict) -> Optional[dict[str, str]]:
    question = record.get("question", "").strip()
    answer = record.get("answer", "").strip()
    if not question and not answer:
        return None
    return {"prompt": question, "response": answer}


def convert_mmlu(record: dict) -> Optional[dict[str, str]]:
    question = record.get("question")
    options = record.get("choices") or record.get("options")
    answer = record.get("answer")
    if answer is None:
        answer = record.get("target")
    if question is None or options is None or answer is None:
        return None
    if isinstance(options, dict):
        options = list(options.values())
    option_lines = [f"{chr(65 + idx)}. {opt}" for idx, opt in enumerate(options)]
    prompt = question.strip() + "\n" + "\n".join(option_lines)
    answer_idx: Optional[int]
    if isinstance(answer, int):
        answer_idx = answer
    elif isinstance(answer, str):
        candidate = answer.strip()
        if candidate.isdigit():
            answer_idx = int(candidate)
        elif len(candidate) == 1 and "A" <= candidate.upper() <= "Z":
            answer_idx = ord(candidate.upper()) - 65
        else:
            answer_idx = None
    else:
        answer_idx = None
    if answer_idx is not None and 0 <= answer_idx < len(options):
        response = chr(65 + answer_idx)
    else:
        response = str(answer).strip()
    return {"prompt": prompt, "response": response}


def _infer_jsonl_records(path: Path) -> list[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    records: list[dict[str, object]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_ifeval(split: str) -> Dataset:
    """Fetch IFEval JSON from the Hub without relying on dataset scripts."""
    normalized = (split or "train").strip().lower()
    if normalized not in {"train", "validation", "test"}:
        raise ValueError(
            "IFEval only exposes train/test/validation splits; "
            f"received '{split}'."
        )
    api = HfApi()
    try:
        files = api.list_repo_files("google/IFEval", repo_type="dataset")
    except Exception as exc:  # pragma: no cover - network/credential issues
        raise RuntimeError("Unable to list files for google/IFEval; check Hugging Face auth") from exc
    candidates: list[str] = []
    for file_name in files:
        lower = file_name.lower()
        if not lower.endswith((".jsonl", ".json", ".jsonl.gz")):
            continue
        if normalized == "train" and any(key in lower for key in ("train", "ifeval")):
            candidates.append(file_name)
        elif normalized == "validation" and any(key in lower for key in ("validation", "valid", "dev")):
            candidates.append(file_name)
        elif normalized == "test" and "test" in lower:
            candidates.append(file_name)
    if not candidates:
        raise RuntimeError(
            f"Unable to locate an IFEval file on the Hub for split '{split}'."
        )
    # Prefer shorter paths (likely top-level files) and deterministic ordering
    candidates.sort(key=lambda path: (len(path.split("/")), path))
    target = candidates[0]
    local_path = Path(
        hf_hub_download(
            repo_id="google/IFEval",
            filename=target,
            repo_type="dataset",
        )
    )
    if local_path.suffix == ".json":
        with local_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and any(
            key in payload for key in ("train", "validation", "test")
        ):
            # Some dumps store splits in a single JSON dict.
            data = payload.get(normalized) or payload.get(normalized[:1]) or []
        else:
            data = payload
    else:
        data = _infer_jsonl_records(local_path)
    if not isinstance(data, list):
        raise ValueError(f"Unexpected IFEval payload shape from '{target}'")
    return Dataset.from_list(data)


def load_alpaca_eval(split: str) -> Dataset:
    """Fetch AlpacaEval prompts/responses via hub file download."""
    if split not in {None, "", "eval", "train"}:
        raise ValueError(f"Unsupported split '{split}' for alpacaeval2; expected 'eval'.")
    filename = "alpaca_eval.json"
    local_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename=filename,
        repo_type="dataset",
    )
    with open(local_path, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Unexpected AlpacaEval payload (expected list of records)")
    return Dataset.from_list(records)


DATASET_REGISTRY: dict[str, dict[str, object]] = {
    "ifeval": {
        "path": (
            "google/IFEval",
            "google/ifeval",
            "openai/ifeval",
            "google/IFEval",
        ),
        "name": None,
        "split": "train",
        "loader": load_ifeval,
        "converter": lambda record: convert_generic(
            record,
            prompt_keys=("prompt", "instruction", "input"),
            response_keys=("response", "output", "completion"),
        ),
    },
    "alpacaeval2": {
        "path": "tatsu-lab/alpaca_eval",
        "name": "alpaca_eval",
        "split": "eval",
        "loader": load_alpaca_eval,
        "converter": lambda record: convert_generic(
            record,
            prompt_keys=("instruction", "input", "prompt"),
            response_keys=("output", "response"),
        ),
    },
    "gsm8k": {
        "path": "gsm8k",
        "name": "main",
        "split": "train",
        "converter": convert_gsm8k,
    },
    "mmlu_en": {
        "path": "cais/mmlu",
        "name": "all",
        "split": "validation",
        "converter": convert_mmlu,
    },
}


def _load_with_fallback(
    candidates: Iterable[str],
    name: str | None,
    split: str,
) -> Dataset | DatasetDict:
    attempts: list[str] = []
    for candidate in candidates:
        try:
            return load_dataset(candidate, name, split=split)
        except DatasetNotFoundError as exc:
            attempts.append(f"{candidate}: {exc}")
    detail = " | ".join(attempts) if attempts else "no attempts made"
    raise DatasetNotFoundError(
        f"Unable to load dataset for split '{split}' from candidates {list(candidates)}; {detail}"
    )


def load_spec(dataset_key: str, split_override: str) -> tuple[Dataset | DatasetDict, DatasetConverter, str]:
    spec = DATASET_REGISTRY[dataset_key]
    path_spec = spec.get("path")
    name = spec.get("name")
    split = split_override or spec.get("split") or "train"
    converter = spec["converter"]
    loader = spec.get("loader")
    if loader is not None:
        dataset = loader(split)
    elif isinstance(path_spec, (tuple, list)):
        dataset = _load_with_fallback(path_spec, name, split)
    else:
        dataset = load_dataset(path_spec, name, split=split)
    return dataset, converter, split


def process_dataset(
    dataset_key: str,
    dataset: Dataset | DatasetDict,
    converter: DatasetConverter,
    output_dir: Path,
    *,
    max_records: int | None,
) -> None:
    if isinstance(dataset, DatasetDict):
        raise ValueError(f"Dataset {dataset_key} returned a DatasetDict; specify --split explicitly")
    output_path = output_dir / f"{dataset_key}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for idx, record in enumerate(dataset):
            if max_records is not None and idx >= max_records:
                break
            converted = converter(record)
            if not converted:
                continue
            writer.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1
    print(f"[{dataset_key}] wrote {written} samples to {output_path}")


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key in args.datasets:
        if dataset_key not in DATASET_REGISTRY:
            print(f"Skipping unknown dataset '{dataset_key}'", flush=True)
            continue
        dataset, converter, split_used = load_spec(dataset_key, args.split)
        print(f"Downloading {dataset_key} (split={split_used})…", flush=True)
        process_dataset(
            dataset_key,
            dataset,
            converter,
            output_dir,
            max_records=args.max_records,
        )


if __name__ == "__main__":
    main()
