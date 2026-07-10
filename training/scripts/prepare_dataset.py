#!/usr/bin/env python3
"""Utility to convert raw text/JSONL corpora into tokenised JSONL datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenise a corpus into JSONL samples")
    parser.add_argument("--input", required=True, help="Path to raw corpus (text or JSONL)")
    parser.add_argument("--output", required=True, help="Path to write tokenised JSONL dataset")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer/model identifier (e.g. meta-llama/Llama-3-8b-instruct)",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "text", "jsonl"),
        default="auto",
        help="Force input parsing mode; default infers from suffix",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="JSON key to read when using single-field JSONL inputs",
    )
    parser.add_argument(
        "--prompt-field",
        help="JSON key containing prompt text; combine with --response-field if provided",
    )
    parser.add_argument(
        "--response-field",
        help="JSON key containing response text; combine with --prompt-field if provided",
    )
    parser.add_argument(
        "--separator",
        default="\n",
        help="Separator inserted between prompt and response when both are present",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum token length per sample (after optional BOS/EOS)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="Slide window by (max_length - stride) tokens when chunking; 0 disables overlap",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=16,
        help="Discard chunks shorter than this many tokens",
    )
    parser.add_argument(
        "--pad-to-max",
        action="store_true",
        help="Pad each chunk to max_length using the tokenizer pad token",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Prepend the tokenizer BOS token if available",
    )
    parser.add_argument(
        "--append-eos",
        action="store_true",
        help="Append the tokenizer EOS token if available",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Allow final chunk shorter than max_length instead of discarding overflow",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on number of raw records processed (helps with quick smoke tests)",
    )
    return parser.parse_args()


def detect_format(path: Path, forced: str) -> str:
    if forced != "auto":
        return forced
    if path.suffix.lower() in {".jsonl", ".json"}:
        return "jsonl"
    return "text"


def iter_texts(
    path: Path,
    fmt: str,
    *,
    text_field: str,
    prompt_field: str | None,
    response_field: str | None,
    separator: str,
    limit: int | None,
) -> Iterator[str]:
    count = 0
    if fmt == "text":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and count >= limit:
                    break
                text = line.strip()
                if not text:
                    continue
                yield text
                count += 1
        return

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text: str | None = None
            if prompt_field and response_field:
                prompt = record.get(prompt_field)
                response = record.get(response_field)
                if prompt is None and response is None:
                    continue
                prompt = prompt or ""
                response = response or ""
                text = f"{prompt}{separator}{response}" if prompt else response
            else:
                raw = record.get(text_field)
                if raw is None:
                    continue
                text = str(raw)
            text = text.strip()
            if not text:
                continue
            yield text
            count += 1


def chunk_tokens(
    tokens: Sequence[int],
    *,
    max_length: int,
    stride: int,
    min_length: int,
    pad_to_max: bool,
    pad_id: int | None,
) -> Iterable[Sequence[int]]:
    if not tokens:
        return []
    window = max_length
    step = window - stride if 0 < stride < window else window
    chunks = []
    for start in range(0, len(tokens), step):
        chunk = list(tokens[start : start + window])
        if len(chunk) < min_length:
            if start == 0 and len(chunk) >= min_length:
                pass
            else:
                continue
        if pad_to_max and len(chunk) < window:
            if pad_id is None:
                raise ValueError("pad-to-max requested but tokenizer has no pad token")
            chunk.extend([pad_id] * (window - len(chunk)))
        chunks.append(chunk)
        if start + window >= len(tokens):
            break
    return chunks


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = detect_format(input_path, args.input_format)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    has_bos = getattr(tokenizer, "bos_token_id", None) is not None
    has_eos = getattr(tokenizer, "eos_token_id", None) is not None
    pad_id = getattr(tokenizer, "pad_token_id", None)

    total_samples = 0
    total_chunks = 0

    with output_path.open("w", encoding="utf-8") as writer:
        for text in iter_texts(
            input_path,
            fmt,
            text_field=args.text_field,
            prompt_field=args.prompt_field,
            response_field=args.response_field,
            separator=args.separator,
            limit=args.limit,
        ):
            encoding = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                truncation=False,
            )
            tokens = encoding["input_ids"]
            if args.add_bos and has_bos:
                tokens = [tokenizer.bos_token_id] + tokens
            if args.append_eos and has_eos:
                tokens = tokens + [tokenizer.eos_token_id]
            if not tokens:
                continue
            chunks = chunk_tokens(
                tokens,
                max_length=args.max_length,
                stride=args.stride,
                min_length=args.min_length,
                pad_to_max=args.pad_to_max,
                pad_id=pad_id,
            )
            if not chunks and args.truncate and len(tokens) >= args.min_length:
                truncated = tokens[: args.max_length]
                if args.pad_to_max and len(truncated) < args.max_length:
                    if pad_id is None:
                        raise ValueError("pad-to-max requested but tokenizer has no pad token")
                    truncated = truncated + [pad_id] * (args.max_length - len(truncated))
                chunks = [truncated]
            for chunk in chunks:
                sample = {"input_ids": chunk, "labels": chunk}
                writer.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_chunks += 1
            total_samples += 1

    print(f"Processed {total_samples} raw records → {total_chunks} tokenised samples", flush=True)


if __name__ == "__main__":
    main()
