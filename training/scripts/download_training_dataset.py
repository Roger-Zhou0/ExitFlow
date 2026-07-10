#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download training corpora for fine-tuning (e.g., Llama-3 8B Instruct)
and emit newline-delimited JSON with keys: {"prompt": str, "response": str}.

Datasets supported:
  Instruction/Dialogue:
    - openhermes25  -> teknium/OpenHermes-2.5 (fallback: HuggingFaceTB/OpenHermes-2.5-H4)
    - dolly15k      -> databricks/databricks-dolly-15k
    - oasst1        -> sablo/oasst2_curated (pre-stitched dialogues, HF Chat format)

  General corpora:
    - rp_wikipedia  -> togethercomputer/RedPajama-Data-1T  (trust_remote_code=True)
    - rp_books      -> togethercomputer/RedPajama-Data-1T  (trust_remote_code=True)
    - rp_arxiv      -> togethercomputer/RedPajama-Data-1T  (trust_remote_code=True)
    - rp_stackexchange -> togethercomputer/RedPajama-Data-1T (trust_remote_code=True)
    - fineweb_top30 -> HuggingFaceFW/fineweb (prefer; quality signals)
    - refinedweb_top30_proxy -> tiiuae/falcon-refinedweb (approximate Top30% via token_count/length)

Examples:
  # Smoke run (small)
  python training/scripts/download_training_dataset.py \
    --output-dir data/train_smoke \
    --datasets openhermes25 dolly15k oasst1 rp_wikipedia \
    --max-records 50

  # Dialogue-only with token budget using Llama-3 tokenizer (requires HF access)
  python training/scripts/download_training_dataset.py \
    --output-dir data/train_dialogue \
    --datasets openhermes25 oasst1 dolly15k \
    --target-tokens 6000000000 \
    --tokenizer meta-llama/Meta-Llama-3-8B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple, Union

from datasets import Dataset, DatasetDict, load_dataset
from datasets.exceptions import DatasetNotFoundError

# Optional: token budgeting
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # allow running without token budgeting

DatasetConverter = Callable[[dict], Optional[dict[str, str]]]
Loader = Callable[[str], Union[Dataset, DatasetDict]]

# -------------------------
# CLI
# -------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch training corpora and export prompt/response JSONL")
    p.add_argument("--output-dir", required=True, help="Directory to write JSONL files")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=[
            "openhermes25", "dolly15k", "oasst1",
            "rp_wikipedia", "rp_books", "rp_arxiv", "rp_stackexchange",
            "fineweb_top30",
        ],
        help="Datasets to download",
    )
    p.add_argument("--max-records", type=int, help="Max records per dataset (smoke tests)")
    p.add_argument("--split", default=None, help="Split override (default per dataset)")
    # Token budgeting
    p.add_argument("--target-tokens", type=int, help="Stop early when total tokens reach this number")
    p.add_argument("--tokenizer", default="gpt2", help="Tokenizer id for budgeting (default: gpt2)")
    p.add_argument("--truncate-response", type=int, default=None,
                   help="Optional: truncate response to N chars (rough guard to cap tokens)")
    return p

# -------------------------
# Converters
# -------------------------
def _strip(x) -> str:
    return x.strip() if isinstance(x, str) else ("" if x is None else str(x))

def convert_openhermes(record: dict) -> Optional[dict[str, str]]:
    """
    OpenHermes 2.5 variants:
      A) {"conversations":[{"from":"human","value":...},{"from":"gpt","value":...}, ...]}
      B) {"messages":[{"role":"user"|"assistant","content":...}, ...]} (H4 variant)
    Take the last user→assistant pair.
    """
    msgs = record.get("conversations") or record.get("messages")
    if not isinstance(msgs, list):
        return None

    norm = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("from")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        text = m.get("content") or m.get("value") or ""
        if role in {"user", "assistant"} and isinstance(text, str):
            norm.append({"role": role, "text": text.strip()})

    last_u = None
    last_pair = None
    for m in norm:
        if m["role"] == "user":
            last_u = m["text"]
        elif m["role"] == "assistant" and last_u is not None:
            last_pair = (last_u, m["text"])
    if not last_pair:
        # fallback to first adjacent user→assistant
        for i in range(len(norm) - 1):
            if norm[i]["role"] == "user" and norm[i + 1]["role"] == "assistant":
                last_pair = (norm[i]["text"], norm[i + 1]["text"])
                break
    if not last_pair:
        return None
    u, a = last_pair
    if not u and not a:
        return None
    return {"prompt": u, "response": a}

def convert_dolly15k(record: dict) -> Optional[dict[str, str]]:
    instr = _strip(record.get("instruction", "")) or _strip(record.get("prompt", ""))
    ctx = _strip(record.get("context", ""))
    resp = _strip(record.get("response", "")) or _strip(record.get("output", ""))
    if not instr and not resp:
        return None
    prompt = instr if not ctx else f"{instr}\n{ctx}"
    return {"prompt": prompt, "response": resp}

def convert_oasst_curated_last_pair(record: dict) -> Optional[dict[str, str]]:
    """
    sablo/oasst2_curated: HF Chat-like format
      {"messages":[{"role":"user"|"assistant","content":...}, ...]}
    Take the last user→assistant pair.
    """
    msgs = record.get("messages")
    if not isinstance(msgs, list):
        return None
    last_u = None
    last_pair = None
    for m in msgs:
        role = m.get("role")
        txt = (m.get("content") or m.get("text") or "").strip()
        if role == "user":
            last_u = txt
        elif role == "assistant" and last_u:
            last_pair = (last_u, txt)
    if not last_pair:
        # fallback: first pair
        for i in range(len(msgs) - 1):
            r1, r2 = msgs[i].get("role"), msgs[i + 1].get("role")
            if r1 == "user" and r2 == "assistant":
                u = (msgs[i].get("content") or msgs[i].get("text") or "").strip()
                a = (msgs[i + 1].get("content") or msgs[i + 1].get("text") or "").strip()
                last_pair = (u, a)
                break
    if not last_pair:
        return None
    return {"prompt": last_pair[0], "response": last_pair[1]}

def convert_text_only(record: dict) -> Optional[dict[str, str]]:
    """
    For pretraining-like corpora. Store raw text in 'response' to keep schema.
    Covers common RedPajama/FineWeb variants.
    """
    # 常见字段尽可能全
    candidates = [
        "text", "content", "raw_content", "body", "document",
        "dump", "content_text", "page", "section_text", "normalized_text",
    ]
    blob = ""
    for k in candidates:
        v = record.get(k, None)
        if isinstance(v, str) and v.strip():
            blob = v
            break

    # 有些源把正文塞在一个 dict 里（极少见）：尝试取 values 里最长的字符串
    if not blob and isinstance(record, dict):
        longest = ""
        for v in record.values():
            if isinstance(v, str) and len(v) > len(longest):
                longest = v
        blob = longest

    if not blob:
        return None

    # StackExchange 等可能是 HTML，做个非常轻量的去标签（避免引入依赖）
    if "<p" in blob or "<div" in blob or "</" in blob:
        import re
        blob = re.sub(r"<[^>]+>", "", blob)  # 去 HTML 标签
        blob = re.sub(r"\s+\n", "\n", blob).strip()

    blob = blob.strip()
    if not blob:
        return None

    return {"prompt": "", "response": blob}


def convert_openorca(record: dict) -> Optional[dict[str, str]]:
    """
    OpenOrca / SlimOrca 常见字段: system_prompt, question, response
    也有些镜像用 input/output 命名，做下兜底。
    """
    sys = _strip(record.get("system_prompt", ""))
    q = _strip(record.get("question", "")) or _strip(record.get("input", ""))
    a = _strip(record.get("response", "")) or _strip(record.get("output", ""))
    if not q and not a:
        return None
    prompt = f"[SYSTEM]\n{sys}\n\n[USER]\n{q}" if sys else q
    return {"prompt": prompt, "response": a}


def convert_flan_v2(record: dict) -> Optional[dict[str, str]]:
    inp = _strip(record.get("inputs", "")) or _strip(record.get("input", "")) \
          or _strip(record.get("instruction", "")) or _strip(record.get("prompt", "")) or _strip(record.get("source", ""))
    tgt = _strip(record.get("targets", "")) or _strip(record.get("target", "")) \
          or _strip(record.get("output", "")) or _strip(record.get("completion", ""))

    if inp or tgt:
        return {"prompt": inp, "response": tgt}

    blob = _strip(record.get("text", "")) or _strip(record.get("sample", "")) \
           or _strip(record.get("source", "")) or _strip(record.get("content", ""))

    if not blob:
        return None

    lines = [ln.rstrip() for ln in blob.splitlines() if ln.strip() != ""]
    if len(lines) >= 2:
        stem, last = "\n".join(lines[:-1]), lines[-1]
        import re
        ans = last
        m = re.search(r"(?:^|\b)(?:answer\s+is|final\s+answer\s*:?)\s*(.+?)\s*$", last, flags=re.I)
        if m:
            ans = m.group(1).strip()
            ans = re.sub(r"^[\(\[\{]\s*|\s*[\)\]\}]$", "", ans)
        return {"prompt": stem, "response": ans}

    return {"prompt": blob, "response": ""}




def convert_ultrachat_last_turn(record: dict) -> Optional[dict[str, str]]:
    """
    HuggingFaceH4/ultrachat_200k: 'messages' 列表 [{role, content}, ...]
    取最后一轮 user→assistant；若没有则取第一对。
    """
    msgs = record.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    last_u = None
    last_pair = None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        txt = _strip(m.get("content", ""))
        if role == "user":
            last_u = txt
        elif role == "assistant" and last_u:
            last_pair = (last_u, txt)
    if not last_pair:
        for i in range(len(msgs) - 1):
            if msgs[i].get("role") == "user" and msgs[i+1].get("role") == "assistant":
                u = _strip(msgs[i].get("content", ""))
                a = _strip(msgs[i+1].get("content", ""))
                last_pair = (u, a)
                break
    if not last_pair:
        return None
    return {"prompt": last_pair[0], "response": last_pair[1]}


# -------------------------
# Loaders
# -------------------------
def _load_with_fallback(
    candidates: Iterable[str],
    name: Optional[str],
    split: str,
) -> Union[Dataset, DatasetDict]:
    attempts = []
    cand_list = list(candidates) if isinstance(candidates, (list, tuple)) else [candidates]
    for cand in cand_list:
        try:
            return load_dataset(cand, name, split=split)
        except DatasetNotFoundError as exc:
            attempts.append(f"{cand}: {exc}")
    detail = " | ".join(attempts) if attempts else "no attempts made"
    raise DatasetNotFoundError(
        f"Unable to load dataset for split '{split}' from candidates {cand_list}; {detail}"
    )

def load_openhermes(split: str) -> Dataset:
    return _load_with_fallback(["teknium/OpenHermes-2.5", "HuggingFaceTB/OpenHermes-2.5-H4"], None, split)  # type: ignore

def load_dolly15k(split: str) -> Dataset:
    return _load_with_fallback("databricks/databricks-dolly-15k", None, split)  # type: ignore

def load_oasst_curated(split: str) -> Dataset:
    # curated dialogues in HF Chat style
    ds = _load_with_fallback("sablo/oasst2_curated", None, "train")
    return ds  # type: ignore


def load_fineweb_top30(split: str = "train"):
    """
    Streaming 版本：不下载整库。
    通过对前 sample_size 条做长度统计估计 70% 分位阈值（近似 Top30%），再重建流并过滤。
    说明：
      - 返回的是 IterableDataset（datasets 的流式迭代器），你的逐条写出逻辑可直接兼容。
      - 若需要更稳的阈值，可把 sample_size 调大。
    """
    from datasets import load_dataset

    # 第一次拉流，用于估计阈值（只消费前缀，不会全量下载）
    ds_stream = load_dataset("HuggingFaceFW/fineweb", split=split, streaming=True)

    def word_count(rec) -> int:
        text = rec.get("content") or rec.get("text") or ""
        return len(str(text).split())

    # 采样前缀估计长度分布
    sample_size = 5000
    sample_lengths = []
    for i, rec in enumerate(ds_stream.take(sample_size)):
        sample_lengths.append(word_count(rec))

    # 如果没有采到样本，直接返回原流（不上筛）
    if not sample_lengths:
        return load_dataset("HuggingFaceFW/fineweb", split=split, streaming=True)

    sample_lengths.sort()
    # 70% 分位（Top30% 的下界）
    k = int(0.70 * (len(sample_lengths) - 1))
    length_threshold = sample_lengths[k]

    # 重新获取一条新流（之前 take 已经消费了）
    ds_stream = load_dataset("HuggingFaceFW/fineweb", split=split, streaming=True)

    # 过滤得到“近似 Top30%”
    ds_stream = ds_stream.filter(lambda x: word_count(x) >= length_threshold)
    return ds_stream


def load_refinedweb_top30_proxy(split: str = "train"):
    """
    Streaming 近似 Top30%：
    - tiiuae/falcon-refinedweb 无公开 quality_score，使用长度作为近似质量信号。
    - 先用小样本估 70% 分位阈值，再重建流并过滤。
    """
    from datasets import load_dataset

    ds_stream = load_dataset("tiiuae/falcon-refinedweb", split=split, streaming=True)

    def word_count(rec) -> int:
        text = rec.get("content") or ""
        return len(str(text).split())

    sample_size = 5000
    sample_lengths = []
    for i, rec in enumerate(ds_stream.take(sample_size)):
        sample_lengths.append(word_count(rec))

    if not sample_lengths:
        return load_dataset("tiiuae/falcon-refinedweb", split=split, streaming=True)

    sample_lengths.sort()
    k = int(0.70 * (len(sample_lengths) - 1))
    length_threshold = sample_lengths[k]

    ds_stream = load_dataset("tiiuae/falcon-refinedweb", split=split, streaming=True)
    ds_stream = ds_stream.filter(lambda x: word_count(x) >= length_threshold)
    return ds_stream

def load_openorca(split: str) -> Dataset:
    # 多个镜像/分叉，按顺序尝试
    candidates = [
        "Open-Orca/OpenOrca",              # 官方主仓
        "Open-Orca/SlimOrca",              # SlimOrca 也可走同一 converter
        "Open-Orca/OpenOrcaV1.1",          # 可能存在的版本/镜像
    ]
    return _load_with_fallback(candidates, None, split)  # type: ignore


def load_slimorca(split: str) -> Dataset:
    candidates = [
        "Open-Orca/SlimOrca",
        "Open-Orca/OpenOrca",              # 兜底
    ]
    return _load_with_fallback(candidates, None, split)  # type: ignore

from datasets import load_dataset

def _load_by_patterns(data_format: str, patterns: list[str]) -> Dataset:
    last_err = None
    for pat in patterns:
        try:
            return load_dataset(data_format, data_files=pat, split="train", streaming=True)  # type: ignore
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Data files not found for format={data_format} via patterns={patterns}\nLast error: {last_err}")

def load_rp_subset(domain: str, split: str) -> Dataset:
    """
    domain: 'wikipedia' | 'book' | 'books' | 'arxiv' | 'stackexchange'
    先尝试 parquet，失败再尝试 jsonl.gz
    """
    parquet_pats = [
        f"hf://datasets/togethercomputer/RedPajama-Data-1T/data/{domain}/*.parquet",
        f"hf://datasets/togethercomputer/RedPajama-Data-1T/{domain}/*.parquet",
        f"hf://datasets/HuggingFaceM4/RedPajama-Data-1T/{domain}/*.parquet",
        f"hf://datasets/openaccess-ai-collective/RedPajama-Data-1T/{domain}/*.parquet",
    ]
    jsonl_pats = [
        f"hf://datasets/togethercomputer/RedPajama-Data-1T/data/{domain}/*.jsonl.gz",
        f"hf://datasets/togethercomputer/RedPajama-Data-1T/{domain}/*.jsonl.gz",
        f"hf://datasets/HuggingFaceM4/RedPajama-Data-1T/{domain}/*.jsonl.gz",
        f"hf://datasets/openaccess-ai-collective/RedPajama-Data-1T/{domain}/*.jsonl.gz",
    ]

    try:
        # 优先 parquet
        return _load_by_patterns("parquet", parquet_pats)
    except Exception:
        # 兜底 jsonl.gz（RedPajama 原始发布格式）
        return _load_by_patterns("json", jsonl_pats)

def load_rp_wikipedia(split: str) -> Dataset:
    return load_rp_subset("wikipedia", split)

def load_rp_books(split: str) -> Dataset:
    # 有的仓叫 'book'，有的叫 'books'
    try:
        return load_rp_subset("book", split)
    except Exception:
        return load_rp_subset("books", split)

def load_rp_arxiv(split: str) -> Dataset:
    return load_rp_subset("arxiv", split)

def load_rp_stackexchange(split: str) -> Dataset:
    return load_rp_subset("stackexchange", split)



from datasets import load_dataset

def load_flan_v2(split: str) -> Dataset:
    # 只用 Parquet 镜像，流式读取，避免 map 全量遍历
    return load_dataset("chiayewken/flan-v2", split=split, streaming=True)  # type: ignore




def load_ultrachat(split: str) -> Dataset:
    # 官方大仓（200k 多轮对话）
    eff_split = split or "train_sft"
    return _load_with_fallback("HuggingFaceH4/ultrachat_200k", "default", eff_split)  # type: ignore


# -------------------------
# Registry
# -------------------------
DATASET_REGISTRY: dict[str, dict] = {
    # Instruction/Dialogue
    "openhermes25": {
        "loader": load_openhermes,
        "split": "train",
        "converter": convert_openhermes,
    },
    "dolly15k": {
        "loader": load_dolly15k,
        "split": "train",
        "converter": convert_dolly15k,
    },
    "oasst1": {
        # use curated pre-stitched dialogues
        "loader": load_oasst_curated,
        "split": "train",
        "converter": convert_oasst_curated_last_pair,
    },


    # General web corpora
    "fineweb_top30": {
        "loader": load_fineweb_top30,
        "split": "train",
        "converter": convert_text_only,
    },
    "refinedweb_top30_proxy": {
        "loader": load_refinedweb_top30_proxy,
        "split": "train",
        "converter": convert_text_only,
    },
    "openorca": {
        "loader": load_openorca,
        "split": "train",
        "converter": convert_openorca,
    },
    "slimorca": {
        "loader": load_slimorca,
        "split": "train",
        "converter": convert_openhermes,   # 同一转换器即可
    },
    "flan_v2": {
        "loader": load_flan_v2,
        "split": "train",
        "converter": convert_flan_v2,
    },
    "ultrachat": {
        "loader": load_ultrachat,
        "split": "train_sft",
        "converter": convert_ultrachat_last_turn,
    },
        "rp_wikipedia": {
        "loader": load_rp_wikipedia,
        "split": "train",
        "converter": convert_text_only,
    },
    "rp_books": {
        "loader": load_rp_books,
        "split": "train",
        "converter": convert_text_only,
    },
    "rp_arxiv": {
        "loader": load_rp_arxiv,
        "split": "train",
        "converter": convert_text_only,
    },
    "rp_stackexchange": {
        "loader": load_rp_stackexchange,
        "split": "train",
        "converter": convert_text_only,
    },

}

# -------------------------
# Token budgeting
# -------------------------
class TokenBudget:
    def __init__(self, target_tokens: Optional[int], tokenizer_id: str, truncate_response: Optional[int]):
        self.target = target_tokens or None
        self.truncate_response = truncate_response
        self.total = 0
        self.enabled = self.target is not None
        self.tok = None
        if self.enabled and AutoTokenizer is not None:
            try:
                self.tok = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
            except Exception:
                # fallback to gpt2 if specified tokenizer unavailable
                try:
                    self.tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
                except Exception:
                    self.tok = None
        if self.enabled and self.tok is None:
            print("[warn] token budgeting enabled but tokenizer unavailable; proceeding without budgeting")
            self.enabled = False

    def count(self, prompt: str, response: str) -> int:
        # Optional rough truncation of response before counting
        if self.truncate_response and response:
            response = response[: self.truncate_response]
        text = (prompt or "") + "\n" + (response or "")
        if self.tok is None:
            # rough char-based heuristic (~4 chars/token)
            return max(1, len(text) // 4)
        return len(self.tok.encode(text, add_special_tokens=False))

    def add_and_check(self, n_tokens: int) -> bool:
        """
        Add tokens and return True to continue; False if budget exhausted.
        """
        if not self.enabled:
            return True
        self.total += n_tokens
        return self.total < self.target

# -------------------------
# Writer
# -------------------------
def process_dataset(
    dataset_key: str,
    dataset: Dataset,
    converter: DatasetConverter,
    output_dir: Path,
    *,
    max_records: Optional[int],
    budget: TokenBudget,
) -> Tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset_key}.jsonl"
    written = 0
    seen = 0
    with out_path.open("w", encoding="utf-8") as w:
        for rec in dataset:
            seen += 1
            if max_records is not None and written >= max_records:
                break
            out = converter(rec)
            if not out:
                continue
            prompt = _strip(out.get("prompt", ""))
            response = _strip(out.get("response", ""))
            if not prompt and not response:
                continue
            n_tok = budget.count(prompt, response)
            w.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")
            written += 1
            if not budget.add_and_check(n_tok):
                print(f"[{dataset_key}] token budget reached; stopping early")
                break
    print(f"[{dataset_key}] wrote {written} samples to {out_path}")
    return written, seen

# -------------------------
# Main
# -------------------------
def load_spec(dataset_key: str, split_override: Optional[str]) -> tuple[Dataset, DatasetConverter, str]:
    spec = DATASET_REGISTRY[dataset_key]
    loader: Loader = spec["loader"]
    split = split_override or spec.get("split") or "train"
    ds = loader(split)
    if isinstance(ds, DatasetDict):
        # force intended split if aggregate provided
        ds = loader(spec.get("split") or "train")  # type: ignore
    return ds, spec["converter"], split

def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budget = TokenBudget(args.target_tokens, args.tokenizer, args.truncate_response)

    total_written = 0
    for key in args.datasets:
        if key not in DATASET_REGISTRY:
            print(f"Skipping unknown dataset '{key}'", flush=True)
            continue
        ds, conv, split_used = load_spec(key, args.split)
        print(f"Downloading {key} (split={split_used})…", flush=True)
        written, _ = process_dataset(
            key, ds, conv, out_dir, max_records=args.max_records, budget=budget
        )
        total_written += written
        if budget.enabled and budget.total >= (budget.target or 0):
            print(f"[all] global token budget reached: {budget.total}")
            break

    print(f"[done] total samples written: {total_written}; approx tokens: {budget.total}")

if __name__ == "__main__":
    main()
