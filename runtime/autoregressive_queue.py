"""Queue-based autoregressive generation demo using SegmentedQueueExecutor.

This script illustrates how to:
- Build multi-replica segment pools from manifest/thresholds.
- Feed prompts as batches into the entry queue.
- Let downstream stages competitively pull work (no manual balancing).
- Track token completion per prompt with AutoRegressiveTracker.

Notes:
- Assumes segment-1 TorchScript expects token ids (int64) as input. If your
  segment export expects embeddings instead, adapt `prepare_input_ids`.
- Exit logits are taken from the forwarder when it decides to exit; if the
  final segment has no forwarder, its output tensor is treated as logits.
- This is a minimal driver to show scheduling; it runs prompts sequentially,
  but stages execute with competitive pull internally.
"""
import argparse
import csv
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repository root is on sys.path for config/runtime imports.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.loader import load_manifest, load_thresholds
from runtime import (
    AutoRegressiveTracker,
    RuntimeFactories,
    build_queue_executor,
)
from runtime.queue_executor import SegmentedQueueExecutor, WorkItem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue-based autoregressive demo with exit heads")
    parser.add_argument("--manifest", required=True, help="Path to config/manifest.json")
    parser.add_argument("--thresholds", required=True, help="Path to thresholds.json")
    parser.add_argument("--questions", required=True, help="Text file or JSONL with prompts")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--queue-capacity", type=int, default=128, help="Per-segment queue capacity")
    parser.add_argument("--show-steps", action="store_true", help="Print step-level info")
    parser.add_argument("--gpu-log-path", help="Override GPU log path (default logs/HIT_inference/<ts>/gpu_log.csv)")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5, help="GPU sampling interval (seconds)")
    return parser.parse_args()


def load_questions(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    if path.suffix == ".jsonl":
        prompts: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                candidate = payload.get("prompt")
                if candidate:
                    prompts.append(candidate)
                    continue
                question = payload.get("question") or payload.get("input") or ""
                options = payload.get("choices") or payload.get("options")
                option_lines: List[str] = []
                if isinstance(options, (list, tuple)):
                    option_lines = [f"{chr(65 + idx)}. {opt}" for idx, opt in enumerate(options)]
                elif isinstance(options, dict):
                    option_lines = [f"{key}. {val}" for key, val in sorted(options.items())]
                prompt_text = question.strip()
                if option_lines:
                    prompt_text = f"{prompt_text}\n" + "\n".join(option_lines)
                prompts.append(prompt_text)
        return prompts
    return [line.strip() for line in path.read_text("utf-8").splitlines() if line.strip()]


def sample_next_token(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return int(token.item())


def select_logits(result) -> torch.Tensor:
    """Pick logits from WorkResult."""
    if result.logits is not None:
        return result.logits
    return result.hidden_state


def compute_final_logits(hidden_state: torch.Tensor, lm_head: torch.nn.Module) -> torch.Tensor:
    """Map final hidden state to vocab logits using the provided lm_head."""
    if hidden_state.dim() == 3:
        hidden_state = hidden_state[:, -1, :]
    head_param = next(lm_head.parameters())
    hidden_state = hidden_state.to(device=head_param.device, dtype=head_param.dtype)
    return lm_head(hidden_state)


def prepare_input_ids(tokenizer: AutoTokenizer, prompt: str) -> List[int]:
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    return encoded.input_ids[0].tolist()


def gather_gpu_stats() -> List[dict]:
    """Best-effort GPU利用率信息，优先用 pynvml，若缺失则返回空。"""
    stats: List[dict] = []
    if not torch.cuda.is_available():
        return stats
    try:
        import pynvml

        pynvml.nvmlInit()
        for idx in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats.append(
                {
                    "id": idx,
                    "gpu_util": util.gpu,  # percent
                    "mem_util": util.memory,  # percent
                    "mem_used_mb": mem.used // 1024 // 1024,
                    "mem_total_mb": mem.total // 1024 // 1024,
                }
            )
        pynvml.nvmlShutdown()
    except Exception:
        # 如果没有 pynvml 或 NVML 失败，则跳过
        pass
    return stats


class GPUSampler:
    """后台线程定期采样 GPU 利用率并写入 CSV。"""

    def __init__(self, interval: float, output_path: Path) -> None:
        self.interval = interval
        self.output_path = output_path
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._buffer: list[dict] = []
        self._nvml_ready = False

    def _init_nvml(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_ready = True
            return True
        except Exception:
            self._nvml_ready = False
            return False

    def _sample_once(self) -> None:
        if not self._nvml_ready:
            return
        now = time.time()
        for idx in range(torch.cuda.device_count()):
            handle = self._nvml.nvmlDeviceGetHandleByIndex(idx)
            util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            self._buffer.append(
                {
                    "ts": now,
                    "gpu": idx,
                    "gpu_util": util.gpu,
                    "mem_util": util.memory,
                    "mem_used_mb": mem.used // 1024 // 1024,
                    "mem_total_mb": mem.total // 1024 // 1024,
                }
            )

    def _loop(self) -> None:
        if not self._init_nvml():
            return
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)
        self._sample_once()
        try:
            self._nvml.nvmlShutdown()
        except Exception:
            pass

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if not self._buffer:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["ts", "gpu", "gpu_util", "mem_util", "mem_used_mb", "mem_total_mb"],
            )
            writer.writeheader()
            writer.writerows(self._buffer)


def run_queue_generation(
    executor: SegmentedQueueExecutor,
    tokenizer: AutoTokenizer,
    lm_head: torch.nn.Module,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    show_steps: bool,
    log_dir: Path,
) -> None:
    tracker = AutoRegressiveTracker()
    eos_id = tokenizer.eos_token_id

    for idx, prompt in enumerate(prompts, start=1):
        request_key = f"req-{idx:04d}"
        base_tokens = prepare_input_ids(tokenizer, prompt)
        generated: List[int] = []
        tracker.register(request_key, max_new_tokens=max_new_tokens, eos_token_id=eos_id)
        print(f"\n=== prompt {idx}/{len(prompts)} ===")
        print(prompt)
        start = time.perf_counter()
        # path accumulator per step
        path_history: List[List[str]] = []

        for step in range(max_new_tokens):
            context_ids = base_tokens + generated
            input_tensor = torch.tensor([context_ids], dtype=torch.long)
            work_id = f"{request_key}-step-{step}-{uuid.uuid4().hex[:6]}"
            handle = executor.submit(
                WorkItem(
                    request_id=work_id,
                    hidden_state=input_tensor,
                    step_index=step,
                    metadata={"prompt_id": request_key},
                )
            )
            result = handle.wait()
            if result is None:
                raise RuntimeError("Executor returned no result")
            logits = select_logits(result)
            if logits is None or logits.dim() > 2:
                logits = compute_final_logits(result.hidden_state, lm_head)
            token_id = sample_next_token(logits, temperature)
            done = tracker.record_step(request_key, token_id, result.segment_id)
            path = result.metadata.get("path", [])
            path.append(result.segment_id)
            path_history.append(path)
            if show_steps:
                token_str = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
                print(
                    f"step {step:02d} | exit@{result.segment_id} | token_id={token_id} | token={token_str} | path={path}"
                )
            generated.append(token_id)
            if done:
                break

        latency_ms = (time.perf_counter() - start) * 1000
        text = tokenizer.decode(generated, skip_special_tokens=True)
        summary = tracker.result(request_key) or {}
        print(f"generated: {text}")
        print(f"latency_ms: {latency_ms:.2f}")
        exits = summary.get("exit_segments", [])
        if exits:
            print(f"exit_segments: {exits}")
        # 持久化路径信息
        path_file = log_dir / f"{request_key}_paths.txt"
        try:
            with path_file.open("w", encoding="utf-8") as f:
                for step_idx, path in enumerate(path_history):
                    f.write(f"step {step_idx:02d}: {' -> '.join(path)}\n")
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    thresholds = load_thresholds(Path(args.thresholds))

    # 默认日志目录：每次运行一个时间戳子目录
    ts = time.strftime("%Y%m%d_%H%M%S")
    base_log_dir = Path("logs") / "HIT_inference" / ts
    base_log_dir.mkdir(parents=True, exist_ok=True)
    default_gpu_log = base_log_dir / "gpu_log.csv"
    default_run_log = base_log_dir / "run.log"
    default_cmd_log = base_log_dir / "commands.md"
    default_stdout = base_log_dir / "stdout.log"
    # 将本次命令记录到 commands.md
    try:
        cmdline = " ".join(sys.argv)
        with default_cmd_log.open("a", encoding="utf-8") as f:
            f.write(f"- {cmdline}\n")
    except Exception:
        pass

    # 复制 stdout 到日志文件
    try:
        tee = default_stdout.open("w", encoding="utf-8")
        # 简单 tee：写到文件，再写到原 stdout
        orig_stdout = sys.stdout
        class Tee:
            def write(self, data):
                orig_stdout.write(data)
                tee.write(data)
            def flush(self):
                orig_stdout.flush()
                tee.flush()
        sys.stdout = Tee()
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(manifest["model"]["base_model"])
    backbone_dir = manifest.get("runtime", {}).get("native_backbone_dir") or manifest["model"]["base_model"]
    backbone = AutoModelForCausalLM.from_pretrained(backbone_dir, torch_dtype=torch.float16, device_map=None)
    lm_head = backbone.lm_head
    sampler: GPUSampler | None = None
    sample_path_env = os.getenv("GPU_LOG_PATH") or args.gpu_log_path
    sample_path = Path(sample_path_env) if sample_path_env else default_gpu_log
    sample_interval = float(os.getenv("GPU_SAMPLE_INTERVAL", str(args.gpu_sample_interval)))
    if torch.cuda.is_available():
        sampler = GPUSampler(sample_interval, sample_path)
        sampler.start()
    executor = build_queue_executor(
        manifest,
        thresholds,
        RuntimeFactories(),
        queue_capacity=args.queue_capacity,
    )
    topology = executor.describe_topology()
    print("\n=== topology ===")
    for seg_id, info in topology.items():
        print(f"{seg_id}: replicas={info['replicas']} devices={info['devices']}")

    prompts = load_questions(Path(args.questions))
    run_queue_generation(
        executor,
        tokenizer,
        lm_head,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        show_steps=args.show_steps,
        log_dir=base_log_dir,
    )
    gpu_stats = gather_gpu_stats()
    if gpu_stats:
        print("\n=== GPU stats (best-effort) ===")
        for entry in gpu_stats:
            print(
                f"gpu{entry['id']}: util={entry['gpu_util']}% mem={entry['mem_util']}% "
                f"used={entry['mem_used_mb']}MB/{entry['mem_total_mb']}MB"
            )
    if sampler is not None:
        sampler.stop()
        print(f"\n[gpu_log] wrote GPU samples to {sample_path}")
    try:
        with default_run_log.open("a", encoding="utf-8") as f:
            f.write(f"command: {' '.join(sys.argv)}\n")
            f.write(f"prompt_file: {args.questions}\n")
            if sampler is not None:
                f.write(f"gpu_log: {sample_path}\n")
            f.write("\n")
    except Exception:
        pass
    try:
        # 恢复 stdout
        sys.stdout = orig_stdout
        tee.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
