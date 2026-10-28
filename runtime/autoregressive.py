"""Full-backbone autoregressive exit-head diagnostic.

All Transformer layers are executed for each token. Intermediate exit heads
are probed after the full forward pass, so this path does not skip compute.
Use the segmented runtime for physical layer skipping.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from config.loader import load_manifest


@dataclass
class ExitDecision:
    forwarder_id: str
    layer_index: int
    segment_id: Optional[str]
    scores: Dict[str, float]


class AutoRegressiveGenerator:
    """Generate token-by-token using pretrained exit heads."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        metadata_path: Path,
        thresholds_path: Optional[Path] = None,
        device: str = "cuda",
        per_step_logging: bool = True,
        gpu_timing: bool = True,
        force_final_exit: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.manifest = load_manifest(manifest_path)
        manifest_model = self.manifest["model"]
        self.dataset = self.manifest.get("evaluation", {}).get("benchmarks", [])

        metadata = json.loads(metadata_path.read_text("utf-8"))
        self.backbone_dir = metadata.get("backbone_dir")
        self.exit_head_paths = metadata.get("exit_heads", {})
        self.thresholds = self._load_thresholds(thresholds_path)

        precision = manifest_model.get("precision")
        self.torch_dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and precision == "bf16")
            else (torch.float16 if (self.device.type == "cuda" and precision in {"fp16", "float16"}) else torch.float32)
        )

        backbone_source = Path(self.backbone_dir) if self.backbone_dir else Path(manifest_model["base_model"])
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone_source,
            low_cpu_mem_usage=True,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_source)
        self.exit_specs = self._build_exit_specs(self.manifest["model"])
        self.exit_heads = {
            exit_id: torch.jit.load(path, map_location=self.device).eval()
            for exit_id, path in self.exit_head_paths.items()
        }
        self.record_per_step = per_step_logging
        self.track_gpu_timing = gpu_timing and self.device.type == "cuda"
        self.choice_map = self._build_choice_token_ids()
        self.force_final_exit = force_final_exit

    def _load_thresholds(self, path: Optional[Path]) -> Dict[str, Dict[str, float]]:
        if path is None:
            return {}
        return json.loads(path.read_text("utf-8"))

    @staticmethod
    def _build_exit_specs(model_section: dict) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        for segment in model_section.get("segments", []):
            forwarder = segment.get("forwarder")
            if not forwarder:
                continue
            specs.append(
                {
                    "id": forwarder["id"],
                    "layer_index": segment["layers"]["end"],
                    "segment_id": segment.get("id"),
                    "next_segment": forwarder.get("next_segment"),
                }
            )
        return specs

    def select_exit_depth(self, input_ids: torch.Tensor) -> ExitDecision:
        """Probe the backbone to select the earliest exit head that passes thresholds."""
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
        if self.force_final_exit:
            decision, _ = self._final_head_decision(outputs)
        else:
            hidden_states = outputs.hidden_states
            decision, _ = self._evaluate_exit_from_hidden(hidden_states)
        return decision

    def _final_token_hidden(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.dim() == 3:
            hidden_state = hidden_state[:, -1, :]
        if hidden_state.dim() == 2:
            return hidden_state
        raise ValueError("Unexpected hidden_state shape: %s" % (tuple(hidden_state.shape),))

    def _run_exit_head(self, forwarder_id: str, hidden: torch.Tensor) -> torch.Tensor:
        head = self.exit_heads.get(forwarder_id)
        if head is None:
            raise KeyError(f"Exit head missing for forwarder {forwarder_id}")
        # align hidden dtype/device to head parameters
        head_param = next(head.parameters(), None)
        if head_param is not None:
            target_device = head_param.device
            target_dtype = head_param.dtype
            hidden = hidden.to(device=target_device, dtype=target_dtype)
        return head(hidden)

    def _score_logits(self, logits: torch.Tensor, exit_id: str) -> Dict[str, float]:
        temperature = self.thresholds.get(exit_id, {}).get("temperature_scale", 1.0)
        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1)
        k = min(2, scaled.shape[-1])
        topk = torch.topk(scaled, k=k)
        top_values = topk.values
        top_logit = float(top_values[0, 0].item())
        if k > 1:
            margin = float((top_values[0, 0] - top_values[0, 1]).item())
        else:
            margin = float("inf")
        entropy = float(-(probs * torch.log(probs + 1e-8)).sum().item())
        return {"logit": top_logit, "margin": margin, "entropy": entropy, "temperature": temperature}

    def _passes_gate(self, scores: Dict[str, float], exit_id: str) -> bool:
        cfg = self.thresholds.get(exit_id, {})
        logit_thr = float(cfg.get("logit_threshold", 0.0))
        margin_thr = float(cfg.get("margin_gate", 0.0))
        entropy_thr = float(cfg.get("entropy_gate", float("inf")))
        return (
            scores["logit"] >= logit_thr
            and scores["margin"] >= margin_thr
            and scores["entropy"] <= entropy_thr
        )

    def _build_choice_token_ids(self) -> Dict[str, List[int]]:
        candidate_variants = {
            "A": ["A", " A", "ĠA", "(A", "A.", "A)"],
            "B": ["B", " B", "ĠB", "(B", "B.", "B)"],
            "C": ["C", " C", "ĠC", "(C", "C.", "C)"],
            "D": ["D", " D", "ĠD", "(D", "D.", "D)"],
        }
        mapping: Dict[str, List[int]] = {}
        for label, variants in candidate_variants.items():
            ids: List[int] = []
            for variant in variants:
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if token_ids:
                    ids.append(token_ids[0])
            mapping[label] = list(dict.fromkeys(ids))
        return mapping

    def _compute_choice_logits(self, logits: torch.Tensor) -> Dict[str, Optional[float]]:
        flat_logits = logits.view(-1)
        results: Dict[str, Optional[float]] = {}
        for label, token_ids in self.choice_map.items():
            best: Optional[float] = None
            for tid in token_ids:
                if tid < flat_logits.size(0):
                    value = float(flat_logits[tid].item())
                    if best is None or value > best:
                        best = value
            results[label] = best
        return results

    def _predict_choice_from_logits(self, choice_logits: Dict[str, Optional[float]]) -> Optional[str]:
        best_label: Optional[str] = None
        best_value = float("-inf")
        for label, value in choice_logits.items():
            if value is None:
                continue
            if value > best_value:
                best_value = value
                best_label = label
        return best_label

    @staticmethod
    def _choice_dict_to_list(choice_logits: Dict[str, Optional[float]]) -> List[Optional[float]]:
        return [choice_logits.get(label) for label in ("A", "B", "C", "D")]

    @staticmethod
    def _select_summary_step(
        per_step: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        # MMLU and other next-token diagnostics must score the first generated
        # token, not whichever later step happens to match the final exit.
        return per_step[0] if per_step else None

    def _summarize_choice_prediction(
        self,
        per_step: List[Dict[str, Any]],
    ) -> Tuple[Optional[List[Optional[float]]], Optional[str]]:
        summary_step = self._select_summary_step(per_step)
        if summary_step is None:
            return None, None
        choice_logits = summary_step.get("choice_logits")
        if not choice_logits:
            return None, None
        choice_list = self._choice_dict_to_list(choice_logits)
        predicted = summary_step.get("predicted_choice")
        if predicted is None:
            predicted = self._predict_choice_from_logits(choice_logits)
        return choice_list, predicted

    def _evaluate_exit_from_hidden(
        self, hidden_states: Tuple[torch.Tensor, ...]
    ) -> Tuple[ExitDecision, torch.Tensor]:
        """Return the decision and logits for the first exit that passes thresholds."""
        last_decision: Optional[ExitDecision] = None
        last_logits: Optional[torch.Tensor] = None
        for spec in self.exit_specs:
            hidden_state = hidden_states[spec["layer_index"]]
            token_hidden = self._final_token_hidden(hidden_state)
            logits = self._run_exit_head(spec["id"], token_hidden)
            scores = self._score_logits(logits, spec["id"])
            decision = ExitDecision(
                forwarder_id=spec["id"],
                layer_index=spec["layer_index"],
                segment_id=spec["segment_id"],
                scores=scores,
            )
            last_decision = decision
            last_logits = logits
            if self._passes_gate(scores, spec["id"]):
                return decision, logits
        if last_decision is None or last_logits is None:
            raise RuntimeError("No exit specs defined in manifest")
        return last_decision, last_logits

    def _final_head_decision(self, model_outputs) -> Tuple[ExitDecision, torch.Tensor]:
        """Use the backbone final LM head logits as the exit decision."""
        logits = model_outputs.logits[:, -1, :]
        scores = self._score_logits(logits, "final_head")
        hidden_states = model_outputs.hidden_states or ()
        layer_index = len(hidden_states) - 1 if hidden_states else -1
        decision = ExitDecision(
            forwarder_id="final_head",
            layer_index=layer_index,
            segment_id=None,
            scores=scores,
        )
        return decision, logits

    @staticmethod
    def _serialize_exit_decision(decision: ExitDecision) -> Dict[str, Any]:
        return {
            "forwarder_id": decision.forwarder_id,
            "layer_index": decision.layer_index,
            "segment_id": decision.segment_id,
            "scores": decision.scores,
        }

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        decode_temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Autoregressively decode tokens using a single exit head."""
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.device)
        probe_start = time.perf_counter()
        exit_decision = self.select_exit_depth(input_ids)
        probe_latency = (time.perf_counter() - probe_start) * 1000
        tokens = input_ids[0].tolist()
        generated: List[int] = []
        per_step: List[Dict[str, Any]] = []
        start = time.perf_counter()
        final_decision = exit_decision
        for step_index in range(max_new_tokens):
            context_ids = torch.tensor([tokens + generated], device=self.device)
            step_cpu_start = time.perf_counter()
            gpu_start = None
            gpu_end = None
            if self.track_gpu_timing:
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
                gpu_start.record()
            with torch.no_grad():
                outputs = self.model(
                    context_ids,
                    output_hidden_states=True,
                )
            if gpu_end is not None:
                gpu_end.record()
            if self.force_final_exit:
                dynamic_decision, logits = self._final_head_decision(outputs)
            else:
                hidden_states = outputs.hidden_states
                dynamic_decision, logits = self._evaluate_exit_from_hidden(hidden_states)
            next_token = self._sample_next_token(logits, decode_temperature)
            base_logits = outputs.logits[:, -1, :]
            k_val = min(5, base_logits.shape[-1])
            base_top = torch.topk(base_logits, k=k_val)
            step_cpu_latency = (time.perf_counter() - step_cpu_start) * 1000
            gpu_latency = None
            if gpu_start is not None and gpu_end is not None:
                torch.cuda.synchronize()
                gpu_latency = float(gpu_start.elapsed_time(gpu_end))
            record = {
                "step": step_index,
                "token_id": int(next_token),
                "token": self.tokenizer.decode([int(next_token)], clean_up_tokenization_spaces=False),
                "scores": dynamic_decision.scores,
                "exit_forwarder_id": dynamic_decision.forwarder_id,
                "exit_layer_index": dynamic_decision.layer_index,
                "exit_segment_id": dynamic_decision.segment_id,
                "gate_passed": (
                    True
                    if self.force_final_exit
                    else self._passes_gate(dynamic_decision.scores, dynamic_decision.forwarder_id)
                ),
                "step_latency_ms": step_cpu_latency,
                "gpu_latency_ms": gpu_latency,
                "base_top_logit": float(base_top.values[0, 0].item()),
                "base_top_tokens": [int(idx.item()) for idx in base_top.indices[0]],
            }
            choice_logits = self._compute_choice_logits(logits)
            record["choice_logits"] = choice_logits
            record["predicted_choice"] = self._predict_choice_from_logits(choice_logits)
            if self.record_per_step:
                per_step.append(record)
            generated.append(int(next_token))
            final_decision = dynamic_decision
            if eos_token_id is not None and int(next_token) == eos_token_id:
                break
        total_latency = (time.perf_counter() - start) * 1000
        generated_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        choice_summary, predicted_choice = self._summarize_choice_prediction(per_step)
        backbone_layer_count = max(
            segment["layers"]["end"] for segment in self.manifest["model"]["segments"]
        )
        result: Dict[str, Any] = {
            "prompt": prompt,
            "execution_mode": "full_backbone_exit_head_diagnostic",
            "physical_layer_skipping": False,
            "backbone_layers_run_per_forward": backbone_layer_count,
            "backbone_forward_calls": len(generated) + 1,  # probe + decoding
            "force_final_exit": self.force_final_exit,
            "probe_included_in_total_latency": False,
            "end_to_end_latency_ms": probe_latency + total_latency,
            "generated_text": generated_text,
            "generated_ids": generated,
            "exit_decision": self._serialize_exit_decision(final_decision),
            "probe_latency_ms": probe_latency,
            "per_step": per_step,
            "total_latency_ms": total_latency,
            "initial_exit_forwarder_id": exit_decision.forwarder_id,
            "initial_exit_layer_index": exit_decision.layer_index,
            "final_exit_forwarder_id": final_decision.forwarder_id,
            "final_exit_layer_index": final_decision.layer_index,
        }
        if choice_summary is not None:
            result["choice_logits"] = choice_summary
        if predicted_choice is not None:
            result["predicted_choice"] = predicted_choice
        return result

    def _sample_next_token(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoregressive generation with exit heads")
    parser.add_argument("--manifest", required=True, help="Path to config/manifest.json")
    parser.add_argument("--metadata", required=True, help="Path to training metadata.json")
    parser.add_argument("--thresholds", help="Path to thresholds.json")
    parser.add_argument("--prompt", required=True, help="Prompt to generate from")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--device", default="cuda", help="Torch device string")
    parser.add_argument("--no-per-step", action="store_true", help="Skip recording detailed per-step logs")
    parser.add_argument("--show-steps", action="store_true", help="Print per-step token info")
    parser.add_argument(
        "--force-final-exit",
        action="store_true",
        help="Bypass intermediate exit heads and always use the backbone final head",
    )
    parser.add_argument("--output", help="Optional path to dump the JSON result")
    return parser.parse_args()


def _print_result(result: Dict[str, Any], show_steps: bool = False) -> None:
    print("\n=== generation summary ===")
    print("prompt:", result["prompt"])
    print("\nGenerated answer:\n")
    print(result["generated_text"])
    print("\n=== timing & exit info ===")
    print("execution mode:", result.get("execution_mode", "legacy_unlabeled"))
    print("physically skips backbone layers:", result.get("physical_layer_skipping", "unknown"))
    print("decode-only latency (ms):", f"{result['total_latency_ms']:.2f}")
    if "end_to_end_latency_ms" in result:
        print("latency including initial probe (ms):", f"{result['end_to_end_latency_ms']:.2f}")
    print("probe latency (ms):", f"{result['probe_latency_ms']:.2f}")
    print(
        "initial exit:",
        result["initial_exit_forwarder_id"],
        "layer:",
        result["initial_exit_layer_index"],
    )
    print(
        "final exit:",
        result["final_exit_forwarder_id"],
        "layer:",
        result["final_exit_layer_index"],
    )
    if show_steps and result["per_step"]:
        print("\nper-step detail:")
        for step in result["per_step"]:
            print(
                f"  step {step['step']:02d} | token_id={step['token_id']} token={step['token']} | "
                f"latency_ms={step['step_latency_ms']:.2f} | logit={step['scores']['logit']:.2f} | "
                f"margin={step['scores']['margin']:.2f} | entropy={step['scores']['entropy']:.2f} | "
                f"exit={step['exit_forwarder_id']}@{step['exit_layer_index']}"
            )


def main() -> None:
    args = parse_args()
    generator = AutoRegressiveGenerator(
        manifest_path=Path(args.manifest),
        metadata_path=Path(args.metadata),
        thresholds_path=Path(args.thresholds) if args.thresholds else None,
        device=args.device,
        per_step_logging=not args.no_per_step,
        force_final_exit=args.force_final_exit,
    )
    result = generator.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        decode_temperature=args.temperature,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    _print_result(result, args.show_steps)


if __name__ == "__main__":
    main()
