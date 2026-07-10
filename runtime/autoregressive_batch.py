"""Batch-run autoregressive early-exit evaluation over a list of prompts."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUTO_PATH = REPO_ROOT / "runtime" / "autoregressive.py"
spec = importlib.util.spec_from_file_location("autoregressive_module", AUTO_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load autoregressive module from {AUTO_PATH}")
autoregressive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autoregressive)
AutoRegressiveGenerator = autoregressive.AutoRegressiveGenerator
_print_result = autoregressive._print_result


def parse_args():
    parser = argparse.ArgumentParser(description="Run autoregressive exits over multiple prompts")
    parser.add_argument("--manifest", required=True, help="Path to config/manifest.json")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json with exit heads")
    parser.add_argument("--thresholds", required=True, help="Threshold JSON for the chosen benchmark")
    parser.add_argument("--questions", required=True, help="Text file with one prompt per line")
    parser.add_argument("--output-dir", required=True, help="Directory to write per-prompt JSON outputs")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--device", default="cuda", help="Torch device string")
    parser.add_argument("--show-steps", action="store_true", help="Print per-step detail for each prompt")
    parser.add_argument("--no-per-step", action="store_true", help="Skip per-step logging to save memory")
    return parser.parse_args()


def load_questions(path):
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    if path.suffix == ".jsonl":
        prompts = []
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
                option_lines = []
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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(Path(args.questions))
    generator = AutoRegressiveGenerator(
        manifest_path=Path(args.manifest),
        metadata_path=Path(args.metadata),
        thresholds_path=Path(args.thresholds),
        device=args.device,
        per_step_logging=not args.no_per_step,
    )

    for index, prompt in enumerate(questions, start=1):
        print(f"\n--- prompt {index}/{len(questions)} ---")
        result = generator.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            decode_temperature=args.temperature,
        )
        _print_result(result, args.show_steps)
        out_path = output_dir / f"question_{index:02d}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
