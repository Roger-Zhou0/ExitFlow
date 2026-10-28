"""CPU-only tests for diagnostic MMLU reporting and launcher contracts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import aggregate_mmlu_results as aggregate


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_log_files_sorted_by_numeric_question_id(tmp_path):
    for index in (100, 2, 10, 1):
        (tmp_path / f"question_{index:02d}.json").write_text("{}", encoding="utf-8")
    assert [p.stem for p in aggregate.iter_log_files(tmp_path)] == [
        "question_01", "question_02", "question_10", "question_100"
    ]


def test_mmlu_prediction_comes_from_first_token_not_later_summary():
    record = {
        "prompt": "A or B or C?",
        "predicted_choice": "A",  # legacy, possibly from a later step
        "choice_logits": {"A": 9.0, "B": 0.0, "C": 0.0, "D": 0.0},
        "per_step": [
            {
                "token": "C",
                "predicted_choice": "C",
                "choice_logits": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 0.0},
            },
            {"token": "A", "predicted_choice": "A"},
        ],
        "total_latency_ms": 40.0,
        "end_to_end_latency_ms": 52.0,
        "execution_mode": "full_backbone_exit_head_diagnostic",
        "physical_layer_skipping": False,
    }
    parsed = aggregate.parse_result(record)
    assert parsed["choice_pred"] == "C"
    assert parsed["total_latency"] == 52.0
    assert parsed["latency_scope"] == "probe_and_decode"
    assert parsed["physical_layer_skipping"] is False


def test_aggregate_refuses_incomplete_dataset(tmp_path, monkeypatch):
    gold = tmp_path / "gold.jsonl"
    gold.write_text('{"answer":"A"}\n{"answer":"B"}\n', encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "question_01.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["aggregate_mmlu_results.py", "--logs", str(logs),
         "--dataset", str(gold), "--output", str(tmp_path / "out.csv")],
    )
    with pytest.raises(ValueError, match="Incomplete or misnumbered"):
        aggregate.main()


@pytest.mark.parametrize("mode,expected_force", [
    ("diagnostic", False),
    ("full_depth", True),
])
def test_mmlu_launcher_preserves_flags_with_stub(mode, expected_force, tmp_path):
    """Exercise bash argument boundaries without requiring Llama or CUDA."""
    scripts = tmp_path / "scripts"
    runtime = tmp_path / "runtime"
    scripts.mkdir()
    runtime.mkdir()
    launcher = scripts / "run_mmlu_benchmark.sh"
    launcher.write_text(
        (REPO_ROOT / "scripts" / "run_mmlu_benchmark.sh").read_text("utf-8"),
        encoding="utf-8",
    )
    (runtime / "autoregressive_batch.py").write_text(
        "import json,os,sys\n"
        "open(os.environ['CAPTURE_ARGS'], 'w').write(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    for name in ("manifest.json", "metadata.json", "thresholds.json", "questions.jsonl"):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    captured = tmp_path / "args.json"
    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "MMLU_MODE": mode,
        "MANIFEST_PATH": str(tmp_path / "manifest.json"),
        "METADATA_PATH": str(tmp_path / "metadata.json"),
        "THRESHOLDS_PATH": str(tmp_path / "thresholds.json"),
        "QUESTIONS_PATH": str(tmp_path / "questions.jsonl"),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "CAPTURE_ARGS": str(captured),
        "DEVICE": "cpu",
        "SEED": "42",
    }
    subprocess.run(["bash", str(launcher)], env=env, capture_output=True,
                   text=True, check=True)
    args = json.loads(captured.read_text("utf-8"))
    assert ("--force-final-exit" in args) is expected_force
    assert args[args.index("--thresholds") + 1] == str(tmp_path / "thresholds.json")
    assert args[args.index("--questions") + 1] == str(tmp_path / "questions.jsonl")
    assert args[args.index("--seed") + 1] == "42"
    assert args[args.index("--output-dir") + 1] == str(tmp_path / "output")
