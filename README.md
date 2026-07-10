# ExitFlow

### Confidence-gated early-exit inference for Llama 3.1 8B

ExitFlow is a research prototype that asks a practical systems question: **can an
LLM stop computing once it is already confident about the next token?** It
splits a frozen Llama 3.1 8B backbone into eight four-layer stages and attaches
small, distillation-trained exit heads at layers 4–28. A local forwarder uses
calibrated logit, margin, and entropy gates to either emit a token or route the
hidden state to the next stage.

The project spans the full ML-systems workflow: dataset preparation,
knowledge-distillation training, threshold calibration, artifact export,
multi-GPU segmented inference, and benchmark aggregation.

> **Project status:** research prototype. The lightweight pipeline and training
> utilities are unit tested; full experiments require gated access to the Llama
> 3.1 8B weights and CUDA hardware. The available benchmark evidence is reported
> below, including its limitations.

## Why it is interesting

- **Fine-grained adaptive compute.** Seven exits provide more choices than a
  shallow/deep two-model cascade.
- **Local decisions.** Each forwarder lives beside its segment, avoiding a
  centralized activation-routing bottleneck.
- **Quality-aware gating.** An exit must pass confidence, top-two margin, and
  entropy checks—not just a single threshold.
- **Frozen-backbone training.** Knowledge distillation trains lightweight exit
  heads without updating the 8B-parameter teacher.
- **End-to-end evaluation.** The runtime records exit depth and latency alongside
  task metrics to expose the quality/compute trade-off.

## Architecture

```mermaid
flowchart LR
    P[Prompt] --> R[Round-robin router]
    R --> S1[Layers 1–4]
    S1 -->|confident| E[Emit token]
    S1 -->|forward| S2[Layers 5–8]
    S2 -->|confident| E
    S2 -->|forward| D[Layers 9–28<br/>five more gated stages]
    D -->|confident| E
    D -->|forward| S8[Layers 29–32<br/>final LM head]
    S8 --> E
```

At each intermediate stage, the exit decision is:

```text
top_logit >= logit_threshold
and top1_minus_top2 >= margin_gate
and entropy <= entropy_gate
```

See [guidance.md](guidance.md) for the research motivation and
[training/docs/workflow.md](training/docs/workflow.md) for the experiment flow.

## Repository map

| Path | Purpose |
| --- | --- |
| `runtime/` | Segment workers, routing, confidence gates, and autoregressive inference |
| `training/` | Distillation, calibration, checkpointing, export, and SLURM jobs |
| `evaluation/` | Benchmark loading, scoring, latency, and exit-depth metrics |
| `config/` | Model topology, runtime placement, and calibrated thresholds |
| `scripts/` | Reproducible inference, aggregation, and plotting commands |
| `tests/` | Fast tests for gating, routing, checkpointing, and batching |

Generated datasets, model artifacts, checkpoints, credentials, and run logs are
intentionally excluded from version control.

## Quick start

Python 3.10+ is recommended.

```bash
git clone https://github.com/Roger-Zhou0/ExitFlow.git
cd ExitFlow
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt
pytest -q
```

The tests use small synthetic modules and do not download Llama weights.

## Reproduce the training workflow

1. Request access to `meta-llama/Llama-3.1-8B-Instruct` and authenticate with
   Hugging Face locally.
2. Download and tokenize a dataset using the scripts in `training/scripts/`.
3. Run a short smoke training job before scheduling a full GPU run.

```bash
python -m training.main \
  --manifest config/manifest.json \
  --dataset data/training/dataset_subset.jsonl \
  --output-dir artifacts/exit-heads-smoke \
  --checkpoints checkpoints/exit-heads-smoke \
  --calibration-path data/training/calibration_subset.jsonl \
  --thresholds-output artifacts/exit-heads-smoke/thresholds.json \
  --device cuda \
  --batch-size 1 \
  --max-steps 20
```

For dataset preparation, resume behavior, artifact layout, and HPC submission,
follow [training/docs/getting_started.md](training/docs/getting_started.md).

## Build the lightweight pipeline

```python
from pathlib import Path

from config.loader import load_manifest, load_thresholds
from runtime import RuntimeFactories, build_segmented_pipeline

manifest = load_manifest(Path("config/manifest.json"))
thresholds = load_thresholds(Path("config/thresholds.json"))
pipeline = build_segmented_pipeline(manifest, thresholds, RuntimeFactories())

result = pipeline.run([2.0, 1.8, 1.6])
print(result["exit_segment"])
```

This path demonstrates orchestration with the repository's compact test
artifacts. The native autoregressive path in `runtime/autoregressive.py` loads
the exported backbone and exit heads for real model inference.

## Evaluation

The evaluation layer supports IFEval, AlpacaEval 2, GSM8K, and an English MMLU
subset. A useful experiment should report at least:

- task accuracy or instruction compliance;
- average and distribution of exit depth;
- median and P95 end-to-end latency;
- hardware, precision, batch size, prompt length, and generation length;
- the threshold configuration used for the run.

Raw run output belongs under `logs/` and can be summarized with
`scripts/aggregate_mmlu_results.py`. Do not commit per-request logs or model
weights; publish a small, provenance-rich summary or release artifact instead.

## Results and claim status

One historical MMLU diagnostic run is recoverable from the original experiment
notes. It is reported here for provenance, not as evidence of a speedup: the
matched full-depth baseline and detailed latency logs were not retained in the
public mirror.

| Workload | Configuration | Accuracy | Exit-depth distribution | Median latency | P95 latency |
| --- | --- | ---: | --- | ---: | ---: |
| MMLU English subset (1,531 prompts) | Early-exit diagnostic | 23.91% (366/1,531) | L4 50.95%, L8 14.24%, L12 3.14%, L16 4.83%, L20 9.73%, L24 2.61%, L28 14.50% | 2,299 ms | 3,097 ms |
| MMLU full-depth baseline | Required for comparison | Not retained | Layer 32 by definition | Not retained | Not retained |
| GSM8K | No publishable run retained | — | — | — | — |

The recovered run used Llama 3.1 8B, the Dolly 15K `tr_0.6` threshold profile,
FP16 inference, gate temperature 2.0, decode temperature 0.8, batch size 1, and
at most 64 generated tokens per prompt on a Perlmutter-oriented CUDA runtime.
Its mean latency was 1,941 ms, its mean final exit depth was 10.96 layers, and
generation averaged 47.64 tokens per sample. The inference logs do not record
the exact GPU allocation, so the latency numbers are not portable across
systems. More importantly, no matched full-depth run survives, so the data
cannot establish a latency reduction or a no-accuracy-loss result.
Accordingly, **this repository does not claim a measured 25% latency reduction**.
Any resume or project description should use that number only if the original
matched run can be recovered and published with its hardware, baseline, prompt
and generation lengths, exit distribution, and latency summary.

The recovered evaluation logs use NERSC `/pscratch` paths, and the checked-in
SLURM jobs target **NERSC Perlmutter**. ORBIT is not identified in the recovered
run metadata and should not be named as the benchmark machine.

<details>
<summary>Dolly 15K <code>tr_0.6</code> gate configuration</summary>

| Exit layer | Minimum logit | Minimum margin | Maximum entropy |
| ---: | ---: | ---: | ---: |
| 4 | 5.1093 | 0.1577 | 11.3805 |
| 8 | 5.7695 | 0.1566 | 11.1596 |
| 12 | 6.6474 | 0.2030 | 11.0985 |
| 16 | 6.5248 | 0.2292 | 11.0756 |
| 20 | 5.5996 | 0.1406 | 11.1767 |
| 24 | 5.1861 | 0.3608 | 11.3411 |
| 28 | 6.1077 | 1.0184 | 11.3519 |

</details>

### Baseline definition

The appropriate comparison is the same model, runtime, precision, batch size,
prompt set, and generation settings with `--force-final-exit`, so every token
uses the layer-32 head. Because both paths disable KV caching, a future speedup
claim must be described as relative to this same-runtime, no-KV-cache baseline—not
relative to an optimized serving engine with KV caching.

## Limitations

- KV caching is disabled in the current research path, so prefixes are
  recomputed during autoregressive generation.
- Confidence thresholds are model- and workload-specific and require
  calibration before comparisons are meaningful.
- Multi-GPU placement in `config/manifest.json` targets a particular
  experimental topology and should be adapted to the host.
- Early exits can trade quality for latency; a baseline using the final head is
  essential for every reported result.

## Relation to prior work

This is an independent Hugging Face/PyTorch implementation and does not build on
the codebase of Chen et al., [*EE-LLM: Large-Scale Training and Inference of
Early-Exit Large Language Models with 3D Parallelism*](https://proceedings.mlr.press/v235/chen24ae.html)
(ICML 2024). Their Megatron-LM framework focuses on scaling early-exit training
and inference with 3D parallelism and KV-cache-compatible generation. ExitFlow
instead explores confidence gates colocated with small model segments and local
activation routing in a single-node research runtime. The rename avoids implying
that this repository is the official EE-LLM implementation.

## Contributing

Bug reports and focused improvements are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. For security
issues, follow [SECURITY.md](SECURITY.md) instead of filing a public issue.

## Acknowledgments

This project builds on PyTorch, Hugging Face Transformers and Datasets, and the
Llama 3.1 model family. Users are responsible for complying with the licenses
and terms of all models and datasets they use.

Developed through a Rutgers WINLAB research collaboration and published with
permission. This repository is a public mirror of the original collaborative
project; its history was intentionally consolidated before release to remove
credentials, generated logs, checkpoints, and datasets.

## License

The repository's original code is available under the [MIT License](LICENSE).
Third-party models and datasets remain subject to their own terms.
