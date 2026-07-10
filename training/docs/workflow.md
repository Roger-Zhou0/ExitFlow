# Training Workflow & Dataflow

This document traces the complete lifecycle of an exit-head knowledge distillation (KD) run, from configuration ingestion to artifact export. It explains how control flows across the training modules and how tensors, checkpoints, and metadata move through the system.

## 1. High-Level Stages

1. **Configuration resolution** – `training.main` parses CLI/SBatch parameters into `TrainingJobConfig`, which loads the project manifest.
2. **Model assembly** – `training.model_factory.load_model_bundle` instantiates the frozen backbone and constructs exit heads for each forwarder-defined exit point.
3. **Dataset materialisation** – raw corpora or downloaded benchmarks are first normalised to prompt/response JSONL via `training/scripts/download_benchmarks.py` (optional) and tokenised with `training/scripts/prepare_dataset.py`. The resulting token files are then loaded by `training.data_pipeline.build_training_dataset`, respecting the manifest token budget, and `training.utils.batch_samples` groups records for the configured batch size.
4. **KD optimisation loop** – `KnowledgeDistillationTrainer.train` runs teacher→student distillation with gradient accumulation, checkpointing, and optional resume.
5. **Artifact export** – `training.exporter.ArtifactExporter` writes backbone weights, TorchScript exit heads, and metadata for runtime integration.
6. **Threshold calibration (optional)** – calibration samples are replayed through the trained heads, thresholds are computed, and the resulting JSON is saved for runtime use.

Each stage feeds the next by producing well-defined artifacts (e.g., manifest dictionaries, sample batches, checkpoints, weight files) that later components consume.

## 2. Control Flow

```text
Download/Prepare scripts → tokenised JSONL → CLI / SBATCH → TrainingJobConfig → run_training_job()
    ├─ load_manifest()                       (model + training spec)
    ├─ load_model_bundle()                   (backbone + exit heads)
    ├─ build_training_dataset()              (tokenised samples)
    ├─ batch_samples()                       (per-step mini-batches)
    ├─ KnowledgeDistillationTrainer.train()  (KD loop)
    │     ↳ CheckpointManager.save()/restore()
    ├─ ArtifactExporter (backbone / exit heads / metadata)
    └─ Optional calibration → ThresholdCalibrator → thresholds.json
```

- **Entry point**: `python -m training.main` or the SLURM script resolves all command line arguments and constructs `TrainingJobConfig` (`training/main.py`).
- **Orchestrator**: `run_training_job` binds everything together (`training/job.py:54`). It is responsible for creating directories, instantiating configs, and sequencing the downstream tasks.

## 3. Model Assembly Pipeline

1. `TrainingJobConfig` extracts base-model ID and forwarder positions from the manifest.
2. `load_model_bundle` downloads/loads the HuggingFace causal LM. The function ties newly created linear exit heads to the base LM logits (duplicating `lm_head` weights when available) to ensure initial compatibility.
3. Each exit head is wrapped in an `ExitHeadSpec` detailing its identifier, target layer index, hidden size, and module reference. These specs are used consistently across training, checkpoints, and export.

Data artefacts produced:
- Frozen backbone (`AutoModelForCausalLM`) on the target device.
- List of `ExitHeadSpec` objects representing the KD-adjustable heads.

## 4. Dataset & Batch Flow

### 4.1 Source ingestion

- Upstream scripts (`download_benchmarks.py`, `prepare_dataset.py`) produce JSONL records shaped as `{"input_ids": [...], "labels": [...]}`. You can also craft these files manually.
- `build_training_dataset` reads those JSONL records, skips blanks, and enforces the manifest `token_budget`.
- Shuffling happens once per dataset realisation, respecting the `--no-shuffle` flag.

### 4.2 Batching and shapes

- `batch_samples` constructs mini-batches of size `batch_size`, returning dictionaries where `input_ids` and `labels` are lists of lists (each sub-list is a token sequence).
- Inside the trainer, each batch is converted into `torch.Tensor` objects. One-dimensional tensors are unsqueezed to maintain `[B, T]` shapes before being sent to the backbone.

### 4.3 Calibration samples

- Optional calibration passes reuse `build_calibration_dataset` to fetch deterministic slices of the corpus.
- `_collect_calibration_samples` forwards these sequences through the trained backbone and exit heads to capture logits for threshold computation.

## 5. KD Loop Mechanics

| Component | Responsibility |
|-----------|----------------|
| `KnowledgeDistillationTrainer._forward_batch` | Runs the frozen backbone with `output_hidden_states=True`, gathers teacher logits, and feeds each exit head. Applies temperature-scaled KL divergence plus CE loss (teacher logits at final token vs student logits per exit). |
| `gradient_accumulation_steps` | Allows multiple mini-batches to accumulate gradients before an optimiser step for large effective batch sizes. |
| `CheckpointManager` | Persists backbone state (optional), exit head weights, optimiser state, RNG state, epoch/step counters, and gradient accumulation offset. Enables job restarts on HPC. |
| Resume logic | On start, restore the latest checkpoint when `--resume` is set, resuming from the stored epoch and sample index. |

The backbone is frozen by default; only exit head parameters participate in optimisation.

## 6. Artifact Export

- **Backbone**: Saved via `save_pretrained` when available, otherwise as a plain state dict. Written to `<output_dir>/backbone/`.
- **Exit heads**: Each `ExitHeadSpec` is scripted (`torch.jit.script`) or traced (fallback) and saved to `<output_dir>/exit_heads/<forwarder_id>.pt`.
- **Metadata**: `metadata.json` maps forwarder IDs to TorchScript paths and records the backbone directory to help runtime bootstrap.

## 7. Threshold Calibration Flow (Optional)

1. Calibration dataset is replayed through the fine-tuned exit heads.
2. For each exit, logits are flattened (top token only if sequence-aware) and stored.
3. `ThresholdCalibrator` produces logit, margin, and entropy gates for each forwarder ID using percentile-based estimates.
4. The resulting dictionary is persisted to the path supplied by `--thresholds-output` (usually under the output directory). Runtime ingests this JSON to configure forwarder decisions.

## 8. Outputs & Runtime Integration

After a successful run, you can expect the following structure (default paths):

```text
artifacts/exit-heads/
├─ backbone/
│  └─ ... (HF save_pretrained files or backbone.pt)
├─ exit_heads/
│  ├─ forwarder@4.pt
│  ├─ forwarder@8.pt
│  ├─ forwarder@12.pt
│  ├─ forwarder@16.pt
│  ├─ forwarder@20.pt
│  ├─ forwarder@24.pt
│  └─ forwarder@28.pt
├─ metadata.json
└─ thresholds.json        (if calibration enabled)

checkpoints/exit-heads/
└─ step_XXXXXXXX.pt       (rolling checkpoints)
```

Runtime configuration (`config/manifest.json`) can then reference the exported TorchScript heads and backbone directory, ensuring seamless loading during segmented inference.
