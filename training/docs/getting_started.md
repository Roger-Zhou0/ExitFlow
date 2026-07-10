# Exit-Head KD Training Guide

This guide walks you through the full training journey—from downloading benchmarks to exporting artifacts ready for segmented inference. Follow the numbered steps to reproduce the workflow on a workstation or high-performance cluster (HPC).

## End-to-End Checklist

1. Acquire raw corpora or benchmarks.
2. Convert them into tokenised JSONL datasets.
3. Verify and adjust manifest/training configuration.
4. Run a small-scale smoke test.
5. Launch full KD training (locally or via SLURM).
6. Monitor progress, checkpoint, and resume if needed.
7. Export artifacts, perform calibration, and validate with the runtime.

## 1. Prerequisites

- Python environment with `torch`, `transformers`, and `pytest` installed. The SLURM script assumes a Conda environment named `llama-kd`.
- Access to the base model defined in `config/manifest.json` (default: `meta-llama/Llama-3-8b-instruct`).
- Tokenised training and calibration datasets in JSONL format, each record containing `input_ids` and `labels` lists.
- Sufficient disk space for checkpoints and exported artifacts.

## 2. Prepare Datasets (Acquisition → Tokenisation)

**目标**：把任意原始语料转换成训练脚本需要的 `{"input_ids": [...], "labels": [...]}` JSONL。

1. **获取原始数据**
   - 公共基准：
     ```bash
     python training/scripts/download_benchmarks.py \
       --output-dir data/raw_benchmarks \
       --datasets ifeval alpacaeval2 gsm8k mmlu_en
     ```
     每个数据集会被写成带 `prompt`/`response` 字段的 JSONL。
   - 自有语料：自定义 JSONL 或纯文本均可，确保每条记录能拼出可读文本。

2. **统一分词 & 切块**
   - 对上一步的 JSONL/text 运行：
     ```bash
     python training/scripts/prepare_dataset.py \
       --input data/raw_benchmarks/ifeval.jsonl \
       --output data/training/dataset.jsonl \
       --tokenizer meta-llama/Llama-3.1-8b-instruct \
       --prompt-field prompt --response-field response \
       --max-length 2048 --stride 256 --append-eos
     ```
   - 脚本会调用指定 tokenizer，拼接 prompt/response、可选 BOS/EOS、按窗口截断，并生成 `input_ids`/`labels` 列表。

3. **整理与校验**
   - 快速检查输出是否为：
     ```json
     {"input_ids": [101, ...], "labels": [101, ...]}
     ```
   - 将训练/校准文件放入 `data/training/`（或在 CLI 里改路径）：
     - `data/training/dataset.jsonl`
     - `data/training/calibration.jsonl`
   - 统计 token 总量，保证不超过 manifest 中的 `token_budget`。需要冒烟时，可用 `--limit` 或 `head` 保留少量样本。

## 3. Review & Edit Configuration

- **Manifest (`config/manifest.json`)**: verify
  - `model.base_model` matches your backbone ID
  - `model.segments[*].forwarder` entries point to the desired exit layers
  - `training.knowledge_distillation` hyperparameters (temperature, loss weights)
  - `training.token_budget`, `training.optimizer`, optional `gradient_accumulation_steps`
- **Thresholds (`config/thresholds.json`)**: optional seed values; final calibrated thresholds overwrite this file if you supply `--thresholds-output` during training.

## 4. Dry-Run Locally (Optional)

在提交全量任务前走一遍快速「冒烟」：

1. **抽样构建缩略数据集**（约 10% 样本即可）：
   ```bash
   head -n $(( $(wc -l < data/training/dataset.jsonl) / 10 )) data/training/dataset.jsonl \
     > data/training/dataset_subset.jsonl
   head -n $(( $(wc -l < data/training/calibration.jsonl) / 10 )) data/training/calibration.jsonl \
     > data/training/calibration_subset.jsonl
   ```

2. **运行训练 CLI 并限制步数**：
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
     --save-interval 50 \
     --max-steps 20
   ```
   - `--max-steps 20` 让训练在几分钟内结束。
   - `--save-interval 50` 触发一次检查点保存，验证恢复路径。

3. **检查产出是否齐全**：
   - 命令行日志应打印 loss、step 等信息。
   - `ls artifacts/exit-heads-smoke` 应看到 `backbone/`、`exit_heads/`、`metadata.json`、`thresholds.json`。
   - 可用 runtime 的 `build_segmented_pipeline` 加载这些产物做一条推理，确保模型文件可用。

冒烟通过后，将 `--dataset` 等参数切回全量文件，并酌情调整 `--batch-size`、`--save-interval` 等，就可以启动正式训练。

## 5. Submit to HPC via SLURM

1. Adjust resource directives in `training/scripts/train_exit_heads.sbatch` (GPUs, CPUs, memory, walltime) to match your cluster policy.
2. Override paths using environment variables or edit defaults:
   ```bash
   export MANIFEST=/path/to/manifest.json
   export DATASET=/scratch/me/dataset.jsonl
   export CALIBRATION=/scratch/me/calibration.jsonl
   export OUTPUT_DIR=/scratch/me/artifacts
   export CHECKPOINT_DIR=/scratch/me/checkpoints
   sbatch training/scripts/train_exit_heads.sbatch
   ```
3. Monitor job logs under `logs/exit_head_kd_<jobid>.out` and `.err`.

## 6. Monitor & Resume Training

- Checkpoint files are written to the directory specified by `--checkpoints` every `--save-interval` optimiser steps.
- To resume a job after interruption, rerun the CLI or submit the SBATCH script with `RESUME=1` (export variable) to pass `--resume`:
  ```bash
  export RESUME=1
  sbatch training/scripts/train_exit_heads.sbatch
  ```
  The trainer automatically loads the most recent `step_*.pt` file.

## 7. Post-Training Outputs

After completion, you should see:

```text
artifacts/exit-heads/
├─ backbone/            # HuggingFace save_pretrained files or backbone.pt
├─ exit_heads/          # TorchScript exit heads keyed by forwarder ID
├─ metadata.json        # Maps forwarders to artifact paths
└─ thresholds.json      # (optional) calibrated gates
```

- Update `config/manifest.json` to reference the new TorchScript head paths and backbone directory if you plan to deploy the runtime on these artifacts.
- Copy checkpoints off scratch storage if you want to keep recovery points.

## 8. Validation (Recommended)

1. Run unit tests to ensure code integrity:
   ```bash
   pytest
   ```
2. Perform a runtime smoke test by loading the exported artifacts through `runtime.build_segmented_pipeline` and verifying early-exit behaviour with sample prompts.

Following these steps ensures a reproducible, resume-capable training workflow that produces artifacts consumable by the segmented inference runtime.
