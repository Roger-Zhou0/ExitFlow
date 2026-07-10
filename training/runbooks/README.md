# Runbook Scripts

This directory collects ready-to-use shell scripts for common workflows on
Perlmutter. They encapsulate the recommended environment variables and CLI
invocations, so you can re-run training/evaluation consistently.

Scripts included:

| Script | Description |
| ------ | ----------- |
| `train_exit_heads_perlmutter.sh` | Launch distributed training with `torchrun` (4 GPUs) and resume support. |
| `evaluate_exit_heads_perlmutter.sh` | Run the exit-head evaluation utility against a dataset. |

All scripts assume you have already obtained a GPU allocation (`salloc …`) and
that the repository root is the working directory. If you need to customise
paths (dataset, checkpoints, etc.), either export the environment variables
listed in each script or pass extra CLI arguments at the end of the command
invocation (they are forwarded to the underlying Python entrypoints).

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATASET_PATH` | `data/prepared_dataset/train` | Directory of training JSONL files |
| `CALIB_PATH` | `data/prepared_dataset/calibration` | Directory of calibration JSONL files |
| `OUTPUT_DIR` | `artifacts/…` | Export directory for trained heads |
| `CHECKPOINT_DIR` | `checkpoints/…` | Directory where checkpoints are written |
