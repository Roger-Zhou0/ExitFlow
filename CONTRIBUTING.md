# Contributing

Thanks for improving ExitFlow. Keep changes focused, reproducible, and easy to
review.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r training/requirements.txt
pytest -q
```

## Pull requests

1. Create a branch from `main`.
2. Add or update tests for behavior changes.
3. Run the test suite locally.
4. Explain the motivation, implementation, and validation in the PR body.
5. Keep generated datasets, checkpoints, model weights, logs, credentials, and
   machine-specific paths out of the commit.

For experimental results, include the hardware, model revision, precision,
dataset version, sampling parameters, thresholds, and random seed. Prefer a
compact summary over raw per-request logs.

## Style

- Use type hints for public Python interfaces.
- Write docstrings that explain intent and constraints.
- Prefer `pathlib.Path` for filesystem paths.
- Keep runtime code separate from one-off analysis scripts.
- Use English for public comments and documentation.

## Reporting bugs

Include a minimal reproduction, expected and actual behavior, Python and CUDA
versions, and relevant configuration with secrets removed.
