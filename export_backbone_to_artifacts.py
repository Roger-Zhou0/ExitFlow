"""Export backbone files into an artifacts/<run>/backbone directory without loading the model into RAM.

This script uses huggingface_hub.snapshot_download to fetch the exact snapshot
of the base model specified by config/manifest.json, then copies those files
into the target artifacts directory and updates metadata.json to point
backbone_dir to that location. No training or parameter changes occur.

Usage:
  python export_backbone_to_artifacts.py \
    --manifest config/manifest.json \
    --artifacts-dir artifacts/trained-exit-heads_kd_round_final

Optional environment:
  HF_HOME / TRANSFORMERS_CACHE control the cache location used by Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from config.loader import load_manifest

try:
    from huggingface_hub import snapshot_download
except Exception as exc:  # pragma: no cover - optional import guard
    raise RuntimeError(
        "huggingface_hub is required. Please ensure transformers/huggingface_hub are installed in your env."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export backbone files into artifacts/<run>/backbone")
    parser.add_argument("--manifest", default="config/manifest.json", help="Path to manifest.json")
    parser.add_argument("--artifacts-dir", required=True, help="Artifacts directory for the run")
    parser.add_argument("--revision", help="Optional HF revision (branch/tag/commit) to pin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    repo_id = manifest["model"]["base_model"]

    print(f"[export] Resolving HF snapshot for: {repo_id}")
    snapshot_path = snapshot_download(repo_id=repo_id, revision=args.revision)
    print(f"[export] Snapshot path: {snapshot_path}")

    backbone_dir = artifacts_dir / "backbone"
    backbone_dir.mkdir(parents=True, exist_ok=True)

    # Copy snapshot contents into backbone directory (dirs_exist_ok ensures idempotency)
    print(f"[export] Copying files to: {backbone_dir}")
    # Copytree cannot copy into a non-empty directory unless dirs_exist_ok is True and Python >=3.8
    # To keep structure identical, copy file-by-file from snapshot.
    src = Path(snapshot_path)
    for item in src.iterdir():
        dest = backbone_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    # Update metadata.json with backbone_dir
    metadata_path = artifacts_dir / "metadata.json"
    metadata = {"backbone_dir": str(backbone_dir), "exit_heads": {}}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text("utf-8"))
        except Exception:
            pass
        metadata["backbone_dir"] = str(backbone_dir)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[export] Updated metadata: {metadata_path}")
    print("[export] Done.")


if __name__ == "__main__":
    main()

