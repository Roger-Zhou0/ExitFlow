#!/usr/bin/env bash
set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
  echo "[error] jq is required for JSON manipulation" >&2
  exit 1
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <threshold_directory>" >&2
  exit 1
fi

threshold_dir="$1"
src="${threshold_dir}/thresholds.json"

if [[ ! -d "${threshold_dir}" ]]; then
  echo "[error] Threshold directory not found: ${threshold_dir}" >&2
  exit 1
fi

if [[ ! -f "${src}" ]]; then
  echo "[error] thresholds.json not found in ${threshold_dir}" >&2
  exit 1
fi

declare -A filters=(
  [margin]='with_entries(.value |= {margin_gate: .margin_gate, temperature_scale: .temperature_scale})'
  [entropy]='with_entries(.value |= {entropy_gate: .entropy_gate, temperature_scale: .temperature_scale})'
  [logit]='with_entries(.value |= {logit_threshold: .logit_threshold, temperature_scale: .temperature_scale})'
)

for variant in margin entropy logit; do
  target="${threshold_dir}/thresholds_${variant}.json"
  jq "${filters[${variant}]}" "${src}" > "${target}.tmp"
  mv "${target}.tmp" "${target}"
  echo "[info] Wrote ${target}"
done
