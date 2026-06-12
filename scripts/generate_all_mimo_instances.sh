#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
out_dir="${1:-$repo_root/data/mimo_instances}"

if [[ $# -gt 0 ]]; then
  shift
fi

python3 "$script_dir/generate_mimo_instances.py" \
  --preset all \
  --out-dir "$out_dir" \
  --channel-normalization by-nr \
  --overwrite \
  "$@"
