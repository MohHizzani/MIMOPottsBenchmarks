#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

phase="tune"
scheduler="slurm"
nodes=""
gpus_per_node=4
instance_dir="$repo_root/data/mimo_instances"
out_dir="$repo_root/results/mimo"
hp_table="$repo_root/results/mimo/hyperparams.tsv"
project="$repo_root"
partition=""
account=""
time_limit="04:00:00"
cpus_per_task=8
julia_threads=8
dry_run=0
include_glob="*.npz"
optimizer="batch"

tune_small=64
tune_medium=32
tune_large=8
tune_chunk_small=16
tune_chunk_medium=8
tune_chunk_large=2

test_chunk_small=256
test_chunk_medium=128
test_chunk_large=16
trials_tune=3
trials_test=8

usage() {
  cat <<USAGE
Usage: $0 --phase tune|test --nodes nodeA,nodeB [options]

Required for SLURM:
  --nodes LIST              Comma-separated node list. Jobs are pinned to these nodes.

Common options:
  --phase PHASE             tune or test. Default: tune.
  --scheduler NAME          slurm or local. Default: slurm.
  --gpus-per-node N         Number of GPU streams to launch per node. Default: 4.
  --instance-dir DIR        Directory containing generated .npz instances.
  --out-dir DIR             Result root. Default: results/mimo.
  --hp-table FILE           Hyperparameter TSV for test phase.
  --include-glob GLOB       Instance filename glob. Default: *.npz.
  --optimizer NAME          Potts optimizer passed to solve_mimo_potts. Default: batch.
  --dry-run                 Print jobs without submitting/running.

SLURM options:
  -p, --partition NAME      Optional SLURM partition.
  --account NAME            Optional account.
  --time HH:MM:SS           Job time limit. Default: 04:00:00.
  --cpus-per-task N         CPU cores per task. Default: 8.
  --julia-threads N         JULIA_NUM_THREADS. Default: 8.

Tune/test sampling:
  --tune-small N            Instances per small scenario for tuning. Default: 64.
  --tune-medium N           Instances per medium scenario for tuning. Default: 32.
  --tune-large N            Instances per large scenario for tuning. Default: 8.
  --test-chunk-small N      Test chunk size for small scenarios. Default: 256.
  --test-chunk-medium N     Test chunk size for medium scenarios. Default: 128.
  --test-chunk-large N      Test chunk size for large scenarios. Default: 16.

Examples:
  $0 --phase tune --nodes gpu001,gpu002 --gpus-per-node 4 --dry-run
  julia --project=. scripts/select_mimo_hyperparams.jl results/mimo/tune outfile=results/mimo/hyperparams.tsv
  $0 --phase test --nodes gpu001,gpu002 --gpus-per-node 4 --hp-table results/mimo/hyperparams.tsv
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) phase="$2"; shift 2 ;;
    --scheduler) scheduler="$2"; shift 2 ;;
    --nodes) nodes="$2"; shift 2 ;;
    --gpus-per-node) gpus_per_node="$2"; shift 2 ;;
    --instance-dir) instance_dir="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --hp-table) hp_table="$2"; shift 2 ;;
    --project) project="$2"; shift 2 ;;
    -p|--partition) partition="$2"; shift 2 ;;
    --account) account="$2"; shift 2 ;;
    --time) time_limit="$2"; shift 2 ;;
    --cpus-per-task) cpus_per_task="$2"; shift 2 ;;
    --julia-threads) julia_threads="$2"; shift 2 ;;
    --include-glob) include_glob="$2"; shift 2 ;;
    --optimizer) optimizer="$2"; shift 2 ;;
    --tune-small) tune_small="$2"; shift 2 ;;
    --tune-medium) tune_medium="$2"; shift 2 ;;
    --tune-large) tune_large="$2"; shift 2 ;;
    --tune-chunk-small) tune_chunk_small="$2"; shift 2 ;;
    --tune-chunk-medium) tune_chunk_medium="$2"; shift 2 ;;
    --tune-chunk-large) tune_chunk_large="$2"; shift 2 ;;
    --test-chunk-small) test_chunk_small="$2"; shift 2 ;;
    --test-chunk-medium) test_chunk_medium="$2"; shift 2 ;;
    --test-chunk-large) test_chunk_large="$2"; shift 2 ;;
    --trials-tune) trials_tune="$2"; shift 2 ;;
    --trials-test) trials_test="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$phase" != "tune" && "$phase" != "test" ]]; then
  echo "--phase must be tune or test" >&2
  exit 2
fi

if [[ "$scheduler" == "slurm" && -z "$nodes" ]]; then
  echo "--nodes is required for --scheduler slurm" >&2
  exit 2
fi

if [[ "$phase" == "test" && ! -f "$hp_table" ]]; then
  echo "test phase requires an existing --hp-table: $hp_table" >&2
  exit 2
fi

mkdir -p "$out_dir/$phase" "$out_dir/logs"

IFS=',' read -r -a node_list <<< "$nodes"
if [[ "$scheduler" == "local" && -z "$nodes" ]]; then
  node_list=("localhost")
fi

devices=()
for node in "${node_list[@]}"; do
  [[ -z "$node" ]] && continue
  devices+=("$node:cpu:-1")
  for ((gpu=0; gpu<gpus_per_node; gpu++)); do
    devices+=("$node:cuda:$gpu")
  done
done

if [[ ${#devices[@]} -eq 0 ]]; then
  echo "No devices available after parsing --nodes/--gpus-per-node" >&2
  exit 2
fi

shell_quote() {
  printf '%q' "$1"
}

join_quoted() {
  local out=""
  for arg in "$@"; do
    if [[ -z "$out" ]]; then
      out="$(shell_quote "$arg")"
    else
      out="$out $(shell_quote "$arg")"
    fi
  done
  printf '%s' "$out"
}

instance_info() {
  python3 - "$1" <<'PYINFO'
import numpy as np
import sys
path = sys.argv[1]
with np.load(path) as data:
    nt = int(data['nt'])
    nr = int(data['nr'])
    instances = int(data['x'].shape[1])
    snrs = int(data['ebnodb'].shape[0])
print(f"{nt}\t{nr}\t{instances}\t{snrs}")
PYINFO
}

size_class() {
  local nt="$1" nr="$2"
  local real_dims=$((2 * nt))
  if (( nt <= 8 && nr <= 8 )); then
    echo "small"
  elif (( real_dims <= 64 && nr <= 64 )); then
    echo "medium"
  else
    echo "large"
  fi
}

choose_snrs() {
  local ns="$1"
  local mid=$(((ns + 1) / 2))
  if (( ns <= 3 )); then
    seq -s, 1 "$ns"
  else
    printf '1,%s,%s' "$mid" "$ns"
  fi
}

tune_count_for() {
  case "$1" in
    small) echo "$tune_small" ;;
    medium) echo "$tune_medium" ;;
    large) echo "$tune_large" ;;
    *) echo "$tune_medium" ;;
  esac
}

tune_chunk_for() {
  case "$1" in
    small) echo "$tune_chunk_small" ;;
    medium) echo "$tune_chunk_medium" ;;
    large) echo "$tune_chunk_large" ;;
    *) echo "$tune_chunk_medium" ;;
  esac
}

test_chunk_for() {
  case "$1" in
    small) echo "$test_chunk_small" ;;
    medium) echo "$test_chunk_medium" ;;
    large) echo "$test_chunk_large" ;;
    *) echo "$test_chunk_medium" ;;
  esac
}

submit_job() {
  local node="$1" backend="$2" gpu="$3" stem="$4" range="$5" snrs="$6" size="$7" instance="$8" index="$9"
  local safe_range="${range//:/-}"
  local job_name="mimo_${phase}_${stem}_${safe_range}_${backend}_${optimizer}"
  local outfile="$out_dir/$phase/${stem}_${safe_range}_${backend}_${optimizer}_${index}.jld2"
  local logfile="$out_dir/logs/${job_name}_%j.log"
  local trials="$trials_tune"
  [[ "$phase" == "test" ]] && trials="$trials_test"

  local cmd_args=(
    julia "--project=$project" "$script_dir/run_mimo_benchmark_job.jl"
    "phase=$phase"
    "instance=$instance"
    "outfile=$outfile"
    "size=$size"
    "backend=$backend"
    "snrs=$snrs"
    "instances=$range"
    "trials=$trials"
    "optimizer=$optimizer"
    "seed=20260612"
    "showProgress=true"
  )
  if [[ "$phase" == "test" ]]; then
    cmd_args+=("hpTable=$hp_table")
  fi

  local command
  command="$(join_quoted "${cmd_args[@]}")"

  if [[ "$scheduler" == "slurm" ]]; then
    local sbatch_args=(sbatch --parsable --exclusive --job-name="$job_name" --nodelist="$node" --cpus-per-task="$cpus_per_task" --time="$time_limit" --output="$logfile")
    [[ -n "$partition" ]] && sbatch_args+=(--partition="$partition")
    [[ -n "$account" ]] && sbatch_args+=(--account="$account")
    if [[ "$backend" == "cuda" ]]; then
      sbatch_args+=(--gres=gpu:1)
    fi
    sbatch_args+=(--export="ALL,JULIA_NUM_THREADS=$julia_threads")
    sbatch_args+=(--wrap="$command")
    if (( dry_run )); then
      printf 'DRYRUN '
      printf '%q ' "${sbatch_args[@]}"
      printf '\n'
    else
      "${sbatch_args[@]}"
    fi
  elif [[ "$scheduler" == "local" ]]; then
    if (( dry_run )); then
      if [[ "$backend" == "cuda" ]]; then
        printf 'DRYRUN CUDA_VISIBLE_DEVICES=%s JULIA_NUM_THREADS=%s %s\n' "$gpu" "$julia_threads" "$command"
      else
        printf 'DRYRUN JULIA_NUM_THREADS=%s %s\n' "$julia_threads" "$command"
      fi
    else
      if [[ "$backend" == "cuda" ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" JULIA_NUM_THREADS="$julia_threads" bash -lc "$command" &
      else
        JULIA_NUM_THREADS="$julia_threads" bash -lc "$command" &
      fi
    fi
  else
    echo "Unsupported --scheduler $scheduler" >&2
    exit 2
  fi
}

mapfile -t files < <(find "$instance_dir" -maxdepth 1 -type f -name "$include_glob" | sort)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No instance files found in $instance_dir matching $include_glob" >&2
  exit 2
fi

job_index=0
for instance in "${files[@]}"; do
  stem="$(basename "$instance" .npz)"
  IFS=$'\t' read -r nt nr total_instances total_snrs < <(instance_info "$instance")
  size="$(size_class "$nt" "$nr")"
  snrs="$(choose_snrs "$total_snrs")"

  if [[ "$phase" == "tune" ]]; then
    take="$(tune_count_for "$size")"
    (( take > total_instances )) && take="$total_instances"
    chunk="$(tune_chunk_for "$size")"
    start=1
    stop="$take"
  else
    take="$(tune_count_for "$size")"
    start=$((take + 1))
    stop="$total_instances"
    chunk="$(test_chunk_for "$size")"
    snrs=":"
    if (( start > stop )); then
      continue
    fi
  fi

  current="$start"
  while (( current <= stop )); do
    end=$((current + chunk - 1))
    (( end > stop )) && end="$stop"
    range="${current}:${end}"
    device="${devices[$((job_index % ${#devices[@]}))]}"
    IFS=':' read -r node backend gpu <<< "$device"
    submit_job "$node" "$backend" "$gpu" "$stem" "$range" "$snrs" "$size" "$instance" "$job_index"
    job_index=$((job_index + 1))
    current=$((end + 1))
  done
done

if [[ "$scheduler" == "local" && "$dry_run" == "0" ]]; then
  wait
fi

if (( dry_run )); then
  echo "Prepared $job_index jobs (dry run)."
else
  echo "Submitted/launched $job_index jobs."
fi
