#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
scheduler="local"
nodes=""
time_limit="02:00:00"
cpus_per_task=4
out_dir="$repo_root/results/mimo/energy_traces"
instance_dir="$repo_root/data/mimo_instances"

usage() {
  cat <<USAGE
Usage: $0 [--scheduler local|slurm] [--nodes nodeA,nodeB] [options]

Options:
  --scheduler NAME       local or slurm. Default: local.
  --nodes LIST           Required for slurm; comma-separated node names.
  --time HH:MM:SS        SLURM time limit. Default: 02:00:00.
  --cpus-per-task N      SLURM CPU count. Default: 4.
  --instance-dir DIR     Scenario directory. Default: data/mimo_instances.
  --out-dir DIR          Trace output directory. Default: results/mimo/energy_traces.
  --dry-run              Print commands without running or submitting.
USAGE
}

dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scheduler) scheduler="$2"; shift 2 ;;
    --nodes) nodes="$2"; shift 2 ;;
    --time) time_limit="$2"; shift 2 ;;
    --cpus-per-task) cpus_per_task="$2"; shift 2 ;;
    --instance-dir) instance_dir="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$scheduler" == "local" || "$scheduler" == "slurm" ]] || { echo "Invalid scheduler: $scheduler" >&2; exit 2; }
[[ "$scheduler" != "slurm" || -n "$nodes" ]] || { echo "--nodes is required for slurm" >&2; exit 2; }

scenarios=(
  certml_2x2_qpsk_rayleigh
  certml_4x4_16qam_rayleigh
  core_8x8_64qam_rayleigh
)
IFS=',' read -r -a node_list <<< "$nodes"
mkdir -p "$out_dir" "$out_dir/logs"

for index in "${!scenarios[@]}"; do
  scenario="${scenarios[$index]}"
  instance="$instance_dir/$scenario.npz"
  outfile="$out_dir/$scenario.jld2"
  [[ -f "$instance" ]] || { echo "Missing scenario: $instance" >&2; exit 1; }
  cmd=(julia "--project=$repo_root" "$script_dir/run_mimo_energy_traces.jl"
    "instance=$instance" "outfile=$outfile" "instances=1,2,3,4,5,6,7,8"
    "snrs=auto" "optimizers=singleflip,batch" "noiseRatios=0,1"
    "trials=32" "numCycles=64" "cyclesScaler=1" "preprocess=qr" "seed=20260612")

  if (( dry_run )); then
    printf '%q ' "${cmd[@]}"; printf '\n'
  elif [[ "$scheduler" == "local" ]]; then
    "${cmd[@]}"
  else
    node="${node_list[$((index % ${#node_list[@]}))]}"
    quoted="$(printf '%q ' "${cmd[@]}")"
    sbatch --parsable --job-name="trace_$scenario" --nodelist="$node" \
      --cpus-per-task="$cpus_per_task" --time="$time_limit" \
      --output="$out_dir/logs/$scenario-%j.log" --wrap="$quoted"
  fi
done
