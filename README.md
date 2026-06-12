# MIMOPottsBenchmarks

Benchmark and smoke-test workspace for `MIMOPotts.jl`.

## Setup

From this directory:

```bash
julia --project=. scripts/setup.jl
```

This develops the sibling solver package at `../MIMOPotts.jl`.

## Smoke Benchmark

By default the smoke script uses the debug MIMO dataset in the original SATField checkout if it exists:

```bash
julia --project=. scripts/run_smoke.jl
```

You can also pass an explicit dataset and output path:

```bash
julia --project=. scripts/run_smoke.jl /path/to/instance.npz results/my_run.jld2
```

The script writes compact JLD2 output with `runDF` and `trialDF`, loadable through `MIMOPotts.load_mimo_potts_results`.

## GPU Smoke Benchmark

If `CUDA.jl` is available in the active Julia environment and a CUDA GPU is functional, compare CPU and GPU backends on the smoke configuration:

```bash
julia --project=. scripts/run_gpu_smoke.jl
```

You can pass an explicit instance path as the first argument.

## MIMO Instance Generation

Generate MIMOPotts-compatible MIMO `.npz` instances directly in this benchmark repo:

```bash
python3 scripts/generate_mimo_instances.py --out-dir data/mimo_instances --preset smoke
```

The output contains the same keys used by `MIMOPotts.load_mimo_potts_instance`: `nt`, `nr`, `modulation`, `channel_model`, `ebnodb`, `no`, `H`, `y`, `x`, `Haug`, `yaug`, and `xaug`. Generated files also record `channel_normalization`, `correlation_rho`, `rician_k`, and `condition_number` metadata.

The default generator preset is now `all`, which contains the recommended benchmark suite:

- `core-paper`: conventional mid-size Rayleigh cases used for BER/SER comparisons.
- `certified-ml`: small exact-solvable cases for comparing against exhaustive or sphere-decoder ML references.
- `robustness`: correlated Rayleigh, Rician, ill-conditioned, and overloaded cases.
- `low-ber`: high-instance tail-SNR cases for stronger rare-error statistics.
- `scaling`: large massive/XL-MIMO stress cases with a smaller SNR grid.

The legacy Rayleigh-only 30-point SNR suite is preserved under `legacy-satfield-rayleigh`, `legacy-future-6g-rayleigh`, and `legacy-all`. The old preset names `satfield-rayleigh` and `future-6g-rayleigh` are kept as aliases for compatibility.

To generate the recommended suite with normalized channels:

```bash
bash scripts/generate_all_mimo_instances.sh data/mimo_instances
```

The wrapper uses `--channel-normalization by-nr` by default. Use `--channel-normalization none` to reproduce the legacy CN(0,1) channel convention, or `--channel-normalization by-nt` for per-receive-antenna signal-power scaling.

Custom scenarios are also supported:

```bash
python3 scripts/generate_mimo_instances.py \
  --preset custom \
  --name custom_16x32_64qam_corr07 \
  --nt 16 --nr 32 --modulation 64QAM \
  --channel-model CorrelatedRayleigh --correlation-rho 0.7 \
  --channel-normalization by-nr \
  --ebnodb 10,15,20,25 \
  --instances 100 \
  --out-dir data/mimo_instances
```
