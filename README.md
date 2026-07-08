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


## MIMO Benchmark Launching

The benchmark workflow is split into tuning and testing:

```bash
# 1. Tune hyperparameters on the front slice of every scenario.
bash scripts/launch_mimo_benchmarks.sh \
  --phase tune \
  --nodes gpu001,gpu002 \
  --gpus-per-node 4 \
  --time 04:00:00

# 2. After the tuning jobs finish, select one hyperparameter row per size class.
julia --project=. scripts/select_mimo_hyperparams.jl \
  results/mimo/tune \
  outfile=results/mimo/hyperparams.tsv

# 3. Test the remaining instances using the selected hyperparameters.
bash scripts/launch_mimo_benchmarks.sh \
  --phase test \
  --nodes gpu001,gpu002 \
  --gpus-per-node 4 \
  --hp-table results/mimo/hyperparams.tsv \
  --time 08:00:00
```

Use `--dry-run` first to inspect the jobs without submitting them. The SLURM launcher pins every job to one of the nodes passed through `--nodes`; for each node it schedules one CPU stream plus `--gpus-per-node` CUDA streams. GPU jobs request `--gres=gpu:1`, leaving SLURM to assign the physical GPU on that node.

The tuning phase uses size-class grids for `noiseRatio`, `cyclesScaler`, and `freeDims`. By default it samples `64` small, `32` medium, and `8` large instances from the front of each scenario, then the test phase starts after that tuning slice. Override these with `--tune-small`, `--tune-medium`, `--tune-large`, and the chunk-size flags shown by `scripts/launch_mimo_benchmarks.sh --help`.

## Plotting MIMO Results

Generate BER figures for every test suite, an all-scenario comparison, runtime scaling, and search diagnostics with:

```bash
julia --project=. scripts/plot_mimo_results.jl \
  input=results/mimo/test \
  outdir=results/mimo/figures
```

The driver loads compact JLD2 shards with Julia and renders PDF and PNG figures through Python with pandas and matplotlib. Quality metrics average the stochastic trials per channel instance; ZF and MMSE are counted once per instance. Steady-state timing excludes the first timed trial of each job because it includes compilation and backend initialization. The output directory also contains the aggregated TSV tables used for every figure.

Inspect tuning sensitivity for every scenario and size class with:

```bash
julia --project=. scripts/plot_mimo_hyperparams.jl \
  input=results/mimo/tune \
  hpTable=results/mimo/hyperparams.tsv \
  outdir=results/mimo/figures/hyperparams
```

These figures use mean BER as the primary metric. Each scenario receives one-dimensional profiles for `noiseRatio`, `cyclesScaler`, and `freeDims`: at every x-axis value, BER is minimized over the other two hyperparameters. Size-class figures apply the same profile definition after normalizing BER by each scenario's best grid value, preventing high-BER scenarios from dominating the selection diagnostic.

## Per-Trial Potts Energy Traces

Run the CPU current-state energy experiment for the 2x2 QPSK, 4x4 16-QAM, and 8x8 64-QAM scenarios with:

```bash
bash scripts/launch_mimo_energy_traces.sh --scheduler local
julia --project=. scripts/plot_mimo_energy_traces.jl \
  input=results/mimo/energy_traces \
  outdir=results/mimo/energy_traces/figures
```

The experiment uses eight instances, low/middle/high SNR, 32 independently seeded trials, all real dimensions free, 64 cycles, and noise ratios zero and one for `singleflip` and `batch`. The compact JLD2 files retain every state and exact current objective. The plotting step validates each energy directly against `||y-Hx||^2`, writes trial and monotonicity tables, and renders every trial separately without substituting a best-so-far trajectory.

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
