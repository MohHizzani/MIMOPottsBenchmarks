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
