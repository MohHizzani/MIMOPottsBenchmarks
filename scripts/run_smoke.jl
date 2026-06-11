using MIMOPotts

repo_root = dirname(@__DIR__)
default_instance = normpath(joinpath(repo_root, "..", "SATField.jl", "dataset", "MIMO", "mimo_instances", "A_dbg_2x2_qpsk_rayleigh.npz"))
instance_path = length(ARGS) >= 1 ? ARGS[1] : default_instance
outfile = length(ARGS) >= 2 ? ARGS[2] : joinpath(repo_root, "results", "smoke.jld2")

isfile(instance_path) || error("MIMO instance not found: $(instance_path)")

out = curunanmimoinstance(instance_path;
    hps = Dict(:noiseRatio => [0.5], :cyclesScaler => [1.0], :freeDims => [2]),
    trials = 2,
    numCycles = 2,
    optimizer = :batch,
    snr_indices = 1,
    instance_indices = 1,
    fixed_candidates_per_dim = 1,
    max_branches = 1,
    cacheCouplings = true,
    preprocess = :qr,
    seed = 5,
    showProgress = true,
    resultFormat = :compact,
    jld2file = outfile,
    saveMetadata = (; script = "run_smoke.jl", instance_path = instance_path),
)

println("Saved $(outfile)")
println("runs: ", size(out.runs), " trials: ", size(out.trials))
