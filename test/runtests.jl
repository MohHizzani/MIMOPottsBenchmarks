using MIMOPotts
using DataFrames
using Test

@testset "benchmark repo can load solver" begin
    @test length(pam_levels("QPSK")) == 2
end

@testset "optional smoke dataset" begin
    repo_root = dirname(@__DIR__)
    instance_path = normpath(joinpath(repo_root, "..", "SATField.jl", "dataset", "MIMO", "mimo_instances", "A_dbg_2x2_qpsk_rayleigh.npz"))
    if isfile(instance_path)
        out = curunanmimoinstance(instance_path;
            hps = Dict(:noiseRatio => [0.5], :cyclesScaler => [1.0], :freeDims => [2]),
            trials = 1,
            numCycles = 2,
            snr_indices = 1,
            instance_indices = 1,
            fixed_candidates_per_dim = 1,
            max_branches = 1,
            showProgress = false,
            resultFormat = :compact,
        )
        @test nrow(out.runs) == 1
        @test nrow(out.trials) == 1
    else
        @info "Skipping dataset smoke test; instance not found" instance_path
    end
end
