#!/usr/bin/env julia

using DataFrames
using MIMOPotts

function parse_args(argv)
    out = Dict{String, String}()
    for arg in argv
        if !occursin("=", arg)
            error("Expected key=value argument, got: $(arg)")
        end
        k, v = split(arg, "="; limit = 2)
        out[k] = v
    end
    return out
end

getstr(args, key, default) = get(args, key, default)
getint(args, key, default) = parse(Int, get(args, key, string(default)))
getfloat(args, key, default) = parse(Float64, get(args, key, string(default)))
getbool(args, key, default) = lowercase(get(args, key, string(default))) in ("1", "true", "yes", "y")

function parse_indices(spec::AbstractString)
    s = strip(spec)
    if s == ":" || lowercase(s) == "all"
        return Colon()
    elseif occursin(":", s)
        parts = split(s, ":")
        length(parts) == 2 || error("Only start:stop ranges are supported, got $(spec)")
        return parse(Int, parts[1]):parse(Int, parts[2])
    else
        vals = [parse(Int, strip(x)) for x in split(s, ",") if !isempty(strip(x))]
        isempty(vals) && error("Empty index list: $(spec)")
        return vals
    end
end

function infer_size_class(inst::MIMOPottsInstance)
    real_dims = 2 * inst.nt
    if inst.nt <= 8 && inst.nr <= 8
        return "small"
    elseif real_dims <= 64 && inst.nr <= 64
        return "medium"
    else
        return "large"
    end
end

function capped(values, max_value)
    out = sort(unique([v for v in values if v <= max_value]))
    isempty(out) && push!(out, max_value)
    return out
end

function tune_hps(size_class::AbstractString, max_free_dims::Int)
    if size_class == "small"
        return Dict(
            :noiseRatio => [0.25, 0.5, 1.0],
            :cyclesScaler => [0.5, 1.0, 10.0, 100.0],
            :freeDims => capped([0, 2, 4, 8], max_free_dims),
        )
    elseif size_class == "medium"
        return Dict(
            :noiseRatio => [0.25, 0.5, 1.0],
            :cyclesScaler => [0.5, 1.0, 10.0, 100.0, 500.0],
            :freeDims => capped([0, 8, 16, 32], max_free_dims),
        )
    elseif size_class == "large"
        return Dict(
            :noiseRatio => [0.25, 0.5, 1.0],
            :cyclesScaler => [0.5, 1.0, 10.0, 100.0, 500.0, 1000.0],
            :freeDims => capped([0, 32, 64, 128, 256], max_free_dims),
        )
    else
        error("Unsupported size class $(size_class)")
    end
end

function default_num_cycles(size_class::AbstractString)
    size_class == "small" && return 64
    size_class == "medium" && return 128
    size_class == "large" && return 256
    error("Unsupported size class $(size_class)")
end

function default_max_branches(size_class::AbstractString)
    size_class == "small" && return 256
    size_class == "medium" && return 128
    size_class == "large" && return 64
    error("Unsupported size class $(size_class)")
end

function default_gpu_batch(size_class::AbstractString)
    size_class == "small" && return 256
    size_class == "medium" && return 512
    size_class == "large" && return 1024
    error("Unsupported size class $(size_class)")
end

function read_hp_table(path::AbstractString)
    rows = Dict{Tuple{String, String}, Dict{String, String}}()
    open(path, "r") do io
        header = String[]
        for (line_no, line) in enumerate(eachline(io))
            s = strip(line)
            isempty(s) && continue
            startswith(s, "#") && continue
            parts = split(s, '\t')
            if isempty(header)
                header = String.(parts)
                continue
            end
            length(parts) == length(header) || error("Bad TSV row $(line_no) in $(path)")
            row = Dict(header[i] => String(parts[i]) for i in eachindex(header))
            optimizer = get(row, "optimizer", "batch")
            rows[(row["sizeClass"], optimizer)] = row
        end
    end
    return rows
end

function test_hps(size_class::AbstractString, optimizer::Symbol, hp_table_path::AbstractString)
    isempty(hp_table_path) && error("phase=test requires hpTable=/path/to/hyperparams.tsv")
    table = read_hp_table(hp_table_path)
    key = (size_class, String(optimizer))
    haskey(table, key) || error("No hyperparameter row for sizeClass=$(size_class), optimizer=$(optimizer) in $(hp_table_path)")
    row = table[key]
    return Dict(
        :noiseRatio => [parse(Float64, row["noiseRatio"])],
        :cyclesScaler => [parse(Float64, row["cyclesScaler"])],
        :freeDims => [parse(Int, row["freeDims"])],
    )
end

args = parse_args(ARGS)
phase = getstr(args, "phase", "tune")
instance_path = getstr(args, "instance", "")
isempty(instance_path) && error("instance=/path/to/file.npz is required")
isfile(instance_path) || error("MIMO instance not found: $(instance_path)")

backend = Symbol(getstr(args, "backend", "cpu"))
if backend === :cuda
    @eval using CUDA
    CUDA.functional() || error("CUDA backend requested but CUDA.functional() is false")
end

inst0 = load_mimo_potts_instance(instance_path; snr_index = 1, instance_index = 1)
size_class = getstr(args, "size", "auto")
size_class = size_class == "auto" ? infer_size_class(inst0) : size_class
max_free_dims = 2 * inst0.nt

snr_indices = parse_indices(getstr(args, "snrs", "1"))
instance_indices = parse_indices(getstr(args, "instances", "1"))
outfile = getstr(args, "outfile", joinpath("results", "mimo", phase, "job.jld2"))

num_cycles = get(args, "numCycles", "auto") == "auto" ? default_num_cycles(size_class) : getint(args, "numCycles", 128)
fixed_candidates_per_dim = getint(args, "fixedCandidatesPerDim", 1)
max_branches = get(args, "maxBranches", "auto") == "auto" ? default_max_branches(size_class) : getint(args, "maxBranches", 128)
gpu_batch_size = get(args, "gpuBatchSize", "auto") == "auto" ? default_gpu_batch(size_class) : getint(args, "gpuBatchSize", 512)
trials = getint(args, "trials", phase == "tune" ? 3 : 8)
optimizer = Symbol(getstr(args, "optimizer", "batch"))
preprocess = Symbol(getstr(args, "preprocess", "qr"))
seed = get(args, "seed", "nothing") == "nothing" ? nothing : parse(Int, args["seed"])
show_progress = getbool(args, "showProgress", true)

hps = if phase == "tune"
    tune_hps(size_class, max_free_dims)
elseif phase == "test"
    test_hps(size_class, optimizer, getstr(args, "hpTable", ""))
else
    error("Unsupported phase $(phase); expected tune or test")
end

metadata = (; 
    script = "run_mimo_benchmark_job.jl",
    phase = phase,
    sizeClass = size_class,
    backend = String(backend),
    optimizer = String(optimizer),
    host = get(ENV, "HOSTNAME", ""),
    cudaVisibleDevices = get(ENV, "CUDA_VISIBLE_DEVICES", ""),
    instancePath = instance_path,
    snrs = getstr(args, "snrs", "1"),
    instances = getstr(args, "instances", "1"),
    hpTable = getstr(args, "hpTable", ""),
)

println("MIMO benchmark job")
println("  phase=$(phase) backend=$(backend) optimizer=$(optimizer) size=$(size_class)")
println("  instance=$(instance_path)")
println("  snrs=$(getstr(args, "snrs", "1")) instances=$(getstr(args, "instances", "1"))")
println("  hps=$(hps)")
println("  outfile=$(outfile)")

out = curunanmimoinstance(instance_path;
    hps = hps,
    trials = trials,
    numCycles = num_cycles,
    optimizer = optimizer,
    snr_indices = snr_indices,
    instance_indices = instance_indices,
    fixed_candidates_per_dim = fixed_candidates_per_dim,
    max_branches = max_branches,
    cacheCouplings = true,
    preprocess = preprocess,
    seed = seed,
    showProgress = show_progress,
    resultFormat = :compact,
    jld2file = outfile,
    saveMetadata = metadata,
    backend = backend,
    gpuBatchSize = gpu_batch_size,
    gpuFloat = Float32,
)

println("Saved $(outfile)")
println("runs: ", size(out.runs), " trials: ", size(out.trials))
