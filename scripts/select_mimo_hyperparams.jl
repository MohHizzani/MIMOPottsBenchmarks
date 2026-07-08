#!/usr/bin/env julia

using DataFrames
using MIMOPotts
using Statistics

function parse_args(argv)
    out = Dict{String, String}()
    positional = String[]
    for arg in argv
        if occursin("=", arg)
            k, v = split(arg, "="; limit = 2)
            out[k] = v
        else
            push!(positional, arg)
        end
    end
    if !isempty(positional)
        out["input"] = positional[1]
    end
    return out
end

function result_files(path::AbstractString)
    if isfile(path)
        return [path]
    elseif isdir(path)
        files = String[]
        for (root, _, names) in walkdir(path)
            for name in names
                endswith(name, ".jld2") && push!(files, joinpath(root, name))
            end
        end
        return sort(files)
    else
        error("Input path does not exist: $(path)")
    end
end

function scalar_meta(meta, key, default = "unknown")
    val = get(meta, key, default)
    return String(val)
end

args = parse_args(ARGS)
input = get(args, "input", joinpath("results", "mimo", "tune"))
outfile = get(args, "outfile", joinpath("results", "mimo", "hyperparams.tsv"))
time_weight = parse(Float64, get(args, "timeWeight", "0.0"))
files = result_files(input)
isempty(files) && error("No .jld2 files found under $(input)")

rows = DataFrame()
for file in files
    loaded = load_mimo_potts_results(file; includeMetadata = true)
    loaded.format === :compact || continue
    runs = loaded.result.runs
    trials = loaded.result.trials
    isempty(runs) && continue
    joined = leftjoin(trials, runs; on = :runId, makeunique = true)
    joined[!, :sizeClass] .= scalar_meta(loaded.metadata, "sizeClass")
    joined[!, :backend] .= scalar_meta(loaded.metadata, "backend")
    if :optimizer in propertynames(joined)
        joined[!, :optimizer] = String.(joined.optimizer)
    else
        joined[!, :optimizer] .= scalar_meta(loaded.metadata, "optimizer", "batch")
    end
    joined[!, :resultFile] .= file
    global rows = isempty(rows) ? joined : vcat(rows, joined; cols = :union)
end

isempty(rows) && error("No compact tuning rows found in $(input)")

key_cols = [:sizeClass, :optimizer, :noiseRatio, :cyclesScaler, :freeDims]
gd = groupby(rows, key_cols)
summary = combine(gd,
    :ber => mean => :meanBer,
    :fer => mean => :meanFer,
    :bestDistance => mean => :meanBestDistance,
    :stepTime => mean => :meanStepTime,
    nrow => :observations,
)
summary[!, :score] = summary.meanBer .+ summary.meanFer .+ time_weight .* summary.meanStepTime
sort!(summary, [:sizeClass, :optimizer, :score, :meanBestDistance, :meanStepTime])

mkpath(dirname(outfile))
open(outfile, "w") do io
    println(io, join(["sizeClass", "optimizer", "noiseRatio", "cyclesScaler", "freeDims", "numCycles", "fixedCandidatesPerDim", "maxBranches", "trials", "gpuBatchSize", "observations", "meanBer", "meanFer", "meanBestDistance", "meanStepTime"], '\t'))
    for key in sort(unique(zip(summary.sizeClass, summary.optimizer)))
        size_class, optimizer = key
        sub = summary[(summary.sizeClass .== size_class) .& (summary.optimizer .== optimizer), :]
        best = sub[1, :]
        num_cycles = size_class == "small" ? 64 : size_class == "medium" ? 128 : 256
        max_branches = size_class == "small" ? 256 : size_class == "medium" ? 128 : 64
        gpu_batch = size_class == "small" ? 256 : size_class == "medium" ? 512 : 1024
        println(io, join([
            best.sizeClass,
            best.optimizer,
            best.noiseRatio,
            best.cyclesScaler,
            best.freeDims,
            num_cycles,
            1,
            max_branches,
            8,
            gpu_batch,
            best.observations,
            best.meanBer,
            best.meanFer,
            best.meanBestDistance,
            best.meanStepTime,
        ], '\t'))
    end
end

println("Wrote $(outfile)")
println("Selected hyperparameters:")
print(read(outfile, String))
