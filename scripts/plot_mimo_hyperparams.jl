#!/usr/bin/env julia

using DataFrames
using MIMOPotts
using Statistics

function parse_args(argv)
    args = Dict{String, String}()
    for arg in argv
        occursin("=", arg) || error("Expected key=value argument, got: $(arg)")
        key, value = split(arg, "="; limit = 2)
        args[key] = value
    end
    return args
end

function result_files(path::AbstractString)
    isfile(path) && return [path]
    isdir(path) || error("Input path does not exist: $(path)")
    files = String[]
    for (root, _, names) in walkdir(path)
        for name in names
            endswith(name, ".jld2") && push!(files, joinpath(root, name))
        end
    end
    return sort(files)
end

scenario_name(source) = splitext(basename(String(source)))[1]
safe_std_error(x) = length(x) <= 1 ? 0.0 : std(x) / sqrt(length(x))

function write_tsv(path::AbstractString, df::DataFrame)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(String.(names(df)), '\t'))
        for row in eachrow(df)
            values = map(names(df)) do name
                value = row[name]
                ismissing(value) ? "" : replace(string(value), '\t' => ' ', '\n' => ' ')
            end
            println(io, join(values, '\t'))
        end
    end
end

args = parse_args(ARGS)
input = get(args, "input", joinpath("results", "mimo", "tune"))
hp_table = get(args, "hpTable", joinpath("results", "mimo", "hyperparams.tsv"))
outdir = get(args, "outdir", joinpath("results", "mimo", "figures", "hyperparams"))
python = get(args, "python", "python3")
files = result_files(input)
isempty(files) && error("No .jld2 files found under $(input)")
isfile(hp_table) || error("Selected hyperparameter table does not exist: $(hp_table)")

rows = NamedTuple[]
seen = Set{Tuple{String, Float64, Int, Float64, Float64, Int}}()
failed = Pair{String, String}[]

for (file_index, file) in enumerate(files)
    loaded = try
        load_mimo_potts_results(file; includeMetadata = true)
    catch err
        push!(failed, file => sprint(showerror, err))
        continue
    end
    loaded.format === :compact || continue
    runs = loaded.result.runs
    trials = loaded.result.trials
    isempty(runs) && continue
    size_class = String(get(loaded.metadata, "sizeClass", "unknown"))
    backend = String(get(loaded.metadata, "backend", "unknown"))
    trial_groups = Dict(first(g.runId) => g for g in groupby(trials, :runId))

    for run in eachrow(runs)
        haskey(trial_groups, run.runId) || continue
        trial = trial_groups[run.runId]
        scenario = scenario_name(run.source)
        key = (scenario, Float64(run.ebnodb), Int(run.instance_index),
               Float64(run.noiseRatio), Float64(run.cyclesScaler), Int(run.freeDims))
        if key in seen
            @warn "Skipping duplicate tuning run" scenario run.ebnodb run.instance_index file
            continue
        end
        push!(seen, key)
        steady = trial[trial.outerTrial .> minimum(trial.outerTrial), :]
        isempty(steady) && (steady = trial)
        push!(rows, (
            sizeClass = size_class,
            scenario = scenario,
            backend = backend,
            ebnodb = Float64(run.ebnodb),
            instanceIndex = Int(run.instance_index),
            noiseRatio = Float64(run.noiseRatio),
            cyclesScaler = Float64(run.cyclesScaler),
            freeDims = Int(run.freeDims),
            pottsBer = mean(trial.ber),
            pottsFer = mean(trial.fer),
            mmseBer = Float64(run.mmseBer),
            zfBer = Float64(run.zfBer),
            steadyTime = median(steady.stepTime),
            totalSteps = mean(trial.totalSteps),
        ))
    end
    file_index % 25 == 0 && println("Loaded $(file_index)/$(length(files)) files")
end

isempty(rows) && error("No compact tuning rows could be loaded from $(input)")
run_df = DataFrame(rows)

function summarize_hp(group)
    cuda = group[group.backend .== "cuda", :]
    timing = isempty(cuda) ? group : cuda
    return (
        nRuns = nrow(group),
        meanBer = mean(group.pottsBer),
        berSE = safe_std_error(group.pottsBer),
        meanFer = mean(group.pottsFer),
        meanMmseBer = mean(group.mmseBer),
        meanZfBer = mean(group.zfBer),
        medianCudaTime = median(timing.steadyTime),
        medianSteps = median(group.totalSteps),
        cudaRuns = nrow(cuda),
    )
end

summary = combine(
    groupby(run_df, [:sizeClass, :scenario, :noiseRatio, :cyclesScaler, :freeDims]),
    summarize_hp,
)
sort!(summary, [:sizeClass, :scenario, :freeDims, :noiseRatio, :cyclesScaler])

mkpath(outdir)
summary_path = joinpath(outdir, "hyperparameter_summary.tsv")
write_tsv(summary_path, summary)

plotter = joinpath(@__DIR__, "plot_mimo_hyperparams.py")
run(`$(python) $(plotter) --summary $(summary_path) --selected $(hp_table) --outdir $(outdir)`)

println("Processed $(length(files) - length(failed))/$(length(files)) files and $(nrow(run_df)) unique tuning runs")
println("Wrote hyperparameter figures and summaries to $(outdir)")
if !isempty(failed)
    println(stderr, "Failed to load $(length(failed)) files:")
    for (file, message) in failed
        println(stderr, "  $(file): $(message)")
    end
end
