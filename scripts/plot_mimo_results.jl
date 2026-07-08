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
suite_name(scenario) = startswith(scenario, "scale_") || startswith(scenario, "scale_overloaded_") ? "scaling" :
    startswith(scenario, "robust_") || startswith(scenario, "robust_overloaded_") ? "robustness" :
    startswith(scenario, "lowber_") ? "lowber" :
    startswith(scenario, "certml_") ? "certml" :
    startswith(scenario, "core_") ? "core" : "other"

function dimensions(scenario::AbstractString)
    m = match(r"(?:^|_)(\d+)x(\d+)(?:_|$)", scenario)
    m !== nothing && return parse(Int, m.captures[1]), parse(Int, m.captures[2])
    m = match(r"(?:^|_)(\d+)Rx_(\d+)Users(?:_|$)", scenario)
    m !== nothing && return parse(Int, m.captures[2]), parse(Int, m.captures[1])
    return 0, 0
end

function modulation_order(scenario::AbstractString)
    occursin(r"(?:^|_)qpsk(?:_|$)"i, scenario) && return 4
    m = match(r"(?:^|_)(\d+)qam(?:_|$)"i, scenario)
    return m === nothing ? 0 : parse(Int, m.captures[1])
end

safe_mean(x) = isempty(x) ? NaN : mean(x)
safe_median(x) = isempty(x) ? NaN : median(x)
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
input = get(args, "input", joinpath("results", "mimo", "test"))
outdir = get(args, "outdir", joinpath("results", "mimo", "figures"))
python = get(args, "python", "python3")
files = result_files(input)
isempty(files) && error("No .jld2 files found under $(input)")

rows = NamedTuple[]
seen = Set{Tuple{String, Float64, Int}}()
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
    backend = String(get(loaded.metadata, "backend", "unknown"))
    trial_groups = Dict(first(g.runId) => g for g in groupby(trials, :runId))

    for run in eachrow(runs)
        haskey(trial_groups, run.runId) || continue
        trial = trial_groups[run.runId]
        scenario = scenario_name(run.source)
        key = (scenario, Float64(run.ebnodb), Int(run.instance_index))
        if key in seen
            @warn "Skipping duplicate test run" scenario run.ebnodb run.instance_index file
            continue
        end
        push!(seen, key)
        nt, nr = dimensions(scenario)
        order = modulation_order(scenario)
        bits_per_frame = nt > 0 && order > 0 ? nt * round(Int, log2(order)) : 0
        steady = trial[trial.outerTrial .> minimum(trial.outerTrial), :]
        isempty(steady) && (steady = trial)
        push!(rows, (
            scenario = scenario,
            suite = suite_name(scenario),
            backend = backend,
            nt = nt,
            nr = nr,
            modulationOrder = order,
            bitsPerFrame = bits_per_frame,
            ebnodb = Float64(run.ebnodb),
            instanceIndex = Int(run.instance_index),
            trials = nrow(trial),
            pottsBer = mean(trial.ber),
            pottsSer = mean(trial.ser),
            pottsFer = mean(trial.fer),
            zfBer = Float64(run.zfBer),
            mmseBer = Float64(run.mmseBer),
            steadyTime = median(steady.stepTime),
            meanTimeWithWarmup = mean(trial.stepTime),
            totalSteps = mean(trial.totalSteps),
            stepFoundBest = mean(trial.stepFoundBest),
            branchesGenerated = Float64(run.branchesGenerated),
            branchesVisited = mean(trial.branchesVisited),
            branchesPruned = mean(trial.branchesPruned),
            radiusUpdates = mean(trial.radiusUpdates),
            bestDistance = mean(trial.bestDistance),
            initialRadius = Float64(run.initialRadius),
        ))
    end
    file_index % 50 == 0 && println("Loaded $(file_index)/$(length(files)) files")
end

isempty(rows) && error("No compact test rows could be loaded from $(input)")
run_df = DataFrame(rows)

function summarize_quality(group)
    n_runs = nrow(group)
    trial_count = sum(group.trials)
    bits = first(group.bitsPerFrame)
    return (
        nt = first(group.nt), nr = first(group.nr),
        modulationOrder = first(group.modulationOrder),
        nRuns = n_runs, nTrials = trial_count,
        pottsBer = mean(group.pottsBer), pottsBerSE = safe_std_error(group.pottsBer),
        pottsSer = mean(group.pottsSer), pottsSerSE = safe_std_error(group.pottsSer),
        pottsFer = mean(group.pottsFer), pottsFerSE = safe_std_error(group.pottsFer),
        zfBer = mean(group.zfBer), zfBerSE = safe_std_error(group.zfBer),
        mmseBer = mean(group.mmseBer), mmseBerSE = safe_std_error(group.mmseBer),
        pottsBits = bits * trial_count, baselineBits = bits * n_runs,
        medianSteadyTime = median(group.steadyTime),
        meanSteps = mean(group.totalSteps),
        meanStepFoundBest = mean(group.stepFoundBest),
        meanBranchesGenerated = mean(group.branchesGenerated),
        meanBranchesVisited = mean(group.branchesVisited),
        meanBranchesPruned = mean(group.branchesPruned),
        meanRadiusUpdates = mean(group.radiusUpdates),
        meanDistanceRatio = mean(group.bestDistance ./ group.initialRadius),
    )
end

summary = combine(groupby(run_df, [:suite, :scenario, :ebnodb]), summarize_quality)
sort!(summary, [:suite, :scenario, :ebnodb])

function summarize_backend(group)
    return (
        nt = first(group.nt), nr = first(group.nr), nRuns = nrow(group),
        medianSteadyTime = median(group.steadyTime),
        q25SteadyTime = quantile(group.steadyTime, 0.25),
        q75SteadyTime = quantile(group.steadyTime, 0.75),
        meanSteps = mean(group.totalSteps),
        medianStepsPerSecond = median(group.totalSteps ./ group.steadyTime),
    )
end

backend_summary = combine(groupby(run_df, [:suite, :scenario, :ebnodb, :backend]), summarize_backend)
sort!(backend_summary, [:suite, :scenario, :backend, :ebnodb])

scenario_summary = combine(groupby(run_df, [:suite, :scenario]), group -> (
    nt = first(group.nt), nr = first(group.nr), modulationOrder = first(group.modulationOrder),
    nRuns = nrow(group), minEbN0 = minimum(group.ebnodb), maxEbN0 = maximum(group.ebnodb),
    meanPottsBer = mean(group.pottsBer), meanMmseBer = mean(group.mmseBer),
    meanZfBer = mean(group.zfBer), medianSteadyTime = median(group.steadyTime),
    medianPrunedFraction = median(group.branchesPruned ./ max.(group.branchesGenerated, 1)),
    medianDistanceImprovement = median(1 .- group.bestDistance ./ group.initialRadius),
    initialBestFraction = mean(group.stepFoundBest .== 0),
    medianBestStepFraction = median(group.stepFoundBest ./ max.(group.totalSteps, 1)),
))
sort!(scenario_summary, [:suite, :scenario])

mkpath(outdir)
summary_path = joinpath(outdir, "test_summary.tsv")
backend_path = joinpath(outdir, "backend_summary.tsv")
scenario_path = joinpath(outdir, "scenario_summary.tsv")
write_tsv(summary_path, summary)
write_tsv(backend_path, backend_summary)
write_tsv(scenario_path, scenario_summary)

plotter = joinpath(@__DIR__, "plot_mimo_results.py")
run(`$(python) $(plotter) --summary $(summary_path) --backend $(backend_path) --scenarios $(scenario_path) --outdir $(outdir)`)

println("Processed $(length(files) - length(failed))/$(length(files)) files and $(nrow(run_df)) unique instance/SNR runs")
println("Wrote figures and summaries to $(outdir)")
if !isempty(failed)
    println(stderr, "Failed to load $(length(failed)) files:")
    for (file, message) in failed
        println(stderr, "  $(file): $(message)")
    end
end
