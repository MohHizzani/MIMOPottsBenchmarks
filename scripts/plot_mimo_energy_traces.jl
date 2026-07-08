#!/usr/bin/env julia

using DataFrames
using JLD2
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

function trace_files(path)
    isfile(path) && return [path]
    isdir(path) || error("Trace input does not exist: $(path)")
    return sort([joinpath(path, name) for name in readdir(path) if endswith(name, ".jld2")])
end

function write_tsv(path, df::DataFrame)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(String.(names(df)), '\t'))
        for row in eachrow(df)
            println(io, join((replace(string(row[name]), '\t' => ' ', '\n' => ' ') for name in names(df)), '\t'))
        end
    end
end

args = parse_args(ARGS)
input = get(args, "input", joinpath("results", "mimo", "energy_traces"))
outdir = get(args, "outdir", joinpath("results", "mimo", "energy_traces", "figures"))
python = get(args, "python", "python3")
files = trace_files(input)
isempty(files) && error("No trace JLD2 files found under $(input)")
mkpath(outdir)

points_path = joinpath(outdir, "trace_points.tsv")
summary_rows = NamedTuple[]
violation_rows = NamedTuple[]
paired_initials = Dict{Tuple{String, Int, Int, Int}, Vector{Int}}()
max_energy_error = Ref(0.0)

open(points_path, "w") do points
    println(points, join(("scenario", "ebnodb", "snrIndex", "instance", "optimizer", "noiseRatio",
        "trial", "step", "energy", "changedCount", "changedDimension", "oldState", "newState"), '\t'))

    for file in files
        metadata = JLD2.load(file, "metadata")
        runs = JLD2.load(file, "traces")
        scenario = String(metadata["scenario"])
        instance_path = String(metadata["instancePath"])
        println("Processing $(scenario): $(length(runs)) configurations")

        for run in runs
            inst = load_mimo_potts_instance(instance_path;
                snr_index = run.snr_index, instance_index = run.instance_index)
            for trial in run.trials
                pair_key = (scenario, run.snr_index, run.instance_index, trial.trial)
                if haskey(paired_initials, pair_key)
                    paired_initials[pair_key] == trial.initial_states ||
                        error("Paired configurations have different initial states for $(pair_key)")
                else
                    paired_initials[pair_key] = copy(trial.initial_states)
                end

                energies = trial.energy
                changes = trial.changed_count
                length(energies) == length(changes) + 1 || error("Malformed trace length")
                deltas = diff(energies)
                changed_steps = findall(>(0), changes)
                uphill_steps = Int[]

                for step in eachindex(energies)
                    values = MIMOPotts.states_to_values(trial.states[:, step], inst.levels)
                    exact = MIMOPotts.mimo_distance(inst.H, inst.y, values)
                    err = abs(exact - energies[step])
                    max_energy_error[] = max(max_energy_error[], err)
                    isapprox(exact, energies[step]; rtol = 1e-10, atol = 1e-10) ||
                        error("Recorded energy mismatch in $(scenario), instance $(run.instance_index), trial $(trial.trial), step $(step - 1)")

                    transition = step == 1 ? 0 : step - 1
                    changed_count = transition == 0 ? 0 : changes[transition]
                    changed_dimension = transition == 0 ? 0 : trial.changed_dimension[transition]
                    old_state = transition == 0 ? 0 : trial.old_state[transition]
                    new_state = transition == 0 ? 0 : trial.new_state[transition]
                    println(points, join((scenario, run.ebnodb, run.snr_index, run.instance_index,
                        String(run.optimizer), run.noise_ratio, trial.trial, step - 1, energies[step],
                        changed_count, changed_dimension, old_state, new_state), '\t'))
                end

                for step in changed_steps
                    tolerance = max(1e-10, 1e-10 * abs(energies[step]))
                    if deltas[step] > tolerance
                        push!(uphill_steps, step)
                        if run.optimizer === :singleflip && iszero(run.noise_ratio)
                            push!(violation_rows, (;
                                scenario, ebnodb = run.ebnodb, snrIndex = run.snr_index,
                                instance = run.instance_index, trial = trial.trial, step,
                                changedDimension = trial.changed_dimension[step],
                                oldState = trial.old_state[step], newState = trial.new_state[step],
                                energyBefore = energies[step], energyAfter = energies[step + 1],
                                absoluteIncrease = deltas[step],
                                relativeIncrease = deltas[step] / max(abs(energies[step]), eps(Float64)),
                            ))
                        end
                    end
                end

                max_uphill = isempty(uphill_steps) ? 0.0 : maximum(deltas[uphill_steps])
                push!(summary_rows, (;
                    scenario, ebnodb = run.ebnodb, snrIndex = run.snr_index,
                    instance = run.instance_index, optimizer = String(run.optimizer),
                    noiseRatio = run.noise_ratio, trial = trial.trial, seed = trial.seed,
                    initialEnergy = first(energies), finalEnergy = last(energies),
                    totalSteps = length(changes), totalStateChanges = sum(changes),
                    changedSteps = length(changed_steps), uphillStateChanges = length(uphill_steps),
                    uphillFraction = isempty(changed_steps) ? 0.0 : length(uphill_steps) / length(changed_steps),
                    maximumUphillIncrease = max_uphill, monotone = isempty(uphill_steps),
                ))
            end
        end
    end
end

summary = DataFrame(summary_rows)
violations = isempty(violation_rows) ? DataFrame(
    scenario=String[], ebnodb=Float64[], snrIndex=Int[], instance=Int[], trial=Int[], step=Int[],
    changedDimension=Int[], oldState=Int[], newState=Int[], energyBefore=Float64[],
    energyAfter=Float64[], absoluteIncrease=Float64[], relativeIncrease=Float64[]) : DataFrame(violation_rows)

summary_path = joinpath(outdir, "trial_summary.tsv")
violations_path = joinpath(outdir, "singleflip_zero_noise_violations.tsv")
write_tsv(summary_path, summary)
write_tsv(violations_path, violations)

plotter = joinpath(@__DIR__, "plot_mimo_energy_traces.py")
run(`$(python) $(plotter) --points $(points_path) --outdir $(outdir)`)

hypothesis = summary[(summary.optimizer .== "singleflip") .& iszero.(summary.noiseRatio), :]
report_path = joinpath(outdir, "monotonicity_report.md")
open(report_path, "w") do io
    nonmonotone = count(!, hypothesis.monotone)
    uphill = sum(hypothesis.uphillStateChanges)
    changed = sum(hypothesis.changedSteps)
    uphill_rate = iszero(changed) ? 0.0 : round(100 * uphill / changed; digits=2)
    max_uphill = isempty(hypothesis.maximumUphillIncrease) ? 0.0 : maximum(hypothesis.maximumUphillIncrease)
    hypothesis_result = iszero(nonmonotone) && iszero(uphill) ? "true" : "false"
    println(io, "# Zero-noise single-flip monotonicity report")
    println(io)
    println(io, "The current-state monotonicity hypothesis is **$(hypothesis_result)** for the tested update rule.")
    println(io)
    println(io, "- Non-monotone trials: $(nonmonotone) / $(nrow(hypothesis))")
    println(io, "- Uphill state changes: $(uphill) / $(changed) ($(uphill_rate)%)")
    println(io, "- Maximum uphill objective increase: $(max_uphill)")
    println(io, "- Median final/initial objective ratio: $(median(hypothesis.finalEnergy ./ hypothesis.initialEnergy))")
    println(io, "- Trials ending below their initial objective: $(count(hypothesis.finalEnergy .< hypothesis.initialEnergy)) / $(nrow(hypothesis))")
    println(io)
    println(io, "| Scenario | Eb/N0 (dB) | Trials | Non-monotone | Uphill changes | Changed steps | Uphill rate |")
    println(io, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for group in groupby(hypothesis, [:scenario, :ebnodb]; sort=true)
        group_uphill = sum(group.uphillStateChanges)
        group_changed = sum(group.changedSteps)
        group_rate = iszero(group_changed) ? 0.0 : round(100 * group_uphill / group_changed; digits=2)
        println(io, "| $(first(group.scenario)) | $(first(group.ebnodb)) | $(nrow(group)) | $(count(!, group.monotone)) | $(group_uphill) | $(group_changed) | $(group_rate)% |")
    end
    println(io)
    println(io, "All curves and statistics use the actual current state at every step; no best-so-far trajectory is substituted.")
end
println("Maximum direct-energy validation error: $(max_energy_error[])")
println("Zero-noise single-flip trials: $(nrow(hypothesis))")
println("Non-monotone trials: $(count(!, hypothesis.monotone))")
println("Uphill state changes: $(sum(hypothesis.uphillStateChanges)) / $(sum(hypothesis.changedSteps))")
println("Violation details: $(violations_path)")
println("Report: $(report_path)")
println("Figures: $(outdir)")
