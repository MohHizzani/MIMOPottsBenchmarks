#!/usr/bin/env julia

using JLD2
using MIMOPotts

function parse_args(argv)
    args = Dict{String, String}()
    for arg in argv
        occursin("=", arg) || error("Expected key=value argument, got: $(arg)")
        key, value = split(arg, "="; limit = 2)
        args[key] = value
    end
    return args
end

parse_int_list(spec) = [parse(Int, strip(x)) for x in split(spec, ',')]
parse_float_list(spec) = [parse(Float64, strip(x)) for x in split(spec, ',')]
parse_symbol_list(spec) = [Symbol(strip(x)) for x in split(spec, ',')]

function selected_snr_indices(path, spec)
    spec != "auto" && return parse_int_list(spec)
    data = MIMOPotts._npzread(path, ["ebnodb"])
    count = length(data["ebnodb"])
    return unique([1, cld(count, 2), count])
end

args = parse_args(ARGS)
instance_path = get(args, "instance", "")
isempty(instance_path) && error("instance=/path/to/scenario.npz is required")
isfile(instance_path) || error("MIMO scenario does not exist: $(instance_path)")

outfile = get(args, "outfile", joinpath("results", "mimo", "energy_traces", splitext(basename(instance_path))[1] * ".jld2"))
instances = parse_int_list(get(args, "instances", "1,2,3,4,5,6,7,8"))
snr_indices = selected_snr_indices(instance_path, get(args, "snrs", "auto"))
optimizers = parse_symbol_list(get(args, "optimizers", "singleflip,batch"))
noise_ratios = parse_float_list(get(args, "noiseRatios", "0,1"))
trials = parse(Int, get(args, "trials", "32"))
num_cycles = parse(Int, get(args, "numCycles", "64"))
cycles_scaler = parse(Float64, get(args, "cyclesScaler", "1"))
batch_rate = parse(Float64, get(args, "batchRate", "0.5"))
preprocess = Symbol(get(args, "preprocess", "qr"))
seed = parse(Int, get(args, "seed", "20260612"))

all(opt -> opt in (:singleflip, :batch), optimizers) ||
    error("Trace experiment supports only singleflip and batch")

println("MIMO current-energy trace")
println("  scenario=$(instance_path)")
println("  snrs=$(snr_indices) instances=$(instances) trials=$(trials)")
println("  optimizers=$(optimizers) noiseRatios=$(noise_ratios)")
println("  numCycles=$(num_cycles) cyclesScaler=$(cycles_scaler) preprocess=$(preprocess)")

traces = Any[]
total = length(snr_indices) * length(instances) * length(optimizers) * length(noise_ratios)
run_index = Ref(0)
for snr_index in snr_indices, instance_index in instances, optimizer in optimizers, noise_ratio in noise_ratios
    run_index[] += 1
    println("  [$(run_index[])/$total] snr=$snr_index instance=$instance_index optimizer=$optimizer noiseRatio=$noise_ratio")
    push!(traces, trace_mimo_potts(instance_path;
        snr_index = snr_index,
        instance_index = instance_index,
        trials = trials,
        num_cycles = num_cycles,
        cycles_scaler = cycles_scaler,
        noise_ratio = noise_ratio,
        noise_stepper = :linear,
        optimizer = optimizer,
        batch_rate = batch_rate,
        preprocess = preprocess,
        seed = seed,
    ))
end

metadata = Dict{String, Any}(
    "schemaVersion" => 1,
    "experiment" => "per-trial-current-energy",
    "scenario" => splitext(basename(instance_path))[1],
    "instancePath" => abspath(instance_path),
    "instances" => instances,
    "snrIndices" => snr_indices,
    "optimizers" => String.(optimizers),
    "noiseRatios" => noise_ratios,
    "trials" => trials,
    "numCycles" => num_cycles,
    "cyclesScaler" => cycles_scaler,
    "batchRate" => batch_rate,
    "preprocess" => String(preprocess),
    "seed" => seed,
)
mkpath(dirname(outfile))
JLD2.jldsave(outfile; metadata, traces)
println("Saved $(outfile)")
