using CUDA
using MIMOPotts

repo_root = dirname(@__DIR__)
default_instance = normpath(joinpath(repo_root, "..", "SATField.jl", "dataset", "MIMO", "mimo_instances", "A_dbg_2x2_qpsk_rayleigh.npz"))
instance_path = length(ARGS) >= 1 ? ARGS[1] : default_instance

isfile(instance_path) || error("MIMO instance not found: $(instance_path)")

CUDA.functional() || error("CUDA.jl is loaded, but no functional CUDA GPU is available.")

kwargs = (;
    snr_index = 1,
    instance_index = 1,
    free_dims = 4,
    fixed_candidates_per_dim = 1,
    max_branches = 8,
    trials = 32,
    num_cycles = 32,
    cycles_scaler = 1.0,
    noise_ratio = 0.5,
    optimizer = :batch,
    cacheCouplings = true,
    preprocess = :qr,
    seed = 11,
)

cpu_time = @elapsed cpu = solve_mimo_potts(instance_path; kwargs..., backend = :cpu)
gpu_time = @elapsed gpu = solve_mimo_potts(instance_path; kwargs..., backend = :cuda, gpuBatchSize = 32, gpuFloat = Float32)

println("instance: ", instance_path)
println("cpu seconds: ", cpu_time)
println("gpu seconds: ", gpu_time)
println("cpu bestDistance: ", cpu.best_distance)
println("gpu bestDistance: ", gpu.best_distance)
println("cpu BER/SER/FER: ", (cpu.ber, cpu.ser, cpu.fer))
println("gpu BER/SER/FER: ", (gpu.ber, gpu.ser, gpu.fer))
