using Pkg

repo_root = dirname(@__DIR__)
solver_path = normpath(joinpath(repo_root, "..", "MIMOPotts.jl"))
Pkg.develop(path = solver_path)
Pkg.instantiate()
