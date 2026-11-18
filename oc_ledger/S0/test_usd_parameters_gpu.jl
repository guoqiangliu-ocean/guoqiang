# S0 check: are arrays inside `UniformStokesDrift(...; parameters=p)` seen by GPU kernels
# after in-place updates between time steps?
#
# Usage (on a Madar GPU node):   julia --project test_usd_parameters_gpu.jl GPU
#        (CPU reference run):    julia --project test_usd_parameters_gpu.jl CPU
#
# Design. u = v = w = 0 at t = 0, no advection, no closure, no buoyancy. The only
# u tendency is ∂t_uˢ(z, t, p) = Σₙ aₙ exp(2 kₙ z), horizontally uniform, so the
# pressure stays zero. aₙ changes between steps (copyto! into the device array)
# and stays fixed inside a step. RK3 weights sum to 1, so after N steps
#   u(z) = Δt Σ_m Σₙ aₙ⁽ᵐ⁾ exp(2 kₙ z)        exactly (to round-off).
# "updated" = the formula with the per-step aₙ; "frozen" = the formula with the aₙ
# at construction. Pass: err_updated ~ 1e-12 and err_frozen = O(1).
# The update is written through model.stokes_drift.parameters, and the script also
# reports whether the model holds the same array object as the user's `p`.

using Oceananigans
using Printf

arch_name = length(ARGS) > 0 ? ARGS[1] : "GPU"
if arch_name == "GPU"
    using CUDA
    CUDA.allowscalar(false)
    arch = GPU()
else
    arch = CPU()
end

FT = Float64
Nz, Lz = 32, 16.0
grid = RectilinearGrid(arch, FT; size=(4, 4, Nz), extent=(4, 4, Lz),
                       topology=(Periodic, Periodic, Bounded))

nb = 8
k_cpu = collect(range(FT(0.05), FT(0.5), length=nb))
a0    = fill(FT(1e-3), nb)
p = (; a = on_architecture(arch, copy(a0)), k = on_architecture(arch, k_cpu))

@inline function spectral_∂t_uˢ(z, t, p)
    s = zero(z)
    @inbounds for n in 1:length(p.a)
        s += p.a[n] * exp(2 * p.k[n] * z)
    end
    return s
end

sd = UniformStokesDrift(; ∂t_uˢ = spectral_∂t_uˢ, parameters = p)
model = NonhydrostaticModel(; grid, stokes_drift = sd, advection = nothing,
                            closure = nothing, timestepper = :RungeKutta3)

same_object = model.stokes_drift.parameters.a === p.a
a_model = model.stokes_drift.parameters.a

Δt, Nsteps = FT(0.1), 10
z = Array(znodes(grid, Center()))
profile(a) = [sum(a .* exp.(2 .* k_cpu .* zz)) for zz in z]
expected_updated = zeros(FT, Nz)
expected_frozen  = zeros(FT, Nz)

for m in 1:Nsteps
    a_new = a0 .* (1 + FT(0.5) * m) .* (1 .+ FT(0.1) .* (1:nb))
    copyto!(a_model, a_new)               # host -> device, no scalar indexing
    time_step!(model, Δt)
    expected_updated .+= Δt .* profile(a_new)
    expected_frozen  .+= Δt .* profile(a0)
end

u = Array(interior(model.velocities.u, 1, 1, :))
umax = maximum(abs, expected_updated)
err_updated = maximum(abs, u .- expected_updated) / umax
err_frozen  = maximum(abs, u .- expected_frozen)  / umax
pass = err_updated < 1e-10 && err_frozen > 1e-2

println("arch                     = ", arch_name)
println("Julia                    = ", VERSION)
println("Oceananigans             = ", pkgversion(Oceananigans))
if arch_name == "GPU"
    println("CUDA.jl                  = ", pkgversion(CUDA))
    println("device                   = ", CUDA.name(CUDA.device()))
    println("CUDA runtime / driver    = ", CUDA.runtime_version(), " / ", CUDA.driver_version())
end
println("model holds same array   = ", same_object)
@printf("rel. err vs updated a    = %.3e\n", err_updated)
@printf("rel. err vs frozen a     = %.3e\n", err_frozen)
println("RESULT                   = ", pass ? "PASS (in-place parameter updates are seen)" :
                                              "FAIL (switch to option 2: hand-written skew-symmetric forcing)")
