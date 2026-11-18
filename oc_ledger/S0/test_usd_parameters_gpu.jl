# S0 check: are arrays inside `UniformStokesDrift(...; parameters=p)` seen by GPU kernels
# after in-place updates between time steps, and does the function receive the RK3
# substage time (needed for C5, N_j(t) = N_j^n + (t - t^n) S_j^n)?
#
# Usage (on a Madar GPU node):   julia --project test_usd_parameters_gpu.jl GPU
#        (CPU reference run):    julia --project test_usd_parameters_gpu.jl CPU
#
# Common setup. u = v = w = 0 at t = 0, no advection, no closure, no buoyancy. The
# only u tendency is ∂t_uˢ(z, t, p), horizontally uniform, so the pressure stays zero.
#
# Test 1 (array update). ∂t_uˢ = Σₙ aₙ exp(2 kₙ z); aₙ is written into the device
# array between steps and is fixed inside a step. RK3 weights sum to 1, so
#   u(z) = Δt Σ_m Σₙ aₙ⁽ᵐ⁾ exp(2 kₙ z)            to round-off.
# Pass: error vs the per-step aₙ ~ 1e-12 and error vs the initial aₙ = O(1).
#
# Test 2 (substage time). ∂t_uˢ = c⁽ᵐ⁾ (t - tⁿ) Σₙ exp(2 kₙ z), with tⁿ and c⁽ᵐ⁾ held
# in 1-element device arrays and reset at the start of each step. Oceananigans' RK3
# is exact for forcing linear in t if the stage times are t, t+8Δt/15, t+2Δt/3:
#   u(z) = (Δt²/2) Σ_m c⁽ᵐ⁾ Σₙ exp(2 kₙ z)        to round-off.
# If the function only saw tⁿ, u would stay 0.
#
# Both updates are written through model.stokes_drift.parameters; the script also
# reports whether the model holds the same array objects as the user's `p`.

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
pass1 = err_updated < 1e-10 && err_frozen > 1e-2

# ---------------- Test 2: substage time ----------------
q = (; c = on_architecture(arch, [FT(0)]), tn = on_architecture(arch, [FT(0)]),
       k = on_architecture(arch, k_cpu))

@inline function linear_in_t_∂t_uˢ(z, t, q)
    s = zero(z)
    @inbounds for n in 1:length(q.k)
        s += exp(2 * q.k[n] * z)
    end
    return @inbounds q.c[1] * (t - q.tn[1]) * s
end

model2 = NonhydrostaticModel(; grid, stokes_drift = UniformStokesDrift(; ∂t_uˢ = linear_in_t_∂t_uˢ, parameters = q),
                             advection = nothing, closure = nothing, timestepper = :RungeKutta3)
c_model, tn_model = model2.stokes_drift.parameters.c, model2.stokes_drift.parameters.tn
shape = [sum(exp.(2 .* k_cpu .* zz)) for zz in z]
expected2 = zeros(FT, Nz)
for m in 1:Nsteps
    cm = FT(1e-3) * (1 + FT(0.3) * m)
    copyto!(c_model,  [cm])
    copyto!(tn_model, [FT(model2.clock.time)])
    time_step!(model2, Δt)
    expected2 .+= cm * Δt^2 / 2 .* shape
end
u2 = Array(interior(model2.velocities.u, 1, 1, :))
err_stage = maximum(abs, u2 .- expected2) / maximum(abs, expected2)
pass2 = err_stage < 1e-10

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
println("TEST 1 array update      = ", pass1 ? "PASS" : "FAIL")
@printf("rel. err, linear-in-t    = %.3e\n", err_stage)
println("TEST 2 substage time     = ", pass2 ? "PASS" : "FAIL")
println("RESULT                   = ", (pass1 && pass2) ?
        "PASS: parameters route usable for C5" :
        "FAIL: per plan, switch to option 2 (hand-written skew-symmetric forcing)")
