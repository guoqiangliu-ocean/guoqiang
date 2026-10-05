# Implementation specification

Reproduction of **Xuan, Deng & Shen (2024) "Effect of an incoming Gaussian wave
packet on underlying turbulence", J. Fluid Mech. 999, A45** (doi:10.1017/jfm.2024.724).

This file is the single source of truth for every module.  Equation numbers
`(2.x)`, `(A x)`, `(D x)` refer to the paper.  Nondimensionalisation: `H = u_* = rho = 1`
(see `wpt/params.py`), so `nu = 1/Re_tau`, `g = 1/Fr^2`, surface stress `tau_0/rho = 1`,
time unit `H/u_*`.

Shared, already written (do not change their public API without updating this spec):
`wpt/params.py` (Params, presets `paper`, `reduced`, `tiny`, `CASES`),
`wpt/grid.py` (Grid: layout, FFTs, staggered vertical operators),
`wpt/wavefields.py` (WaveFields container, `zero_wave_fields`).

---------------------------------------------------------------------------
## 0. Physical problem (paper section 2)

Total velocity `U = u_phi + u`, `u_phi = grad(phi)`, `lap(phi) = 0` (2.3-2.4).
Rotational part (2.5)-(2.6):

    du/dt + (u.grad)u = -u_phi.grad(u) - u.grad(u_phi) - grad(p) + nu lap(u) - div(tau_sgs) + f
    div(u) = 0

with the uniform adverse mean pressure gradient `f = (-1, 0, 0)` (i.e. dp/dx = tau_0/H,
section 2.2), Coriolis and buoyancy neglected.

Wave: deep water, 2-D (x, z), evolved by HOS (Dommermuth & Yue 1987) independently of
the turbulence (no feedback, no wave viscous damping), eq. (2.11), (2.13).
Initial packet (2.17)-(2.19):

    eta(x,0)  = A0(x) cos(k0 x)
    phis(x,0) = A0(x) (omega0/k0) sin(k0 x)          (omega0 = sqrt(g k0))
    A0(x)     = a0 exp(-(x-x0)^2 / (2 chi^2))        (periodic distance x - x0 wrapped to [-Lx/2, Lx/2))
    a0 = alpha/k0,  chi = 1/(eps k0),  k0 H = 12,  eps = 0.13,  alpha in {0.12, 0.09, 0.06}

Base flow: shear-driven turbulent layer, Re_tau = 2000 (paper) / 500 (reduced preset),
surface stress 1 in +x, free-slip bottom (w = 0, du/dz = dv/dz = 0 at z = -H).

Boundary conditions at the mean surface z = 0 (Appendix A, truncated at O(alpha)):

    (A6)  w_s = d(eta u_s)/dx + d(eta v_s)/dy
    (A7)  du/dz|_s = tau_x^r/nu - dw_s/dx + 2 eta_x (2 du_s/dx + dv_s/dy)
                     + eta_y (du_s/dy + dv_s/dx) - eta (w_zx|_s + u_zz|_s)
    (A8)  dv/dz|_s = tau_y^r/nu - dw_s/dy + 2 eta_y (2 dv_s/dy + du_s/dx)
                     + eta_x (dv_s/dx + du_s/dy) - eta (w_zy|_s + v_zz|_s)

where `w_z|_s = -(du_s/dx + dv_s/dy)` (continuity), `w_zx|_s = -d/dx(du_s/dx + dv_s/dy)`,
`w_zy|_s = -d/dy(du_s/dx + dv_s/dy)`, `eta_y = 0` for the 2-D packet, and (2.15) to O(alpha)

    tau_x^r/nu = Re_tau * 1  -  [include_irrot_stress] * 2 phi_xz|_{z=0}
    tau_y^r/nu = 0

(`WaveFields.irrot_stress_x` holds `2 phi_xz|_{z=0}`.)  The subscript `s` denotes the value
at z = 0 of the rotational field.  (A7) is (A 7) of the paper solved for du/dz:
`tau^r/(rho nu) = (w_x+u_z) - 2 eta_x (u_x - w_z) - eta_y (u_y+v_x) + eta (w_zx + u_zz)`.

---------------------------------------------------------------------------
## 1. Grid and storage (see wpt/grid.py docstring)

* centres: u, v, p, nu_t, all diagnostics `(Nx, Ny, Nz)`; faces: w `(Nx, Ny, Nz+1)`,
  `w[...,0] = 0` (bottom), `w[...,Nz] = w_s` (surface, from A6).
* spectral layout `(Nx, Ny//2+1, nz)` from `grid.fft` (scipy rfft2 over axes (0,1),
  unnormalised forward).  `grid.dealias` = 2/3 mask.  Every prognostic spectral field
  is kept dealiased (truncated) at all times.
* `grid.dzc[k]` cell width, `grid.dzf[k]` centre-to-centre distance at face k
  (half cells at k = 0 and k = Nz).  `h_t = grid.dzf[-1] = -zc[-1]` is the distance
  from the top centre to the surface.

---------------------------------------------------------------------------
## 2. Wave solver — `wpt/hos.py`

```python
class HOS:
    def __init__(self, p: Params): ...      # N = p.hos_N points on [0, Lx), order M = p.hos_order
    x: np.ndarray        # (N,)
    k: np.ndarray        # (N//2+1,) rfft wavenumbers 2*pi*m/Lx
    eta: np.ndarray      # (N,) surface elevation
    phis: np.ndarray     # (N,) surface potential phi(x, eta(x))
    t: float
    def init_packet(self) -> None           # eq. (2.17)-(2.19) with p.alpha, p.eps_bw, p.x0
    def init_monochromatic(self, a: float, k: float) -> None   # linear progressive wave (tests/validation)
    def rhs(self, eta, phis) -> (deta_dt, dphis_dt)            # Zakharov/HOS equations
    def step(self, dt) -> None             # classical RK4
    def phi_modes(self) -> np.ndarray      # (N//2+1,) complex: sum_m of the HOS modal amplitudes,
                                           #  numpy.fft.rfft convention (unnormalised), such that
                                           #  phi(x, z) = irfft(phi_modes * exp(|k| z), n=N) for z <= 0
    def energy(self) -> float              # (1/Lx) * [ 0.5*g*int eta^2 + 0.5*int phis*W*(...) ] — see below
```

HOS (deep water, West et al. 1987 / Dommermuth & Yue 1987), evolution equations

    eta_t  = -eta_x phis_x + (1 + eta_x^2) W
    phis_t = -g eta - 0.5 phis_x^2 + 0.5 (1 + eta_x^2) W^2

with the order-M perturbation expansion `phi = sum_{m=1}^M phi^(m)`,
`phi^(1)|_0 = phis`, `phi^(m)|_0 = -sum_{l=1}^{m-1} eta^l/l! d^l phi^(m-l)/dz^l |_0`,
each `phi^(m)(x,z) = sum_k phihat^(m)_k exp(|k| z) exp(i k x)` (so `d^l/dz^l -> |k|^l`), and
`W = sum_{m=1}^{M} sum_{l=0}^{M-m} eta^l/l! d^{l+1} phi^(m)/dz^{l+1} |_0`.
Dealiasing: zero-pad to (M+1)/2 * N for products, or (simpler, acceptable) truncate
the spectrum of every product at |m| < N/(M+1)*... — the implementer must choose a
standard scheme and document it; a mild exponential low-pass filter on eta/phis
(e.g. keep |m| < 0.9 * N/2) after each RK4 step is allowed and must be documented.
Energy (deep water, per unit length): `E = (1/Lx) * int [ 0.5 g eta^2 + 0.5 phis * W_lin ] dx`
where `W_lin` is the first-order vertical velocity; for the test it is enough that the
relative change of the full HOS energy `0.5*g*eta^2 + 0.5*phis*(eta_t)` stays < 1e-4 over
20 periods for alpha = 0.12.

Required tests `tests/test_hos.py`: (i) small amplitude (alpha = 1e-3) monochromatic wave
matches linear theory phase after 10 periods; (ii) packet (alpha = 0.12 and 1e-3) centre of
wave-energy moves at c_g = c0/2 within 2 % over 20 periods; (iii) M = 3 Stokes wave
(alpha = 0.1) frequency correction omega = omega0 (1 + alpha^2/2) within 15 % of the
correction; (iv) energy conservation; (v) `phi_modes` gives the linear analytic
`u = a omega e^{kz} cos(kx - omega t)`, `w = a omega e^{kz} sin(kx - omega t)` at small alpha.
Use the physical parameters of `Params` (g = 1/Fr^2 ~ 7.15e6, k0 = 12) so that the time
step range used by the LES (dt ~ 1e-5, T0 ~ 6.8e-4) is exercised.

---------------------------------------------------------------------------
## 3. Wave fields on the LES grid (`wpt/coupling.py`) and surface BCs (`wpt/bc.py`)

```python
# wpt/coupling.py
def wave_fields_from_hos(hos: HOS, grid: Grid, p: Params) -> WaveFields
# wpt/bc.py  (owned by the LES implementer)
def surface_bc(u, v, w_top_prev, grid, wf: WaveFields, p: Params, g_prev=None) -> dict
def surface_w(u_s, v_s, eta, grid) -> np.ndarray
```
`wave_fields_from_hos` only needs `hos.phi_modes()`, `hos.eta`, `hos.t`, `hos.N` (tests may use a stub object).

`wave_fields_from_hos`: evaluate phi and its derivatives from `hos.phi_modes()` on the LES
x grid (Nx points; hos_N is a multiple of Nx).  Convert the HOS rfft coefficients
`c_m` (length N/2+1) to LES coefficients `(Nx/N) * c_m` for `m < Nx/2` and evaluate with
an Nx-point inverse rfft; additionally zero all `|m| >= Nx/3` (dealiasing of products).
For each retained mode, `exp(|k| z)` with z = zc (centres) and z = zf (faces).
Derivatives: `phi_x -> i k`, `phi_z -> |k|`, `phi_xx -> -k^2`, `phi_xz -> i k |k|`,
`phi_zz -> k^2`.  `eta`, `eta_x` from `hos.eta` with the same spectral resampling.
`irrot_stress_x = 2 phi_xz(x, 0)` if `p.include_irrot_stress` else 0.  If `p.wave_on` is
False return `zero_wave_fields`.

`surface_w(u_s, v_s, eta, grid)`: eq. (A6), `d(eta u_s)/dx + d(eta v_s)/dy` evaluated
pseudo-spectrally with dealiasing (inputs physical (Nx,Ny) and eta (Nx,1)); result has
zero horizontal mean.

`surface_bc(u, v, w_top_prev, grid, wf, p, g_prev)`: given physical rotational velocities
u, v at centres `(Nx,Ny,Nz)` (and `w_top_prev` = current surface w, `(Nx,Ny)`), return
dict with keys

* `'u_s', 'v_s'`: surface values `q_s = q[...,-1] + h_t * dqdz_s` (h_t = grid.dzf[-1]),
* `'dudz_s', 'dvdz_s'`: eq. (A7), (A8),
* `'w_s'`: eq. (A6) evaluated with the returned u_s, v_s and `wf.eta`.

Algorithm (explicit, O(alpha) corrections lagged): (1) provisional `q_s` from linear
extrapolation of the top two centres; (2) horizontal derivatives of q_s spectrally (with
dealiasing for products with eta); (3) `u_zz|_s` approximated by the top-cell second
derivative `[g - (u[-1]-u[-2])/dzf[-2]] / dzc[-1]` with `g = g_prev['dudz_s']` if given else
the extrapolation slope (same for v); (4) evaluate (A7), (A8); (5) recompute `q_s` with
the new gradient; (6) `w_s` from (A6).  `dw_s/dx` uses `w_top_prev`.

Tests `tests/test_coupling.py` (wave fields) and `tests/test_bc.py` (surface BCs): (i) wave fields vs analytic linear wave
(`phi = (a omega/k) e^{kz} sin(kx - omega t)`) at alpha = 1e-3: all 12 fields to 1e-6 rel.;
(ii) `dwphi_dz == -duphi_dx`, `duphi_dz == dwphi_dx`; (iii) (A6)-(A8) against a hand-coded
evaluation for analytic u(x,y,z), v, eta (e.g. trigonometric) on a fine grid with
second-order convergence; (iv) with eta = 0 and no irrotational stress, `dudz_s = Re_tau`,
`dvdz_s = 0`, `w_s = 0`.

---------------------------------------------------------------------------
## 4. LES solver — `wpt/les.py`

```python
class LESSolver:
    def __init__(self, p: Params, grid: Grid, sgs=None)   # sgs: object with .compute(...) (section 5) or None
    # state (all dealiased):
    uh, vh: complex (Nx, Nyh, Nz);  wh: complex (Nx, Nyh, Nz+1);  ph: complex (Nx, Nyh, Nz)
    u, v: real (Nx, Ny, Nz);  w: real (Nx, Ny, Nz+1);  p: real (Nx, Ny, Nz)
    t: float; nstep: int; bc: dict (last surface_bc result); nut: (Nx,Ny,Nz) or None
    def set_state(self, u, v, w, t=0.0) -> None    # dealias + project to div-free, compute bc
    def step(self, dt, wf_n: WaveFields, wf_np1: WaveFields) -> None
    def divergence(self) -> np.ndarray             # discrete div at centres (physical)
    def cfl(self, dt, wf=None) -> dict             # horizontal/vertical Courant numbers
    def save(self, path) / load(path)              # npz with u,v,w,t,nstep,AB2 history
```

### 4.1 Time advance (one step n -> n+1)
AB2 (Euler for the very first step or after `set_state`) for explicit terms; Crank-Nicolson
for the implicit vertical operator; pressure-correction projection (Kim & Moin 1985):

1. `bc_np1 = surface_bc(u^n, v^n, w_s^n, grid, wf_np1, p, g_prev=bc_n)` (lagged).
2. Explicit terms `E^n` (physical, then `grid.fft` + `grid.truncate`):
   * `-N(u)`: divergence form (section 4.2);
   * wave coupling `-C(u)` (section 4.3), *excluding* `wphi d/dz` if `p.implicit_wave_vadv`;
   * horizontal viscous `nu (d_xx + d_yy) u` (spectral, `-nu k^2`);
   * `-div(tau_sgs)` (section 5) when an SGS model is given;
   * mean pressure gradient `-1` in the x equation (horizontal-mean mode only).
3. Implicit vertical operator, per physical column (x,y), on the variable's own grid:
   `A q = nu d2q/dz2 - [implicit_wave_vadv] * wphi(x,z) dq/dz`.
   Solve `(I - dt/2 A^{n+1}) q* = q^n + dt (3/2 E^n - 1/2 E^{n-1}) + dt/2 A^n q^n`
   with vectorised Thomas algorithm (loop over k, vectorised over (Nx,Ny)).
   Note `A` depends on (x, z) through wphi -> the solve is done in physical space.
   * u, v (centres): Neumann at top with flux `dudz_s`/`dvdz_s` from `bc_np1` (used in both
     A^n and A^{n+1}), zero gradient at bottom.  Second derivative at centre k:
     `[ (q[k+1]-q[k])/dzf[k+1] - (q[k]-q[k-1])/dzf[k] ] / dzc[k]` with boundary face
     gradients replaced by the Neumann data.  First derivative at centre k: average of
     the two face gradients (`grid.ddz_c`).
   * w (faces 1..Nz-1 unknown; w[0] = 0, w[Nz] = w_s^{n+1} from bc_np1 Dirichlet):
     second derivative at face k `[ (w[k+1]-w[k])/dzc[k] - (w[k]-w[k-1])/dzc[k-1] ] / dzf[k]`,
     first derivative central `(w[k+1]-w[k-1])/(zf[k+1]-zf[k-1])`.
4. `q* -> spectral`, truncate.  Projection: for each (kx,ky) solve the tridiagonal
   `L_p phi = div(u*)/dt`, `L_p = Dz_f2c Gz_c2f - k^2` with homogeneous Neumann at both
   boundary faces (w* already carries the boundary values); for the (0,0) mode fix the
   gauge (e.g. phi[0] = 0 and mean-zero afterwards).  Update
   `u = u* - dt i kx phi`, `v = v* - dt i ky phi`, `w[1:-1] = w*[1:-1] - dt (phi[k]-phi[k-1])/dzf[k]`.
   The discrete divergence `i kx u + i ky v + (w[k+1]-w[k])/dzc[k]` must vanish to round-off.
   Store `ph = phi` (pressure up to O(dt)).
5. inverse FFT to refresh physical u, v, w, p; `t += dt`; keep `E^n` for AB2.

### 4.2 Nonlinear term N_i = d_j(u_j u_i) (divergence form, dealiased)
* x eq. (centres): `d_x(u u) + d_y(v u) + [w[k+1] uf[k+1] - w[k] uf[k]] / dzc[k]`,
  `uf = grid.c2f(u, bottom=u[...,0], top=bc['u_s'])` (bottom value irrelevant as w = 0).
* y eq.: same with v.
* z eq. (interior faces k = 1..Nz-1): `d_x(uf w) + d_y(vf w) + [wc[k]^2 - wc[k-1]^2] / dzf[k]`,
  `wc = grid.f2c(w)`.  Boundary faces: 0 (not used).
Products: transform dealiased spectra to physical, multiply, transform back, truncate.

### 4.3 Wave coupling C_i = uphi_j d_j u_i + u_j d_j uphi_i (advective form, eq. 2.5)
* x (centres): `uphi_c du/dx + wphi_c du/dz + u duphi_dx_c + wc duphi_dz_c`
* y (centres): `uphi_c dv/dx + wphi_c dv/dz`
* z (faces):   `uphi_f dw/dx + wphi_f dw/dz + uf dwphi_dx_f + w dwphi_dz_f`
with `du/dz` from `grid.ddz_c(u, 0, dudz_s)`, `dw/dz` from `grid.ddz_f`, horizontal
derivatives spectral.  The `wphi d/dz` parts are moved to the implicit operator when
`p.implicit_wave_vadv` (default True); otherwise they stay explicit.

### 4.4 Required tests `tests/test_les.py` (use `tiny`-like small grids, sgs=None)
(i) projection: random field -> divergence < 1e-10 * max|u|/dx;
(ii) exact viscous decay `u = exp(-nu (ky^2+kz^2) t) sin(ky y) cos(kz (z+1))`, kz = pi,
     v = w = 0, tau-forcing and mean pressure gradient switched off (helper flags), error
     O(dz^2) and O(dt^2);
(iii) laminar shear-driven layer with large nu (e.g. Re_tau = 2) converges to
     `u = Re_tau/2 (z+1)^2 + C` with `int u dz` conserved (= initial);
(iv) momentum budget: `d/dt int u dV = (1 - 1) * Lx Ly = 0` exactly in discrete form for a
     random divergence-free initial state without SGS (horizontal mean of x-momentum
     changes only through surface stress and forcing);
(v) with WaveFields from a linear wave and u = 0, tau-forcing off and irrot stress off,
     u stays exactly 0; with irrot stress on, the response is the linear Stokes layer
     (compare to analytic oscillatory boundary layer for a fine vertical grid);
(vi) energy: inviscid (nu -> 0 explicit-free variant) random field conserves kinetic
     energy to O(dt^2) over a few steps (skip if impractical; document).
For the solver to be testable add constructor/attribute switches
`self.surface_stress_on = True`, `self.mean_pgrad_on = True`.

---------------------------------------------------------------------------
## 5. SGS model — `wpt/sgs.py`

```python
class DynamicSmagorinsky:
    def __init__(self, grid: Grid, p: Params)
    def compute(self, u, v, w, bc: dict) -> dict
       # returns {'div_x': (Nx,Ny,Nz), 'div_y': (Nx,Ny,Nz), 'div_z': (Nx,Ny,Nz+1),
       #          'nut': (Nx,Ny,Nz), 'cs2': (Nz,)}
class Smagorinsky(DynamicSmagorinsky):  # constant Cs = p.cs_const with van Driest damping
                                       # using distance from the surface: (1 - exp(-d+/25))
```
`tau_ij = -2 nu_t S_ij` (deviatoric; the trace is absorbed in pressure), built from the
*rotational* velocity u only (the wave field is irrotational and fully resolved; document
this choice).  `S_ij` at centres: horizontal derivatives spectral; `du/dz`, `dv/dz` from
`grid.ddz_c` with surface data `bc['dudz_s']`, `bc['dvdz_s']` and zero at the bottom;
`dw/dz` from `grid.ddz_f2c(w)`; `dw/dx`, `dw/dy` at centres from `grid.f2c` of face values.
`Delta = (dx dy dzc)^(1/3)`.  Dynamic procedure (Germano 1991 / Lilly 1992): test filter =
sharp spectral cut-off at half the resolved horizontal wavenumbers (x and y only),
`Delta_hat/Delta = 2^(2/3)`; `L_ij = hat(u_i u_j) - hat(u_i) hat(u_j)`,
`M_ij = 2 Delta^2 [ hat(|S| S_ij) - (Delta_hat/Delta)^2 |hat S| hat(S_ij) ]`,
`Cs^2(z) = max(0, <L_ij M_ij>_xy / <M_ij M_ij>_xy)` (guard zero denominators),
`nu_t = Cs^2 Delta^2 |S|`.  Divergence of tau in staggered form: `d_x tau_11 + d_y tau_12 +
[tau_13f[k+1] - tau_13f[k]]/dzc[k]` etc., where `tau_13f` at faces uses face-interpolated
`nu_t` and `(du/dz)_f + (dw/dx)_f`; `tau_13f = tau_23f = 0` at the bottom and the surface
faces (the imposed surface stress is carried by the resolved viscous flux).  `div_z` at
interior faces uses centre `tau_33` differences and face `tau_13, tau_23`.
Tests `tests/test_sgs.py`: zero model for uniform/linear shear fields, nu_t >= 0,
known Cs for synthetic random field is finite, symmetry of tau, divergence of constant
nu_t * S equals nu_t * lap(u) for a smooth divergence-free field (consistency).

---------------------------------------------------------------------------
## 6. Statistics — `wpt/stats.py` (paper sections 3-5, Appendix D)

Ensemble + spanwise average `<f>(x,z,t)` (3.1); fluctuation `f' = f - <f>`.
Packet-following average (3.2) over `t in [t1, t2]` (defaults 0.01, 0.04) at
`x' = x - (x0 + cg t)` (periodic).

```python
class PacketAccumulator:      # one per ensemble member (run), mergeable
    def __init__(self, grid, p, sample_times, budget=True)
    def add_sample(self, i_sample, solver, wf: WaveFields) -> None
    def merge(self, other) -> None
    def save(self, path) / @classmethod load(path)
def packet_results(acc: PacketAccumulator, grid, p, window=3.5) -> dict   # final averaged quantities
class BaseAccumulator:        # base-flow (no wave) horizontal/time statistics
    add_sample(solver), save/load, results() -> profiles dict
```

Per sample time `t_i` (identical across runs) accumulate **ensemble sums over runs of
spanwise sums** (so that ensemble means are formed only at the end):
* first and second moments on the (x, z) plane at centres: `u, v, w(c), u^2, v^2, w^2, u w`,
  vorticity `ox, oy, oz` and squares, where `w` at centres = `grid.f2c(w)`,
  `ox = dw/dy - dv/dz`, `oy = du/dz - dw/dx`, `oz = dv/dx - du/dy` at centres (vertical
  derivatives via `grid.ddz_c` with surface data `solver.bc['dudz_s'], ['dvdz_s']`).
  Shape `(n_samples, Nx, Nz)`; counts `n_runs * Ny`.
* spanwise spectra for `my >= 1` (exclude my = 0; for even Ny exclude Nyquist): with
  `uh_y = rfft(u, axis=1)/Ny`, `Phi_u = 2 |uh_y|^2` (one-sided, so `sum_my Phi = <u'^2>_y`
  for zero-mean); same for v, w(c).  These need no ensemble mean, so they are
  **shifted to the packet frame immediately** (Fourier shift in x by `-(x0 + cg t_i)`)
  and summed over samples: shape `(Nx, nky, Nz)`.
* spectral budget terms (if `budget`), each `B = 4 Re[conj(uh_i) Fh]` with the
  y-transform `Fh` of the term F in the u'_i equation (D1), shifted to the packet frame
  and summed (shape `(Nx, nky, Nz)` per term and component):
  - `Pw_i`: F = `-u'_k d(uphi)_i/dx_k` (x: `-u duphi_dx_c - wc duphi_dz_c`;
    z: `-u dwphi_dx_c - wc dwphi_dz_c`; y: 0), all at centres;
  - `Pis_i` (pressure-strain): `4 Re[conj(d_i u_i hat) p hat]` (no sum; x: spectral d/dx;
    y: `i ky vh`; z: `dw/dz` at centres `grid.ddz_f2c(w)`); `sum_i Pis_i = 0` check;
  - `Tp_i` (pressure diffusion) for i = x, z: `-d/dx_i (4 Re[conj(uh_i) ph])` (x: spectral
    derivative in x of the correlation field; z: centred FD in z, one-sided at ends);
    `Tp_y = 0`;
  - `Aphi_i`: F = `-uphi_c du_i/dx - wphi_c du_i/dz`;
  - `Amean_i`: F = `-ubar du_i/dx - wbar du_i/dz` with `ubar, wbar` the spanwise mean of
    the run (approximation of the ensemble mean, document it);
  - `lhs_i`: computed at the end as `-cg d Phi_i/dx'` from the averaged spectra.
  All terms of component w use w at centres.
Normalisation of `packet_results` output (dict, saved with `np.savez`):
`xp` (x' within `|x'| <= window*chi`), `zc`, `k0z`, `ky`, `lam_y`, packet-frame averages of
`U, V, W, uu, vv, ww, uw` (Reynolds stresses = `<u^2> - <u>^2` formed per sample before
time averaging), `oxox, oyoy, ozoz` (enstrophy components likewise), spectra
`Phi_u, Phi_v, Phi_w` (shape `(nxp, nky, Nz)`), budget arrays `Pw_x, ...` and metadata
(`alpha, Re_tau, nu, omega0, k0, chi, cg, n_runs, n_samples`).  Packet-frame values at
`x'` are obtained by spectral (Fourier) interpolation.  Plotting conventions (paper):
`x'/chi`, `k0 z`, enstrophy / `(u_*^2/nu)^2 = Re_tau^2`, stresses / `u_*^2`, premultiplied
spectra `ky Phi / dky`, budget terms premultiplied and divided by `alpha u_*^2 omega0`,
changes Delta relative to the leading edge `x' = +3 chi`.

`BaseAccumulator`: plane+time averages of U(z), V, u'^2, v'^2, w'^2, u'w' (w' at centres),
`nu dU/dz`, SGS shear stress `<tau_13>` (if solver.nut available), total stress check
`-<u'w'> + nu dU/dz - <tau13> = (1 + z)` in equilibrium.

Tests `tests/test_stats.py`: synthetic fields with known answers (e.g. u = cos(3 y) ->
Phi_u at my = 3 equals 1/2; Reynolds stress of `<u>` + random ensembles; Fourier shift
exactness; pressure-strain sum zero for div-free synthetic field; merge == single big run).

---------------------------------------------------------------------------
## 7. Drivers and figures (`scripts/`)

* `run_base.py`: spin-up base flow (optionally on a coarse grid then spectrally
  interpolated), write `data/base/snap_XXX.npz` snapshots separated by `>= 1 H/u_*`, and
  base statistics.
* `run_packet.py --preset reduced --case W12 --members 0 1 ...`: for each member load a
  base snapshot, create HOS with the case alpha, integrate to `t2`, accumulate statistics
  at `n_samples` instants in `[t1, t2]` (choose the sampling interval so that the
  carrier phase `omega0 t/2` advances by an irrational-looking fraction of 2 pi), save
  `data/<case>/acc_member_XX.npz`; `combine.py` merges and writes
  `data/<case>/results.npz`.
* `plot_figures.py`: paper figures 2-17 and tables 1-2 analogues from the results files
  (plus base-flow validation figure), PNG into `figures/`.

---------------------------------------------------------------------------
## 8. Documented deviations from the paper (keep README in sync)

1. `reduced` preset: Re_tau = 500 (paper 2000), grid 192 x 128 x 48 (paper 512 x 256 x 112),
   fewer ensemble members (paper 30).
2. Vertical wave advection `w_phi d/dz` treated implicitly (CN) — the paper uses AB2 for
   all advection with presumably smaller dt.
3. SGS model built from the rotational velocity; dynamic coefficient plane-averaged.
4. `x0` not given in the paper; default `x0 = Lx/4`.
5. Budget advection by the mean rotational velocity uses each run's spanwise mean.
