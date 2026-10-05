"""Tests of the LES solver (SPEC.md section 4.4)."""
import math

import numpy as np
import pytest

from wpt.grid import Grid
from wpt.les import LESSolver, thomas_factor, thomas_solve
from wpt.params import Params
from wpt.wavefields import WaveFields, zero_wave_fields


def make_params(**kw):
    base = dict(Re_tau=300.0, Lx=2 * math.pi, Ly=math.pi, Nx=16, Ny=16, Nz=16,
                z_stretch=1.5, workers=1, wave_on=True)
    base.update(kw)
    return Params(**base)


def linear_wave_fields(g: Grid, a, k, omega, t, irrot=True) -> WaveFields:
    """Analytic linear deep-water wave phi = (a omega/k) e^{kz} sin(kx - omega t)."""
    x = g.x.reshape(-1, 1, 1)
    th = k * x - omega * t
    out = {}
    for tag, z in (("c", g.zc), ("f", g.zf)):
        e = np.exp(k * z).reshape(1, 1, -1)
        out["uphi_" + tag] = a * omega * e * np.cos(th)
        out["wphi_" + tag] = a * omega * e * np.sin(th)
        pxx = -a * omega * k * e * np.sin(th)
        pxz = a * omega * k * e * np.cos(th)
        out["duphi_dx_" + tag] = pxx
        out["duphi_dz_" + tag] = pxz
        out["dwphi_dx_" + tag] = pxz.copy()
        out["dwphi_dz_" + tag] = -pxx
    th2 = th[:, :, 0]
    eta = a * np.cos(th2)
    eta_x = -a * k * np.sin(th2)
    irr = 2 * a * omega * k * np.cos(th2) if irrot else np.zeros_like(eta)
    return WaveFields(t=t, eta=eta, eta_x=eta_x, irrot_stress_x=irr, **out)


def random_state(g, seed=0, amp=1.0):
    rng = np.random.default_rng(seed)
    u = amp * rng.standard_normal((g.Nx, g.Ny, g.Nz))
    v = amp * rng.standard_normal((g.Nx, g.Ny, g.Nz))
    w = amp * rng.standard_normal((g.Nx, g.Ny, g.Nz + 1))
    return u, v, w


def kinetic_energy(s, g):
    # centre weights dzc, face weights dzf (w = 0 on the boundary faces)
    return 0.5 * (np.sum(s.u ** 2 * g.dzc) + np.sum(s.v ** 2 * g.dzc)
                  + np.sum(s.w ** 2 * g.dzf)) * g.dx * g.dy


# ---------------------------------------------------------------------------
def test_thomas():
    rng = np.random.default_rng(1)
    n = 9
    a = rng.uniform(-1, 0, (3, 1, n))
    c = rng.uniform(-1, 0, (3, 1, n))
    b = 3.0 + rng.uniform(0, 1, (3, 1, n))
    d = rng.standard_normal((3, 4, n))
    x = thomas_solve(thomas_factor(a, b, c), d.copy())
    for i in range(3):
        M = np.diag(b[i, 0]) + np.diag(a[i, 0, 1:], -1) + np.diag(c[i, 0, :-1], 1)
        for j in range(4):
            assert np.allclose(M @ x[i, j], d[i, j], atol=1e-13)


# (i) projection --------------------------------------------------------------
@pytest.mark.parametrize("stretch", [0.0, 2.0])
def test_projection_divergence_free(stretch):
    p = make_params(Nx=24, Ny=16, Nz=20, z_stretch=stretch)
    g = Grid(p)
    s = LESSolver(p, g)
    s.set_state(*random_state(g, 3))
    umax = max(np.abs(s.u).max(), np.abs(s.v).max(), np.abs(s.w).max())
    div = np.abs(s.divergence()).max()
    assert div < 1e-10 * umax / g.dx
    assert np.all(s.w[..., 0] == 0.0) and np.abs(s.w[..., -1]).max() < 1e-14
    # spectral fields stay truncated
    for a in (s.uh, s.vh, s.wh):
        assert np.abs(a[~np.broadcast_to(g.dealias, a.shape)]).max() == 0.0
    # projecting again leaves the field unchanged (idempotent)
    u0 = s.u.copy()
    s.set_state(s.u, s.v, s.w)
    assert np.abs(s.u - u0).max() < 1e-12 * umax
    # stays divergence free over steps (nonlinear, with waves)
    wf0 = linear_wave_fields(g, 0.01, 3.0, 50.0, 0.0)
    for n in range(3):
        wf1 = linear_wave_fields(g, 0.01, 3.0, 50.0, (n + 1) * 1e-3)
        s.step(1e-3, wf0, wf1)
        wf0 = wf1
    umax = max(np.abs(s.u).max(), np.abs(s.w).max())
    assert np.abs(s.divergence()).max() < 1e-10 * umax / g.dx


# (ii) exact viscous decay -----------------------------------------------------
def _decay_run(Nz, dt, T, stretch, discrete_exact=False):
    p = make_params(Re_tau=10.0, Nx=8, Ny=8, Nz=Nz, z_stretch=stretch)
    g = Grid(p)
    s = LESSolver(p, g, surface_stress_on=False, mean_pgrad_on=False)
    ky, kz = 2.0, math.pi
    Y = g.y.reshape(1, -1, 1)
    Z = g.zc.reshape(1, 1, -1)
    u0 = np.sin(ky * Y) * np.cos(kz * (Z + 1.0)) * np.ones((g.Nx, 1, 1))
    s.set_state(u0, 0 * u0, np.zeros((g.Nx, g.Ny, g.Nz + 1)))
    wf = zero_wave_fields(g)
    n = int(round(T / dt))
    for _ in range(n):
        s.step(dt, wf, wf)
    if discrete_exact:
        dz = g.dzc[0]
        lam = p.nu * (ky ** 2 + 4 * math.sin(kz * dz / 2) ** 2 / dz ** 2)
    else:
        lam = p.nu * (ky ** 2 + kz ** 2)
    ue = math.exp(-lam * s.t) * u0
    assert np.abs(s.v).max() < 1e-13 and np.abs(s.w).max() < 1e-13
    return np.abs(s.u - ue).max()


@pytest.mark.parametrize("stretch", [0.0, 1.5])
def test_viscous_decay_space(stretch):
    errs = [_decay_run(Nz, 2e-3, 0.2, stretch) for Nz in (8, 16, 32)]
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(2)]
    print("space", stretch, errs, orders)
    assert min(orders) > 1.8


def test_viscous_decay_time():
    errs = [_decay_run(16, dt, 0.4, 0.0, discrete_exact=True) for dt in (0.04, 0.02, 0.01)]
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(2)]
    print("time", errs, orders)
    assert min(orders) > 1.8


# (iii) laminar shear-driven layer --------------------------------------------
def test_laminar_shear_layer():
    p = make_params(Re_tau=2.0, Nx=8, Ny=8, Nz=16, z_stretch=0.0)
    g = Grid(p)
    s = LESSolver(p, g)
    rng = np.random.default_rng(5)
    u0 = 0.3 * np.cos(math.pi * (g.zc + 1)) * np.ones((g.Nx, g.Ny, 1)) + 0.7
    u0 = u0 + 0.05 * rng.standard_normal(u0.shape)
    s.set_state(u0, np.zeros_like(u0), np.zeros((g.Nx, g.Ny, g.Nz + 1)))
    M0 = np.mean(np.sum(s.u * g.dzc, axis=-1))
    wf = zero_wave_fields(g)
    dt = 0.01
    for _ in range(800):
        s.step(dt, wf, wf)
    M1 = np.mean(np.sum(s.u * g.dzc, axis=-1))
    assert abs(M1 - M0) < 1e-12
    prof = 0.5 * p.Re_tau * (g.zc + 1) ** 2
    prof += M0 - np.sum(prof * g.dzc)
    err = np.abs(s.u - prof).max()
    print("laminar err", err)
    assert err < 1e-6
    # continuous solution Re/2 (z+1)^2 + C, C = M0 - Re/6, to O(dz^2)
    cont = 0.5 * p.Re_tau * (g.zc + 1) ** 2 + M0 - p.Re_tau / 6.0
    assert np.abs(s.u.mean(axis=(0, 1)) - cont).max() < 0.5 * p.Re_tau * g.dzc[0] ** 2
    assert np.allclose(s.bc["dudz_s"], p.Re_tau)


# (iv) momentum budget ---------------------------------------------------------
@pytest.mark.parametrize("pgrad", [True, False])
def test_momentum_budget(pgrad):
    p = make_params(Re_tau=300.0, Nx=16, Ny=16, Nz=12, z_stretch=2.0)
    g = Grid(p)
    s = LESSolver(p, g, mean_pgrad_on=pgrad)
    s.set_state(*random_state(g, 7))
    wf = zero_wave_fields(g)
    area = g.Lx * g.Ly

    def mom():
        return np.sum(s.u * g.dzc) * g.dx * g.dy, np.sum(s.v * g.dzc) * g.dx * g.dy

    Mx0, My0 = mom()
    dts = [2e-3, 2e-3, 1e-3, 1.5e-3, 1.5e-3, 1.5e-3]          # variable step AB2 too
    for dt in dts:
        s.step(dt, wf, wf)
    Mx1, My1 = mom()
    expect = 0.0 if pgrad else sum(dts) * 1.0 * area       # nu * Re_tau = 1
    print("momentum", Mx1 - Mx0 - expect, My1 - My0)
    assert abs(Mx1 - Mx0 - expect) < 1e-12 * max(1.0, abs(Mx0))
    assert abs(My1 - My0) < 1e-12 * max(1.0, abs(My0))
    assert np.abs(s.w.mean(axis=(0, 1))).max() < 1e-14


# (v) linear wave: no response without forcing; Stokes layer with irrot stress --
def test_wave_zero_rotational_stays_zero():
    p = make_params(Nx=48, Ny=8, Nz=16, include_irrot_stress=False)
    g = Grid(p)
    s = LESSolver(p, g, surface_stress_on=False, mean_pgrad_on=False)
    z = np.zeros((g.Nx, g.Ny, g.Nz))
    s.set_state(z, z, np.zeros((g.Nx, g.Ny, g.Nz + 1)))
    k, a = 12.0, 0.12 / 12.0
    om = math.sqrt(p.g * k)
    dt = 1e-5
    wf0 = linear_wave_fields(g, a, k, om, 0.0, irrot=False)
    for n in range(5):
        wf1 = linear_wave_fields(g, a, k, om, (n + 1) * dt, irrot=False)
        s.step(dt, wf0, wf1)
        wf0 = wf1
    for f in (s.u, s.v, s.w, s.p):
        assert np.all(f == 0.0)


def _stokes_layer_error(dt_per_period, cn_both_new=False, Nz=64):
    p = make_params(Re_tau=100.0, Nx=8, Ny=4, Nz=Nz, z_stretch=3.0)
    g = Grid(p)
    s = LESSolver(p, g, surface_stress_on=False, mean_pgrad_on=False)
    s.cn_flux_both_new = cn_both_new
    a, k, om = 1e-6, 1.0, 200.0
    nu = p.nu
    S = -2 * a * om * k                       # dudz_s = -2 phi_xz(0)
    # exact linear solution of the unsteady Stokes equations for u ~ exp(i(kx - om t)) with
    # u_z(0) = S, w(0) = 0, free-slip bottom: oscillatory Stokes layer A exp(mz),
    # m^2 = k^2 - i om/nu, plus the irrotational correction (relative size ~ k delta)
    # B cosh(k(z+1)) that restores w(0) = 0.
    m = np.sqrt(k ** 2 - 1j * om / nu)
    m = m if m.real > 0 else -m
    A = S * m / (m ** 2 - k ** 2)
    B = 1j * A / (m * math.sinh(k))

    def prof(z):
        return A * np.exp(m * z) + 1j * k * B * np.cosh(k * (z + 1.0))

    def u_exact(t):
        X = g.x.reshape(-1, 1, 1)
        Z = g.zc.reshape(1, 1, -1)
        return np.real(prof(Z) * np.exp(1j * (k * X - om * t))) * np.ones((1, g.Ny, 1))

    u0 = u_exact(0.0)
    ux = np.real(1j * k * prof(g.zc.reshape(1, 1, -1))
                 * np.exp(1j * k * g.x.reshape(-1, 1, 1))) * np.ones((1, g.Ny, 1))
    w0 = np.zeros((g.Nx, g.Ny, g.Nz + 1))
    w0[..., 1:] = -np.cumsum(ux * g.dzc, axis=-1)
    w0[..., -1] = 0.0
    wf0 = linear_wave_fields(g, a, k, om, 0.0)
    s.set_state(u0, np.zeros_like(u0), w0, wf=wf0)
    T = 2 * math.pi / om
    dt = T / dt_per_period
    nsteps = 2 * dt_per_period
    for n in range(nsteps):
        wf1 = linear_wave_fields(g, a, k, om, (n + 1) * dt)
        s.step(dt, wf0, wf1)
        wf0 = wf1
    ue = u_exact(s.t)
    return np.abs(s.u - ue).max() / np.abs(ue).max(), s


def test_stokes_layer_irrot_stress():
    err32, _ = _stokes_layer_error(100, Nz=32)
    err, s = _stokes_layer_error(100, Nz=64)
    print("stokes layer rel err Nz=32, 64:", err32, err)
    assert err < 3e-3
    assert err32 / err > 3.0                  # ~second order in dz
    # SPEC variant (dudz_s^{n+1} in both CN halves) is only first order in time
    e1, _ = _stokes_layer_error(100, cn_both_new=True)
    e2, _ = _stokes_layer_error(200, cn_both_new=True)
    print("stokes layer rel err (SPEC CN flux), 100 / 200 steps per period:", e1, e2)
    assert err < e1 and 1.6 < e1 / e2 < 2.4


# (vi) inviscid energy conservation -------------------------------------------
def _energy_change(dt, T=0.02):
    p = make_params(Re_tau=1e30, Nx=16, Ny=16, Nz=16, z_stretch=0.0)
    g = Grid(p)
    s = LESSolver(p, g, surface_stress_on=False, mean_pgrad_on=False)
    rng = np.random.default_rng(11)
    u, v, w = random_state(g, 11)
    s.set_state(u, v, w)
    E0 = kinetic_energy(s, g)
    wf = zero_wave_fields(g)
    for _ in range(int(round(T / dt))):
        s.step(dt, wf, wf)
    return abs(kinetic_energy(s, g) - E0) / E0


def test_inviscid_energy():
    e = [_energy_change(dt) for dt in (2e-3, 1e-3, 5e-4)]
    orders = [math.log2(e[i] / e[i + 1]) for i in range(2)]
    print("energy", e, orders)
    assert e[-1] < 1e-4
    assert min(orders) > 1.8


# save / load / cfl --------------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    p = make_params(Nx=16, Ny=8, Nz=10)
    g = Grid(p)
    s = LESSolver(p, g)
    s.set_state(*random_state(g, 2))
    wf0 = linear_wave_fields(g, 0.01, 2.0, 30.0, 0.0)
    wf1 = linear_wave_fields(g, 0.01, 2.0, 30.0, 1e-3)
    wf2 = linear_wave_fields(g, 0.01, 2.0, 30.0, 2e-3)
    s.step(1e-3, wf0, wf1)
    s.save(tmp_path / "ck.npz")
    s2 = LESSolver(p, g)
    s2.load(tmp_path / "ck.npz")
    s.step(1e-3, wf1, wf2)
    s2.step(1e-3, wf1, wf2)
    assert np.abs(s.u - s2.u).max() < 1e-12 * np.abs(s.u).max()
    assert np.abs(s.w - s2.w).max() < 1e-12 * np.abs(s.w).max()
    assert s2.nstep == 2 and abs(s2.t - 2e-3) < 1e-15
    c = s.cfl(1e-3, wf2)
    assert c["max"] > 0 and np.isfinite(c["max"])


# stability with the paper's wave amplitude on the reduced vertical grid ------
def _wave_run(uzz_method, nsteps):
    from wpt.params import reduced
    from wpt.initial import mean_profile
    # Lx = pi/3 so that k0 = 12 is the mode m = 2; reduced vertical grid (Nz = 48)
    p = reduced(Lx=math.pi / 3, Ly=math.pi / 3, Nx=8, Ny=8, workers=1, alpha=0.12)
    g = Grid(p)
    s = LESSolver(p, g, uzz_method=uzz_method)
    rng = np.random.default_rng(4)
    U = mean_profile(g, p.Re_tau)
    env = np.exp(g.zc / 0.1)
    u = U + env * rng.standard_normal((g.Nx, g.Ny, g.Nz))
    v = env * rng.standard_normal((g.Nx, g.Ny, g.Nz))
    w = np.zeros((g.Nx, g.Ny, g.Nz + 1))
    a, k, om, dt = p.a0, p.k0, p.omega0, p.dt
    wf0 = linear_wave_fields(g, a, k, om, 0.0)
    s.set_state(u, v, w, wf=wf0)
    with np.errstate(all="ignore"):
        for n in range(nsteps):
            wf1 = linear_wave_fields(g, a, k, om, (n + 1) * dt)
            s.step(dt, wf0, wf1)
            wf0 = wf1
            if not np.isfinite(s.v).all():
                return np.inf
    return np.abs(s.v).max()


def test_wave_amplitude_stability():
    """Paper wave amplitude (a0 = 4 dz_top): the default 'wave_scale' surface u_zz stays
    bounded over ~8 wave periods; the grid-scale 'cubic' stencil blows up."""
    vmax = _wave_run("wave_scale", 400)
    print("wave_scale vmax", vmax)
    assert vmax < 20.0
    vbad = _wave_run("cubic", 150)
    print("cubic vmax", vbad)
    assert vbad > 100.0
