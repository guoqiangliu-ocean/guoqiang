"""Tests of the HOS wave solver (SPEC.md section 2).

Physical scales of Params are kept (g = 1/Fr^2 ~ 7.15e6, k0 = 12,
omega0 ~ 9264, T0 ~ 6.8e-4) and the solver is stepped with the LES time step
dt = 1.4e-5 (reduced preset), so omega0 dt ~ 0.13.  Monochromatic tests use
Lx = 2 pi (k0 = mode 12) and small N to stay fast; packet tests use the paper
domain Lx = 6 pi.
"""
import math
import time

import numpy as np
import pytest

from wpt.params import Params, reduced
from wpt.hos import HOS

DT = 1.4e-5


def _mono_params(**kw):
    base = dict(Lx=2.0 * math.pi, hos_N=64, workers=1)
    base.update(kw)
    return Params(**base)


def _fund(h, m0):
    return np.fft.rfft(h.eta)[m0]


def _energy_centre(h):
    """Circular centroid of the wave energy density 0.5 g eta^2 + 0.5 phis eta_t."""
    de, _ = h.rhs(h.eta, h.phis)
    e = 0.5 * h.g * h.eta ** 2 + 0.5 * h.phis * de
    c = np.sum(e * np.exp(2j * np.pi * h.x / h.Lx))
    return np.angle(c) * h.Lx / (2.0 * np.pi)


def _run(h, t_end, dt=DT):
    n = int(round(t_end / dt))
    for _ in range(n):
        h.step(dt)
    return n


# ---------------------------------------------------------------------------
def test_interface_and_init_packet():
    p = Params(hos_N=256, workers=1)
    h = HOS(p)
    assert h.x.shape == (256,) and h.k.shape == (129,)
    assert np.isclose(h.k[1], 2 * math.pi / p.Lx)
    h.init_packet()
    assert h.eta.shape == (256,) and h.phis.shape == (256,) and h.t == 0.0
    # eqs (2.17)-(2.19)
    d = h.x - p.x0
    A0 = p.a0 * np.exp(-d ** 2 / (2 * p.chi ** 2))
    assert np.max(np.abs(h.eta - A0 * np.cos(p.k0 * h.x))) < 1e-10 * p.a0
    assert np.max(np.abs(h.phis - A0 * p.c0 * np.sin(p.k0 * h.x))) < 1e-10 * p.a0 * p.c0
    assert np.isclose(np.abs(h.eta).max(), p.a0, rtol=1e-3)
    # periodic wrapping of x - x0: packet centred near the right end of the domain
    q = p.with_(x0_frac=0.97)
    h2 = HOS(q)
    h2.init_packet()
    dw = np.mod(h2.x - q.x0 + q.Lx / 2, q.Lx) - q.Lx / 2
    A0w = q.a0 * np.exp(-dw ** 2 / (2 * q.chi ** 2))
    assert np.max(np.abs(h2.eta - A0w * np.cos(q.k0 * h2.x))) < 1e-10 * q.a0
    # envelope continues across x = 0 (wrap actually used: A0 there is O(a0))
    assert A0w[0] > 0.1 * q.a0
    assert abs(_energy_centre(h2) - (q.x0 - q.Lx)) < 1e-3 * q.chi


def test_linear_rhs_exact():
    """For a tiny monochromatic wave the HOS rhs equals the linear one."""
    p = _mono_params()
    h = HOS(p)
    a = 1e-6 / p.k0
    h.init_monochromatic(a, p.k0)
    de, dp = h.rhs(h.eta, h.phis)
    om = p.omega0
    assert np.max(np.abs(de - a * om * np.sin(p.k0 * h.x))) < 1e-5 * a * om
    assert np.max(np.abs(dp + p.g * a * np.cos(p.k0 * h.x))) < 1e-5 * p.g * a


def _with_padding(h, Np):
    """Force the product grid of an HOS instance to Np points (test helper)."""
    h.Np = Np
    h.nkp = Np // 2 + 1
    return h


def test_dealiasing_exact():
    """Products are alias-free: for a random *full-band* state (all modes
    1..N/2-1 populated, |eta_x| ~ 0.3) the rhs computed with the default
    padding Np = (M+1)N/2 equals, to round-off, the rhs computed with an
    8x over-resolved product grid (same truncation of every phi^(m)).  The
    same comparison without padding (Np = N) shows a large aliasing error, so
    the test is sensitive."""
    rng = np.random.default_rng(1)
    N = 64
    nk = N // 2 + 1
    for M in (2, 3, 4):
        p = _mono_params(hos_N=N, hos_order=M)
        h = HOS(p)
        assert h.Np >= (M + 1) * N / 2
        ref = _with_padding(HOS(p), 8 * N)
        bad = _with_padding(HOS(p), N)
        m = np.arange(nk)
        env = np.exp(-((m - 12) / 10.0) ** 2) + 0.05
        env[0] = env[-1] = 0.0
        eh = env * (rng.standard_normal(nk) + 1j * rng.standard_normal(nk))
        ph = env * (rng.standard_normal(nk) + 1j * rng.standard_normal(nk))
        eta = np.fft.irfft(eh, N)
        eta *= 0.3 / np.abs(np.fft.irfft(1j * h.k * eh, N)).max()      # max slope 0.3
        phs = np.fft.irfft(ph, N)
        phs *= 0.3 * p.c0 / p.k0 / np.abs(phs).max()
        r = [hh.rhs(eta, phs) for hh in (h, ref, bad)]
        def err(a, b, i):
            return np.abs(a[i] - b[i]).max() / np.abs(b[i]).max()
        e_ok = max(err(r[0], r[1], 0), err(r[0], r[1], 1))
        e_bad = max(err(r[2], r[1], 0), err(r[2], r[1], 1))
        print(f"M={M}: Np={h.Np}, alias error {e_ok:.1e} (no padding: {e_bad:.1e})")
        assert e_ok < 1e-12
        assert e_bad > 1e-4


# ---------------------------------------------------------------------------
# (i) small-amplitude monochromatic wave: linear phase after 10 periods
def test_linear_phase_10_periods():
    p = _mono_params(hos_N=64)
    h = HOS(p)
    a = 1e-3 / p.k0
    h.init_monochromatic(a, p.k0)
    m0 = 12
    _run(h, 10 * p.T0)
    om = p.omega0
    c = _fund(h, m0)
    ph_err = np.angle(c * np.exp(1j * om * h.t))          # expected arg = -omega t
    amp_err = abs(abs(c) * 2 / h.N - a) / a
    field_err = np.max(np.abs(h.eta - a * np.cos(p.k0 * h.x - om * h.t))) / a
    print(f"linear phase error after {h.t / p.T0:.2f} T0: {ph_err:.2e} rad, "
          f"amp err {amp_err:.2e}, max field err {field_err:.2e}")
    assert abs(ph_err) < 1e-3          # RK4 dispersion ~1.4e-4 rad; nonlinear 3e-5 rad
    assert amp_err < 1e-4              # RK4 damping ~1.6e-5
    assert field_err < 2e-3


# ---------------------------------------------------------------------------
# (ii) packet centre of energy moves at c_g = c0/2 within 2 % over 20 periods
@pytest.mark.parametrize("alpha", [1e-3, 0.12])
def test_packet_group_velocity(alpha):
    p = Params(alpha=alpha, hos_N=384, workers=1)
    h = HOS(p)
    h.init_packet()
    xc0 = _energy_centre(h)
    E0 = h.energy()
    t0 = time.perf_counter()
    _run(h, 20 * p.T0)
    wall = time.perf_counter() - t0
    d = np.mod(_energy_centre(h) - xc0 + p.Lx / 2, p.Lx) - p.Lx / 2
    cg_num = d / h.t
    rel = cg_num / p.cg - 1.0
    dE = h.energy() / E0 - 1.0
    print(f"alpha={alpha}: cg/cg0 - 1 = {rel:.4e}, dE/E = {dE:.2e}, wall {wall:.2f}s")
    assert d > 0.4 * p.Lx / 2               # it really moved (~5.2 = 10 wavelengths)
    assert abs(rel) < 0.02
    if alpha < 0.01:
        assert abs(rel) < 0.005             # linear: only the finite-bandwidth effect
    # (iv) energy conservation over 20 periods
    assert abs(dE) < 1e-4


# ---------------------------------------------------------------------------
# (iii) third-order Stokes wave frequency correction
def _measure_omega(h, m0, nper, T0):
    ts, ph = [], []
    n = int(round(nper * T0 / DT))
    for _ in range(n):
        h.step(DT)
        ts.append(h.t)
        ph.append(np.angle(_fund(h, m0)))
    return -np.polyfit(ts, np.unwrap(ph), 1)[0]


def test_stokes_frequency_correction():
    alpha = 0.1
    corr = 0.5 * alpha ** 2
    res = {}
    for M in (1, 3):
        p = _mono_params(hos_N=64, hos_order=M)
        h = HOS(p)
        h.init_stokes(alpha / p.k0, p.k0)
        om = _measure_omega(h, 12, 10, p.T0)
        res[M] = (om / p.omega0 - 1.0) / corr
    print(f"Stokes alpha=0.1: measured/theoretical correction M=1: {res[1]:.4f}, M=3: {res[3]:.4f}")
    assert abs(res[3] - 1.0) < 0.15
    assert abs(res[1]) < 0.05          # the linear model has no correction


def test_stokes_from_linear_init():
    """Same with the plain linear initial condition (free harmonics present)."""
    alpha = 0.1
    p = _mono_params(hos_N=64, hos_order=3)
    h = HOS(p)
    h.init_monochromatic(alpha / p.k0, p.k0)
    om = _measure_omega(h, 12, 10, p.T0)
    r = (om / p.omega0 - 1.0) / (0.5 * alpha ** 2)
    print(f"Stokes from linear init: correction ratio {r:.4f}")
    assert abs(r - 1.0) < 0.15


# ---------------------------------------------------------------------------
# (iv) energy conservation of the steep monochromatic wave
def test_energy_conservation_stokes():
    p = _mono_params(hos_N=64)
    h = HOS(p)
    h.init_stokes(0.12 / p.k0, p.k0)
    E0 = h.energy()
    Elin = h.energy("linear")
    assert np.isclose(E0, 0.5 * p.g * (0.12 / p.k0) ** 2, rtol=0.05)
    assert np.isclose(E0, Elin, rtol=0.05)
    Es = []
    for i in range(int(round(20 * p.T0 / DT))):
        h.step(DT)
        if i % 25 == 0:
            Es.append(h.energy())
    dev = np.max(np.abs(np.array(Es) / E0 - 1.0))
    print(f"Stokes alpha=0.12, 20 T0: max |E/E0-1| = {dev:.2e}")
    assert dev < 1e-4
    # halving dt reduces the (RK4) drift by ~2^5
    h2 = HOS(p)
    h2.init_stokes(0.12 / p.k0, p.k0)
    _run(h2, 5 * p.T0, DT / 2)
    h1 = HOS(p)
    h1.init_stokes(0.12 / p.k0, p.k0)
    _run(h1, 5 * p.T0, DT)
    d1 = abs(h1.energy() / E0 - 1)
    d2 = abs(h2.energy() / E0 - 1)
    print(f"energy drift 5 T0: dt {d1:.2e}, dt/2 {d2:.2e}, ratio {d1 / d2:.1f}")
    assert d1 / d2 > 16


# ---------------------------------------------------------------------------
# (v) phi_modes reproduces the linear analytic velocity field
def _uw_from_modes(h, z):
    c = h.phi_modes()
    k = h.k
    ez = np.exp(np.abs(k) * z)
    u = np.fft.irfft(1j * k * c * ez, n=h.N)
    w = np.fft.irfft(np.abs(k) * c * ez, n=h.N)
    return u, w


def test_phi_modes_linear_velocity():
    p = _mono_params(hos_N=64)
    k0, om = p.k0, p.omega0
    a = 1e-3 / k0
    errs = []
    for init in ("stokes", "mono"):
        h = HOS(p)
        if init == "stokes":
            om_s = h.init_stokes(a, k0)
        else:
            h.init_monochromatic(a, k0)
            om_s = om
        _run(h, 3 * p.T0)
        assert h.phi_modes().shape == (h.N // 2 + 1,) and np.iscomplexobj(h.phi_modes())
        for z in (0.0, -0.05, -0.2, -0.5):
            u, w = _uw_from_modes(h, z)
            th = k0 * h.x - om_s * h.t
            ue = a * om * np.exp(k0 * z) * np.cos(th)
            we = a * om * np.exp(k0 * z) * np.sin(th)
            scale = a * om * np.exp(k0 * z)
            eu = np.max(np.abs(u - ue)) / scale
            ew = np.max(np.abs(w - we)) / scale
            errs.append((init, z, eu, ew))
    for init, z, eu, ew in errs:
        print(f"{init:6s} z={z:5.2f}: rel err u {eu:.2e}, w {ew:.2e}")
        tol = 5e-4 if init == "stokes" else 2e-3       # O(alpha) free harmonic for 'mono'
        assert eu < tol and ew < tol


def test_phi_modes_dirichlet_residual():
    """phi(x, eta(x)) evaluated directly (exp(|k| eta) summation) from
    phi_modes must reproduce phis to O(alpha^(M+1)) (Stokes wave alpha = 0.1,
    N = 128 so that the harmonics are not truncated); the Taylor-expanded W
    likewise matches phi_z(x, eta)."""
    alpha = 0.1
    res = {}
    for M in (1, 2, 3, 4):
        p = _mono_params(hos_N=128, hos_order=M)
        h = HOS(p)
        h.init_stokes(alpha / p.k0, p.k0)
        c = h.phi_modes()
        k = h.k
        wts = np.full(k.size, 2.0)
        wts[0] = 1.0
        wts[-1] = 1.0
        E = np.exp(np.outer(h.eta, np.abs(k)) + 1j * np.outer(h.x, k))
        phi_eta = (E * (wts * c)).real.sum(1) / h.N
        phiz_eta = (E * (wts * np.abs(k) * c)).real.sum(1) / h.N
        scale = np.abs(h.phis).max()
        r_phi = np.abs(phi_eta - h.phis).max() / scale
        r_w = np.abs(phiz_eta - h.surface_W()).max() / (scale * p.k0)
        res[M] = (r_phi, r_w)
        print(f"M={M}: Dirichlet residual {r_phi:.2e}, W residual {r_w:.2e}")
    for M in (1, 2, 3):
        assert res[M + 1][0] < 0.2 * res[M][0]      # ~alpha per order
    assert res[3][0] < 2e-3 and res[3][1] < 6e-3
    assert res[4][1] < 0.2 * res[3][1]


# ---------------------------------------------------------------------------
def test_filter_optional():
    """The optional low-pass filter (off by default) only touches modes far
    above the 3rd harmonic: with N = 384 the filtered and unfiltered W12
    packets differ by < 2e-3 a0 after 3 periods, and the filter is the
    identity below 0.9 N/2."""
    p = Params(alpha=0.12, hos_N=384, workers=1)
    h0 = HOS(p)
    assert h0._filter is None
    hf = HOS(p, filter_frac=0.9)
    m = np.arange(hf.nk)
    assert np.all(hf._filter[m <= 0.9 * hf.N / 2] == 1.0) and hf._filter[-1] == 0.0
    h0.init_packet()
    hf.init_packet()
    _run(h0, 3 * p.T0)
    _run(hf, 3 * p.T0)
    d = np.max(np.abs(h0.eta - hf.eta)) / p.a0
    print(f"filter effect after 3 T0: {d:.2e} a0")
    assert np.all(np.isfinite(hf.eta)) and d < 2e-3


# ---------------------------------------------------------------------------
# convergence: RK4 order in time, spectral convergence in N
def test_time_convergence_rk4():
    p = Params(alpha=0.12, hos_N=256, workers=1)
    T = 2 * p.T0
    n0 = int(round(T / DT))
    out = {}
    for f in (1, 2, 4, 16):
        h = HOS(p)
        h.init_packet()
        _run(h, T, T / (n0 * f))
        out[f] = h.eta.copy()
    e = [np.abs(out[f] - out[16]).max() / p.a0 for f in (1, 2, 4)]
    orders = [math.log2(e[0] / e[1]), math.log2(e[1] / e[2])]
    print(f"RK4 errors (dt={T / n0:.2e}, /2, /4): {e[0]:.2e} {e[1]:.2e} {e[2]:.2e}; "
          f"orders {orders[0]:.2f} {orders[1]:.2f}")
    assert e[0] < 1e-4
    assert all(3.7 < o < 4.4 for o in orders)


def test_spatial_convergence():
    """Packet W12 after 2 periods: spectral coefficients for N = 192, 256, 384
    against N = 768 converge rapidly (L1 norm of the coefficient error)."""
    p = Params(alpha=0.12, workers=1)
    T = 2 * p.T0
    spec = {}
    for N in (192, 256, 384, 768):
        h = HOS(p.with_(hos_N=N))
        h.init_packet()
        _run(h, T)
        spec[N] = np.fft.rfft(h.eta) / N
    err = {}
    for N in (192, 256, 384):
        r = spec[768].copy()
        r[:N // 2 + 1] -= spec[N]
        err[N] = 2 * np.abs(r).sum() / p.a0
    print("spatial errors (L1 of spectrum / a0): " +
          ", ".join(f"N={N}: {e:.2e}" for N, e in err.items()))
    assert err[192] > err[256] > err[384]
    assert err[384] < 2e-3


def test_performance_N768_M3():
    p = reduced(workers=1)
    h = HOS(p)
    assert h.N == 768 and h.M == 3
    h.init_packet()
    for _ in range(5):
        h.step(p.dt)
    best = np.inf
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20):
            h.step(p.dt)
        best = min(best, (time.perf_counter() - t0) / 20)
    print(f"RK4 step N=768 M=3: {best * 1e3:.3f} ms")
    assert best < 5e-3
