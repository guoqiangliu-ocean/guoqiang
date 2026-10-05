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


def test_dealiasing_exact():
    """The rhs is alias-free: a band-limited state (modes <= N/8) gives the same
    retained rhs on N and on 2N points (zero-padded) to round-off, for M = 3, 4."""
    rng = np.random.default_rng(1)
    for M in (3, 4):
        p = _mono_params(hos_N=64, hos_order=M)
        h1 = HOS(p)
        h2 = HOS(p.with_(hos_N=128))
        nk = 64 // 8 + 1
        eh = np.zeros(33, complex)
        ph = np.zeros(33, complex)
        amp = 0.12 / p.k0 * 64 / 4
        eh[1:nk] = amp * (rng.standard_normal(nk - 1) + 1j * rng.standard_normal(nk - 1))
        ph[1:nk] = p.c0 * amp * (rng.standard_normal(nk - 1) + 1j * rng.standard_normal(nk - 1))
        eta1 = np.fft.irfft(eh, 64)
        phs1 = np.fft.irfft(ph, 64)
        eta2 = np.fft.irfft(np.r_[eh, np.zeros(32)] * 2, 128)
        phs2 = np.fft.irfft(np.r_[ph, np.zeros(32)] * 2, 128)
        assert np.allclose(eta2[::2], eta1) and np.allclose(phs2[::2], phs1)
        # the state must actually be nonlinear for the test to mean something
        assert np.abs(np.gradient(eta1, h1.x)).max() > 0.1
        de1, dp1 = h1.rhs(eta1, phs1)
        de2, dp2 = h2.rhs(eta2, phs2)
        r1e, r2e = np.fft.rfft(de1), np.fft.rfft(de2)[:33] / 2
        r1p, r2p = np.fft.rfft(dp1), np.fft.rfft(dp2)[:33] / 2
        # compare modes that the 2N grid also resolves without truncation effects
        err_e = np.abs(r1e[:32] - r2e[:32]).max() / np.abs(r2e).max()
        err_p = np.abs(r1p[:32] - r2p[:32]).max() / np.abs(r2p).max()
        assert err_e < 1e-12 and err_p < 1e-12, (M, err_e, err_p)
        # the finer grid has real content above the coarse band: band-limited to
        # M * N/8 < N/2 so nothing is lost -- check the 2N rhs is zero above N/2
        assert np.abs(np.fft.rfft(de2)[33:]).max() < 1e-12 * np.abs(np.fft.rfft(de2)).max()


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
    """phi(x, eta(x)) evaluated directly from phi_modes must reproduce phis to
    O(alpha^(M+1)) (Stokes wave alpha = 0.1); W likewise matches phi_z(x, eta)."""
    alpha = 0.1
    res = {}
    for M in (1, 2, 3, 4):
        p = _mono_params(hos_N=64, hos_order=M)
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
        assert res[M + 1][0] < 0.3 * res[M][0]
    assert res[3][0] < 1e-3 and res[3][1] < 1e-3


# ---------------------------------------------------------------------------
def test_filter_and_long_stability():
    """Optional low-pass filter leaves the packet unchanged to < 1e-6 relative
    energy over a few periods; filtered/unfiltered agree."""
    p = Params(alpha=0.12, hos_N=256, workers=1)
    h0 = HOS(p)
    hf = HOS(p, filter_frac=0.9)
    h0.init_packet()
    hf.init_packet()
    _run(h0, 3 * p.T0)
    _run(hf, 3 * p.T0)
    assert np.max(np.abs(h0.eta - hf.eta)) < 1e-6 * p.a0
    assert np.all(np.isfinite(hf.eta))


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
