"""Tests for wpt/coupling.py (SPEC.md section 3, wave-field part).

The HOS module is replaced by a stub exposing only ``N``, ``t``, ``eta``,
``k`` and ``phi_modes()`` built from analytic linear deep-water waves.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from wpt.coupling import wave_fields_from_hos, resample_rfft, clear_cache
from wpt.grid import Grid
from wpt.params import tiny, reduced

FIELDS_C = ["uphi_c", "wphi_c", "duphi_dx_c", "duphi_dz_c", "dwphi_dx_c", "dwphi_dz_c"]
FIELDS_F = ["uphi_f", "wphi_f", "duphi_dx_f", "duphi_dz_f", "dwphi_dx_f", "dwphi_dz_f"]


class StubHOS:
    """Linear superposition of progressive deep-water waves
    phi = sum_j (a_j omega_j/k_j) e^{k_j z} sin(k_j x - omega_j t + th_j),
    eta = sum_j a_j cos(k_j x - omega_j t + th_j), sampled on an N-point grid."""

    def __init__(self, Lx, N, g, modes, t=0.0):
        self.N = N
        self.Lx = Lx
        self.g = g
        self.t = t
        self.x = np.arange(N) * Lx / N
        self.k = 2 * math.pi / Lx * np.arange(N // 2 + 1)
        self.modes = modes          # list of (m, a, th)
        th = self._phases(self.x[:, None])
        a, kk, om = self._amp()
        self.eta = (a * np.cos(th)).sum(axis=1)
        self._phis = (a * om / kk * np.sin(th)).sum(axis=1)

    def _amp(self):
        m = np.array([q[0] for q in self.modes], float)
        a = np.array([q[1] for q in self.modes], float)
        k = 2 * math.pi / self.Lx * m
        return a, k, np.sqrt(self.g * k)

    def _phases(self, x):
        a, k, om = self._amp()
        th0 = np.array([q[2] for q in self.modes], float)
        return k * x - om * self.t + th0

    def phi_modes(self):
        # linear wave: phi(x, 0) is the surface potential, phi = irfft(c e^{kz})
        return np.fft.rfft(self._phis)

    # analytic fields at (x, z) for the subset of modes with m < mcut
    def exact(self, x, z, mcut=np.inf):
        a, k, om = self._amp()
        m = np.array([q[0] for q in self.modes])
        sel = m < mcut
        a, k, om = a[sel], k[sel], om[sel]
        th0 = np.array([q[2] for q in self.modes], float)[sel]
        X = x[:, None, None]
        Z = np.asarray(z)[None, :, None]
        th = k * X - om * self.t + th0
        E = np.exp(k * Z)
        aw = a * om
        out = {
            "uphi": (aw * E * np.cos(th)).sum(-1),
            "wphi": (aw * E * np.sin(th)).sum(-1),
            "duphi_dx": (-aw * k * E * np.sin(th)).sum(-1),
            "duphi_dz": (aw * k * E * np.cos(th)).sum(-1),
            "dwphi_dx": (aw * k * E * np.cos(th)).sum(-1),
            "dwphi_dz": (aw * k * E * np.sin(th)).sum(-1),
        }
        th_s = k * x[:, None] - om * self.t + th0
        out["eta"] = (a * np.cos(th_s)).sum(-1)
        out["eta_x"] = (-a * k * np.sin(th_s)).sum(-1)
        out["irrot"] = (2 * aw * k * np.cos(th_s)).sum(-1)
        return out


def _relerr(a, b):
    a = np.asarray(a).reshape(np.shape(b))
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)


def _check_all(wf, stub, grid, tol, mcut=np.inf, report=None):
    ex_c = stub.exact(grid.x, grid.zc, mcut)
    ex_f = stub.exact(grid.x, grid.zf, mcut)
    errs = {}
    for name in ["uphi", "wphi", "duphi_dx", "duphi_dz", "dwphi_dx", "dwphi_dz"]:
        errs[name + "_c"] = _relerr(getattr(wf, name + "_c")[:, 0, :], ex_c[name])
        errs[name + "_f"] = _relerr(getattr(wf, name + "_f")[:, 0, :], ex_f[name])
    errs["eta"] = _relerr(wf.eta[:, 0], ex_c["eta"])
    errs["eta_x"] = _relerr(wf.eta_x[:, 0], ex_c["eta_x"])
    errs["irrot_stress_x"] = _relerr(wf.irrot_stress_x[:, 0], ex_c["irrot"])
    if report is not None:
        report.update(errs)
    for k, e in errs.items():
        assert e < tol, (k, e)
    return errs


def _shapes(wf, grid):
    Nx, Nz = grid.Nx, grid.Nz
    for n in FIELDS_C:
        assert getattr(wf, n).shape == (Nx, 1, Nz), n
    for n in FIELDS_F:
        assert getattr(wf, n).shape == (Nx, 1, Nz + 1), n
    for n in ("eta", "eta_x", "irrot_stress_x"):
        assert getattr(wf, n).shape == (Nx, 1), n


def _small(**kw):
    base = dict(Nx=48, Ny=8, Nz=16, hos_N=192, workers=1)
    base.update(kw)
    return tiny(**base)


# ---------------------------------------------------------------------------
# (i) analytic linear wave, alpha = 1e-3, all 12 fields + surface quantities
@pytest.mark.parametrize("t", [0.0, 3.7e-4])
def test_linear_wave_monochromatic(t):
    p = _small()
    grid = Grid(p)
    m0 = int(round(p.k0 * p.Lx / (2 * math.pi)))   # k0 H = 12 -> m0 = 12 (Lx = 2 pi)
    a = 1e-3 / p.k0
    stub = StubHOS(p.Lx, p.hos_N, p.g, [(m0, a, 0.0)], t=t)
    wf = wave_fields_from_hos(stub, grid, p)
    _shapes(wf, grid)
    assert wf.t == t
    errs = _check_all(wf, stub, grid, 1e-6)
    print("monochromatic max rel err:", max(errs.values()))


def test_linear_wave_multimode():
    p = _small()
    grid = Grid(p)
    rng = np.random.default_rng(1)
    ms = [1, 3, 7, 12, 13, 15]                       # all < Nx/3 = 16
    modes = [(m, 1e-3 / (2 * math.pi * m / p.Lx) * rng.uniform(0.2, 1.0),
              rng.uniform(0, 2 * math.pi)) for m in ms]
    stub = StubHOS(p.Lx, p.hos_N, p.g, modes, t=1.234e-3)
    wf = wave_fields_from_hos(stub, grid, p)
    errs = _check_all(wf, stub, grid, 1e-6)
    print("multimode max rel err:", max(errs.values()))


def test_linear_wave_reduced_domain():
    # paper domain Lx = 6 pi (k0 -> m0 = 36), reduced-preset vertical grid, packet-like spectrum
    p = reduced(Ny=8, workers=1)
    grid = Grid(p)
    m0 = int(round(p.k0 * p.Lx / (2 * math.pi)))
    assert m0 == 36
    modes = [(m, 1e-3 / p.k0 * math.exp(-0.5 * ((m - m0) / 3.0) ** 2), 0.3 * m)
             for m in range(m0 - 10, m0 + 11)]
    stub = StubHOS(p.Lx, p.hos_N, p.g, modes, t=2.0e-3)
    wf = wave_fields_from_hos(stub, grid, p)
    _shapes(wf, grid)
    _check_all(wf, stub, grid, 1e-6)


# ---------------------------------------------------------------------------
# (ii) irrotationality / incompressibility identities and independent FD checks
def test_identities():
    p = _small()
    grid = Grid(p)
    rng = np.random.default_rng(2)
    modes = [(m, 1e-4 * rng.uniform(0.5, 1), rng.uniform(0, 6.3)) for m in (2, 5, 9, 14)]
    wf = wave_fields_from_hos(StubHOS(p.Lx, p.hos_N, p.g, modes, t=5e-4), grid, p)
    for s in ("c", "f"):
        g = lambda n: getattr(wf, f"{n}_{s}")
        np.testing.assert_array_equal(g("dwphi_dz"), -g("duphi_dx"))
        np.testing.assert_array_equal(g("duphi_dz"), g("dwphi_dx"))
        # spectral x-derivatives of uphi, wphi agree with the gradient fields
        kx = grid.kx1d(2)
        for fld, d in (("uphi", "duphi_dx"), ("wphi", "dwphi_dx")):
            dx = grid.ifft_x(1j * kx * grid.fft_x(g(fld)))
            assert _relerr(dx, g(d)) < 1e-12
    # second-order FD in z of face values matches the centre values (independent check)
    for fld, d in (("uphi", "duphi_dz"), ("wphi", "dwphi_dz")):
        fd = grid.ddz_f2c(getattr(wf, fld + "_f"))
        assert _relerr(fd, getattr(wf, d + "_c")) < 0.05
    # surface stress equals 2 * top-face phi_xz
    np.testing.assert_allclose(wf.irrot_stress_x[:, 0], 2 * wf.duphi_dz_f[:, 0, -1], rtol=0, atol=0)
    # eta_x is the spectral derivative of eta
    dex = grid.ifft_x(1j * grid.kx1d(1) * grid.fft_x(wf.eta))
    assert _relerr(dex, wf.eta_x) < 1e-12


def test_fd_convergence_in_z():
    """The vertical structure exp(k z) is exact: a 2nd-order FD of uphi_f across
    cells converges to duphi_dz_c at order 2 under grid refinement."""
    errs = []
    for Nz in (16, 32, 64):
        p = _small(Nz=Nz, z_stretch=0.0)
        grid = Grid(p)
        stub = StubHOS(p.Lx, p.hos_N, p.g, [(12, 1e-3 / 12, 0.4)])
        wf = wave_fields_from_hos(stub, grid, p)
        errs.append(_relerr(grid.ddz_f2c(wf.uphi_f), wf.duphi_dz_c))
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(2)]
    print("FD errors", errs, "orders", orders)
    assert all(1.9 < o < 2.1 for o in orders)


# ---------------------------------------------------------------------------
# resampling: hos_N = 4 Nx with energy above Nx/3 (incl. aliases and HOS Nyquist)
def test_resampling_truncation():
    p = _small(Nx=48, hos_N=192)
    grid = Grid(p)
    Nx, N = p.Nx, p.hos_N
    rng = np.random.default_rng(3)
    kept = [1, 4, 12, 15]                     # < Nx/3 = 16
    dropped = [16, 20, 24, 30, 36, 48 - 12, 48 + 4, 60, 95]   # >= Nx/3, includes
    # Nx - 12 (aliases onto 12 by pointwise sampling), Nx + 4 (aliases onto 4),
    # the LES Nyquist 24 and modes up to just below the HOS Nyquist
    modes = [(m, 1e-3 / (2 * math.pi * m / p.Lx), rng.uniform(0, 6.3)) for m in kept + dropped]
    stub = StubHOS(p.Lx, N, p.g, modes, t=7e-4)
    # add a HOS Nyquist component (m = N/2 = 96) directly to the spectrum and eta
    nyq_amp = 0.37
    c0 = stub.phi_modes()
    c_nyq = c0.copy()
    c_nyq[N // 2] += nyq_amp * N
    stub.phi_modes = lambda: c_nyq
    stub.eta = stub.eta + 1e-4 * np.cos(math.pi * np.arange(N))
    wf = wave_fields_from_hos(stub, grid, p)
    errs = _check_all(wf, stub, grid, 1e-10, mcut=Nx / 3.0)
    print("resampling max rel err:", max(errs.values()))
    # no energy at |m| >= Nx/3 in any output field
    for n in FIELDS_C + FIELDS_F + ["eta", "eta_x", "irrot_stress_x"]:
        f = getattr(wf, n)
        fh = np.fft.fft(f, axis=0)
        hi = ~grid.dealias_x
        assert np.max(np.abs(fh[hi])) <= 1e-12 * max(np.max(np.abs(fh)), 1e-300), n


def test_resample_rfft_scaling_and_nyquist():
    # coarse HOS grid whose Nyquist lies inside the retained band (N < 2 Nx/3)
    Nx, N, Lx = 48, 24, 2 * math.pi
    nm = 16
    x_h = np.arange(N) * Lx / N
    x_l = np.arange(Nx) * Lx / Nx
    f_h = 0.3 + 0.8 * np.cos(3 * x_h + 0.2) + 0.5 * np.cos(12 * x_h)   # m = 12 = N/2 Nyquist
    cl = resample_rfft(np.fft.rfft(f_h), N, Nx, nm)
    spec = np.zeros(Nx // 2 + 1, complex)
    spec[:nm] = cl
    f_l = np.fft.irfft(spec, n=Nx)
    np.testing.assert_allclose(f_l, 0.3 + 0.8 * np.cos(3 * x_l + 0.2) + 0.5 * np.cos(12 * x_l),
                               atol=1e-13)
    # fine HOS grid: plain (Nx/N) rescaling
    N = 4 * Nx
    x_h = np.arange(N) * Lx / N
    f_h = 0.3 + 0.8 * np.sin(7 * x_h) + 2.0 * np.cos(40 * x_h)
    cl = resample_rfft(np.fft.rfft(f_h), N, Nx, nm)
    spec[:] = 0
    spec[:nm] = cl
    np.testing.assert_allclose(np.fft.irfft(spec, n=Nx), 0.3 + 0.8 * np.sin(7 * x_l), atol=1e-13)


# ---------------------------------------------------------------------------
def test_switches():
    p = _small()
    grid = Grid(p)
    stub = StubHOS(p.Lx, p.hos_N, p.g, [(12, 1e-3 / 12, 0.0)], t=0.5)
    wf = wave_fields_from_hos(stub, grid, p.with_(include_irrot_stress=False))
    assert np.all(wf.irrot_stress_x == 0)
    assert np.max(np.abs(wf.uphi_c)) > 0
    wf0 = wave_fields_from_hos(stub, grid, p.with_(wave_on=False))
    _shapes(wf0, grid)
    assert wf0.t == 0.5
    for n in FIELDS_C + FIELDS_F + ["eta", "eta_x", "irrot_stress_x"]:
        assert np.all(getattr(wf0, n) == 0)
    # outputs do not alias each other (safe for in-place modification)
    wf = wave_fields_from_hos(stub, grid, p)
    arrs = [getattr(wf, n) for n in FIELDS_C + FIELDS_F + ["eta", "eta_x", "irrot_stress_x"]]
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            assert not np.shares_memory(arrs[i], arrs[j])


def test_bad_input_shapes():
    p = _small()
    grid = Grid(p)
    stub = StubHOS(p.Lx, p.hos_N, p.g, [(12, 1e-4, 0.0)])
    stub.N = p.hos_N + 2
    with pytest.raises(ValueError):
        wave_fields_from_hos(stub, grid, p)


# ---------------------------------------------------------------------------
def test_performance_reduced():
    p = reduced(workers=1)
    grid = Grid(p)
    m0 = 36
    modes = [(m, 1e-2 / p.k0 * math.exp(-0.5 * ((m - m0) / 5.0) ** 2), 0.1 * m)
             for m in range(1, 200)]
    stub = StubHOS(p.Lx, p.hos_N, p.g, modes, t=1e-3)
    clear_cache()
    wave_fields_from_hos(stub, grid, p)          # warm-up (builds the cache)
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        wave_fields_from_hos(stub, grid, p)
        ts.append(time.perf_counter() - t0)
    med = float(np.median(ts))
    print(f"wave_fields_from_hos reduced grid: median {med*1e3:.2f} ms, min {min(ts)*1e3:.2f} ms")
    assert med < 15e-3


# ---------------------------------------------------------------------------
def test_with_real_hos_if_available():
    """Integration cross-check with wpt.hos.HOS (skipped if not importable):
    small-amplitude monochromatic wave, a few RK4 steps, compare with linear theory.
    The tolerance covers the RK4 phase error of HOS (not a coupling error)."""
    hos_mod = pytest.importorskip("wpt.hos")
    p = reduced(Ny=8, workers=1)
    grid = Grid(p)
    h = hos_mod.HOS(p)
    if not hasattr(h, "init_monochromatic"):
        pytest.skip("HOS.init_monochromatic not available")
    a, k, om = 1e-3 / p.k0, p.k0, p.omega0
    h.init_monochromatic(a, k)
    for _ in range(10):
        h.step(p.T0 / 80)
    wf = wave_fields_from_hos(h, grid, p)
    assert wf.t == h.t
    th = k * grid.x[:, None] - om * h.t
    for zz, s in ((grid.zc, "c"), (grid.zf, "f")):
        E = np.exp(k * zz)[None, :]
        assert _relerr(getattr(wf, "uphi_" + s)[:, 0, :], a * om * E * np.cos(th)) < 5e-5
        assert _relerr(getattr(wf, "wphi_" + s)[:, 0, :], a * om * E * np.sin(th)) < 5e-5
        assert _relerr(getattr(wf, "duphi_dz_" + s)[:, 0, :], a * om * k * E * np.cos(th)) < 5e-5
    assert _relerr(wf.eta[:, 0], a * np.cos(th[:, 0])) < 5e-5
