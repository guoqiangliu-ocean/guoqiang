"""Tests for wpt/sgs.py (SPEC.md section 5).

All tests use small grids and workers=1.  The surface Neumann data are given
by hand-built ``bc`` dicts (keys 'dudz_s', 'dvdz_s').
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from wpt.grid import Grid
from wpt.params import tiny
from wpt.sgs import DynamicSmagorinsky, Smagorinsky, make_sgs

MODELS = [DynamicSmagorinsky, Smagorinsky]


def make_grid(Nx=32, Ny=32, Nz=16, Lx=2 * math.pi, Ly=2 * math.pi, **kw):
    p = tiny(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, workers=1, **kw)
    return p, Grid(p)


def zero_bc(g, dudz=0.0, dvdz=0.0):
    return {"dudz_s": np.full((g.Nx, g.Ny), float(dudz)),
            "dvdz_s": np.full((g.Nx, g.Ny), float(dvdz))}


def discrete_div(g, u, v, w):
    """Discrete divergence of the solver (SPEC 4.1 step 4) at centres."""
    uh, vh = g.fft(u), g.fft(v)
    d = g.ifft(g.ddx(uh) + g.ddy(vh))
    return d + g.ddz_f2c(w)


def random_divfree(g, seed=0, slope=-4.0 / 3.0, surface_w=True, mean_shear=0.0):
    """Random, dealiased, discretely divergence-free field (u, v at centres,
    w at faces, w[...,0] = 0) with horizontal amplitude spectrum |q_k| ~ k^slope
    (k^-5/3 energy spectrum) and smooth random vertical structure."""
    rng = np.random.default_rng(seed)
    kh = np.sqrt(g.k2)                                   # (Nx, Nyh, 1)
    amp = np.where(kh > 0, np.maximum(kh, 1e-30) ** slope, 0.0)

    def rnd(nz):
        a = rng.standard_normal((g.Nx, g.Ny, nz))
        # smooth in z by a 3-point running mean
        a[..., 1:-1] = (a[..., :-2] + 2 * a[..., 1:-1] + a[..., 2:]) / 4.0
        return g.fft(a) * amp * g.dealias

    uh = rnd(g.Nz)
    vh = rnd(g.Nz)
    wh = rnd(g.Nz + 1)
    wh[..., 0] = 0.0
    if not surface_w:
        wh[..., -1] = 0.0
    wh[0, 0, :] = 0.0
    wz = g.ddz_f2c(wh)
    ikx = 1j * g.kx * np.ones_like(uh)
    iky = 1j * g.ky * np.ones_like(uh)
    pos = (g.ky[0, :, 0] > 0)
    # ky > 0: solve continuity for v
    vh[:, pos] = -(ikx[:, pos] * uh[:, pos] + wz[:, pos]) / iky[:, pos]
    # ky = 0, kx != 0: solve continuity for u
    kx0 = g.kx[:, 0, 0] != 0
    uh[kx0, 0] = -wz[kx0, 0] / ikx[kx0, 0]
    uh *= g.dealias
    vh *= g.dealias
    wh *= g.dealias
    u, v, w = g.ifft(uh), g.ifft(vh), g.ifft(wh)
    # normalise to rms ~ 1 and add a mean shear profile
    s = np.sqrt(np.mean(u ** 2) + np.mean(v ** 2))
    u, v, w = u / s, v / s, w / s
    if mean_shear:
        u = u + mean_shear * (g.zc + 1.0)
    return u, v, w


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("Model", MODELS)
def test_uniform_field_gives_zero(Model):
    p, g = make_grid()
    m = Model(g, p)
    u = np.full((g.Nx, g.Ny, g.Nz), 1.3)
    v = np.full((g.Nx, g.Ny, g.Nz), -0.4)
    w = np.zeros((g.Nx, g.Ny, g.Nz + 1))
    out = m.compute(u, v, w, zero_bc(g))
    assert out["div_x"].shape == (g.Nx, g.Ny, g.Nz)
    assert out["div_y"].shape == (g.Nx, g.Ny, g.Nz)
    assert out["div_z"].shape == (g.Nx, g.Ny, g.Nz + 1)
    assert out["nut"].shape == (g.Nx, g.Ny, g.Nz)
    assert out["cs2"].shape == (g.Nz,)
    for k in ("div_x", "div_y", "div_z", "nut"):
        assert np.abs(out[k]).max() < 1e-12, k
    if Model is DynamicSmagorinsky:
        assert np.all(out["cs2"] == 0.0)


@pytest.mark.parametrize("profile", ["linear", "cosine"])
def test_dynamic_zero_for_horizontally_uniform_shear(profile):
    """Linear (or any horizontally uniform) shear has no resolved scales
    between the test filter and the grid filter: L_ij = 0 -> Cs^2 = 0."""
    p, g = make_grid()
    m = DynamicSmagorinsky(g, p)
    z = g.zc
    if profile == "linear":
        a, b = 3.0, -1.5
        u = np.broadcast_to(a * (z + 1.0) + 0.7, (g.Nx, g.Ny, g.Nz)).copy()
        v = np.broadcast_to(b * z, (g.Nx, g.Ny, g.Nz)).copy()
        bc = zero_bc(g, a, b)
    else:
        u = np.broadcast_to(np.cos(3 * z), (g.Nx, g.Ny, g.Nz)).copy()
        v = np.broadcast_to(np.sin(2 * z), (g.Nx, g.Ny, g.Nz)).copy()
        bc = zero_bc(g, -3 * np.sin(0.0), 2 * np.cos(0.0))
    w = np.zeros((g.Nx, g.Ny, g.Nz + 1))
    out = m.compute(u, v, w, bc)
    S, Smag = m.strain_rate(u, v, w, bc)
    assert Smag.max() > 0.1                    # the field *is* sheared ...
    assert np.all(out["cs2"] == 0.0)           # ... but the dynamic model is off
    assert np.abs(out["nut"]).max() == 0.0
    for k in ("div_x", "div_y", "div_z"):
        assert np.abs(out[k]).max() < 1e-12


def test_smagorinsky_linear_shear_profile():
    """Constant-Cs model: Cs(z) = cs (1 - exp(-d+/25)), nu_t = Cs^2 Delta^2 |S|
    with |S| = a in the interior (S_13 = a/2), and the stress is carried
    between the boundary faces only (momentum conserving)."""
    p, g = make_grid(Re_tau=300.0, cs_const=0.12)
    m = Smagorinsky(g, p)
    a = 2.0
    u = np.broadcast_to(a * g.zc, (g.Nx, g.Ny, g.Nz)).copy()
    v = np.zeros_like(u)
    w = np.zeros((g.Nx, g.Ny, g.Nz + 1))
    out = m.compute(u, v, w, zero_bc(g, a, 0.0))
    dplus = -g.zc * p.Re_tau
    cs2 = (0.12 * (1 - np.exp(-dplus / 25.0))) ** 2
    np.testing.assert_allclose(out["cs2"], cs2, rtol=1e-14)
    delta2 = (g.dx * g.dy * g.dzc) ** (2.0 / 3.0)
    Smag = np.full(g.Nz, a)
    Smag[0] = a / 2.0                          # zero-gradient bottom face
    np.testing.assert_allclose(out["nut"][0, 0], cs2 * delta2 * Smag, rtol=1e-12)
    assert np.all(out["nut"] >= 0)
    # horizontal-mean momentum is conserved: sum_k div_x dzc = 0
    tot = np.sum(out["div_x"].mean(axis=(0, 1)) * g.dzc)
    assert abs(tot) < 1e-12 * np.abs(out["div_x"]).max()
    # interior: d/dz(-nu_t a) with face-interpolated nu_t
    nutf = g.c2f(out["nut"][0, 0], 0.0, 0.0)
    tau13 = -nutf * a
    tau13[0] = tau13[-1] = 0.0
    np.testing.assert_allclose(out["div_x"][0, 0], np.diff(tau13) / g.dzc,
                               rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("Model", MODELS)
def test_nut_nonnegative_and_finite(Model):
    p, g = make_grid(Nx=32, Ny=32, Nz=16)
    m = Model(g, p)
    for seed in range(3):
        u, v, w = random_divfree(g, seed=seed, mean_shear=5.0)
        out = m.compute(u, v, w, zero_bc(g, 5.0, 0.0))
        assert np.all(np.isfinite(out["nut"]))
        assert np.all(out["nut"] >= 0.0)
        assert np.all(out["cs2"] >= 0.0)
        for k in ("div_x", "div_y", "div_z"):
            assert np.all(np.isfinite(out[k]))


def test_dynamic_cs_random_field_finite():
    """Synthetic div-free random field with a k^-5/3 horizontal spectrum:
    Cs^2(z) must be finite, >= 0, non-trivial and of sensible magnitude."""
    p, g = make_grid(Nx=48, Ny=48, Nz=16)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=3)
    assert np.abs(discrete_div(g, u, v, w)).max() < 1e-10 * np.abs(u).max() / g.dx
    out = m.compute(u, v, w, zero_bc(g))
    cs2 = out["cs2"]
    print("Cs(z) random k^-5/3 field:", np.round(np.sqrt(cs2), 4))
    assert np.all(np.isfinite(cs2)) and np.all(cs2 >= 0)
    # a Gaussian random-phase field has no systematic inter-scale transfer, so
    # <L M> fluctuates in sign between levels; only finiteness / bounds are known
    raw = m.last_num / m.last_den
    print("raw <LM>/<MM>:", raw)
    assert np.all(np.isfinite(raw))
    assert np.count_nonzero(cs2) >= 1
    assert cs2.max() < 1.0                    # Cs < 1
    np.testing.assert_array_equal(cs2, np.maximum(raw, 0.0))
    assert np.all(np.isfinite(m.last_num)) and np.all(m.last_den > 0)


def test_tau_symmetry_and_trace():
    p, g = make_grid(Nx=32, Ny=32, Nz=16)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=1, mean_shear=2.0)
    bc = zero_bc(g, 2.0, 0.0)
    tau = m.stress_tensor(u, v, w, bc)
    scale = np.abs(tau).max()
    assert scale > 0
    assert np.abs(tau - tau.transpose(1, 0, 2, 3, 4)).max() == 0.0
    # deviatoric: trace = -2 nu_t div(u) = 0 for a discretely div-free field
    tr = tau[0, 0] + tau[1, 1] + tau[2, 2]
    assert np.abs(tr).max() < 1e-10 * scale
    # consistent with -2 nu_t S_ij used by compute()
    out = m.compute(u, v, w, bc)
    S, _ = m.strain_rate(u, v, w, bc)
    idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    for n, (i, j) in enumerate(idx):
        np.testing.assert_allclose(tau[i, j], -2 * out["nut"] * S[n],
                                   rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("Model", MODELS)
def test_xy_swap_equivariance(Model):
    """Exchanging x and y (u <-> v) on a square domain maps div_x <-> div_y."""
    p, g = make_grid(Nx=32, Ny=32, Nz=12)
    m = Model(g, p)
    u, v, w = random_divfree(g, seed=2, mean_shear=1.0)
    rng = np.random.default_rng(5)
    dus = rng.standard_normal((g.Nx, g.Ny))
    dvs = rng.standard_normal((g.Nx, g.Ny))
    o1 = m.compute(u, v, w, {"dudz_s": dus, "dvdz_s": dvs})
    T = lambda a: np.swapaxes(a, 0, 1).copy()
    o2 = m.compute(T(v), T(u), T(w), {"dudz_s": T(dvs), "dvdz_s": T(dus)})
    sc = np.abs(o1["div_x"]).max()
    assert np.abs(T(o2["div_y"]) - o1["div_x"]).max() < 1e-11 * sc
    assert np.abs(T(o2["div_x"]) - o1["div_y"]).max() < 1e-11 * sc
    assert np.abs(T(o2["div_z"]) - o1["div_z"]).max() < 1e-11 * sc
    np.testing.assert_allclose(o2["cs2"], o1["cs2"], rtol=1e-10, atol=1e-16)


def _analytic(g):
    """Smooth div-free field with S_13 = S_23 = 0 at z = 0 and z = -1:
    u = pi a cos(pi z), v = pi b cos(pi z), w = F sin(pi z),
    a = sin x cos 2y, b = cos x sin y, F = -(a_x + b_y)."""
    X = g.x[:, None, None]
    Y = g.y[None, :, None]
    pi = math.pi

    def fields(z):
        a = np.sin(X) * np.cos(2 * Y)
        b = np.cos(X) * np.sin(Y)
        F = -np.cos(X) * (np.cos(2 * Y) + np.cos(Y))
        lapF = np.cos(X) * (5 * np.cos(2 * Y) + 2 * np.cos(Y))
        c, s = np.cos(pi * z), np.sin(pi * z)
        u = pi * a * c
        v = pi * b * c
        w = F * s
        lu = pi * c * a * (-5 - pi ** 2)
        lv = pi * c * b * (-2 - pi ** 2)
        lw = s * (lapF - pi ** 2 * F)
        return u, v, w, lu, lv, lw

    u, v, _, lu, lv, _ = fields(g.zc)
    _, _, w, _, _, lw = fields(g.zf)
    return u, v, w, lu, lv, lw


@pytest.mark.parametrize("stretch", [0.0, 1.5])
def test_constant_nut_divergence_consistency(stretch):
    """div(-2 nu_t S) = -nu_t lap(u) for constant nu_t and a smooth
    div-free field.  Second-order convergence in dz everywhere on a uniform
    grid; on the stretched grid second order in the interior and first order
    in the boundary cells (property of the SPEC staggered operator
    [(q_{k+1}-q_k)/dzf_{k+1} - (q_k-q_{k-1})/dzf_k]/dzc_k next to a boundary
    face when dzc_{k-1}/dzc_k = 1 + O(dz); identical to the solver's viscous
    operator)."""
    nut0 = 0.37
    errs, errs_in = [], []
    Nzs = (16, 32, 64)
    for Nz in Nzs:
        p, g = make_grid(Nx=16, Ny=16, Nz=Nz, z_stretch=stretch)
        m = DynamicSmagorinsky(g, p)
        u, v, w, lu, lv, lw = _analytic(g)
        out = m.divergence_from_nut(u, v, w, zero_bc(g), nut0)
        assert np.all(out["div_z"][..., [0, -1]] == 0.0)
        ex = np.abs(out["div_x"] + nut0 * lu).max(axis=(0, 1)) / np.abs(nut0 * lu).max()
        ey = np.abs(out["div_y"] + nut0 * lv).max(axis=(0, 1)) / np.abs(nut0 * lv).max()
        ez = np.abs(out["div_z"] + nut0 * lw).max(axis=(0, 1))[1:-1] / np.abs(nut0 * lw).max()
        errs.append((ex.max(), ey.max(), ez.max()))
        errs_in.append((ex[1:-1].max(), ey[1:-1].max(), ez[1:-1].max()))
    errs, errs_in = np.array(errs), np.array(errs_in)
    orders = np.log2(errs[:-1] / errs[1:])
    orders_in = np.log2(errs_in[:-1] / errs_in[1:])
    print(f"stretch={stretch}: rel. max errors (x,y,z) =\n{errs}\norders =\n{orders}"
          f"\ninterior errors =\n{errs_in}\ninterior orders =\n{orders_in}")
    assert np.all(orders_in[-1] > 1.8)
    assert np.all(errs_in[-1] < 1e-3)
    if stretch == 0.0:
        assert np.all(orders[-1] > 1.8)
    else:
        assert np.all(orders[-1] > 0.9)
        assert np.all(errs[-1] < 1e-2)


def test_momentum_conservation_and_boundary_faces():
    p, g = make_grid(Nx=32, Ny=24, Nz=16, Ly=math.pi)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=4, mean_shear=3.0)
    rng = np.random.default_rng(1)
    bc = {"dudz_s": 3.0 + rng.standard_normal((g.Nx, g.Ny)),
          "dvdz_s": rng.standard_normal((g.Nx, g.Ny))}
    out = m.compute(u, v, w, bc)
    assert out["cs2"].max() > 0
    for key in ("div_x", "div_y"):
        tot = np.sum(out[key].mean(axis=(0, 1)) * g.dzc)
        assert abs(tot) < 1e-13 * np.abs(out[key]).max(), key
    for key in ("tau13_f", "tau23_f", "div_z"):
        assert np.all(out[key][..., 0] == 0.0) and np.all(out[key][..., -1] == 0.0)
    # outputs are dealiased: no energy outside the 2/3 mask
    for key in ("div_x", "div_y", "div_z"):
        sp = g.fft(out[key])
        assert np.abs(sp[~np.broadcast_to(g.dealias, sp.shape)]).max() < \
            1e-12 * np.abs(sp).max()
        np.testing.assert_allclose(g.ifft(out[key + "_h"]), out[key], atol=1e-13 * np.abs(out[key]).max())


def test_galilean_invariance_dynamic():
    p, g = make_grid(Nx=32, Ny=32, Nz=12)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=6)
    o1 = m.compute(u, v, w, zero_bc(g))
    o2 = m.compute(u + 4.0, v - 2.5, w, zero_bc(g))
    np.testing.assert_allclose(o2["cs2"], o1["cs2"], rtol=1e-9, atol=1e-14)
    sc = np.abs(o1["div_x"]).max()
    assert np.abs(o2["div_x"] - o1["div_x"]).max() < 1e-9 * sc


def test_surface_neumann_data_enter_strain():
    """S_13 in the top cell is the average of the interior face gradient and
    the imposed surface gradient; the bottom face gradient is zero."""
    p, g = make_grid(Nx=16, Ny=16, Nz=8)
    m = DynamicSmagorinsky(g, p)
    u = np.broadcast_to(g.zc ** 2, (g.Nx, g.Ny, g.Nz)).copy()
    v = np.zeros_like(u)
    w = np.zeros((g.Nx, g.Ny, g.Nz + 1))
    S, _ = m.strain_rate(u, v, w, zero_bc(g, 7.0, -2.0))
    dfu = np.diff(g.zc ** 2) / g.dzf[1:-1]
    np.testing.assert_allclose(S[4][0, 0, -1], 0.5 * 0.5 * (dfu[-1] + 7.0))
    np.testing.assert_allclose(S[4][0, 0, 0], 0.5 * 0.5 * (dfu[0] + 0.0))
    np.testing.assert_allclose(S[5][0, 0, -1], 0.5 * 0.5 * (-2.0))


def test_update_interval_and_factory():
    p, g = make_grid(Nx=24, Ny=24, Nz=8)
    m = make_sgs(g, p.with_(sgs="dynamic"))
    assert isinstance(m, DynamicSmagorinsky)
    assert isinstance(make_sgs(g, p.with_(sgs="smagorinsky")), Smagorinsky)
    assert make_sgs(g, p.with_(sgs="none")) is None
    m.update_interval = 3
    u, v, w = random_divfree(g, seed=7)
    c0 = m.compute(u, v, w, zero_bc(g))["cs2"]
    u2, v2, w2 = random_divfree(g, seed=8)
    c1 = m.compute(u2, v2, w2, zero_bc(g))["cs2"]
    np.testing.assert_array_equal(c0, c1)       # reused
    m.compute(u2, v2, w2, zero_bc(g))
    c3 = m.compute(u2, v2, w2, zero_bc(g))["cs2"]  # 4th call -> recomputed
    assert not np.array_equal(c3, c0)


def test_truncate_input_switch():
    """For already-dealiased input, skipping the input truncation is exact."""
    p, g = make_grid(Nx=24, Ny=24, Nz=8)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=9, mean_shear=1.0)
    o1 = m.compute(u, v, w, zero_bc(g, 1.0))
    m.truncate_input = False
    o2 = m.compute(u, v, w, zero_bc(g, 1.0))
    sc = np.abs(o1["div_x"]).max()
    assert np.abs(o1["div_x"] - o2["div_x"]).max() < 1e-12 * sc
    np.testing.assert_allclose(o1["cs2"], o2["cs2"], rtol=1e-10, atol=1e-16)


def test_timing_small():
    """Smoke timing on a 64 x 48 x 24 grid (single thread)."""
    p, g = make_grid(Nx=64, Ny=48, Nz=24, Ly=math.pi)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=10)
    m.compute(u, v, w, zero_bc(g))
    t0 = time.perf_counter()
    m.compute(u, v, w, zero_bc(g))
    dt = time.perf_counter() - t0
    print(f"dynamic compute 64x48x24, workers=1: {dt:.3f} s")
    assert dt < 5.0


def _reference_cs2(g, u, v, w, dus, dvs):
    """Straightforward (unoptimised) Germano-Lilly procedure: full rfft2
    transforms, explicit 3x3 deviatoric tensors."""
    tm = (np.abs(g.mx) < g.Nx / 6).reshape(-1, 1, 1) & (g.my < g.Ny / 6).reshape(1, -1, 1)
    filt = lambda a: g.ifft(g.fft(a) * tm)
    D = lambda a: g.ifft(g.fft(a) * g.dealias)
    u, v, w = D(u), D(v), D(w)
    D2 = lambda a2: np.fft.irfft2(np.fft.rfft2(a2) * g.dealias[:, :, 0], s=a2.shape)
    dus, dvs = D2(dus), D2(dvs)          # Neumann data are truncated as well
    dx = lambda a: g.ifft(g.ddx(g.fft(a)))
    dy = lambda a: g.ifft(g.ddy(g.fft(a)))

    def grad(u, v, w, dus, dvs):
        A = np.empty((3, 3) + u.shape)
        A[0, 0], A[0, 1], A[0, 2] = dx(u), dy(u), g.ddz_c(u, 0.0, dus)
        A[1, 0], A[1, 1], A[1, 2] = dx(v), dy(v), g.ddz_c(v, 0.0, dvs)
        A[2, 0], A[2, 1], A[2, 2] = g.f2c(dx(w)), g.f2c(dy(w)), g.ddz_f2c(w)
        S = 0.5 * (A + A.transpose(1, 0, 2, 3, 4))
        return S, np.sqrt(2 * np.sum(S * S, axis=(0, 1)))

    S, Sm = grad(u, v, w, dus, dvs)
    sf = lambda a2: np.fft.irfft2(np.fft.rfft2(a2) * tm[:, :, 0], s=a2.shape)
    Sf, Sfm = grad(filt(u), filt(v), filt(w), sf(dus), sf(dvs))
    vel = [u, v, g.f2c(w)]
    d2 = (g.dx * g.dy * g.dzc) ** (2 / 3)
    L = np.empty_like(S)
    M = np.empty_like(S)
    for i in range(3):
        for j in range(3):
            L[i, j] = filt(vel[i] * vel[j]) - filt(vel[i]) * filt(vel[j])
            M[i, j] = 2 * d2 * (filt(Sm * S[i, j]) - 2 ** (4 / 3) * Sfm * Sf[i, j])
    for T in (L, M):
        tr = (T[0, 0] + T[1, 1] + T[2, 2]) / 3
        for i in range(3):
            T[i, i] -= tr
    num = np.sum(L * M, axis=(0, 1, 2, 3))
    den = np.sum(M * M, axis=(0, 1, 2, 3))
    return np.maximum(num / den, 0.0), num / den


@pytest.mark.parametrize("Ny,Ly", [(32, 2 * math.pi), (24, math.pi)])
def test_dynamic_matches_reference_implementation(Ny, Ly):
    p, g = make_grid(Nx=36, Ny=Ny, Nz=10, Ly=Ly)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=11, mean_shear=4.0)
    # add some non-dealiased noise: the model must truncate its input
    rng = np.random.default_rng(3)
    u = u + 0.05 * rng.standard_normal(u.shape)
    rng2 = np.random.default_rng(4)
    dus = 4.0 + 0.5 * rng2.standard_normal((g.Nx, g.Ny))
    dvs = 0.5 * rng2.standard_normal((g.Nx, g.Ny))
    out = m.compute(u, v, w, {"dudz_s": dus, "dvdz_s": dvs})
    ref, raw = _reference_cs2(g, u, v, w, dus, dvs)
    print("Cs^2 model:", out["cs2"], "\nraw ratio ref:", raw)
    np.testing.assert_allclose(m.last_num / m.last_den, raw, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(out["cs2"], ref, rtol=1e-9, atol=1e-14)


# ---------------------------------------------------------------------------
# adversarial tests added in review
def _analytic_strain(g, z_c, z_f):
    """Exact S_ij of the field of :func:`_analytic` (centres for the
    centre-located components, faces for S_13, S_23)."""
    X = g.x[:, None, None]
    Y = g.y[None, :, None]
    pi = math.pi
    a = np.sin(X) * np.cos(2 * Y)
    b = np.cos(X) * np.sin(Y)
    ax, ay = np.cos(X) * np.cos(2 * Y), -2 * np.sin(X) * np.sin(2 * Y)
    bx, by = -np.sin(X) * np.sin(Y), np.cos(X) * np.cos(Y)
    F = -(ax + by)
    Fx = np.sin(X) * (np.cos(2 * Y) + np.cos(Y))
    Fy = np.cos(X) * (2 * np.sin(2 * Y) + np.sin(Y))
    c, s = np.cos(pi * z_c), np.sin(pi * z_c)
    S11, S22, S33 = pi * ax * c, pi * by * c, pi * F * c
    S12 = 0.5 * pi * c * (ay + bx)
    S13c = 0.5 * s * (-pi ** 2 * a + Fx)
    S23c = 0.5 * s * (-pi ** 2 * b + Fy)
    sf = np.sin(pi * z_f)
    S13f = 0.5 * sf * (-pi ** 2 * a + Fx)
    S23f = 0.5 * sf * (-pi ** 2 * b + Fy)
    return S11, S22, S33, S12, S13c, S23c, S13f, S23f


def _nut_field(g, z):
    X = g.x[:, None, None]
    Y = g.y[None, :, None]
    nut = 0.3 + 0.1 * np.cos(X) * np.sin(Y) + 0.1 * z ** 2
    nx = -0.1 * np.sin(X) * np.sin(Y) + 0 * z
    ny = 0.1 * np.cos(X) * np.cos(Y) + 0 * z
    nz = 0.2 * z + 0 * X * Y
    return nut, nx, ny, nz


@pytest.mark.parametrize("stretch", [0.0, 1.5])
def test_variable_nut_divergence_convergence(stretch):
    """div(-2 nu_t S) for nu_t(x, y, z) against the exact
    -nu_t lap(u_i) - 2 S_ij d_j nu_t.  Catches errors in the face interpolation
    of nu_t, in the off-diagonal factors and in the staggering of tau_33.
    Second order on the uniform grid (all cells) and in the interior of the
    stretched grid."""
    errs, errs_in = [], []
    for Nz in (16, 32, 64):
        p, g = make_grid(Nx=16, Ny=16, Nz=Nz, z_stretch=stretch)
        m = DynamicSmagorinsky(g, p)
        u, v, w, lu, lv, lw = _analytic(g)
        nut_c, nxc, nyc, nzc = _nut_field(g, g.zc)
        nut_f, nxf, nyf, nzf = _nut_field(g, g.zf)
        S11, S22, S33, S12, S13c, S23c, S13f, S23f = _analytic_strain(g, g.zc, g.zf)
        # exact divergence; S_31 / S_32 / S_33 at faces for the z equation
        S33f = math.pi * np.cos(math.pi * g.zf) * (
            -(np.cos(g.x[:, None, None]) * np.cos(2 * g.y[None, :, None])
              + np.cos(g.x[:, None, None]) * np.cos(g.y[None, :, None])))
        ex_x = -nut_c * lu - 2 * (S11 * nxc + S12 * nyc + S13c * nzc)
        ex_y = -nut_c * lv - 2 * (S12 * nxc + S22 * nyc + S23c * nzc)
        ex_z = -nut_f * lw - 2 * (S13f * nxf + S23f * nyf + S33f * nzf)
        out = m.divergence_from_nut(u, v, w, zero_bc(g), nut_c)
        e = [np.abs(out["div_x"] - ex_x).max(axis=(0, 1)) / np.abs(ex_x).max(),
             np.abs(out["div_y"] - ex_y).max(axis=(0, 1)) / np.abs(ex_y).max(),
             np.abs(out["div_z"] - ex_z).max(axis=(0, 1))[1:-1] / np.abs(ex_z).max()]
        errs.append([a.max() for a in e])
        errs_in.append([a[1:-1].max() for a in e])
    errs, errs_in = np.array(errs), np.array(errs_in)
    orders = np.log2(errs[:-1] / errs[1:])
    orders_in = np.log2(errs_in[:-1] / errs_in[1:])
    print(f"stretch={stretch}: errors\n{errs}\norders\n{orders}\ninterior orders\n{orders_in}")
    assert np.all(orders_in[-1] > 1.8)
    assert np.all(errs_in[-1] < 2e-3)
    if stretch == 0.0:
        assert np.all(orders[-1] > 1.8)
    else:
        assert np.all(orders[-1] > 0.9)


@pytest.mark.parametrize("Model", MODELS)
def test_discrete_dissipation_identity(Model):
    """Summation by parts: with w_s = 0 the discrete SGS work
    sum(u div_x dzc + v div_y dzc + w div_z dzf) equals
    sum 2 nu_t (S11^2+S22^2+S33^2+2 S12^2) dzc + sum nu_t,f (a_13^2 + a_23^2) dzf
    >= 0 (a_13 = du/dz + dw/dx at interior faces).  Catches sign / weight /
    staggering errors in the divergence."""
    p, g = make_grid(Nx=32, Ny=24, Nz=12, Ly=math.pi, z_stretch=1.5)
    m = Model(g, p)
    u, v, w = random_divfree(g, seed=12, mean_shear=2.0, surface_w=False)
    rng = np.random.default_rng(2)
    bc = {"dudz_s": 2.0 + rng.standard_normal((g.Nx, g.Ny)),
          "dvdz_s": rng.standard_normal((g.Nx, g.Ny))}
    out = m.compute(u, v, w, bc)
    assert np.abs(w[..., -1]).max() == 0.0
    work = (np.sum((u * out["div_x"] + v * out["div_y"]) * g.dzc)
            + np.sum((w * out["div_z"]) * g.dzf))
    S, _ = m.strain_rate(u, v, w, bc)
    nut = out["nut"]
    centre = np.sum(2 * nut * (S[0] ** 2 + S[1] ** 2 + S[2] ** 2 + 2 * S[3] ** 2) * g.dzc)
    nutf = g.c2f(nut, 0.0, 0.0)
    uz = g.ddz_c2f(u)
    vz = g.ddz_c2f(v)
    wx = g.ifft(g.ddx(g.fft(w)))
    wy = g.ifft(g.ddy(g.fft(w)))
    face = np.sum((nutf * ((uz + wx) ** 2 + (vz + wy) ** 2))[..., 1:-1] * g.dzf[1:-1])
    print(Model.__name__, "work", work, "centre+face", centre + face)
    assert centre + face > 0
    assert abs(work - (centre + face)) < 1e-11 * (centre + face)


@pytest.mark.parametrize("Nx,Ny,Ly", [(33, 27, math.pi), (35, 30, 2 * math.pi), (30, 25, math.pi)])
def test_dynamic_reference_odd_sizes(Nx, Ny, Ly):
    """Pruned test-filter transforms must equal full rfft2 + mask also for odd
    and non-multiple-of-6 grid sizes."""
    p, g = make_grid(Nx=Nx, Ny=Ny, Nz=8, Ly=Ly)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=13, mean_shear=3.0)
    rng = np.random.default_rng(7)
    v = v + 0.03 * rng.standard_normal(v.shape)          # non-dealiased noise
    dus = 3.0 + 0.3 * rng.standard_normal((g.Nx, g.Ny))
    dvs = 0.3 * rng.standard_normal((g.Nx, g.Ny))
    out = m.compute(u, v, w, {"dudz_s": dus, "dvdz_s": dvs})
    ref, raw = _reference_cs2(g, u, v, w, dus, dvs)
    np.testing.assert_allclose(m.last_num / m.last_den, raw, rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(out["cs2"], ref, rtol=1e-9, atol=1e-14)


def test_dynamic_invariances():
    """Cs^2(z) is invariant under velocity scaling (L and M both quadratic) and
    under periodic translations by whole grid cells; outputs translate."""
    p, g = make_grid(Nx=32, Ny=32, Nz=10)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=14, mean_shear=2.0)
    rng = np.random.default_rng(8)
    dus = 2.0 + rng.standard_normal((g.Nx, g.Ny))
    dvs = rng.standard_normal((g.Nx, g.Ny))
    o1 = m.compute(u, v, w, {"dudz_s": dus, "dvdz_s": dvs})
    raw1 = m.last_num / m.last_den
    lam = 37.0
    o2 = m.compute(lam * u, lam * v, lam * w, {"dudz_s": lam * dus, "dvdz_s": lam * dvs})
    np.testing.assert_allclose(m.last_num / m.last_den, raw1, rtol=1e-10, atol=1e-16)
    sc = np.abs(o1["div_x"]).max()
    assert np.abs(o2["div_x"] - lam ** 2 * o1["div_x"]).max() < 1e-9 * lam ** 2 * sc
    sh = lambda a: np.roll(np.roll(a, 5, axis=0), -3, axis=1)
    o3 = m.compute(sh(u), sh(v), sh(w), {"dudz_s": sh(dus), "dvdz_s": sh(dvs)})
    np.testing.assert_allclose(m.last_num / m.last_den, raw1, rtol=1e-10, atol=1e-16)
    for k in ("div_x", "div_y", "div_z", "nut"):
        assert np.abs(o3[k] - sh(o1[k])).max() < 1e-11 * np.abs(o1[k]).max(), k


@pytest.mark.parametrize("truncate_input", [True, False])
def test_inputs_not_modified(truncate_input):
    p, g = make_grid(Nx=24, Ny=24, Nz=8)
    m = DynamicSmagorinsky(g, p)
    m.truncate_input = truncate_input
    u, v, w = random_divfree(g, seed=15, mean_shear=1.0)
    bc = zero_bc(g, 1.0, 0.5)
    copies = [a.copy() for a in (u, v, w, bc["dudz_s"], bc["dvdz_s"])]
    for _ in range(2):
        m.compute(u, v, w, bc)
    for a, b in zip((u, v, w, bc["dudz_s"], bc["dvdz_s"]), copies):
        np.testing.assert_array_equal(a, b)


def test_scalar_bc_and_float32_input():
    """Scalar Neumann data and float32 inputs are accepted and give the same
    result as full float64 arrays."""
    p, g = make_grid(Nx=24, Ny=24, Nz=8)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=16, mean_shear=1.5)
    o1 = m.compute(u, v, w, zero_bc(g, 1.5, 0.0))
    o2 = m.compute(u, v, w, {"dudz_s": 1.5, "dvdz_s": 0.0})
    for k in ("div_x", "div_y", "div_z", "nut", "cs2"):
        np.testing.assert_allclose(o2[k], o1[k], rtol=1e-12, atol=1e-14)
        assert o2[k].dtype == np.float64
    o3 = m.compute(u.astype(np.float32), v.astype(np.float32), w.astype(np.float32),
                   {"dudz_s": 1.5, "dvdz_s": 0.0})
    assert o3["div_x"].dtype == np.float64
    assert np.abs(o3["div_x"] - o1["div_x"]).max() < 1e-4 * np.abs(o1["div_x"]).max()


def test_galilean_invariance_large_translation():
    """A large uniform translation (here 1e6, as an extreme mean current) must
    not trip the round-off guard of <M M>: Cs^2 unchanged to the precision
    left after the shift."""
    p, g = make_grid(Nx=32, Ny=32, Nz=12)
    m = DynamicSmagorinsky(g, p)
    u, v, w = random_divfree(g, seed=17, mean_shear=1.0)
    bc = zero_bc(g, 1.0, 0.0)
    o1 = m.compute(u, v, w, bc)
    raw1 = m.last_num / m.last_den
    assert o1["cs2"].max() > 0
    U = 1.0e6
    o2 = m.compute(u + U, v - U, w, bc)
    raw2 = m.last_num / m.last_den
    print("max |d raw|:", np.abs(raw2 - raw1).max(), "max raw:", np.abs(raw1).max())
    assert np.abs(raw2 - raw1).max() < 1e-6 * np.abs(raw1).max()
    np.testing.assert_allclose(o2["cs2"], o1["cs2"], rtol=1e-6, atol=1e-8 * o1["cs2"].max())


def test_bc_none_means_zero_gradients():
    p, g = make_grid(Nx=16, Ny=16, Nz=8)
    m = Smagorinsky(g, p)
    u, v, w = random_divfree(g, seed=18)
    o1 = m.compute(u, v, w, zero_bc(g))
    o2 = m.compute(u, v, w, None)
    np.testing.assert_array_equal(o1["div_x"], o2["div_x"])
