"""Tests for wpt/stats.py (SPEC section 6) with synthetic fields and a stub solver."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.fft as sfft

from wpt.grid import Grid
from wpt.params import Params
from wpt.stats import (BUDGET_KEYS, MOMENT_KEYS, BaseAccumulator, PacketAccumulator,
                       fourier_shift_x, packet_frame_x, packet_results)

WAVE_KEYS = ("uphi_c", "wphi_c", "duphi_dx_c", "duphi_dz_c", "dwphi_dx_c", "dwphi_dz_c")


def make_params(**kw):
    base = dict(Re_tau=300.0, Lx=2.0 * math.pi, Ly=2.0 * math.pi, Nx=32, Ny=16, Nz=8,
                z_stretch=1.5, hos_N=64, workers=1)
    base.update(kw)
    return Params(**base)


@pytest.fixture(scope="module")
def setup():
    p = make_params()
    return p, Grid(p)


def integer_times(p, g, ns):
    """Sample times for which x0 + cg t is an integer number of grid cells."""
    dx = g.dx
    n0 = p.x0 / dx
    assert abs(n0 - round(n0)) < 1e-12
    shifts = np.array([3, 7, 12, 20, 29][:ns])
    return shifts * dx / p.cg, (shifts + int(round(n0))) % g.Nx


def bandlimited(g, rng, shape, mx_max, my_frac=1.0 / 3.0):
    """Random real field with |mx| < mx_max and my < Ny*my_frac."""
    a = rng.standard_normal(shape)
    ah = sfft.rfft2(a, axes=(0, 1))
    keep = (np.abs(g.mx) < mx_max).reshape(-1, 1) & (g.my < g.Ny * my_frac).reshape(1, -1)
    ah *= keep.reshape(keep.shape + (1,) * (len(shape) - 2))
    return sfft.irfft2(ah, s=(g.Nx, g.Ny), axes=(0, 1))


def random_wave_fields(g, rng, mx_max, t=None):
    wf = SimpleNamespace(t=t)
    for k in WAVE_KEYS:
        a = rng.standard_normal((g.Nx, g.Nz))
        ah = np.fft.rfft(a, axis=0)
        ah[np.arange(ah.shape[0]) >= mx_max] = 0.0
        setattr(wf, k, np.fft.irfft(ah, n=g.Nx, axis=0)[:, None, :])
    return wf


def stub(u, v, w, p_, dudz_s, dvdz_s, t, nut=None):
    return SimpleNamespace(u=u, v=v, w=w, p=p_, bc={"dudz_s": dudz_s, "dvdz_s": dvdz_s},
                           t=t, nut=nut)


def random_state(g, rng, mx_max=None, scale=1.0, mean=None):
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    if mx_max is None:
        f = lambda s: rng.standard_normal(s)
    else:
        f = lambda s: bandlimited(g, rng, s, mx_max)
    u = scale * f((Nx, Ny, Nz))
    if mean is not None:
        u = u + mean
    v = scale * f((Nx, Ny, Nz))
    w = scale * f((Nx, Ny, Nz + 1))
    w[..., 0] = 0.0
    pr = f((Nx, Ny, Nz))
    bu = f((Nx, Ny, 1))[..., 0]
    bv = f((Nx, Ny, 1))[..., 0]
    return u, v, w, pr, bu, bv


# ----------------------------------------------------------------------
# brute-force reference (lab frame, physical space, then np.roll)
def dx_lab(g, a):
    ah = sfft.rfft2(a, axes=(0, 1))
    kx = np.array(g.kx).reshape((g.Nx, 1) + (1,) * (a.ndim - 2))
    kx = kx.copy()
    kx[g.Nx // 2] = 0.0
    return sfft.irfft2(1j * kx * ah, s=(g.Nx, g.Ny), axes=(0, 1))


def dy_lab(g, a):
    ah = sfft.rfft2(a, axes=(0, 1))
    ky = np.array(g.ky).reshape((1, g.Nyh) + (1,) * (a.ndim - 2)).copy()
    ky[:, -1] = 0.0
    return sfft.irfft2(1j * ky * ah, s=(g.Nx, g.Ny), axes=(0, 1))


def lab_quantities(g, u, v, w, pr, bu, bv):
    wc = g.f2c(w)
    q = dict(u=u, v=v, w=wc, p=pr)
    q["dudx"], q["dvdx"], q["dwdx"] = dx_lab(g, u), dx_lab(g, v), dx_lab(g, wc)
    q["dudz"] = g.ddz_c(u, 0.0, bu)
    q["dvdz"] = g.ddz_c(v, 0.0, bv)
    q["dwdz"] = g.ddz_f2c(w)
    q["dudy"], q["dwdy"] = dy_lab(g, u), dy_lab(g, wc)
    q["ox"] = q["dwdy"] - q["dvdz"]
    q["oy"] = q["dudz"] - q["dwdx"]
    q["oz"] = q["dvdx"] - q["dudy"]
    return q


def yhat(a, Ny, nky):
    return np.fft.rfft(a, axis=1)[:, 1:nky + 1] / Ny


# ----------------------------------------------------------------------
def test_fourier_shift_exact_1d():
    L, N = 3.7, 24
    x = np.arange(N) * L / N
    rng = np.random.default_rng(1)
    amps, phs = rng.standard_normal(11), rng.uniform(0, 2 * np.pi, 11)

    def f(xx):
        return sum(a * np.cos(2 * np.pi * m / L * xx + ph)
                   for m, (a, ph) in enumerate(zip(amps, phs)))
    for s in (0.123, -1.9, 5.55, 0.0):
        err = np.abs(fourier_shift_x(f(x), s, L) - f(x + s)).max()
        assert err < 1e-12, (s, err)
    # 2-D, other axis
    a2 = np.stack([f(x), 2 * f(x)])
    assert np.allclose(fourier_shift_x(a2, 0.7, L, axis=1), np.stack([f(x + 0.7), 2 * f(x + 0.7)]),
                       atol=1e-12)


def test_fourier_shift_exact_in_accumulator(setup):
    """Band-limited u: packet-frame mean and (pointwise) second moment are exact
    at x'_j = j dx for a non-integer shift."""
    p, g = setup
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    x = g.x.reshape(-1, 1, 1)
    y = g.y.reshape(1, -1, 1)
    z = g.zc.reshape(1, 1, -1)
    kx = 2 * np.pi / g.Lx

    def F(xx):        # spanwise mean part, |mx| <= 10 < Nx/3
        return np.cos(3 * kx * xx + 0.3) + 0.5 * np.sin(10 * kx * xx - 1.0)

    def Hh(xx):       # amplitude of the cos(2y) part
        return 1.0 + 0.7 * np.cos(5 * kx * xx + 0.1)
    gz = 1.0 + z
    u = F(x) * gz + Hh(x) * np.cos(2 * y) * gz ** 2
    zeros = np.zeros((Nx, Ny, Nz))
    w = np.zeros((Nx, Ny, Nz + 1))
    t = 0.0123456                       # non-integer shift
    acc = PacketAccumulator(g, p, [t], budget=False)
    acc.add_sample(0, stub(u, zeros, w, zeros, np.zeros((Nx, Ny)), np.zeros((Nx, Ny)), t), None)
    s = p.x0 + p.cg * t
    assert abs((s / g.dx) - round(s / g.dx)) > 0.1
    xp = np.arange(Nx).reshape(-1, 1) * g.dx
    zz = g.zc.reshape(1, -1)
    U_ex = F(xp + s) * (1 + zz)
    uu_ex = 0.5 * (Hh(xp + s) * (1 + zz) ** 2) ** 2
    U = acc.mom["u"][0] / Ny
    uu = acc.mom["uu"][0] / Ny - U ** 2
    assert np.abs(U - U_ex).max() < 1e-12
    assert np.abs(uu - uu_ex).max() < 1e-12
    # spectrum at my = 2 is (H gz^2)^2 / 2, all others zero
    Phi = acc.spec["Phi_u"]
    assert np.abs(Phi[:, 1, :] - uu_ex).max() < 1e-12
    assert np.abs(np.delete(Phi, 1, axis=1)).max() < 1e-12


def test_spectrum_cos3y(setup):
    p, g = setup
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    u = np.broadcast_to(np.cos(3 * g.y).reshape(1, -1, 1), (Nx, Ny, Nz)).copy()
    zeros = np.zeros((Nx, Ny, Nz))
    w = np.zeros((Nx, Ny, Nz + 1))
    acc = PacketAccumulator(g, p, [0.02], budget=True)
    acc.add_sample(0, stub(u, zeros, w, zeros, np.zeros((Nx, Ny)), np.zeros((Nx, Ny)), 0.02), None)
    r = packet_results(acc, g, p)
    ky = r["ky"]
    i3 = int(np.argmin(np.abs(ky - 3.0)))
    assert abs(ky[i3] - 3.0) < 1e-12
    assert np.allclose(r["Phi_u"][:, i3, :], 0.5, atol=1e-13)
    assert np.abs(np.delete(r["Phi_u"], i3, axis=1)).max() < 1e-13
    assert np.allclose(r["Phi_u"].sum(axis=1), r["uu"], atol=1e-13)
    assert np.allclose(r["uu"], 0.5, atol=1e-13)
    assert np.allclose(r["U"], 0.0, atol=1e-13)
    # oz = -du/dy = 3 sin(3y): <oz^2> = 4.5
    assert np.allclose(r["ozoz"], 4.5, atol=1e-11)
    assert np.allclose(r["oxox"], 0.0, atol=1e-11) and np.allclose(r["oyoy"], 0.0, atol=1e-11)
    # window cut
    assert np.all(np.abs(r["xp"]) <= 3.5 * p.chi + 1e-12)
    assert np.all(np.diff(r["xp"]) > 0)
    assert r["Phi_u"].shape == (r["xp"].size, (Ny - 1) // 2, Nz)


def test_reynolds_stress_ensemble_bruteforce(setup):
    """<u> + random ensembles (not band-limited), integer shifts: compare every
    packet-frame mean / stress / enstrophy component with a brute-force
    evaluation, and the stress with its statistical value."""
    p, g = setup
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    rng = np.random.default_rng(2)
    times, nsh = integer_times(p, g, 3)
    nruns = 6
    xx = g.x.reshape(-1, 1, 1)
    zz = g.zc.reshape(1, 1, -1)
    acc = PacketAccumulator(g, p, times, budget=False)
    store = {k: [[] for _ in times] for k in ("u", "v", "w", "ox", "oy", "oz")}
    for r in range(nruns):
        for i, t in enumerate(times):
            mean = np.sin(xx + 0.5 * i) * (1 + zz) * 3.0     # <u>(x, z, t)
            u, v, w, pr, bu, bv = random_state(g, rng, scale=0.5, mean=mean)
            acc.add_sample(i, stub(u, v, w, pr, bu, bv, t), None)
            q = lab_quantities(g, u, v, w, pr, bu, bv)
            for k in store:
                store[k][i].append(np.roll(q[k], -nsh[i], axis=0))
    assert acc.n_runs == nruns and np.all(acc.n_added == nruns)
    res = packet_results(acc, g, p, window=1e9)       # full domain
    xw, order = packet_frame_x(Nx, g.Lx)
    assert np.allclose(res["xp"], xw[order])
    ref = {}
    for k in store:
        arr = np.array(store[k])                      # (ns, runs, Nx, Ny, Nz)
        ref[k] = arr.mean(axis=(1, 3))
        ref[k + "2"] = (arr ** 2).mean(axis=(1, 3))
    ref_uw = (np.array(store["u"]) * np.array(store["w"])).mean(axis=(1, 3))
    tol = 1e-11
    assert np.abs(res["U"] - ref["u"].mean(0)[order]).max() < tol
    assert np.abs(res["W"] - ref["w"].mean(0)[order]).max() < tol
    for key, k in (("uu", "u"), ("vv", "v"), ("ww", "w"),
                   ("oxox", "ox"), ("oyoy", "oy"), ("ozoz", "oz")):
        rs = (ref[k + "2"] - ref[k] ** 2).mean(0)[order]
        assert np.abs(res[key] - rs).max() < tol * max(1.0, np.abs(rs).max()), key
    rs = (ref_uw - ref["u"] * ref["w"]).mean(0)[order]
    assert np.abs(res["uw"] - rs).max() < tol
    # statistical check: u = <u> + N(0, 0.25) -> <u'^2> ~ 0.25 * (n-1)/n
    nn = nruns * Ny
    assert abs(res["uu"].mean() - 0.25 * (nn - 1) / nn) < 0.02
    assert abs(res["uw"].mean()) < 0.02
    # the mean flow is recovered in the packet frame
    s = p.x0 + p.cg * times
    Uex = np.mean([np.sin(xw[order][:, None] + s[i] + 0.5 * i) * (1 + g.zc) * 3.0
                   for i in range(3)], axis=0)
    assert np.abs(res["U"] - Uex).max() < 0.15
    assert res["uu_t"].shape == (3, Nx, Nz)


def _budget_case(p, g, rng, times, nsh, mx_max):
    """One run with band-limited fields; returns accumulator and brute-force terms."""
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    nky = (Ny - 1) // 2
    acc = PacketAccumulator(g, p, times, budget=True)
    ref = {k: 0.0 for k in BUDGET_KEYS + ("Phi_u", "Phi_v", "Phi_w", "lhsx_u")}
    for i, t in enumerate(times):
        xx = g.x.reshape(-1, 1, 1)
        u, v, w, pr, bu, bv = random_state(g, rng, mx_max=mx_max)
        u = u + 2.0 * np.cos(xx) * (1 + g.zc)               # non-zero spanwise mean
        wf = random_wave_fields(g, rng, mx_max, t)
        acc.add_sample(i, stub(u, v, w, pr, bu, bv, t), wf)
        q = lab_quantities(g, u, v, w, pr, bu, bv)
        R = {k: np.roll(a, -nsh[i], axis=0) for k, a in q.items()}
        Wv = {k: np.roll(getattr(wf, k), -nsh[i], axis=0) for k in WAVE_KEYS}
        ubar = R["u"].mean(axis=1, keepdims=True)
        vbar = R["v"].mean(axis=1, keepdims=True)
        wbar = R["w"].mean(axis=1, keepdims=True)
        dbar = {"ux": R["dudx"].mean(1, keepdims=True), "uz": R["dudz"].mean(1, keepdims=True),
                "vx": R["dvdx"].mean(1, keepdims=True), "vz": R["dvdz"].mean(1, keepdims=True),
                "wx": R["dwdx"].mean(1, keepdims=True), "wz": R["dwdz"].mean(1, keepdims=True)}
        del vbar
        h = {k: yhat(R[k], Ny, nky) for k in ("u", "v", "w", "p")}

        def B(c, F):
            return 4.0 * np.real(np.conj(h[c]) * yhat(F, Ny, nky))
        ref["Phi_u"] = ref["Phi_u"] + 2 * np.abs(h["u"]) ** 2
        ref["Phi_v"] = ref["Phi_v"] + 2 * np.abs(h["v"]) ** 2
        ref["Phi_w"] = ref["Phi_w"] + 2 * np.abs(h["w"]) ** 2
        ref["Pw_x"] += B("u", -R["u"] * Wv["duphi_dx_c"] - R["w"] * Wv["duphi_dz_c"])
        ref["Pw_z"] += B("w", -R["u"] * Wv["dwphi_dx_c"] - R["w"] * Wv["dwphi_dz_c"])
        ref["Pr_x"] += B("u", -R["u"] * dbar["ux"] - R["w"] * dbar["uz"])
        ref["Pr_y"] += B("v", -R["u"] * dbar["vx"] - R["w"] * dbar["vz"])
        ref["Pr_z"] += B("w", -R["u"] * dbar["wx"] - R["w"] * dbar["wz"])
        hp = h["p"]
        ref["Pis_x"] += 4 * np.real(np.conj(yhat(R["dudx"], Ny, nky)) * hp)
        ref["Pis_y"] += 4 * np.real(np.conj(yhat(dy_lab(g, R["v"]), Ny, nky)) * hp)
        ref["Pis_z"] += 4 * np.real(np.conj(yhat(R["dwdz"], Ny, nky)) * hp)
        # pressure diffusion: spectral x-derivative of the correlation (exact here
        # because |mx| < Nx/4 keeps the product un-aliased)
        Cx = 4 * np.real(np.conj(h["u"]) * hp)
        Cxh = np.fft.fft(Cx, axis=0)
        kx = 2 * np.pi / g.Lx * np.fft.fftfreq(Nx, 1.0 / Nx)
        ref["Tp_x"] -= np.real(np.fft.ifft(1j * kx[:, None, None] * Cxh, axis=0))
        ref["Tp_z"] -= np.gradient(4 * np.real(np.conj(h["w"]) * hp), g.zc, axis=-1,
                                   edge_order=2)
        for c, comp, dx_, dz_ in (("u", "x", "dudx", "dudz"), ("v", "y", "dvdx", "dvdz"),
                                  ("w", "z", "dwdx", "dwdz")):
            ref["Aphi_" + comp] += B(c, -Wv["uphi_c"] * R[dx_] - Wv["wphi_c"] * R[dz_])
            ref["Amean_" + comp] += B(c, -ubar * R[dx_] - wbar * R[dz_])
            Phi = 2 * np.abs(h[c]) ** 2
            ref["dPhidx_" + comp] += np.real(np.fft.ifft(1j * kx[:, None, None]
                                                         * np.fft.fft(Phi, axis=0), axis=0))
    return acc, ref


def test_budget_terms_bruteforce(setup):
    p, g = setup
    rng = np.random.default_rng(3)
    times, nsh = integer_times(p, g, 2)
    acc, ref = _budget_case(p, g, rng, times, nsh, mx_max=g.Nx / 4)
    errs = {}
    for k in BUDGET_KEYS + ("Phi_u", "Phi_v", "Phi_w"):
        scale = max(np.abs(ref[k]).max(), 1e-300)
        errs[k] = np.abs(acc.spec[k] - ref[k]).max() / scale
    bad = {k: e for k, e in errs.items() if e > 1e-10}
    assert not bad, bad
    # results dict: lhs = -cg dPhi/dx', A = Aphi + Amean, zeros for Pw_y, Tp_y
    r = packet_results(acc, g, p)
    assert np.allclose(r["lhs_z"], -p.cg * r["dPhidx_z"])
    assert np.allclose(r["A_x"], r["Aphi_x"] + r["Amean_x"])
    assert np.all(r["Pw_y"] == 0) and np.all(r["Tp_y"] == 0)


def test_pressure_strain_sums_to_zero(setup):
    """Discretely divergence-free synthetic field: sum_i Pis_i = 0."""
    p, g = setup
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    rng = np.random.default_rng(4)
    keep = g.dealias
    Uh = g.fft(rng.standard_normal((Nx, Ny, Nz))) * keep
    Vh = g.fft(rng.standard_normal((Nx, Ny, Nz))) * keep
    Wh = g.fft(rng.standard_normal((Nx, Ny, Nz + 1))) * keep
    Wh[..., 0] = 0.0
    kx, ky = g.kx, g.ky
    Wh[0, 0, :] = 0.0                       # (0,0) mode: dw/dz = 0, w(bottom) = 0
    dW = g.ddz_f2c(Wh)
    # kx = 0, ky != 0: v from continuity
    kx0 = (kx[:, 0, 0] == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vfix = -dW / (1j * ky)
        ufix = -(1j * ky * Vh + dW) / (1j * kx)
    Vh[kx0, 1:, :] = vfix[kx0, 1:, :]
    Uh[~kx0] = -(1j * ky * Vh[~kx0] + dW[~kx0]) / (1j * kx[~kx0])
    Uh *= keep
    Vh *= keep
    u, v, w = g.ifft(Uh), g.ifft(Vh), g.ifft(Wh)
    div = g.ifft(1j * kx * g.fft(u) + 1j * ky * g.fft(v)) + g.ddz_f2c(w)
    assert np.abs(div).max() < 1e-12 * np.abs(u).max() / g.dx
    pr = g.ifft(g.fft(rng.standard_normal((Nx, Ny, Nz))) * keep)
    t = 0.0171                               # non-integer shift
    acc = PacketAccumulator(g, p, [t], budget=True)
    wf = random_wave_fields(g, rng, Nx / 3, t)
    acc.add_sample(0, stub(u, v, w, pr, np.zeros((Nx, Ny)), np.zeros((Nx, Ny)), t), wf)
    S = acc.spec
    tot = S["Pis_x"] + S["Pis_y"] + S["Pis_z"]
    scale = max(np.abs(S[k]).max() for k in ("Pis_x", "Pis_y", "Pis_z"))
    assert scale > 1e-3
    assert np.abs(tot).max() < 1e-12 * scale


def test_merge_equals_single_run_and_io(setup, tmp_path):
    p, g = setup
    rng = np.random.default_rng(5)
    times = np.array([0.011, 0.023, 0.037])
    states = [[random_state(g, rng) for _ in times] for _ in range(3)]
    wfs = [random_wave_fields(g, rng, g.Nx / 3, t) for t in times]
    big = PacketAccumulator(g, p, times)
    parts = [PacketAccumulator(g, p, times) for _ in range(3)]
    for r in range(3):
        for i, t in enumerate(times):
            if r == 2 and i == 2:
                continue                    # incomplete member: one sample missing
            st = stub(*states[r][i], t)
            big.add_sample(i, st, wfs[i])
            parts[r].add_sample(i, st, wfs[i])
    parts[1].save(tmp_path / "a1.npz")
    loaded = PacketAccumulator.load(tmp_path / "a1.npz")
    assert loaded.budget and loaded.Nx == g.Nx
    m = parts[0]
    m.merge(loaded)
    m.merge(parts[2])
    assert np.array_equal(m.n_added, big.n_added) and list(m.n_added) == [3, 3, 2]
    for k in MOMENT_KEYS:
        assert np.allclose(m.mom[k], big.mom[k], rtol=1e-13, atol=1e-12)
    for k in big.spec:
        assert np.allclose(m.spec[k], big.spec[k], rtol=1e-13, atol=1e-12)
    r1, r2 = packet_results(m, g, p), packet_results(big, g, p)
    for k, a in r2.items():
        if isinstance(a, np.ndarray) and a.dtype.kind == "f":
            assert np.allclose(r1[k], a, rtol=1e-12, atol=1e-12), k
    assert r1["n_runs"] == 3 and r1["n_samples"] == 3
    # incompatible merges are refused; sample time mismatch is detected
    other = PacketAccumulator(g, p, times[:2])
    with pytest.raises(ValueError):
        m.merge(other)
    with pytest.raises(ValueError):
        big.add_sample(0, stub(*states[0][0], 0.5), wfs[0])
    # wave fields of another instant (e.g. one step behind) are refused
    with pytest.raises(ValueError):
        big.add_sample(0, stub(*states[0][0], times[0]), wfs[1])


def test_memory_estimate():
    """Memory of one accumulator follows the documented formula (no allocation
    of the reduced preset here: checked with the tiny test grid)."""
    p = make_params()
    g = Grid(p)
    ns = 5
    acc = PacketAccumulator(g, p, np.linspace(0.01, 0.04, ns), budget=True)
    nky = (g.Ny - 1) // 2
    expect = 8 * (len(MOMENT_KEYS) * ns * g.Nx * g.Nz
                  + (3 + len(BUDGET_KEYS)) * g.Nx * nky * g.Nz)
    assert abs(acc.nbytes - expect) < 1000


# ----------------------------------------------------------------------
def test_base_accumulator(setup, tmp_path):
    p, g = setup
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    nu = p.nu
    # U(z) with discrete face gradient nu dU/dz_f = 1 + z_f at interior faces
    U = np.zeros(Nz)
    for k in range(1, Nz):
        U[k] = U[k - 1] + (1 + g.zf[k]) * g.dzf[k] / nu
    y = g.y.reshape(1, -1, 1)
    A = (1 + g.zc) ** 2
    Bf = np.sin(np.pi * (g.zf + 1))
    u = U + A * np.cos(2 * y) + np.zeros((Nx, 1, 1))
    v = 0.3 * np.sin(5 * y) * np.ones((Nx, 1, Nz))
    w = Bf * np.cos(2 * y) * np.ones((Nx, 1, 1))
    dudz_s = np.full((Nx, Ny), 1.0 / nu)
    sol = stub(u, v, w, np.zeros((Nx, Ny, Nz)), dudz_s, np.zeros((Nx, Ny)), 1.0)
    acc = BaseAccumulator(g, p)
    acc.add_sample(sol)
    sol.t = 2.0
    acc.add_sample(sol)
    r = acc.results()
    Bc = g.f2c(Bf)
    assert np.allclose(r["U"], U)
    assert np.allclose(r["uu"], 0.5 * A ** 2)
    assert np.allclose(r["ww"], 0.5 * Bc ** 2)
    assert np.allclose(r["uw"], 0.5 * A * Bc)
    assert np.allclose(r["vv"], 0.045)
    assert np.allclose(r["nu_dUdz"], nu * g.ddz_c(U, 0.0, 1.0 / nu))
    # face flux: uf' = interpolated A cos 2y -> <w uf'> = Bf Af / 2
    Af = g.c2f(A)
    assert np.allclose(r["uw_f"][1:-1], 0.5 * Bf[1:-1] * Af[1:-1])
    # total stress without fluctuations is exactly 1 + z at the faces
    sol0 = stub(np.broadcast_to(U, (Nx, Ny, Nz)).copy(), np.zeros((Nx, Ny, Nz)),
                np.zeros((Nx, Ny, Nz + 1)), np.zeros((Nx, Ny, Nz)), dudz_s,
                np.zeros((Nx, Ny)), 0.0)
    b0 = BaseAccumulator(g, p)
    b0.add_sample(sol0)
    r0 = b0.results()
    assert np.allclose(r0["total_f"], r0["total_f_expected"], atol=1e-12)
    # constant eddy viscosity adds nu_t dU/dz to the total stress at interior faces
    sol0.nut = np.full((Nx, Ny, Nz), 2.0 * nu)
    b1 = BaseAccumulator(g, p)
    b1.add_sample(sol0)
    r1 = b1.results()
    assert np.allclose(r1["tau13_f"][1:-1], -2.0 * (1 + g.zf[1:-1]))
    assert np.allclose(r1["tau13"], -2.0 * nu * g.ddz_c(U, 0.0, 1.0 / nu))
    assert np.allclose(r1["total_f"][1:-1], 3.0 * (1 + g.zf[1:-1]))
    # spectra: v = 0.3 sin(5y) -> Phi_v(ky=5) = 0.045
    assert np.allclose(r["Phi_v"][4], 0.045) and np.allclose(np.delete(r["Phi_v"], 4, 0), 0)
    # io + merge
    acc.save(tmp_path / "b.npz")
    b2 = BaseAccumulator.load(tmp_path / "b.npz")
    b2.merge(acc)
    r2 = b2.results()
    assert r2["n_samples"] == 4 and r2["t_first"] == 1.0
    for k in ("U", "uu", "uw", "uw_f", "Phi_v", "total"):
        assert np.allclose(r2[k], r[k])


# ----------------------------------------------------------------------
# adversarial tests added in review
class _Trig:
    """Real trigonometric series in x with random (y, z) coefficients, |m| <= M
    < Nx/2, so that it can be evaluated (with its x derivative) at arbitrary x:
    an independent reference for the Fourier shift to the packet frame."""

    def __init__(self, rng, Lx, M, tail):
        self.k = 2 * np.pi / Lx * np.arange(M + 1)
        self.A = rng.standard_normal((M + 1,) + tail)
        self.B = rng.standard_normal((M + 1,) + tail)
        self.B[0] = 0.0

    def __call__(self, X, deriv=False):
        kx = np.outer(X, self.k)
        C, Sn = np.cos(kx), np.sin(kx)
        if deriv:
            C, Sn = -self.k * Sn, self.k * C
        return np.tensordot(C, self.A, axes=(1, 0)) + np.tensordot(Sn, self.B, axes=(1, 0))


def _dy(a, Ny):
    ah = np.fft.rfft(a, axis=1)
    ky = np.arange(ah.shape[1]) * 2 * np.pi / (2 * np.pi)   # Ly = 2 pi in make_params
    ky = ky.astype(float)
    if Ny % 2 == 0:
        ky[-1] = 0.0
    return np.fft.irfft(1j * ky.reshape(1, -1, 1) * ah, n=Ny, axis=1)


@pytest.mark.parametrize("shape", [(32, 16, 8), (27, 15, 7)])
def test_all_terms_noninteger_shift_analytic(shape):
    """Every moment, spectrum and budget term for NON-integer packet shifts (and
    odd Nx, Ny) against an independent evaluation of trigonometric series at
    x'_j + s (no FFT shift involved in the reference)."""
    Nx, Ny, Nz = shape
    p = make_params(Nx=Nx, Ny=Ny, Nz=Nz)
    g = Grid(p)
    rng = np.random.default_rng(11)
    M = Nx // 3 - 1
    nky = (Ny - 1) // 2
    times = np.array([0.0113, 0.0257, 0.0391])
    acc = PacketAccumulator(g, p, times, budget=True)
    refm = {k: np.zeros((times.size, Nx, Nz)) for k in MOMENT_KEYS}
    refs = {k: 0.0 for k in BUDGET_KEYS + ("Phi_u", "Phi_v", "Phi_w")}
    for i, t in enumerate(times):
        s = p.x0 + p.cg * t
        assert abs(s / g.dx - round(s / g.dx)) > 0.05
        T = {k: _Trig(rng, g.Lx, M, (Ny, Nz)) for k in ("u", "v", "p")}
        T["w"] = _Trig(rng, g.Lx, M, (Ny, Nz + 1))
        T["w"].A[..., 0] = 0.0
        T["w"].B[..., 0] = 0.0
        T["bu"] = _Trig(rng, g.Lx, M, (Ny,))
        T["bv"] = _Trig(rng, g.Lx, M, (Ny,))
        TW = {k: _Trig(rng, g.Lx, M, (1, Nz)) for k in WAVE_KEYS}
        # lab frame input
        X = g.x
        wf = SimpleNamespace(t=t, **{k: TW[k](X) for k in WAVE_KEYS})
        acc.add_sample(i, stub(T["u"](X), T["v"](X), T["w"](X), T["p"](X),
                               T["bu"](X), T["bv"](X), t), wf)
        # independent reference at x'_j + s
        Xp = X + s
        u, v, wf_, pr = T["u"](Xp), T["v"](Xp), T["w"](Xp), T["p"](Xp)
        ux, vx, wfx, px = (T[k](Xp, True) for k in ("u", "v", "w", "p"))
        Wv = {k: TW[k](Xp) for k in WAVE_KEYS}
        wc, wcx = g.f2c(wf_), g.f2c(wfx)
        uz = g.ddz_c(u, 0.0, T["bu"](Xp))
        vz = g.ddz_c(v, 0.0, T["bv"](Xp))
        wz = g.ddz_f2c(wf_)
        ox = _dy(wc, Ny) - vz
        oy = uz - wcx
        oz = vx - _dy(u, Ny)
        for k, a in (("u", u), ("v", v), ("w", wc), ("uu", u * u), ("vv", v * v),
                     ("ww", wc * wc), ("uw", u * wc), ("ox", ox), ("oy", oy), ("oz", oz),
                     ("oxox", ox * ox), ("oyoy", oy * oy), ("ozoz", oz * oz)):
            refm[k][i] = a.sum(axis=1)
        h = {k: yhat(a, Ny, nky) for k, a in (("u", u), ("v", v), ("w", wc), ("p", pr),
                                               ("ux", ux), ("vx", vx), ("wx", wcx),
                                               ("px", px))}
        hp = h["p"]

        def B(c, F):
            return 4.0 * np.real(np.conj(h[c]) * yhat(F, Ny, nky))
        bar = {k: a.mean(axis=1, keepdims=True) for k, a in
               (("u", u), ("w", wc), ("ux", ux), ("uz", uz), ("vx", vx), ("vz", vz),
                ("wx", wcx), ("wz", wz))}
        for c in ("u", "v", "w"):
            refs["Phi_" + c] = refs["Phi_" + c] + 2 * np.abs(h[c]) ** 2
        refs["Pw_x"] += B("u", -u * Wv["duphi_dx_c"] - wc * Wv["duphi_dz_c"])
        refs["Pw_z"] += B("w", -u * Wv["dwphi_dx_c"] - wc * Wv["dwphi_dz_c"])
        refs["Pr_x"] += B("u", -u * bar["ux"] - wc * bar["uz"])
        refs["Pr_y"] += B("v", -u * bar["vx"] - wc * bar["vz"])
        refs["Pr_z"] += B("w", -u * bar["wx"] - wc * bar["wz"])
        refs["Pis_x"] += 4 * np.real(np.conj(h["ux"]) * hp)
        refs["Pis_y"] += 4 * np.real(np.conj(yhat(_dy(v, Ny), Ny, nky)) * hp)
        refs["Pis_z"] += 4 * np.real(np.conj(yhat(wz, Ny, nky)) * hp)
        refs["Tp_x"] -= 4 * np.real(np.conj(h["ux"]) * hp + np.conj(h["u"]) * h["px"])
        refs["Tp_z"] -= np.gradient(4 * np.real(np.conj(h["w"]) * hp), g.zc, axis=-1,
                                    edge_order=2)
        for c, comp, dx_, dz_ in (("u", "x", ux, uz), ("v", "y", vx, vz), ("w", "z", wcx, wz)):
            refs["Aphi_" + comp] += B(c, -Wv["uphi_c"] * dx_ - Wv["wphi_c"] * dz_)
            refs["Amean_" + comp] += B(c, -bar["u"] * dx_ - bar["w"] * dz_)
            refs["dPhidx_" + comp] += 4 * np.real(np.conj(h[c]) * h[c[0] + "x"])
    bad = {}
    for k in MOMENT_KEYS:
        e = np.abs(acc.mom[k] - refm[k]).max() / np.abs(refm[k]).max()
        if e > 1e-11:
            bad[k] = e
    for k in refs:
        e = np.abs(acc.spec[k] - refs[k]).max() / np.abs(refs[k]).max()
        if e > 1e-11:
            bad[k] = e
    assert not bad, bad


def test_tp_z_pis_z_second_order():
    """Vertical pressure diffusion / pressure strain converge at second order in
    the max norm (including the surface and bottom cells) on the stretched grid."""
    errs_t, errs_p = [], []
    for Nz in (12, 24, 48, 96):
        p = make_params(Nx=4, Ny=4, Nz=Nz)
        g = Grid(p)
        Nx, Ny = g.Nx, g.Ny

        def a(z):
            return np.sin(2.0 * z) + 0.5 * z ** 2

        def da(z):
            return 2.0 * np.cos(2.0 * z) + z

        def b(z):
            return np.exp(1.3 * z) + 0.2

        def db(z):
            return 1.3 * np.exp(1.3 * z)
        cy = np.cos(g.y).reshape(1, -1, 1)
        w = a(g.zf) * cy * np.ones((Nx, 1, 1))
        pr = b(g.zc) * cy * np.ones((Nx, 1, 1))
        z0 = np.zeros((Nx, Ny, Nz))
        acc = PacketAccumulator(g, p, [0.02], budget=True)
        acc.add_sample(0, stub(z0, z0, w, pr, np.zeros((Nx, Ny)), np.zeros((Nx, Ny)), 0.02),
                       None)
        zc = g.zc
        # c_w = c_p-like amplitude/2 at my = 1 -> 4 Re(conj c_w c_p) = a b
        tp_ex = -(da(zc) * b(zc) + a(zc) * db(zc))
        pis_ex = da(zc) * b(zc)
        errs_t.append(np.abs(acc.spec["Tp_z"][:, 0, :] - tp_ex).max())
        errs_p.append(np.abs(acc.spec["Pis_z"][:, 0, :] - pis_ex).max())
    ot = np.log2(np.array(errs_t[:-1]) / np.array(errs_t[1:]))
    op = np.log2(np.array(errs_p[:-1]) / np.array(errs_p[1:]))
    assert np.all(ot[1:] > 1.8), (errs_t, ot)
    assert np.all(op[1:] > 1.8), (errs_p, op)


def test_packet_results_rejects_other_params(setup):
    p, g = setup
    acc = PacketAccumulator(g, p, [0.02], budget=False)
    z0 = np.zeros((g.Nx, g.Ny, g.Nz))
    acc.add_sample(0, stub(z0, z0, np.zeros((g.Nx, g.Ny, g.Nz + 1)), z0,
                           np.zeros((g.Nx, g.Ny)), np.zeros((g.Nx, g.Ny)), 0.02), None)
    packet_results(acc, g, p.with_(workers=3))          # workers is irrelevant
    with pytest.raises(ValueError):
        packet_results(acc, g, p.with_(alpha=0.06))      # other case -> wrong metadata
    with pytest.raises(ValueError):
        packet_results(acc, g, p.with_(x0_frac=0.5))     # other frame


def test_save_load_without_suffix(setup, tmp_path):
    p, g = setup
    acc = PacketAccumulator(g, p, [0.02], budget=False)
    acc.save(tmp_path / "acc")                 # np.savez appends .npz
    b = PacketAccumulator.load(tmp_path / "acc")
    assert b.n_samples == 1 and not b.budget
    ba = BaseAccumulator(g, p)
    ba.save(str(tmp_path / "base"))
    assert BaseAccumulator.load(str(tmp_path / "base")).n == 0


def test_base_accumulator_sgs_average_over_sgs_samples(setup):
    """Samples without nu_t (e.g. right after set_state) must not dilute <tau13>."""
    p, g = setup
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    U = (1 + g.zc) ** 2
    u = np.broadcast_to(U, (Nx, Ny, Nz)).copy()
    z0 = np.zeros((Nx, Ny, Nz))
    dudz_s = np.full((Nx, Ny), 2.0)
    s0 = stub(u, z0, np.zeros((Nx, Ny, Nz + 1)), z0, dudz_s, np.zeros((Nx, Ny)), 0.0)
    acc = BaseAccumulator(g, p)
    acc.add_sample(s0)                       # nut None
    s0.nut = np.full((Nx, Ny, Nz), 0.01)
    acc.add_sample(s0)
    r = acc.results()
    assert r["n_sgs"] == 1 and r["n_samples"] == 2
    assert np.allclose(r["tau13"], -0.01 * g.ddz_c(U, 0.0, 2.0))
    assert np.allclose(r["tau13_f"][1:-1], -0.01 * (U[1:] - U[:-1]) / g.dzf[1:-1])


def test_with_real_les_solver():
    """Integration with the actual LESSolver interface: after one step of a
    discretely divergence-free state, sum_i Pis_i = 0 to round-off and the
    surface data in solver.bc are accepted."""
    les = pytest.importorskip("wpt.les")
    from wpt.wavefields import zero_wave_fields
    p = make_params(Nx=24, Ny=12, Nz=8, sgs="none")
    g = Grid(p)
    rng = np.random.default_rng(7)
    sol = les.LESSolver(p, g, None)
    u = bandlimited(g, rng, (g.Nx, g.Ny, g.Nz), g.Nx / 3)
    v = bandlimited(g, rng, (g.Nx, g.Ny, g.Nz), g.Nx / 3)
    w = bandlimited(g, rng, (g.Nx, g.Ny, g.Nz + 1), g.Nx / 3)
    w[..., 0] = 0.0
    w[..., -1] = 0.0
    sol.set_state(u, v, w, t=0.0)
    wf0 = zero_wave_fields(g, 0.0)
    dt = 1e-4
    sol.step(dt, wf0, zero_wave_fields(g, dt))
    assert np.abs(sol.p).max() > 0
    acc = PacketAccumulator(g, p, [sol.t], budget=True)
    acc.add_sample(0, sol, zero_wave_fields(g, sol.t))
    S = acc.spec
    tot = S["Pis_x"] + S["Pis_y"] + S["Pis_z"]
    scale = max(np.abs(S[k]).max() for k in ("Pis_x", "Pis_y", "Pis_z"))
    assert scale > 0
    assert np.abs(tot).max() < 1e-10 * scale, np.abs(tot).max() / scale
