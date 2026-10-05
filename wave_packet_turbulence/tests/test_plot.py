"""Tests for scripts/plot_figures.py.

Synthetic ``results.npz`` files are produced by the real ``wpt.stats``
pipeline (``PacketAccumulator`` + ``packet_results`` and ``BaseAccumulator``),
either from random stub-solver states (end-to-end: every figure must be
written) or from prescribed accumulator sums with known answers (tables,
relative changes, spectral/budget normalisation, depth integration).
"""
from __future__ import annotations

import importlib.util
import math
import os
import types

import numpy as np
import pytest

from wpt.grid import Grid
from wpt.params import CASES, Params
from wpt.stats import BaseAccumulator, PacketAccumulator, packet_results, wrap_periodic
from wpt.wavefields import zero_wave_fields

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "plot_figures", os.path.join(ROOT, "scripts", "plot_figures.py"))
PF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PF)


def small_params(alpha=0.12, **kw):
    base = dict(alpha=alpha, Nx=48, Ny=8, Nz=12, workers=1, wave_on=True)
    base.update(kw)
    return Params(**base)


# ----------------------------------------------------------------------
# stub solver -> real accumulators
def _stub_solver(g, rng, t, scale=1.0):
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    z = g.zc
    prof = (1.0 + np.exp(6.0 * z)) * scale
    u = 5.0 * (1.0 + z) + prof * rng.standard_normal((Nx, Ny, Nz))
    v = prof * rng.standard_normal((Nx, Ny, Nz))
    w = np.zeros((Nx, Ny, Nz + 1))
    w[..., 1:-1] = 0.5 * rng.standard_normal((Nx, Ny, Nz - 1))
    p = rng.standard_normal((Nx, Ny, Nz))
    bc = {"dudz_s": g.p.Re_tau + rng.standard_normal((Nx, Ny)),
          "dvdz_s": rng.standard_normal((Nx, Ny)),
          "u_s": u[..., -1].copy()}
    nut = 1e-3 * (1.0 + rng.random((Nx, Ny, Nz)))
    return types.SimpleNamespace(u=u, v=v, w=w, p=p, bc=bc, t=t, nut=nut)


def make_results_random(alpha, seed=0, n_samples=3, n_members=2):
    p = small_params(alpha)
    g = Grid(p)
    times = np.linspace(p.t1, p.t2, n_samples)
    acc = PacketAccumulator(g, p, times, budget=True)
    rng = np.random.default_rng(seed)
    for _m in range(n_members):
        for i, t in enumerate(times):
            # amplitude growing towards the trailing side of the packet frame
            acc.add_sample(i, _stub_solver(g, rng, t, scale=1.0 + alpha), zero_wave_fields(g, t))
    return packet_results(acc, g, p)


def make_base_stats(seed=1, n=3):
    p = small_params(wave_on=False)
    g = Grid(p)
    acc = BaseAccumulator(g, p)
    rng = np.random.default_rng(seed)
    for k in range(n):
        acc.add_sample(_stub_solver(g, rng, 0.1 * k))
    return acc.results()


def write_case(data_dir, case, res):
    d = os.path.join(data_dir, case)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "results.npz"), **res)


def expected_files(cases, base=True, budget=True):
    names = []
    if base:
        names.append("fig_base_flow.png")
    for c in cases:
        names += [f"fig02_enstrophy_{c}.png", f"fig03_enstrophy_ratio_{c}.png",
                  f"fig07_reynolds_{c}.png", f"fig08_reynolds_ratio_{c}.png",
                  f"fig12_spectrum_w_{c}.png", f"fig13_spectrum_v_{c}.png",
                  f"fig14_spectrum_u_{c}.png"]
        if budget:
            names += [f"fig15_budget_w_{c}.png", f"fig16_budget_v_{c}.png",
                      f"fig17_budget_u_{c}.png", f"fig_budget_closure_{c}.png"]
    if cases:
        names += ["fig04_enstrophy_change_x.png", "fig05_enstrophy_change_profile.png",
                  "fig06_enstrophy_x_alpha2.png", "fig09_reynolds_change_x.png",
                  "fig10_reynolds_change_x_alpha2.png",
                  "fig11_reynolds_change_profile_alpha2.png",
                  "table1_enstrophy.csv", "table1_enstrophy.txt",
                  "table2_reynolds.csv", "table2_reynolds.txt"]
    return names


# ----------------------------------------------------------------------
def test_all_figures_end_to_end(tmp_path):
    data, out = tmp_path / "data", tmp_path / "fig"
    for n, (case, alpha) in enumerate(sorted(CASES.items())):
        write_case(str(data), case, make_results_random(alpha, seed=n))
    os.makedirs(data / "base")
    np.savez(data / "base" / "base_stats.npz", **make_base_stats())
    rc = PF.main(["--data-dir", str(data), "--out", str(out), "--strict"])
    assert rc == 0
    for name in expected_files(["W12", "W09", "W06"]):
        path = out / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
        if name.endswith(".png"):
            with open(path, "rb") as fh:
                assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    # tables: one row per case, CSV header + 3 rows
    rows = (out / "table2_reynolds.csv").read_text().strip().splitlines()
    assert len(rows) == 4 and rows[0].startswith("Case,alpha,Eu_pct")
    assert [r.split(",")[0] for r in rows[1:]] == ["W12", "W09", "W06"]


def test_missing_cases_and_base_skipped(tmp_path):
    data, out = tmp_path / "data", tmp_path / "fig"
    write_case(str(data), "W09", make_results_random(0.09))
    rc = PF.main(["--data-dir", str(data), "--out", str(out)])
    assert rc == 0
    produced = sorted(os.listdir(out))
    assert sorted(expected_files(["W09"], base=False)) == produced
    # nothing at all: still succeeds, nothing written
    out2 = tmp_path / "fig2"
    rc = PF.main(["--data-dir", str(tmp_path / "empty"), "--out", str(out2)])
    assert rc == 0
    assert os.listdir(out2) == []


def test_no_budget_skips_budget_figures(tmp_path):
    p = small_params(0.06)
    g = Grid(p)
    times = np.linspace(p.t1, p.t2, 2)
    acc = PacketAccumulator(g, p, times, budget=False)
    rng = np.random.default_rng(3)
    for i, t in enumerate(times):
        acc.add_sample(i, _stub_solver(g, rng, t), None)
    write_case(str(tmp_path / "data"), "W06", packet_results(acc, g, p))
    out = tmp_path / "fig"
    rc = PF.main(["--data-dir", str(tmp_path / "data"), "--out", str(out),
                  "--cases", "W06", "--strict"])
    assert rc == 0
    assert sorted(os.listdir(out)) == sorted(expected_files(["W06"], base=False, budget=False))


# ----------------------------------------------------------------------
# prescribed accumulator sums with known answers
C_X = {"oxox": 0.10, "oyoy": 0.05, "ozoz": 0.02, "uu": 0.03, "vv": 0.07, "ww": 0.2}


def q_of(xc):
    """Cubic in x'/chi (reproduced exactly by the not-a-knot spline):
    q(3) = 0 (leading edge), q(-3) = 2 (trailing edge), q(0) = 1."""
    return 1.0 - xc ** 3 / 27.0


def make_results_prescribed(alpha=0.12, Nz=24):
    p = small_params(alpha, Nz=Nz)
    g = Grid(p)
    times = np.linspace(p.t1, p.t2, 2)
    acc = PacketAccumulator(g, p, times, budget=True)
    acc.n_added[:] = 1
    xc = wrap_periodic(np.arange(g.Nx) * g.dx, g.Lx) / p.chi      # packet frame x'_j/chi
    ez = np.exp(p.k0 * g.zc)
    for k, c in C_X.items():
        F = ez[None, :] * (2.0 + c * q_of(xc)[:, None])            # (Nx, Nz)
        acc.mom[k][:] = g.Ny * F[None]
    my = np.arange(1, acc.nky + 1)
    for k in acc.spec:
        # B(x', ky, z) = q(x') * my * e^{k0 z} (times n_added.sum())
        acc.spec[k][:] = 2.0 * q_of(xc)[:, None, None] * my[None, :, None] * ez[None, None, :]
    return packet_results(acc, g, p), p


def test_table_values_exact(tmp_path):
    res, p = make_results_prescribed(0.12)
    cd = PF.CaseData("W12", res)
    for k, c in C_X.items():
        r, Fl, Ft = PF.integrated_change(cd, k)
        # F = e^{k0 z} (2 + c q): Delta F / F_lead = c q(-3) / (2 + c q(3)) = c
        assert abs(r - c) < 1e-12, (k, r, c)
        assert Fl > 0
        # relative change at fixed depths and trailing-edge profile
        for kz in PF.DEPTHS:
            rx = PF.rel_change_x(cd, k, kz)
            np.testing.assert_allclose(rx, 0.5 * c * q_of(cd.xc), atol=1e-12)
        np.testing.assert_allclose(PF.rel_change_profile(cd, k), c, atol=1e-12)
    out = tmp_path / "t"
    os.makedirs(out)
    text = PF.table1([cd], str(out), [])
    assert "10.0 %" in text and f"{0.1 / 0.12 ** 2:.1f}" in text
    PF.table2([cd], str(out), [])
    vals = (out / "table2_reynolds.csv").read_text().splitlines()[1].split(",")
    np.testing.assert_allclose([float(v) for v in vals[2:]],
                               [3.0, 0.03 / 0.0144, 7.0, 0.07 / 0.0144,
                                20.0, 0.2 / 0.0144], rtol=1e-6)


def test_spectrum_and_budget_normalisation():
    res, p = make_results_prescribed(0.09)
    cd = PF.CaseData("W09", res)
    my = np.arange(1, cd.ky.size + 1)
    ez = np.exp(p.k0 * cd.zc)
    # spectra: Phi = q(x') my e^{k0 z}; premultiplied ky Phi / dky = q my^2 e^{k0 z}
    lead = PF.premult(cd, cd.at_x(cd["Phi_w"], PF.X_LEAD))
    trail = PF.premult(cd, cd.at_x(cd["Phi_w"], PF.X_TRAIL))
    np.testing.assert_allclose(lead, 0.0, atol=1e-12)
    np.testing.assert_allclose(trail, 2.0 * my[:, None] ** 2 * ez[None, :], rtol=1e-12)
    # budget: Delta at x' = 0 from x' = 3 chi: (q(0)-q(3)) my^2 e^{k0z} / (alpha omega0)
    for key in ("Pw_z", "Pis_y", "Tp_x", "A_x"):
        Z = PF.budget_change(cd, key)
        fac = 2.0 if key.startswith("A_") else 1.0     # A = Aphi + Amean
        np.testing.assert_allclose(
            Z, fac * my[:, None] ** 2 * ez[None, :] / (p.alpha * p.omega0), rtol=1e-12)
    assert math.isclose(cd.budget_scale, p.alpha * p.omega0, rel_tol=1e-14)


def test_integrate_depth_second_order():
    k0 = 12.0
    exact = (1.0 - math.exp(-2.0)) / k0
    errs = []
    Ns = (24, 48, 96, 192)
    for Nz in Ns:
        g = Grid(Params(Nx=8, Ny=4, Nz=Nz, workers=1))
        val = PF.integrate_depth(np.exp(k0 * g.zc), g.zc, -2.0 / k0)
        errs.append(abs(val - exact) / exact)
    orders = [math.log(errs[i] / errs[i + 1], 2) for i in range(len(errs) - 1)]
    print("integrate_depth rel errors", errs, "orders", orders)
    assert errs[-1] < 1e-3
    assert min(orders) > 1.8
    # vectorised over leading axes, and linear functions are integrated exactly
    g = Grid(Params(Nx=8, Ny=4, Nz=16, workers=1))
    f = np.stack([1.0 + 0 * g.zc, 3.0 * g.zc + 2.0])
    np.testing.assert_allclose(PF.integrate_depth(f, g.zc, -0.3), [0.3, -1.5 * 0.09 + 0.6],
                               rtol=1e-12)


def test_at_x_window_and_grid_point():
    res, _ = make_results_prescribed(0.06)
    cd = PF.CaseData("W06", res)
    a = cd["uu"]
    np.testing.assert_array_equal(cd.at_x(a, 0.0), a[np.argmin(np.abs(cd.xc))])
    with pytest.raises(ValueError):
        cd.at_x(a, cd.xc[-1] + 0.5)
