#!/usr/bin/env python3
"""Figures 2-17 and tables 1-2 of Xuan, Deng & Shen (2024), JFM 999 A45, from
the packet-following statistics written by ``scripts/combine.py``
(``data/<case>/results.npz``, see ``wpt.stats.packet_results``), plus a
base-flow validation figure from ``data/base/base_stats.npz``
(``wpt.stats.BaseAccumulator.results``).

Usage::

    python3 scripts/plot_figures.py [--data-dir data] [--out figures]
                                    [--cases W12 W09 W06] [--strict]

Conventions (paper sections 4-5; units H = u_* = rho = 1)
---------------------------------------------------------
* abscissa ``x'/chi`` in [-3, 3]; the leading edge is ``x' = +3 chi`` (the packet
  moves towards +x), the trailing edge ``x' = -3 chi``, the core ``x' = 0``;
* depth ``k0 z`` in [-2, 0] (figs 2-11) or [-1.5, 0] (spectra, figs 12-17);
* enstrophy divided by ``(u_*^2/nu)^2 = Re_tau^2``, stresses by ``u_*^2``;
* changes ``Delta f = f(x') - f(x' = 3 chi)`` relative to the leading edge;
* spectra premultiplied, ``ky Phi / dky`` (``Phi`` from stats.py is the energy
  of one discrete spanwise mode, so ``Phi/dky`` is the spectral density),
  versus ``lambda_y / H`` on a log axis;
* spectral-budget terms premultiplied, ``ky B / dky``, and divided by
  ``alpha u_*^2 omega0``, shown as changes from the leading edge at ``x' = 0``;
* tables 1-2: integrals from ``k0 z = -2`` to the surface (eq. 4.1, 4.2).

Numerical choices (documented deviations)
-----------------------------------------
* ``results.npz`` only holds the window ``|x'| <= window chi`` (default 3.5), so
  the values at ``x' = +-3 chi`` are obtained by a not-a-knot cubic spline in
  ``x'`` (scipy ``CubicSpline``) instead of the Fourier interpolation mentioned
  in SPEC section 6 (that would need the full periodic record).  ``x' = 0`` is a
  grid point of the packet frame, so the spline is exact there.
* values at a fixed depth (``k0 z = -0.5, -0.8``) are linear interpolations
  between cell centres.
* depth integrals use the trapezoidal rule on the stretched cell-centre grid,
  with nodes ``[z_D, centres inside (z_D, 0), 0]``; the value at ``z_D = -2/k0`` is
  linearly interpolated and the surface value ``z = 0`` (not stored) is linearly
  extrapolated from the two top centres.
* colour levels are chosen from the 0.5-99.5 % percentiles of the data shown,
  with common levels for panels that the paper plots on a common scale
  (ratio maps, spectra at leading/trailing edge, budget terms).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import traceback

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wpt.params import CASES  # noqa: E402

# paper styling: W12 blue circles, W09 orange squares, W06 green down-triangles
CASE_STYLE = {
    "W12": dict(color="C0", marker="o", ls="-"),
    "W09": dict(color="C1", marker="s", ls="--"),
    "W06": dict(color="C2", marker="v", ls="-."),
}
DEPTHS = (-0.5, -0.8)                 # k0 z of figs 4, 6a, 9, 10
DEPTH_LS = {-0.5: "-", -0.8: "--"}
X_LEAD, X_TRAIL, X_CORE = 3.0, -3.0, 0.0   # x'/chi
KZ_INT = -2.0                         # lower limit k0 D of eq. (4.1), (4.2)
ZLIM_MAIN = (-2.0, 0.0)
ZLIM_SPEC = (-1.5, 0.0)

REQUIRED_KEYS = ("xp", "zc", "k0z", "alpha", "chi", "k0", "Re_tau", "omega0",
                 "uu", "vv", "ww", "oxox", "oyoy", "ozoz")
SPEC_KEYS = ("ky", "lam_y", "dky", "Phi_u", "Phi_v", "Phi_w")
BUDGET_FIG_KEYS = ("Pw_x", "Pw_z", "Pis_x", "Pis_y", "Pis_z", "Tp_x", "Tp_z",
                   "A_x", "A_y", "A_z")

ENST = (("oxox", r"\omega_x"), ("oyoy", r"\omega_y"), ("ozoz", r"\omega_z"))
REYN = (("uu", "u"), ("vv", "v"), ("ww", "w"))


def _log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# data access
class CaseData:
    """Packet-following results of one case (dict from ``packet_results``)."""

    def __init__(self, name, d):
        self.name = name
        self.d = d
        self.alpha = float(d["alpha"])
        self.chi = float(d["chi"])
        self.k0 = float(d["k0"])
        self.Re_tau = float(d["Re_tau"])
        self.omega0 = float(d["omega0"])
        self.xp = np.asarray(d["xp"], dtype=float)
        order = np.argsort(self.xp, kind="stable")
        if not np.all(order == np.arange(self.xp.size)):
            raise ValueError(f"{name}: xp is not sorted")
        self.xc = self.xp / self.chi                    # x'/chi
        self.zc = np.asarray(d["zc"], dtype=float)
        self.k0z = np.asarray(d["k0z"], dtype=float)
        self.has_spec = all(k in d for k in SPEC_KEYS)
        self.has_budget = bool(np.asarray(d.get("budget", False))) and \
            all(k in d for k in BUDGET_FIG_KEYS)
        if self.has_spec:
            self.ky = np.asarray(d["ky"], dtype=float)
            self.lam_y = np.asarray(d["lam_y"], dtype=float)
            self.dky = float(d["dky"])
        bs = d.get("budget_scale")
        # stats.py: budget_scale = alpha * omega0 (u_* = 1)
        self.budget_scale = float(bs) if bs is not None else self.alpha * self.omega0

    def __getitem__(self, k):
        return np.asarray(self.d[k], dtype=float)

    def at_x(self, a, xc0):
        """Value of ``a`` (x' along axis 0) at x'/chi = xc0 (cubic spline)."""
        lo, hi = self.xc[0], self.xc[-1]
        tol = 1e-9 * max(1.0, abs(xc0))
        if xc0 < lo - tol or xc0 > hi + tol:
            raise ValueError(f"{self.name}: x'/chi = {xc0} outside the stored window "
                             f"[{lo:.3f}, {hi:.3f}]")
        j = np.flatnonzero(np.abs(self.xc - xc0) <= tol)
        if j.size:                                     # grid point: exact
            return np.asarray(a, dtype=float)[j[0]]
        return CubicSpline(self.xc, np.asarray(a, dtype=float), axis=0)(xc0)

    def at_k0z(self, a, kz):
        """Linear interpolation along the last (z) axis at k0 z = kz."""
        return interp_last(np.asarray(a, dtype=float), self.k0z, kz)

    def lead(self, key):
        return self.at_x(self[key], X_LEAD)

    def trail(self, key):
        return self.at_x(self[key], X_TRAIL)


def interp_last(f, z, z0):
    """Linear interpolation (or extrapolation beyond the end intervals) of
    f(..., Nz), given at ascending z, at the scalar z0."""
    j = int(np.clip(np.searchsorted(z, z0), 1, z.size - 1))
    w = (z0 - z[j - 1]) / (z[j] - z[j - 1])
    return (1.0 - w) * f[..., j - 1] + w * f[..., j]


def integrate_depth(f, zc, z_bot, z_top=0.0):
    """Trapezoidal integral of f(..., Nz) (cell centres zc, ascending) over
    [z_bot, z_top] on the stretched grid.  Nodes: z_bot (linear
    interpolation), all centres strictly inside, z_top (linear interpolation,
    or extrapolation from the two top centres when z_top > zc[-1])."""
    f = np.asarray(f, dtype=float)
    zc = np.asarray(zc, dtype=float)
    if z_bot < zc[0]:
        raise ValueError(f"z_bot = {z_bot} below the first centre {zc[0]}")
    inside = (zc > z_bot) & (zc < z_top)
    zn = np.concatenate(([z_bot], zc[inside], [z_top]))
    fn = np.concatenate((interp_last(f, zc, z_bot)[..., None], f[..., inside],
                         interp_last(f, zc, z_top)[..., None]), axis=-1)
    return np.trapezoid(fn, zn, axis=-1)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_cases(data_dir, cases):
    out = {}
    for c in cases:
        path = os.path.join(data_dir, c, "results.npz")
        if not os.path.exists(path):
            _log(f"[skip] {c}: {path} not found")
            continue
        try:
            d = load_npz(path)
        except Exception as e:   # corrupt file
            _log(f"[skip] {c}: cannot read {path}: {e}")
            continue
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            _log(f"[skip] {c}: {path} lacks keys {missing}")
            continue
        out[c] = CaseData(c, d)
        _log(f"[load] {c}: alpha={out[c].alpha} n_runs={int(d.get('n_runs', -1))} "
             f"n_samples={int(d.get('n_samples', -1))} nxp={out[c].xp.size} "
             f"Nz={out[c].zc.size} spectra={out[c].has_spec} budget={out[c].has_budget}")
    return out


# ----------------------------------------------------------------------
# plotting helpers
def _levels(arrs, n=17, sym=False, pct=(0.5, 99.5)):
    v = np.concatenate([np.ravel(np.asarray(a, dtype=float)) for a in arrs])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(-1.0, 1.0, n) if sym else np.linspace(0.0, 1.0, n)
    if sym:
        m = float(np.percentile(np.abs(v), pct[1]))
        if not m > 0:
            m = 1.0
        return np.linspace(-m, m, n)
    lo, hi = (float(x) for x in np.percentile(v, pct))
    if not hi > lo:
        d = max(abs(lo), 1.0) * 1e-3
        lo, hi = lo - d, hi + d
    return np.linspace(lo, hi, n)


def _cbar(fig, cs, ax, fmt="%.3g"):
    cb = fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.22, aspect=30,
                      shrink=0.95, format=fmt)
    cb.ax.tick_params(labelsize=7)
    ticks = cs.levels[::4] if len(cs.levels) > 8 else cs.levels
    cb.set_ticks(ticks)
    return cb


def _envelope_strip(ax, eps_k0chi=None, label="(a)"):
    """Sketch of the wave envelope above a panel (dashed lines) with an arrow
    showing the group velocity direction (+x'), as in the paper."""
    s = ax.inset_axes([0.0, 1.04, 1.0, 0.17])
    x = np.linspace(-3, 3, 600)
    env = np.exp(-0.5 * x ** 2)
    k0chi = eps_k0chi if eps_k0chi else 1.0 / 0.13
    s.plot(x, env, "k--", lw=0.8)
    s.plot(x, -env, "k--", lw=0.8)
    s.plot(x, env * np.cos(k0chi * x), color="0.5", lw=0.5)
    s.annotate("", xy=(0.9, 0.0), xytext=(-0.9, 0.0),
               arrowprops=dict(arrowstyle="->", lw=0.9))
    s.text(0.0, 0.25, r"$c_g$", ha="center", va="bottom", fontsize=7)
    s.set_xlim(-3, 3)
    s.set_ylim(-1.25, 1.25)
    s.set_xticks([])
    s.set_yticks([])
    s.text(-0.12, 1.15, label, transform=s.transAxes, fontsize=10)
    return s


def _contour_xz(fig, ax, cd, Z, levels, cmap="viridis", label="(a)", ylab=True):
    """Contours of Z(x', z) (shape (nxp, Nz)) over x'/chi in [-3,3], k0z in [-2,0]."""
    X, Y = np.meshgrid(cd.xc, cd.k0z, indexing="ij")
    cs = ax.contourf(X, Y, np.ma.masked_invalid(Z), levels=levels, cmap=cmap,
                     extend="both")
    ax.set_xlim(-3, 3)
    ax.set_ylim(*ZLIM_MAIN)
    ax.set_xticks(np.arange(-3, 4))
    ax.set_xlabel(r"$x'/\chi$")
    if ylab:
        ax.set_ylabel(r"$k_0z$")
    _envelope_strip(ax, cd.k0 * cd.chi, label)
    return cs


def _contour_spec(fig, ax, cd, Z, levels, cmap="viridis", label="(a)", ylab=True):
    """Contours of Z(ky, z) (shape (nky, Nz)) over lambda_y/H (log) and k0 z."""
    X, Y = np.meshgrid(cd.lam_y, cd.k0z, indexing="ij")
    cs = ax.contourf(X, Y, np.ma.masked_invalid(Z), levels=levels, cmap=cmap,
                     extend="both")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.set_xlim(cd.lam_y.min(), cd.lam_y.max())
    ax.set_ylim(*ZLIM_SPEC)
    ax.set_xlabel(r"$\lambda_y/H$")
    if ylab:
        ax.set_ylabel(r"$k_0z$")
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=10)
    return cs


def _zmask(cd, zlim):
    return (cd.k0z >= zlim[0] - 1e-12) & (cd.k0z <= zlim[1])


def _xmask(cd):
    return np.abs(cd.xc) <= 3.0 + 1e-9


def _ratio(a, b):
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(a, dtype=float) / b
    r[~np.isfinite(r)] = np.nan
    return r


def _save(fig, out, name, produced):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    produced.append(path)
    _log(f"[fig] {path}")


PANEL = "abcdef"


# ----------------------------------------------------------------------
# figures 2, 3, 7, 8: contour maps of one case
def fig_contours(cd, out, produced, which):
    """which in {'enst', 'enst_ratio', 'reyn', 'reyn_ratio'}."""
    comps = ENST if which.startswith("enst") else REYN
    ratio = which.endswith("ratio")
    xm, zm = _xmask(cd), _zmask(cd, ZLIM_MAIN)
    Zs = []
    for key, _sym in comps:
        a = cd[key]
        if which == "enst":
            Z = a / cd.Re_tau ** 2           # / (u_*^2/nu)^2
        elif which == "reyn":
            Z = a                            # / u_*^2 (= 1)
        else:
            Z = _ratio(a, cd.lead(key)[None, :])
        Zs.append(Z)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    common = _levels([Z[xm][:, zm] for Z in Zs]) if ratio else None
    for n, ((key, sym), Z, ax) in enumerate(zip(comps, Zs, axs)):
        lv = common if ratio else _levels([Z[xm][:, zm]])
        cs = _contour_xz(fig, ax, cd, Z, lv, label=f"({PANEL[n]})", ylab=(n == 0))
        cb = _cbar(fig, cs, ax)
        if which == "enst":
            lab = rf"$\widehat{{{sym}'^2}}/(u_*^2/\nu)^2$"
        elif which == "reyn":
            lab = rf"$\widehat{{{sym}'^2}}/u_*^2$"
        else:
            lab = rf"$\widehat{{{sym}'^2}}/\widehat{{{sym}'^2}}|_{{x'=3\chi}}$"
        cb.set_label(lab, fontsize=9)
    num = {"enst": 2, "enst_ratio": 3, "reyn": 7, "reyn_ratio": 8}[which]
    stem = {"enst": "enstrophy", "enst_ratio": "enstrophy_ratio",
            "reyn": "reynolds", "reyn_ratio": "reynolds_ratio"}[which]
    fig.suptitle(f"Figure {num} analogue, case {cd.name} "
                 rf"($\alpha$ = {cd.alpha:g}, $Re_\tau$ = {cd.Re_tau:g})", y=1.06)
    _save(fig, out, f"fig{num:02d}_{stem}_{cd.name}.png", produced)


# ----------------------------------------------------------------------
# relative changes
def rel_change_x(cd, key, kz):
    """Delta f / f|_{x'=3chi} along x' at k0 z = kz."""
    f = cd.at_k0z(cd[key], kz)               # (nxp,)
    f0 = cd.at_x(f, X_LEAD)
    return (f - f0) / f0


def rel_change_profile(cd, key, xc0=X_TRAIL):
    """Delta f / f|_{x'=3chi} at x' = xc0 chi versus z."""
    f = cd[key]
    f0 = cd.lead(key)
    return _ratio(cd.at_x(f, xc0) - f0, f0)


def _case_label(cd):
    return rf"{cd.name} ($\alpha$={cd.alpha:g})"


def fig_change_x(cds, out, produced, comps, num, stem, alpha2=False, only=None):
    comps = [c for c in comps if only is None or c[0] in only]
    ncol = len(comps)
    fig, axs = plt.subplots(1, ncol, figsize=(4.0 * ncol, 3.4), squeeze=False)
    axs = axs[0]
    for n, ((key, sym), ax) in enumerate(zip(comps, axs)):
        for cd in cds:
            st = CASE_STYLE.get(cd.name, dict(color=None, marker="x"))
            xm = _xmask(cd)
            fac = 1.0 / cd.alpha ** 2 if alpha2 else 1.0
            me = max(1, int(xm.sum()) // 8)
            for kz in DEPTHS:
                r = rel_change_x(cd, key, kz) * fac
                ax.plot(cd.xc[xm], r[xm], ls=DEPTH_LS[kz], color=st["color"],
                        marker=st["marker"], ms=3.5, markevery=me, lw=1.1,
                        label=f"{_case_label(cd)}, $k_0z$={kz:g}")
        ax.set_xlim(-3, 3)
        ax.set_xticks(np.arange(-3, 4))
        ax.set_xlabel(r"$x'/\chi$")
        a2 = r"\alpha^2" if alpha2 else ""
        ax.set_ylabel(rf"$\Delta\widehat{{{sym}'^2}}/({a2}\widehat{{{sym}'^2}}|_{{x'=3\chi}})$"
                      if alpha2 else
                      rf"$\Delta\widehat{{{sym}'^2}}/\widehat{{{sym}'^2}}|_{{x'=3\chi}}$")
        ax.axhline(0.0, color="0.6", lw=0.5)
        ax.text(-0.15, 1.03, f"({PANEL[n]})", transform=ax.transAxes, fontsize=10)
    axs[0].legend(fontsize=6, loc="upper left", frameon=False)
    fig.suptitle(f"Figure {num} analogue (solid: $k_0z=-0.5$, dashed: $k_0z=-0.8$)", y=1.04)
    fig.tight_layout()
    _save(fig, out, f"fig{num:02d}_{stem}.png", produced)


def fig_change_profile(cds, out, produced, comps, num, stem, alpha2=False,
                       case_ls=False, only=None):
    comps = [c for c in comps if only is None or c[0] in only]
    ncol = len(comps)
    fig, axs = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.6), squeeze=False)
    axs = axs[0]
    for n, ((key, sym), ax) in enumerate(zip(comps, axs)):
        for cd in cds:
            st = CASE_STYLE.get(cd.name, dict(color=None, marker="x", ls="-"))
            zm = _zmask(cd, ZLIM_MAIN)
            fac = 1.0 / cd.alpha ** 2 if alpha2 else 1.0
            r = rel_change_profile(cd, key) * fac
            if case_ls:      # fig 11: line styles distinguish the cases
                ax.plot(r[zm], cd.k0z[zm], ls=st["ls"], color=st["color"], lw=1.1,
                        label=_case_label(cd))
            else:
                ax.plot(r[zm], cd.k0z[zm], ls="-", color=st["color"],
                        marker=st["marker"], ms=3.5,
                        markevery=max(1, int(zm.sum()) // 10), lw=1.1,
                        label=_case_label(cd))
        ax.set_ylim(*ZLIM_MAIN)
        ax.axvline(0.0, color="0.6", lw=0.5)
        ax.set_ylabel(r"$k_0z$" if n == 0 else "")
        a2 = r"\alpha^2" if alpha2 else ""
        ax.set_xlabel(rf"$\Delta\widehat{{{sym}'^2}}/({a2}\widehat{{{sym}'^2}}|_{{x'=3\chi}})$"
                      if alpha2 else
                      rf"$\Delta\widehat{{{sym}'^2}}/\widehat{{{sym}'^2}}|_{{x'=3\chi}}$")
        ax.text(-0.15, 1.03, f"({PANEL[n]})", transform=ax.transAxes, fontsize=10)
    axs[-1].legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle(f"Figure {num} analogue: trailing edge $x'=-3\\chi$ vs leading edge "
                 f"$x'=3\\chi$", y=1.04)
    fig.tight_layout()
    _save(fig, out, f"fig{num:02d}_{stem}.png", produced)


def fig06(cds, out, produced):
    """Normalised streamwise-enstrophy change: (a) along x', (b) profile at x'=-3chi."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for cd in cds:
        st = CASE_STYLE.get(cd.name, dict(color=None, marker="x"))
        xm, zm = _xmask(cd), _zmask(cd, ZLIM_MAIN)
        fac = 1.0 / cd.alpha ** 2
        for kz in DEPTHS:
            r = rel_change_x(cd, "oxox", kz) * fac
            a.plot(cd.xc[xm], r[xm], ls=DEPTH_LS[kz], color=st["color"],
                   marker=st["marker"], ms=3.5, markevery=max(1, int(xm.sum()) // 8),
                   lw=1.1, label=f"{_case_label(cd)}, $k_0z$={kz:g}")
        r = rel_change_profile(cd, "oxox") * fac
        b.plot(r[zm], cd.k0z[zm], color=st["color"], marker=st["marker"], ms=3.5,
               markevery=max(1, int(zm.sum()) // 10), lw=1.1, label=_case_label(cd))
    a.set_xlim(-3, 3)
    a.set_xticks(np.arange(-3, 4))
    a.set_xlabel(r"$x'/\chi$")
    a.set_ylabel(r"$\Delta\widehat{\omega_x'^2}/(\alpha^2\widehat{\omega_x'^2}|_{x'=3\chi})$")
    a.legend(fontsize=6, frameon=False)
    b.set_ylim(*ZLIM_MAIN)
    b.set_ylabel(r"$k_0z$")
    b.set_xlabel(r"$\Delta\widehat{\omega_x'^2}/(\alpha^2\widehat{\omega_x'^2}|_{x'=3\chi})$")
    b.axvline(0.0, color="0.6", lw=0.5)
    b.legend(fontsize=7, frameon=False)
    for n, ax in enumerate((a, b)):
        ax.text(-0.15, 1.03, f"({PANEL[n]})", transform=ax.transAxes, fontsize=10)
    fig.suptitle(r"Figure 6 analogue: streamwise enstrophy change normalised by $\alpha^2$",
                 y=1.04)
    fig.tight_layout()
    _save(fig, out, "fig06_enstrophy_x_alpha2.png", produced)


# ----------------------------------------------------------------------
# tables 1 and 2
def integrated_change(cd, key, kz_bot=KZ_INT):
    """(Delta F / F|_{x'=3chi}) with F = int_{z_D}^0 f dz (eq. 4.1 / 4.2)."""
    zb = kz_bot / cd.k0
    F_lead = float(integrate_depth(cd.lead(key), cd.zc, zb))
    F_trail = float(integrate_depth(cd.trail(key), cd.zc, zb))
    return (F_trail - F_lead) / F_lead, F_lead, F_trail


def _write_table(out, stem, header, rows, fmt_rows, title, produced):
    csv_path = os.path.join(out, stem + ".csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([r[0]] + [f"{v:.12g}" if isinstance(v, float) else v for v in r[1:]])
    widths = [max(len(h), max((len(fr[i]) for fr in fmt_rows), default=0)) + 2
              for i, h in enumerate(header)]
    lines = [title, "".join(h.rjust(wd) for h, wd in zip(header, widths))]
    for fr in fmt_rows:
        lines.append("".join(c.rjust(wd) for c, wd in zip(fr, widths)))
    text = "\n".join(lines)
    txt_path = os.path.join(out, stem + ".txt")
    with open(txt_path, "w") as fh:
        fh.write(text + "\n")
    print(text, flush=True)
    produced += [csv_path, txt_path]
    _log(f"[table] {csv_path}")
    return text


def table1(cds, out, produced):
    header = ["Case", "alpha", "dOx/Ox_pct", "dOx/(a2 Ox)", "dOy/Oy_pct", "dOz/Oz_pct"]
    rows, fr = [], []
    for cd in cds:
        rx = integrated_change(cd, "oxox")[0]
        ry = integrated_change(cd, "oyoy")[0]
        rz = integrated_change(cd, "ozoz")[0]
        a2 = cd.alpha ** 2
        rows.append([cd.name, cd.alpha, 100 * rx, rx / a2, 100 * ry, 100 * rz])
        fr.append([cd.name, f"{cd.alpha:.2f}", f"{100 * rx:.1f} %", f"{rx / a2:.1f}",
                   f"{100 * ry:.1f} %", f"{100 * rz:.1f} %"])
    title = ("Table 1 analogue: relative changes in the enstrophy components integrated "
             "from k0z = -2 to the surface (trailing vs leading edge)")
    return _write_table(out, "table1_enstrophy", header, rows, fr, title, produced)


def table2(cds, out, produced):
    header = ["Case", "alpha", "Eu_pct", "Eu/a2", "Ev_pct", "Ev/a2", "Ew_pct", "Ew/a2"]
    rows, fr = [], []
    for cd in cds:
        a2 = cd.alpha ** 2
        r = [integrated_change(cd, k)[0] for k in ("uu", "vv", "ww")]
        row, f = [cd.name, cd.alpha], [cd.name, f"{cd.alpha:.2f}"]
        for v in r:
            row += [100 * v, v / a2]
            f += [f"{100 * v:.1f} %", f"{v / a2:.1f}"]
        rows.append(row)
        fr.append(f)
    title = ("Table 2 analogue: relative changes in the Reynolds normal stresses integrated "
             "from k0z = -2 to the surface (trailing vs leading edge)")
    return _write_table(out, "table2_reynolds", header, rows, fr, title, produced)


# ----------------------------------------------------------------------
# figures 12-14: premultiplied spanwise spectra
def premult(cd, B):
    """ky B / dky for B of shape (nky, Nz)."""
    return cd.ky[:, None] * np.asarray(B, dtype=float) / cd.dky


def fig_spectrum(cd, out, produced, comp):
    num = {"w": 12, "v": 13, "u": 14}[comp]
    Phi = cd["Phi_" + comp]
    lead = premult(cd, cd.at_x(Phi, X_LEAD))
    trail = premult(cd, cd.at_x(Phi, X_TRAIL))
    ratio = _ratio(trail, lead)
    zm = _zmask(cd, ZLIM_SPEC)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
    lv = _levels([lead[:, zm], trail[:, zm]])
    for n, (Z, ax) in enumerate(zip((lead, trail), axs[:2])):
        cs = _contour_spec(fig, ax, cd, Z, lv, label=f"({PANEL[n]})", ylab=(n == 0))
        _cbar(fig, cs, ax, fmt="%.3g")
    axs[0].set_title(r"$k_y\Phi_%s/u_*^2$ at $x'=3\chi$ (leading)" % comp, fontsize=9)
    axs[1].set_title(r"$k_y\Phi_%s/u_*^2$ at $x'=-3\chi$ (trailing)" % comp, fontsize=9)
    cs = _contour_spec(fig, axs[2], cd, ratio, _levels([ratio[:, zm]]), label="(c)",
                       ylab=False)
    _cbar(fig, cs, axs[2], fmt="%.3g")
    axs[2].set_title("trailing / leading", fontsize=9)
    fig.suptitle(f"Figure {num} analogue: premultiplied spectrum of ${comp}'$, case "
                 f"{cd.name}", y=1.03)
    _save(fig, out, f"fig{num:02d}_spectrum_{comp}_{cd.name}.png", produced)


# ----------------------------------------------------------------------
# figures 15-17: spectral budget at the packet core
BUDGET_TERMS = {
    "w": (("Pw_z", r"\Delta P^w"), ("Pis_z", r"\Delta\Pi^s"), ("Tp_z", r"\Delta T^p"),
          ("A_z", r"\Delta A")),
    "v": (("Pis_y", r"\Delta\Pi^s"), ("A_y", r"\Delta A")),
    "u": (("Pw_x", r"\Delta P^w"), ("Pis_x", r"\Delta\Pi^s"), ("Tp_x", r"\Delta T^p"),
          ("A_x", r"\Delta A")),
}


def budget_change(cd, key, xc0=X_CORE):
    """Premultiplied change from the leading edge, ky (B(x') - B(3chi))/dky,
    normalised by alpha u_*^2 omega0; shape (nky, Nz)."""
    B = cd[key]
    dB = cd.at_x(B, xc0) - cd.at_x(B, X_LEAD)
    return premult(cd, dB) / cd.budget_scale


def fig_budget(cd, out, produced, comp):
    num = {"w": 15, "v": 16, "u": 17}[comp]
    terms = BUDGET_TERMS[comp]
    Zs = [budget_change(cd, k) for k, _ in terms]
    zm = _zmask(cd, ZLIM_SPEC)
    lv = _levels([Z[:, zm] for Z in Zs], sym=True, n=21)
    nt = len(terms)
    nrow = 1 if nt <= 2 else 2
    ncol = 2
    fig, axs = plt.subplots(nrow, ncol, figsize=(10, 4.0 * nrow), squeeze=False)
    for n, ((key, lab), Z) in enumerate(zip(terms, Zs)):
        ax = axs.flat[n]
        cs = _contour_spec(fig, ax, cd, Z, lv, cmap="bwr", label=f"({PANEL[n]})",
                           ylab=(n % 2 == 0))
        _cbar(fig, cs, ax, fmt="%.2g")
        ax.set_title(rf"${lab}$  ($k_y\,\cdot/(\alpha u_*^2\omega_0)$)", fontsize=9)
    fig.suptitle(f"Figure {num} analogue: spectral budget of ${comp}'$ at the packet core "
                 f"$x'=0$, case {cd.name}", y=1.01)
    _save(fig, out, f"fig{num:02d}_budget_{comp}_{cd.name}.png", produced)


def fig_budget_closure(cd, out, produced):
    """Extra diagnostic: ky-integrated budget at x' = 0 (changes from the
    leading edge): lhs = -cg dPhi/dx' versus Delta(Pw + Pis + Tp + A) and the
    full computed rhs (incl. Pr), normalised by alpha u_*^2 omega0."""
    needed = [f"{t}_{c}" for c in "xyz" for t in ("lhs", "rhs", "Pw", "Pis", "Tp", "A")]
    if not all(k in cd.d for k in needed):
        _log(f"[skip] budget closure {cd.name}: missing lhs/rhs keys")
        return
    zm = _zmask(cd, ZLIM_MAIN)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, c in zip(axs, "xyz"):
        def dsum(k):
            B = cd[f"{k}_{c}"]
            return (cd.at_x(B, X_CORE) - cd.at_x(B, X_LEAD)).sum(axis=0) / cd.budget_scale
        lhs = dsum("lhs")
        dom = dsum("Pw") + dsum("Pis") + dsum("Tp") + dsum("A")
        rhs = dsum("rhs")
        ax.plot(lhs[zm], cd.k0z[zm], "k-", lw=1.4, label=r"$\Delta(-c_g\partial\Phi/\partial x')$")
        ax.plot(dom[zm], cd.k0z[zm], "C3--", lw=1.1,
                label=r"$\Delta(P^w+\Pi^s+T^p+A)$")
        ax.plot(rhs[zm], cd.k0z[zm], "C0:", lw=1.1, label=r"$\Delta$ rhs (incl. $P^r$)")
        ax.set_ylim(*ZLIM_MAIN)
        ax.axvline(0, color="0.6", lw=0.5)
        ax.set_xlabel(r"$\sum_{k_y}$ / $(\alpha u_*^2\omega_0)$")
        ax.set_title(f"${'uvw'['xyz'.index(c)]}'$ at $x'=0$", fontsize=9)
    axs[0].set_ylabel(r"$k_0z$")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Budget closure check, case {cd.name} (turbulent transport, viscous/SGS "
                 r"and $\partial\Phi/\partial t$ not computed)", y=1.03)
    fig.tight_layout()
    _save(fig, out, f"fig_budget_closure_{cd.name}.png", produced)


# ----------------------------------------------------------------------
# base-flow validation
BASE_KEYS = ("zc", "zf", "U", "uu", "vv", "ww", "uw", "uw_f", "nu_dUdz_f", "tau13_f",
             "Re_tau")


def fig_base(b, out, produced):
    zc = np.asarray(b["zc"], dtype=float)
    zf = np.asarray(b["zf"], dtype=float)
    Re = float(b["Re_tau"])
    nu = 1.0 / Re
    U = np.asarray(b["U"], dtype=float)
    dUs = float(np.asarray(b["nu_dUdz_f"])[-1]) / nu        # surface shear (Re_tau ideally)
    Us = U[-1] + dUs * (0.0 - zc[-1])                        # extrapolated surface velocity
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    a = axs[0, 0]
    a.plot(U, zc, "k-")
    a.set_xlabel(r"$U/u_*$")
    a.set_ylabel(r"$z/H$")
    a.set_title(f"(a) mean velocity ($U_s\\approx${Us:.1f})", fontsize=9)
    a = axs[0, 1]
    zp = -zc * Re
    a.semilogx(zp, Us - U, "k-", label="LES")
    zz = np.logspace(-1, math.log10(max(zp.max(), 10.0)), 200)
    a.semilogx(zz[zz < 15], zz[zz < 15], "C0--", lw=0.9, label=r"$z^+$")
    a.semilogx(zz[zz > 5], np.log(zz[zz > 5]) / 0.41 + 5.0, "C3--", lw=0.9,
               label=r"$\ln z^+/0.41+5.0$")
    a.set_xlabel(r"$z^+=-z u_*/\nu$")
    a.set_ylabel(r"$(U_s-U)/u_*$")
    a.set_title("(b) velocity defect in surface units", fontsize=9)
    a.legend(fontsize=7, frameon=False)
    a = axs[0, 2]
    for k, lab, c in (("uu", r"$\langle u'^2\rangle$", "C0"), ("vv", r"$\langle v'^2\rangle$", "C1"),
                      ("ww", r"$\langle w'^2\rangle$", "C2")):
        a.plot(np.asarray(b[k], dtype=float), zc, color=c, label=lab)
    a.plot(-np.asarray(b["uw"], dtype=float), zc, "k--", label=r"$-\langle u'w'\rangle$")
    a.set_xlabel(r"stress$/u_*^2$")
    a.set_ylabel(r"$z/H$")
    a.set_title("(c) Reynolds stresses", fontsize=9)
    a.legend(fontsize=7, frameon=False)
    a = axs[1, 0]
    rs = -np.asarray(b["uw_f"], dtype=float)
    vs = np.asarray(b["nu_dUdz_f"], dtype=float)
    ss = -np.asarray(b["tau13_f"], dtype=float)
    tot = rs + vs + ss
    a.plot(rs, zf, "C0-", label=r"$-\langle u'w'\rangle$")
    a.plot(vs, zf, "C1-", label=r"$\nu\, dU/dz$")
    a.plot(ss, zf, "C2-", label=r"$-\langle\tau_{13}\rangle$ (SGS)")
    a.plot(tot, zf, "k-", lw=1.6, label="total")
    a.plot(1.0 + zf, zf, "r--", lw=1.0, label=r"$1+z/H$")
    a.set_xlabel(r"stress$/u_*^2$")
    a.set_ylabel(r"$z/H$")
    err = float(np.max(np.abs(tot - (1.0 + zf))))
    a.set_title(f"(d) total stress balance, max error {err:.3f}", fontsize=9)
    a.legend(fontsize=7, frameon=False)
    for ax, comp, lab in ((axs[1, 1], "Phi_u", "(e)"), (axs[1, 2], "Phi_w", "(f)")):
        if comp in b and "ky" in b:
            ky = np.asarray(b["ky"], dtype=float)
            dky = ky[0] if ky.size else 1.0
            Z = ky[:, None] * np.asarray(b[comp], dtype=float) / dky
            X, Y = np.meshgrid(2 * np.pi / ky, zc, indexing="ij")
            cs = ax.contourf(X, Y, Z, levels=_levels([Z]), cmap="viridis", extend="both")
            ax.set_xscale("log")
            ax.set_xlabel(r"$\lambda_y/H$")
            ax.set_ylabel(r"$z/H$")
            ax.set_title(f"{lab} $k_y\\Phi_{comp[-1]}/u_*^2$ (base flow)", fontsize=9)
            _cbar(fig, cs, ax, fmt="%.3g")
        else:
            ax.set_axis_off()
    n = int(np.asarray(b.get("n_samples", 0)))
    fig.suptitle(f"Base flow (no waves): $Re_\\tau$ = {Re:g}, {n} samples", y=1.0)
    fig.tight_layout()
    _save(fig, out, "fig_base_flow.png", produced)
    _log(f"[base] U_s = {Us:.3f}, surface dU/dz = {dUs:.2f} (Re_tau = {Re:g}), "
         f"max |total stress - (1+z)| = {err:.4f}")


# ----------------------------------------------------------------------
def make_all(data_dir, out, cases, strict=False):
    """Produce every figure/table that the available data allow.  Returns
    (produced paths, list of (name, error) failures)."""
    os.makedirs(out, exist_ok=True)
    produced, failures = [], []

    def run(name, fn, *a):
        try:
            fn(*a)
        except Exception as e:     # keep going; report at the end
            plt.close("all")
            if strict:
                raise
            failures.append((name, repr(e)))
            _log(f"[fail] {name}: {e}")
            traceback.print_exc()

    base_path = os.path.join(data_dir, "base", "base_stats.npz")
    if os.path.exists(base_path):
        b = load_npz(base_path)
        miss = [k for k in BASE_KEYS if k not in b]
        if miss:
            _log(f"[skip] base flow: {base_path} lacks keys {miss}")
        else:
            run("base", fig_base, b, out, produced)
    else:
        _log(f"[skip] base flow: {base_path} not found")

    cd_map = load_cases(data_dir, cases)
    cds = [cd_map[c] for c in cases if c in cd_map]
    if not cds:
        _log("[skip] no packet results found; figures 2-17 and tables 1-2 not produced")
        return produced, failures

    for cd in cds:
        for which in ("enst", "enst_ratio", "reyn", "reyn_ratio"):
            run(f"{which}_{cd.name}", fig_contours, cd, out, produced, which)
    run("fig04", fig_change_x, cds, out, produced, ENST, 4, "enstrophy_change_x")
    run("fig05", fig_change_profile, cds, out, produced, ENST, 5, "enstrophy_change_profile")
    run("fig06", fig06, cds, out, produced)
    run("table1", table1, cds, out, produced)
    run("fig09", fig_change_x, cds, out, produced, REYN, 9, "reynolds_change_x")
    run("fig10", fig_change_x, cds, out, produced, REYN, 10, "reynolds_change_x_alpha2", True)
    run("fig11", fig_change_profile, cds, out, produced, REYN, 11,
        "reynolds_change_profile_alpha2", True, True)
    run("table2", table2, cds, out, produced)
    for cd in cds:
        if cd.has_spec:
            for comp in ("w", "v", "u"):
                run(f"spectrum_{comp}_{cd.name}", fig_spectrum, cd, out, produced, comp)
        else:
            _log(f"[skip] spectra {cd.name}: keys missing")
        if cd.has_budget:
            for comp in ("w", "v", "u"):
                run(f"budget_{comp}_{cd.name}", fig_budget, cd, out, produced, comp)
            run(f"closure_{cd.name}", fig_budget_closure, cd, out, produced)
        else:
            _log(f"[skip] budget figures {cd.name}: budget not accumulated")
    return produced, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    ap.add_argument("--cases", nargs="+", default=["W12", "W09", "W06"],
                    choices=sorted(CASES))
    ap.add_argument("--strict", action="store_true",
                    help="raise on the first plotting error instead of continuing")
    a = ap.parse_args(argv)
    produced, failures = make_all(a.data_dir, a.out, a.cases, strict=a.strict)
    _log(f"{len(produced)} files written to {a.out}")
    if failures:
        _log(f"{len(failures)} failures: {[f[0] for f in failures]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
