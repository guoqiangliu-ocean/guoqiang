"""Turbulence statistics (paper section 3, sections 4-5 and Appendix D).

Averaging (paper eq. 3.1-3.2)
-----------------------------
``<f>(x, z, t)``  ensemble + spanwise average, ``f' = f - <f>``;
``f^(x', z) = 1/(t2-t1) int <f>(x' + x0 + cg t, z, t) dt`` packet-following
average, with ``x' = x - (x0 + cg t)`` (periodic).  The time integral is
replaced by the mean over the ``n_samples`` sample instants ``t_i`` (identical
in every ensemble member).

Packet-frame interpolation
--------------------------
At every sample the *linear* fields (u, v, w, p, their derivatives, the
surface gradients and the wave fields) are shifted to the packet frame by an
exact Fourier shift in x, ``f'(x') = f(x' + s)``, ``s = x0 + cg t_i``, applied in
the horizontal spectral space (Nyquist mode treated with ``cos(k_N s)``, i.e.
the real trigonometric interpolant).  All products (moments, spectra, budget
terms) are then formed point-wise on the packet-frame grid
``x'_j = j dx`` (wrapped to ``[-Lx/2, Lx/2)``).  For band-limited (dealiased)
fields this gives the *exact* value of every product at ``x'_j``; no aliasing
error from interpolating a quadratic quantity is introduced.

Spanwise spectra (my >= 1, Nyquist excluded)
--------------------------------------------
``c_q(x', my, z) = rfft_y(q)/Ny``, ``Phi_q = 2 |c_q|^2`` (one-sided, so that
``sum_my Phi_q = <q'^2>_y`` for a zero spanwise mean).  Spanwise sums of moments
are evaluated with the discrete Parseval identity directly from ``c_q``.

Spectral budget (Appendix D, eq. D6) -- each term ``B = 4 Re[conj(c_i) F_i]``
with ``F_i`` the y-transform of a term of the u'_i equation (D1):

* ``Pw_i``    wave production, ``F = -u'_k d(uphi)_i/dx_k``               (Pw_y = 0)
* ``Pr_i``    current production, ``F = -u'_k d<u_i>/dx_k`` (extra, not in SPEC 6)
* ``Pis_i``   pressure strain ``4 Re[conj(d_i c_i) c_p]`` (no sum)
* ``Tp_i``    pressure diffusion ``-d/dx_i 4 Re[conj(c_i) c_p]``            (Tp_y = 0);
  d/dz by ``np.gradient(..., zc, edge_order=2)``: second-order three-point
  formula on the non-uniform centres, second-order one-sided at the two ends
* ``Aphi_i``  advection by the wave ``F = -uphi du_i/dx - wphi du_i/dz``
* ``Amean_i`` advection by the mean current ``F = -ubar du_i/dx - wbar du_i/dz``
* ``lhs_i = -cg dPhi_i/dx'`` (the frame-translation term of eq. 5.1).

``<u>`` in Pr/Amean is approximated by the spanwise mean of the run (SPEC 8.5);
the spanwise-mean advection ``-vbar d/dy`` contributes exactly zero to B.
For my >= 1 the Fourier coefficients of u' equal those of u (the mean is
y-independent), so spectra, Pw, Pis, Tp, Aphi need no ensemble mean at all.
``dPhi/dx'`` and ``Tp_x`` are accumulated per sample with the product rule
(exact point values) instead of differentiating the aliased averaged spectra.
Vertical velocity terms use w at centres, ``grid.f2c(w)``; ``dw/dz`` at centres
is ``grid.ddz_f2c(w)``; du/dz, dv/dz use ``grid.ddz_c`` with the surface data
``solver.bc['dudz_s'], ['dvdz_s']`` and zero gradient at the bottom.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os

import numpy as np
import scipy.fft as sfft

from .grid import Grid
from .params import Params

MOMENT_KEYS = ("u", "v", "w", "uu", "vv", "ww", "uw",
               "ox", "oy", "oz", "oxox", "oyoy", "ozoz")
SPEC_KEYS = ("Phi_u", "Phi_v", "Phi_w")
BUDGET_KEYS = ("Pw_x", "Pw_z",
               "Pr_x", "Pr_y", "Pr_z",
               "Pis_x", "Pis_y", "Pis_z",
               "Tp_x", "Tp_z",
               "Aphi_x", "Aphi_y", "Aphi_z",
               "Amean_x", "Amean_y", "Amean_z",
               "dPhidx_x", "dPhidx_y", "dPhidx_z")
_WAVE_KEYS = ("uphi_c", "wphi_c", "duphi_dx_c", "duphi_dz_c", "dwphi_dx_c", "dwphi_dz_c")


# ----------------------------------------------------------------------
# helpers
def wrap_periodic(x, L):
    """Map x to [-L/2, L/2)."""
    return (np.asarray(x) + 0.5 * L) % L - 0.5 * L


def shift_phase(N: int, L: float, s: float, rfft: bool = False) -> np.ndarray:
    """Phase factors e^{i k s} so that multiplying the x-spectrum of a periodic
    function f sampled at x_j = j L/N gives the samples of f(x + s) (real
    trigonometric interpolant: the Nyquist mode gets cos(k_N s))."""
    if rfft:
        m = np.arange(N // 2 + 1, dtype=float)
    else:
        m = np.fft.fftfreq(N, d=1.0 / N)
    k = 2.0 * math.pi / L * m
    ph = np.exp(1j * k * s)
    if N % 2 == 0:
        iN = N // 2
        ph[iN] = math.cos(k[iN] * s)
    return ph


def fourier_shift_x(a: np.ndarray, s: float, L: float, axis: int = 0,
                    workers: int = 1) -> np.ndarray:
    """Exact Fourier shift of real samples ``a`` (periodic of period L along
    ``axis``): returns the samples of ``f(x + s)``."""
    a = np.asarray(a, dtype=float)
    N = a.shape[axis]
    ah = sfft.rfft(a, axis=axis, workers=workers)
    shp = [1] * a.ndim
    shp[axis] = N // 2 + 1
    ah *= shift_phase(N, L, s, rfft=True).reshape(shp)
    return sfft.irfft(ah, n=N, axis=axis, workers=workers)


def _y_weights(Ny: int) -> np.ndarray:
    """Parseval weights for one-sided rfft coefficients along y."""
    wt = np.full(Ny // 2 + 1, 2.0)
    wt[0] = 1.0
    if Ny % 2 == 0:
        wt[-1] = 1.0
    return wt


def _nky(Ny: int) -> int:
    """Number of retained spanwise modes my = 1..nky (Nyquist excluded)."""
    return (Ny - 1) // 2


def _abs2(a: np.ndarray) -> np.ndarray:
    return a.real * a.real + a.imag * a.imag


def _cre(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Re(conj(a) b)."""
    return a.real * b.real + a.imag * b.imag


def _cim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Im(conj(a) b)."""
    return a.real * b.imag - a.imag * b.real


def _params_from_dict(d: dict) -> Params:
    names = {f.name for f in dataclasses.fields(Params)}
    return Params(**{k: v for k, v in d.items() if k in names})


def _npz_path(path):
    """np.savez appends '.npz' to a path without it; accept both spellings."""
    path = os.fspath(path)
    if not os.path.exists(path) and os.path.exists(path + ".npz"):
        return path + ".npz"
    return path


# physical/grid parameters that must agree between accumulators (merge) and
# between an accumulator and the Params passed to packet_results
_COMPAT_KEYS = ("Lx", "Ly", "H", "Re_tau", "alpha", "Fr", "k0H", "eps_bw", "x0_frac",
                "Nx", "Ny", "Nz", "z_stretch", "wave_on", "include_irrot_stress")


def _check_params(pa: dict, pb: dict, what: str = "parameter") -> None:
    for k in _COMPAT_KEYS:
        a, b = pa.get(k), pb.get(k)
        if isinstance(a, float) or isinstance(b, float):
            same = a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12,
                                                                     abs_tol=0.0)
        else:
            same = a == b
        if not same:
            raise ValueError(f"{what} {k} differs: {a} vs {b}")


def _deriv_wavenumbers(grid: Grid):
    """i kx (Nx,1,1) and ky (1,Nyh,1) with the Nyquist modes zeroed (the
    derivative of the real trigonometric interpolant vanishes there at the
    grid points)."""
    kx = np.array(grid.kx, dtype=float)
    if grid.Nx % 2 == 0:
        kx[grid.Nx // 2] = 0.0
    ky = np.array(grid.ky, dtype=float)
    if grid.Ny % 2 == 0:
        ky[:, -1] = 0.0
    return 1j * kx, ky


def _ddx_real(a: np.ndarray, L: float, workers: int = 1) -> np.ndarray:
    """Spectral d/dx along axis 0 of a real periodic array (Nyquist zeroed)."""
    N = a.shape[0]
    ah = sfft.rfft(a, axis=0, workers=workers)
    k = 2.0 * math.pi / L * np.arange(N // 2 + 1)
    if N % 2 == 0:
        k[-1] = 0.0
    ah *= (1j * k).reshape((-1,) + (1,) * (a.ndim - 1))
    return sfft.irfft(ah, n=N, axis=0, workers=workers)


def _ddz_c_complex(grid: Grid, qc: np.ndarray, top: np.ndarray) -> np.ndarray:
    """grid.ddz_c for complex arrays with complex surface data, zero at bottom."""
    df = np.zeros(qc.shape[:-1] + (grid.Nz + 1,), dtype=qc.dtype)
    df[..., 1:-1] = (qc[..., 1:] - qc[..., :-1]) / grid.dzf[1:-1]
    df[..., -1] = top
    return 0.5 * (df[..., 1:] + df[..., :-1])


# ----------------------------------------------------------------------
class PacketAccumulator:
    """Packet-frame statistics of one or several ensemble members.

    Storage (float64), all in the packet frame on the full periodic grid x'_j:

    * ``mom[k]``  (n_samples, Nx, Nz): sums over runs of spanwise sums of
      ``u, v, w, u^2, v^2, w^2, u w, ox, oy, oz, ox^2, oy^2, oz^2`` (w at centres);
      count for sample i is ``n_added[i] * Ny``;
    * ``spec[k]`` (Nx, nky, Nz): sums over runs and samples of ``Phi_u, Phi_v, Phi_w``
      and, if ``budget``, of the budget terms ``BUDGET_KEYS``; count ``n_added.sum()``.

    Note: moments are ensemble-averaged per sample and then time-averaged with
    equal sample weights (eq. 3.2), whereas the spectra/budget sums are divided by
    ``n_added.sum()``.  Both agree when every member contributed every sample; with
    an incomplete member the spectra weight the samples by their member count.
    """

    def __init__(self, grid: Grid, p: Params, sample_times, budget: bool = True):
        self.grid = grid
        self.p = p
        self.budget = bool(budget)
        self.sample_times = np.asarray(sample_times, dtype=float).ravel()
        ns = self.sample_times.size
        self.Nx, self.Ny, self.Nz = grid.Nx, grid.Ny, grid.Nz
        self.nky = _nky(grid.Ny)
        self.n_added = np.zeros(ns, dtype=np.int64)
        self.mom = {k: np.zeros((ns, self.Nx, self.Nz)) for k in MOMENT_KEYS}
        keys = SPEC_KEYS + (BUDGET_KEYS if self.budget else ())
        self.spec = {k: np.zeros((self.Nx, self.nky, self.Nz)) for k in keys}
        self.meta = self._make_meta(p)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_meta(p: Params) -> dict:
        return {"params": p.to_dict()}

    @property
    def n_runs(self) -> int:
        return int(self.n_added.max()) if self.n_added.size else 0

    @property
    def n_samples(self) -> int:
        return int(self.sample_times.size)

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.mom.values())
                   + sum(a.nbytes for a in self.spec.values())
                   + self.n_added.nbytes + self.sample_times.nbytes)

    def shift(self, t: float) -> float:
        """Packet-frame offset s(t) = x0 + cg t (x = x' + s)."""
        return self.p.x0 + self.p.cg * t

    # ------------------------------------------------------------------
    def add_sample(self, i_sample: int, solver, wf) -> None:
        """Add the state of ``solver`` (one ensemble member) at sample ``i_sample``.

        ``solver`` must expose physical u, v (Nx,Ny,Nz), w (Nx,Ny,Nz+1), p
        (Nx,Ny,Nz), bc['dudz_s'], bc['dvdz_s'] (Nx,Ny) and t.  ``wf`` is the
        WaveFields at the same time (used only if ``budget``)."""
        i = int(i_sample)
        t_i = float(self.sample_times[i])
        t_sol = float(getattr(solver, "t", t_i))
        if abs(t_sol - t_i) > 1e-6 * max(abs(t_i), 1e-2):
            raise ValueError(f"solver.t = {t_sol!r} does not match sample time "
                             f"t[{i}] = {t_i!r}")
        if self.budget and wf is not None and getattr(wf, "t", None) is not None:
            # the wave fields must be those of the same instant (a one-step lag
            # shifts the carrier phase by omega0 dt and biases Pw and Aphi)
            if abs(float(wf.t) - t_i) > 1e-6 * max(abs(t_i), 1e-2):
                raise ValueError(f"wave fields at t = {wf.t!r} do not match sample time "
                                 f"t[{i}] = {t_i!r}")
        g = self.grid
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        W = self.p.workers
        nky = self.nky
        sl = slice(1, nky + 1)
        s = self.shift(t_i)

        # phase (shift to packet frame) and y-normalisation in one factor
        ph = (shift_phase(Nx, g.Lx, s) / Ny).reshape(Nx, 1, 1)
        ikx, ky = _deriv_wavenumbers(g)                    # (Nx,1,1), (1,Nyh,1)

        def f2(a):
            return sfft.rfft2(a, axes=(0, 1), workers=W)

        def ifx(a):
            return sfft.ifft(a, axis=0, workers=W, overwrite_x=True)

        bc = solver.bc
        # ---- horizontal spectra, shifted, then back to physical x (packet frame)
        Uh = f2(solver.u)
        Uh *= ph
        cdudx = ifx(ikx * Uh)
        cu = ifx(Uh)
        del Uh
        Vh = f2(solver.v)
        Vh *= ph
        cdvdx = ifx(ikx * Vh)
        cv = ifx(Vh)
        del Vh
        Wh = f2(solver.w)
        Wh *= ph
        cdwdx = ifx(ikx * g.f2c(Wh))
        cwf = ifx(Wh)
        del Wh
        cw = g.f2c(cwf)
        cdwdz = g.ddz_f2c(cwf)
        del cwf
        ph2 = ph[:, :, 0]
        cbu = ifx(f2(np.asarray(bc["dudz_s"], dtype=float)) * ph2)
        cbv = ifx(f2(np.asarray(bc["dvdz_s"], dtype=float)) * ph2)
        cdudz = _ddz_c_complex(g, cu, cbu)
        cdvdz = _ddz_c_complex(g, cv, cbv)

        # ---- moments (spanwise sums via discrete Parseval)
        wt = _y_weights(Ny) * Ny                            # (Nyh,)

        def ysum(E):
            return np.einsum("xmz,m->xz", E, wt, optimize=False)

        M = self.mom
        Euu = _abs2(cu)
        Evv = _abs2(cv)
        Eww = _abs2(cw)
        Euw = _cre(cu, cw)
        M["u"][i] += Ny * cu[:, 0].real
        M["v"][i] += Ny * cv[:, 0].real
        M["w"][i] += Ny * cw[:, 0].real
        M["uu"][i] += ysum(Euu)
        M["vv"][i] += ysum(Evv)
        M["ww"][i] += ysum(Eww)
        M["uw"][i] += ysum(Euw)
        iky = 1j * ky
        co = iky * cw
        co -= cdvdz                                        # ox = dw/dy - dv/dz
        M["ox"][i] += Ny * co[:, 0].real
        M["oxox"][i] += ysum(_abs2(co))
        co = cdudz - cdwdx                                 # oy = du/dz - dw/dx
        M["oy"][i] += Ny * co[:, 0].real
        M["oyoy"][i] += ysum(_abs2(co))
        co = iky * cu
        np.subtract(cdvdx, co, out=co)                     # oz = dv/dx - du/dy
        M["oz"][i] += Ny * co[:, 0].real
        M["ozoz"][i] += ysum(_abs2(co))
        del co

        # ---- spanwise spectra
        S = self.spec
        Euu_s, Evv_s, Eww_s, Euw_s = Euu[:, sl], Evv[:, sl], Eww[:, sl], Euw[:, sl]
        S["Phi_u"] += 2.0 * Euu_s
        S["Phi_v"] += 2.0 * Evv_s
        S["Phi_w"] += 2.0 * Eww_s

        if self.budget:
            self._add_budget(solver, wf, s, sl, ikx, ky, cu, cv, cw, cdudx, cdvdx, cdwdx,
                             cdudz, cdvdz, cdwdz, Euu_s, Evv_s, Eww_s, Euw_s)
        self.n_added[i] += 1

    # ------------------------------------------------------------------
    def _add_budget(self, solver, wf, s, sl, ikx, ky, cu, cv, cw, cdudx, cdvdx, cdwdx,
                    cdudz, cdvdz, cdwdz, Euu, Evv, Eww, Euw):
        g = self.grid
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        W = self.p.workers
        S = self.spec
        ky = ky[:, sl]

        # pressure, shifted (+ x derivative for Tp_x)
        ph = (shift_phase(Nx, g.Lx, s) / Ny).reshape(Nx, 1, 1)
        Ph = sfft.rfft2(solver.p, axes=(0, 1), workers=W)[:, sl]
        Ph *= ph
        cdpdx = sfft.ifft(ikx * Ph, axis=0, workers=W, overwrite_x=True)
        cp = sfft.ifft(Ph, axis=0, workers=W, overwrite_x=True)
        del Ph

        # wave fields in the packet frame, (Nx, 1, Nz) each
        if wf is None:
            wv = {k: np.zeros((Nx, 1, Nz)) for k in _WAVE_KEYS}
        else:
            stack = np.stack([np.broadcast_to(np.asarray(getattr(wf, k), dtype=float),
                                              (Nx, 1, Nz))[:, 0, :] for k in _WAVE_KEYS])
            sh = fourier_shift_x(stack, s, g.Lx, axis=1, workers=W)
            wv = {k: sh[n][:, None, :] for n, k in enumerate(_WAVE_KEYS)}

        # spanwise means of the run (approximation of <u_i>), (Nx, 1, Nz)
        ubar, vbar, wbar = cu[:, :1].real, cv[:, :1].real, cw[:, :1].real
        dubdx, dvbdx, dwbdx = cdudx[:, :1].real, cdvdx[:, :1].real, cdwdx[:, :1].real
        dubdz, dvbdz, dwbdz = cdudz[:, :1].real, cdvdz[:, :1].real, cdwdz[:, :1].real
        del vbar

        cu, cv, cw = cu[:, sl], cv[:, sl], cw[:, sl]
        cdudx, cdvdx, cdwdx = cdudx[:, sl], cdvdx[:, sl], cdwdx[:, sl]
        cdudz, cdvdz, cdwdz = cdudz[:, sl], cdvdz[:, sl], cdwdz[:, sl]

        # production by wave straining (Pw_y = 0)
        S["Pw_x"] -= 4.0 * (Euu * wv["duphi_dx_c"] + Euw * wv["duphi_dz_c"])
        S["Pw_z"] -= 4.0 * (Euw * wv["dwphi_dx_c"] + Eww * wv["dwphi_dz_c"])
        # production by the mean current
        S["Pr_x"] -= 4.0 * (Euu * dubdx + Euw * dubdz)
        S["Pr_y"] -= 4.0 * (_cre(cv, cu) * dvbdx + _cre(cv, cw) * dvbdz)
        S["Pr_z"] -= 4.0 * (Euw * dwbdx + Eww * dwbdz)
        # pressure strain and pressure diffusion
        a = _cre(cdudx, cp)
        S["Pis_x"] += 4.0 * a
        a += _cre(cu, cdpdx)
        S["Tp_x"] -= 4.0 * a
        # conj(i ky cv) cp = -i ky conj(cv) cp -> Re = ky Im(conj(cv) cp)
        S["Pis_y"] += 4.0 * ky * _cim(cv, cp)
        S["Pis_z"] += 4.0 * _cre(cdwdz, cp)
        cwp = 4.0 * _cre(cw, cp)
        if Nz >= 3:
            S["Tp_z"] -= np.gradient(cwp, g.zc, axis=-1, edge_order=2)
        else:
            S["Tp_z"] -= np.gradient(cwp, g.zc, axis=-1)
        del a, cwp, cp, cdpdx
        # advection by wave and by the mean current, and dPhi/dx'
        for comp, c, cx, cz in (("x", cu, cdudx, cdudz), ("y", cv, cdvdx, cdvdz),
                                ("z", cw, cdwdx, cdwdz)):
            Gx = _cre(c, cx)
            Gz = _cre(c, cz)
            S["Aphi_" + comp] -= 4.0 * (wv["uphi_c"] * Gx + wv["wphi_c"] * Gz)
            S["Amean_" + comp] -= 4.0 * (ubar * Gx + wbar * Gz)
            S["dPhidx_" + comp] += 4.0 * Gx

    # ------------------------------------------------------------------
    def _check_compatible(self, other: "PacketAccumulator") -> None:
        if (self.Nx, self.Ny, self.Nz) != (other.Nx, other.Ny, other.Nz):
            raise ValueError("grid shapes differ")
        if self.budget != other.budget:
            raise ValueError("budget flags differ")
        if self.sample_times.shape != other.sample_times.shape or \
                not np.allclose(self.sample_times, other.sample_times, rtol=1e-12, atol=1e-14):
            raise ValueError("sample times differ")
        _check_params(self.meta["params"], other.meta["params"])

    def merge(self, other: "PacketAccumulator") -> None:
        """Add the sums of another accumulator (other ensemble members)."""
        self._check_compatible(other)
        for k in self.mom:
            self.mom[k] += other.mom[k]
        for k in self.spec:
            self.spec[k] += other.spec[k]
        self.n_added += other.n_added

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        d = {"sample_times": self.sample_times, "n_added": self.n_added,
             "budget": np.array(self.budget),
             "meta_json": np.array(json.dumps(self.meta))}
        for k, a in self.mom.items():
            d["mom_" + k] = a
        for k, a in self.spec.items():
            d["spec_" + k] = a
        np.savez(path, **d)

    @classmethod
    def load(cls, path) -> "PacketAccumulator":
        with np.load(_npz_path(path), allow_pickle=False) as z:
            return cls._from_npz(z)

    @classmethod
    def _from_npz(cls, z) -> "PacketAccumulator":
        meta = json.loads(str(z["meta_json"]))
        p = _params_from_dict(meta["params"])
        obj = cls.__new__(cls)
        obj.p = p
        obj.grid = Grid(p)
        obj.meta = meta
        obj.budget = bool(z["budget"])
        obj.sample_times = np.array(z["sample_times"], dtype=float)
        obj.n_added = np.array(z["n_added"], dtype=np.int64)
        obj.Nx, obj.Ny, obj.Nz = p.Nx, p.Ny, p.Nz
        obj.nky = _nky(p.Ny)
        obj.mom = {k: np.array(z["mom_" + k]) for k in MOMENT_KEYS}
        keys = SPEC_KEYS + (BUDGET_KEYS if obj.budget else ())
        obj.spec = {k: np.array(z["spec_" + k]) for k in keys}
        return obj


# ----------------------------------------------------------------------
def packet_frame_x(Nx: int, Lx: float):
    """x'_j = j dx wrapped to [-Lx/2, Lx/2) and the index order that sorts it."""
    xw = wrap_periodic(np.arange(Nx) * (Lx / Nx), Lx)
    return xw, np.argsort(xw, kind="stable")


def packet_results(acc: PacketAccumulator, grid: Grid, p: Params, window: float = 3.5) -> dict:
    """Final packet-frame averages (eq. 3.1-3.2) restricted to |x'| <= window chi.

    Units: H = u_* = 1 (see params.py); no plotting normalisation is applied,
    the scale factors are returned as metadata (``enst_scale = Re_tau^2``,
    ``budget_scale = alpha u_*^2 omega0``)."""
    if (grid.Nx, grid.Ny, grid.Nz) != (acc.Nx, acc.Ny, acc.Nz):
        raise ValueError("grid does not match the accumulator")
    # the packet-frame shift and lhs use x0, cg (alpha, Fr, k0H, ...): p must be
    # the configuration the accumulator was filled with
    _check_params(acc.meta["params"], p.to_dict(), "Params passed to packet_results:")
    Nx, Ny = acc.Nx, acc.Ny
    xw, order = packet_frame_x(Nx, grid.Lx)
    chi = p.chi
    sel = order[np.abs(xw[order]) <= window * chi * (1.0 + 1e-12)]
    valid = acc.n_added > 0
    if not np.any(valid):
        raise ValueError("accumulator is empty")
    cnt = (acc.n_added[valid] * Ny).astype(float)[:, None, None]

    m = {k: acc.mom[k][valid][:, sel, :] / cnt for k in MOMENT_KEYS}
    out = {}
    out["xp"] = xw[sel]
    out["xp_chi"] = xw[sel] / chi
    out["zc"] = np.array(grid.zc)
    out["k0z"] = p.k0 * grid.zc
    my = np.arange(1, acc.nky + 1)
    ky = 2.0 * math.pi / grid.Ly * my
    out["ky"] = ky
    out["lam_y"] = 2.0 * math.pi / ky
    out["dky"] = 2.0 * math.pi / grid.Ly
    out["t_samples"] = acc.sample_times[valid]
    for a, b in (("U", "u"), ("V", "v"), ("W", "w"),
                 ("OX", "ox"), ("OY", "oy"), ("OZ", "oz")):
        out[a] = m[b].mean(axis=0)
    for key, sq, a, b in (("uu", "uu", "u", "u"), ("vv", "vv", "v", "v"),
                          ("ww", "ww", "w", "w"), ("uw", "uw", "u", "w"),
                          ("oxox", "oxox", "ox", "ox"), ("oyoy", "oyoy", "oy", "oy"),
                          ("ozoz", "ozoz", "oz", "oz")):
        rt = m[sq] - m[a] * m[b]            # per-sample Reynolds stress
        out[key + "_t"] = rt
        out[key] = rt.mean(axis=0)

    ntot = float(acc.n_added.sum())
    for k, a in acc.spec.items():
        out[k] = a[sel] / ntot
    nxp = sel.size
    zero = np.zeros((nxp, acc.nky, acc.Nz))
    if acc.budget:
        out["Pw_y"] = zero
        out["Tp_y"] = zero.copy()
        for c in ("x", "y", "z"):
            out["A_" + c] = out["Aphi_" + c] + out["Amean_" + c]
            out["lhs_" + c] = -p.cg * out["dPhidx_" + c]
            out["rhs_" + c] = (out["Pw_" + c] + out["Pr_" + c] + out["Pis_" + c]
                               + out["Tp_" + c] + out["A_" + c])
    out.update(alpha=p.alpha, Re_tau=p.Re_tau, nu=p.nu, omega0=p.omega0, k0=p.k0,
               chi=chi, cg=p.cg, x0=p.x0, Lx=grid.Lx, Ly=grid.Ly, window=window,
               n_runs=acc.n_runs, n_samples=int(valid.sum()), n_added=acc.n_added,
               budget=acc.budget, enst_scale=p.Re_tau ** 2,
               budget_scale=p.alpha * p.omega0)
    return out


# ----------------------------------------------------------------------
class BaseAccumulator:
    """Plane (x, y) + time averages of the base flow (no waves).

    Centre profiles (Nz,): U, V, W(c), u'^2, v'^2, w'^2, u'w', u'v', v'w' (w' at
    centres), nu dU/dz, <tau13> (SGS, tau = -2 nu_t S), spanwise spectra.
    Face profiles (Nz+1,) of the discrete x-momentum fluxes used by the
    solver: Reynolds flux <w' u'_f>, nu dU/dz at faces, SGS tau13_f -- the
    total ``-<w'u'_f> + nu dU/dz_f - tau13_f`` must equal ``1 + z_f`` in
    equilibrium (tau_s = 1 at the surface, mean pressure gradient 1).

    ``<tau13>`` is averaged over the samples that provided ``solver.nut`` only
    (``n_sgs``).  ``solver.nut`` is the eddy viscosity evaluated from the state at
    the beginning of the last step, so it lags u, v, w by one time step."""

    C_KEYS = ("U", "V", "W", "uu", "vv", "ww", "uw", "uv", "vw", "nu_dUdz", "tau13")
    F_KEYS = ("uw_f", "nu_dUdz_f", "tau13_f")
    S_KEYS = ("Phi_u", "Phi_v", "Phi_w")

    def __init__(self, grid: Grid, p: Params):
        self.grid = grid
        self.p = p
        Nz = grid.Nz
        self.nky = _nky(grid.Ny)
        self.n = 0
        self.n_sgs = 0
        self.t_first = np.nan
        self.t_last = np.nan
        self.c = {k: np.zeros(Nz) for k in self.C_KEYS}
        self.f = {k: np.zeros(Nz + 1) for k in self.F_KEYS}
        self.s = {k: np.zeros((self.nky, Nz)) for k in self.S_KEYS}
        self.meta = {"params": p.to_dict()}

    def add_sample(self, solver) -> None:
        g, p = self.grid, self.p
        W = p.workers
        u, v, w = solver.u, solver.v, solver.w
        bc = solver.bc
        nu = p.nu
        Nx, Ny = g.Nx, g.Ny
        wc = g.f2c(w)
        U = u.mean(axis=(0, 1))
        V = v.mean(axis=(0, 1))
        Wc = wc.mean(axis=(0, 1))
        up, vp, wp = u - U, v - V, wc - Wc
        n2 = 1.0 / (Nx * Ny)
        c = self.c
        c["U"] += U
        c["V"] += V
        c["W"] += Wc
        c["uu"] += np.einsum("xyz,xyz->z", up, up) * n2
        c["vv"] += np.einsum("xyz,xyz->z", vp, vp) * n2
        c["ww"] += np.einsum("xyz,xyz->z", wp, wp) * n2
        c["uw"] += np.einsum("xyz,xyz->z", up, wp) * n2
        c["uv"] += np.einsum("xyz,xyz->z", up, vp) * n2
        c["vw"] += np.einsum("xyz,xyz->z", vp, wp) * n2
        dudz_s = np.asarray(bc["dudz_s"], dtype=float)
        dUs = float(dudz_s.mean())
        c["nu_dUdz"] += nu * g.ddz_c(U, 0.0, dUs)
        # face fluxes
        top = bc.get("u_s") if isinstance(bc, dict) else None
        uf = g.c2f(u, bottom=u[..., 0], top=top)
        Wf = w.mean(axis=(0, 1))
        Uf = uf.mean(axis=(0, 1))
        f = self.f
        f["uw_f"] += np.einsum("xyz,xyz->z", w, uf) * n2 - Wf * Uf
        nd = np.zeros(g.Nz + 1)
        nd[1:-1] = (U[1:] - U[:-1]) / g.dzf[1:-1]
        nd[-1] = dUs
        f["nu_dUdz_f"] += nu * nd
        nut = getattr(solver, "nut", None)
        if nut is not None:
            nut = np.asarray(nut, dtype=float)
            # centres: tau13 = -nu_t (du/dz + dw/dx)
            dudz = g.ddz_c(u, 0.0, dudz_s)
            dwdx_c = _ddx_real(wc, g.Lx, W)
            c["tau13"] += -np.einsum("xyz,xyz->z", nut, dudz + dwdx_c) * n2
            # faces (as in sgs.py): nu_t,f (du/dz + dw/dx)_f, zero at k = 0, Nz
            nutf = g.wlo[1:-1] * nut[..., :-1] + g.whi[1:-1] * nut[..., 1:]
            dudz_f = (u[..., 1:] - u[..., :-1]) / g.dzf[1:-1]
            dwdx_f = _ddx_real(w[..., 1:-1], g.Lx, W)
            f["tau13_f"][1:-1] += -np.einsum("xyz,xyz->z", nutf, dudz_f + dwdx_f) * n2
            self.n_sgs += 1
        # spanwise spectra (x-averaged)
        sl = slice(1, self.nky + 1)
        for k, q in (("Phi_u", u), ("Phi_v", v), ("Phi_w", wc)):
            qh = sfft.rfft(q, axis=1, workers=W)[:, sl] / Ny
            self.s[k] += 2.0 * _abs2(qh).mean(axis=0)
        t = float(getattr(solver, "t", np.nan))
        if self.n == 0:
            self.t_first = t
        self.t_last = t
        self.n += 1

    def merge(self, other: "BaseAccumulator") -> None:
        _check_params(self.meta["params"], other.meta["params"])
        for d, o in ((self.c, other.c), (self.f, other.f), (self.s, other.s)):
            for k in d:
                d[k] += o[k]
        if self.n == 0:
            self.t_first = other.t_first
        self.n += other.n
        self.n_sgs += other.n_sgs
        self.t_last = other.t_last if np.isfinite(other.t_last) else self.t_last

    def results(self) -> dict:
        if self.n == 0:
            raise ValueError("BaseAccumulator is empty")
        g, p = self.grid, self.p
        n = float(self.n)
        # SGS stresses are averaged over the samples that had solver.nut (it is
        # None right after set_state); dividing by n would bias them low
        n_sgs = float(max(self.n_sgs, 1))
        sgs_keys = ("tau13", "tau13_f")
        out = {k: a / (n_sgs if k in sgs_keys else n) for k, a in self.c.items()}
        out.update({k: a / (n_sgs if k in sgs_keys else n) for k, a in self.f.items()})
        out.update({k: a / n for k, a in self.s.items()})
        out["zc"] = np.array(g.zc)
        out["zf"] = np.array(g.zf)
        out["ky"] = 2.0 * math.pi / g.Ly * np.arange(1, self.nky + 1)
        out["total"] = -out["uw"] + out["nu_dUdz"] - out["tau13"]
        out["total_expected"] = 1.0 + g.zc
        out["total_f"] = -out["uw_f"] + out["nu_dUdz_f"] - out["tau13_f"]
        out["total_f_expected"] = 1.0 + g.zf
        out.update(n_samples=self.n, n_sgs=self.n_sgs, t_first=self.t_first,
                   t_last=self.t_last, Re_tau=p.Re_tau, nu=p.nu)
        return out

    def save(self, path) -> None:
        d = {"n": np.array(self.n), "n_sgs": np.array(self.n_sgs),
             "t_first": np.array(self.t_first), "t_last": np.array(self.t_last),
             "meta_json": np.array(json.dumps(self.meta))}
        for pre, dd in (("c_", self.c), ("f_", self.f), ("s_", self.s)):
            for k, a in dd.items():
                d[pre + k] = a
        np.savez(path, **d)

    @classmethod
    def load(cls, path) -> "BaseAccumulator":
        with np.load(_npz_path(path), allow_pickle=False) as z:
            return cls._from_npz(z)

    @classmethod
    def _from_npz(cls, z) -> "BaseAccumulator":
        meta = json.loads(str(z["meta_json"]))
        p = _params_from_dict(meta["params"])
        obj = cls(Grid(p), p)
        obj.meta = meta
        obj.n = int(z["n"])
        obj.n_sgs = int(z["n_sgs"])
        obj.t_first = float(z["t_first"])
        obj.t_last = float(z["t_last"])
        for pre, dd in (("c_", obj.c), ("f_", obj.f), ("s_", obj.s)):
            for k in dd:
                dd[k] = np.array(z[pre + k])
        return obj
