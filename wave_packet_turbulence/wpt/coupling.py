"""Irrotational (wave) fields on the LES grid from the HOS solution.

SPEC.md section 3.  The HOS solver provides the modal amplitudes of the
velocity potential on its own ``N``-point periodic x grid,

    phi(x, z) = irfft(c * exp(|k| z), n=N),   c = hos.phi_modes()  (len N//2+1),

with the unnormalised numpy rfft convention.  On the LES x grid (``Nx`` points,
same period ``Lx``) the same Fourier mode has coefficient ``(Nx/N) c_m``.  Only
the modes ``m < Nx/3`` are retained (2/3-rule dealiasing of the LES products),
so the LES Nyquist mode is always zero.

For every retained mode the vertical structure ``exp(k z)`` (``k >= 0`` for the
rfft half-spectrum, so ``|k| = k``) is tabulated once per grid at the cell
centres ``zc`` and faces ``zf`` (cached).  With ``A_m(z) = k c_m exp(k z)``
(spectrum of ``wphi = phi_z``) all fields follow from four inverse real FFTs:

    uphi     = phi_x  <-  i A          wphi     = phi_z  <-  A
    duphi_dx = phi_xx <- -k A          duphi_dz = phi_xz <-  i k A
    dwphi_dx = phi_xz (= duphi_dz)     dwphi_dz = phi_zz  = -phi_xx

``irrot_stress_x = 2 phi_xz(x, 0)`` is the top-face value of ``phi_xz``
(``zf[-1] = 0`` exactly), and ``eta``, ``eta_x`` are obtained from ``hos.eta``
with the same spectral resampling and truncation.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.fft as sfft

from .grid import Grid
from .params import Params
from .wavefields import WaveFields, zero_wave_fields

__all__ = ["wave_fields_from_hos", "resample_rfft", "clear_cache"]

# ---------------------------------------------------------------------------
# cache of per-grid tables
_CACHE: dict = {}
_CACHE_MAX = 8


class _Tables:
    """Precomputed quantities for one LES grid."""

    def __init__(self, grid: Grid):
        Nx, Nz = grid.Nx, grid.Nz
        self.Nx, self.Nz = Nx, Nz
        self.Nxh = Nx // 2 + 1
        # retained rfft modes: 0 <= m < Nx/3 (same rule as grid.dealias_x)
        self.nm = int(math.ceil(Nx / 3.0))          # m = 0 .. nm-1
        m = np.arange(self.nm)
        assert np.all(m < Nx / 3.0)
        self.k = 2.0 * math.pi / grid.Lx * m        # (nm,) >= 0
        z = np.concatenate([grid.zc, grid.zf])      # (2Nz+1,)
        self.nzt = z.size
        # k exp(k z) and k^2 exp(k z): (nm, 2Nz+1); exp(k z) <= 1 for z <= 0
        e = np.exp(self.k[:, None] * z[None, :])
        self.kE = self.k[:, None] * e
        self.k2E = self.k[:, None] * self.kE
        self.ik = 1j * self.k


def _tables(grid: Grid) -> _Tables:
    key = (grid.Nx, float(grid.Lx), grid.Nz, grid.zc.tobytes(), grid.zf.tobytes())
    t = _CACHE.get(key)
    if t is None:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        t = _Tables(grid)
        _CACHE[key] = t
    return t


def clear_cache() -> None:
    _CACHE.clear()


# ---------------------------------------------------------------------------
def resample_rfft(c: np.ndarray, N: int, Nx: int, nm: int) -> np.ndarray:
    """Map rfft coefficients of an ``N``-point signal (unnormalised, length
    ``N//2+1``) to the coefficients of the same trigonometric polynomial on an
    ``Nx``-point grid, keeping only modes ``m < nm`` (``nm <= Nx/2`` so the LES
    Nyquist mode is never filled).

    Returns a complex array of length ``nm``.  Modes ``m > N/2`` that do not
    exist on the HOS grid are zero.  If the HOS Nyquist mode ``m = N/2`` (N
    even) falls inside the retained range (only when ``N < 2 nm``), it carries
    ``(1/N) c cos(k x)`` on the HOS grid, i.e. half the amplitude of an
    ordinary mode, so its LES coefficient is ``(Nx/N) Re(c)/2``.
    """
    c = np.asarray(c)
    out = np.zeros(nm, dtype=complex)
    mmax = min(nm, N // 2 + 1)
    out[:mmax] = c[:mmax]
    if N % 2 == 0 and N // 2 < nm:
        out[N // 2] = 0.5 * c[N // 2].real
    out *= Nx / N
    return out


def wave_fields_from_hos(hos, grid: Grid, p: Params) -> WaveFields:
    """Sample the HOS wave field on the LES grid (SPEC.md section 3).

    Only ``hos.phi_modes()``, ``hos.eta``, ``hos.t`` and ``hos.N`` are used.
    All returned fields are dealiased in x (``|mx| < Nx/3``)."""
    t_now = float(getattr(hos, "t", 0.0)) if hos is not None else 0.0
    if not p.wave_on:
        return zero_wave_fields(grid, t=t_now)

    tb = _tables(grid)
    Nx, Nz, nm, Nxh = tb.Nx, tb.Nz, tb.nm, tb.Nxh
    N = int(hos.N)
    if N <= 0:
        raise ValueError("hos.N must be positive")

    c = np.asarray(hos.phi_modes())
    if c.shape != (N // 2 + 1,):
        raise ValueError(f"phi_modes() has shape {c.shape}, expected ({N // 2 + 1},)")
    eta_hos = np.asarray(hos.eta, dtype=float)
    if eta_hos.shape != (N,):
        raise ValueError(f"hos.eta has shape {eta_hos.shape}, expected ({N},)")

    cl = resample_rfft(c, N, Nx, nm)                       # (nm,)
    el = resample_rfft(sfft.rfft(eta_hos), N, Nx, nm)      # (nm,)

    workers = p.workers
    # ---- 3-D (x, z) fields: four spectra stacked -> one batched irfft ----
    S = np.zeros((4, Nxh, tb.nzt), dtype=complex)
    A = cl[:, None] * tb.kE                                # wphi spectrum
    kA = cl[:, None] * tb.k2E                              # k * A
    S[0, :nm] = 1j * A                                     # uphi = phi_x
    S[1, :nm] = A                                          # wphi = phi_z
    S[2, :nm] = -kA                                        # phi_xx
    S[3, :nm] = 1j * kA                                    # phi_xz
    R = sfft.irfft(S, n=Nx, axis=1, workers=workers)       # (4, Nx, 2Nz+1)

    def cen(a):
        return np.ascontiguousarray(a[:, :Nz]).reshape(Nx, 1, Nz)

    def fac(a):
        return np.ascontiguousarray(a[:, Nz:]).reshape(Nx, 1, Nz + 1)

    uphi, wphi, pxx, pxz = R[0], R[1], R[2], R[3]

    # ---- surface quantities ----
    E = np.zeros((2, Nxh), dtype=complex)
    E[0, :nm] = el
    E[1, :nm] = tb.ik * el
    es = sfft.irfft(E, n=Nx, axis=1, workers=workers)      # (2, Nx)
    eta = es[0].reshape(Nx, 1).copy()
    eta_x = es[1].reshape(Nx, 1).copy()

    if p.include_irrot_stress:
        irr = (2.0 * pxz[:, -1]).reshape(Nx, 1).copy()     # zf[-1] = 0
    else:
        irr = np.zeros((Nx, 1))

    pxz_c = cen(pxz)
    pxz_f = fac(pxz)
    pxx_c = cen(pxx)
    pxx_f = fac(pxx)
    return WaveFields(
        t=t_now, eta=eta, eta_x=eta_x, irrot_stress_x=irr,
        uphi_c=cen(uphi), wphi_c=cen(wphi),
        duphi_dx_c=pxx_c, duphi_dz_c=pxz_c,
        dwphi_dx_c=pxz_c.copy(), dwphi_dz_c=-pxx_c,
        uphi_f=fac(uphi), wphi_f=fac(wphi),
        duphi_dx_f=pxx_f, duphi_dz_f=pxz_f,
        dwphi_dx_f=pxz_f.copy(), dwphi_dz_f=-pxx_f,
    )
