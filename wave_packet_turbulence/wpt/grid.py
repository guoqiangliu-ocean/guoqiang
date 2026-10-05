"""Computational grid and elementary operators.

Horizontal directions (x, y) are periodic and treated pseudo-spectrally.
The vertical direction z in [-H, 0] uses a staggered finite-difference grid:

* u, v, p (and every derived scalar) live at the Nz cell centres ``zc``;
* w lives at the Nz + 1 cell faces ``zf`` (``zf[0] = -H`` bottom,
  ``zf[Nz] = 0`` mean free surface).

Array layout
------------
physical, centres : (Nx, Ny, Nz)        real
physical, faces   : (Nx, Ny, Nz + 1)    real
spectral          : (Nx, Ny//2 + 1, nz) complex, from ``scipy.fft.rfft2`` over
                    axes (0, 1) (complex FFT in x, real FFT in y), unnormalised
                    forward transform (``norm='backward'``).
2-D wave quantities that do not depend on y are stored with shape (Nx, 1, nz)
or (Nx, 1) so that they broadcast against 3-D arrays.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.fft as sfft

from .params import Params


class Grid:
    def __init__(self, p: Params):
        self.p = p
        self.Nx, self.Ny, self.Nz = p.Nx, p.Ny, p.Nz
        self.Lx, self.Ly, self.H = p.Lx, p.Ly, p.H
        self.dx = p.Lx / p.Nx
        self.dy = p.Ly / p.Ny
        self.workers = p.workers
        self.Nyh = p.Ny // 2 + 1

        self.x = np.arange(p.Nx) * self.dx
        self.y = np.arange(p.Ny) * self.dy

        # ---- vertical faces: tanh clustering towards the surface z = 0 ----
        # s = distance index from the surface (0 at z = 0, 1 at z = -H)
        s = np.linspace(0.0, 1.0, p.Nz + 1)
        gam = p.z_stretch
        if gam > 0:
            zf_from_top = -p.H * (1.0 - np.tanh(gam * (1.0 - s)) / math.tanh(gam))
        else:
            zf_from_top = -p.H * s
        self.zf = np.ascontiguousarray(zf_from_top[::-1])   # ascending, zf[0] = -H, zf[-1] = 0
        self.zf[0] = -p.H
        self.zf[-1] = 0.0
        self.zc = 0.5 * (self.zf[1:] + self.zf[:-1])
        self.dzc = self.zf[1:] - self.zf[:-1]               # cell widths, (Nz,)
        # centre-to-centre distances at faces; boundary entries are half cells
        dzf = np.empty(p.Nz + 1)
        dzf[1:-1] = self.zc[1:] - self.zc[:-1]
        dzf[0] = self.zc[0] - self.zf[0]
        dzf[-1] = self.zf[-1] - self.zc[-1]
        self.dzf = dzf

        # linear interpolation weights centre -> interior face k (1..Nz-1):
        # q_f[k] = wlo[k] * q_c[k-1] + whi[k] * q_c[k]
        wlo = np.zeros(p.Nz + 1)
        whi = np.zeros(p.Nz + 1)
        wlo[1:-1] = (self.zc[1:] - self.zf[1:-1]) / dzf[1:-1]
        whi[1:-1] = (self.zf[1:-1] - self.zc[:-1]) / dzf[1:-1]
        self.wlo, self.whi = wlo, whi

        # ---- wavenumbers for the rfft2 layout ----
        mx = np.fft.fftfreq(p.Nx, d=1.0 / p.Nx)            # integer mode numbers
        my = np.fft.rfftfreq(p.Ny, d=1.0 / p.Ny)
        self.mx = mx
        self.my = my
        self.kx = (2.0 * math.pi / p.Lx * mx).reshape(p.Nx, 1, 1)
        self.ky = (2.0 * math.pi / p.Ly * my).reshape(1, self.Nyh, 1)
        self.k2 = self.kx ** 2 + self.ky ** 2                  # (Nx, Nyh, 1)
        # 2/3-rule dealiasing mask: keep |mx| < Nx/3 and my < Ny/3
        keep = (np.abs(mx) < p.Nx / 3.0).reshape(p.Nx, 1, 1) & \
               (my < p.Ny / 3.0).reshape(1, self.Nyh, 1)
        self.dealias = keep
        # 1-D x mask for y-independent wave fields
        self.dealias_x = (np.abs(mx) < p.Nx / 3.0)

    # ------------------------------------------------------------------
    # horizontal transforms
    def fft(self, a: np.ndarray) -> np.ndarray:
        return sfft.rfft2(a, axes=(0, 1), workers=self.workers)

    def ifft(self, ah: np.ndarray) -> np.ndarray:
        return sfft.irfft2(ah, s=(self.Nx, self.Ny), axes=(0, 1), workers=self.workers)

    def ddx(self, ah: np.ndarray) -> np.ndarray:
        return 1j * self.kx * ah

    def ddy(self, ah: np.ndarray) -> np.ndarray:
        return 1j * self.ky * ah

    def truncate(self, ah: np.ndarray) -> np.ndarray:
        """Apply the 2/3 dealiasing mask in place and return ah."""
        ah *= self.dealias
        return ah

    # 1-D (x only) helpers for y-independent wave fields, shape (Nx, ...)
    def fft_x(self, a: np.ndarray) -> np.ndarray:
        return sfft.fft(a, axis=0, workers=self.workers)

    def ifft_x(self, ah: np.ndarray) -> np.ndarray:
        return sfft.ifft(ah, axis=0, workers=self.workers).real

    def kx1d(self, ndim_after: int = 0) -> np.ndarray:
        """kx as a column (Nx, 1, ..., 1) with ``ndim_after`` trailing axes."""
        return (2.0 * math.pi / self.Lx * self.mx).reshape((self.Nx,) + (1,) * ndim_after)

    # ------------------------------------------------------------------
    # vertical staggered operators (act on the last axis; work for real or
    # complex arrays, physical or spectral, since coefficients are real)
    def c2f(self, qc: np.ndarray, bottom=None, top=None) -> np.ndarray:
        """Centres (..., Nz) -> faces (..., Nz+1).  Interior faces are linearly
        interpolated; boundary faces take ``bottom``/``top`` if given (arrays
        broadcastable to qc[..., 0]) or are linearly extrapolated otherwise."""
        Nz = self.Nz
        qf = np.empty(qc.shape[:-1] + (Nz + 1,), dtype=qc.dtype)
        qf[..., 1:-1] = self.wlo[1:-1] * qc[..., :-1] + self.whi[1:-1] * qc[..., 1:]
        if bottom is None:
            slope = (qc[..., 1] - qc[..., 0]) / self.dzf[1]
            qf[..., 0] = qc[..., 0] - slope * self.dzf[0]
        else:
            qf[..., 0] = bottom
        if top is None:
            slope = (qc[..., -1] - qc[..., -2]) / self.dzf[-2]
            qf[..., -1] = qc[..., -1] + slope * self.dzf[-1]
        else:
            qf[..., -1] = top
        return qf

    def f2c(self, qf: np.ndarray) -> np.ndarray:
        """Faces (..., Nz+1) -> centres (..., Nz) (cell midpoint average)."""
        return 0.5 * (qf[..., 1:] + qf[..., :-1])

    def ddz_f2c(self, qf: np.ndarray) -> np.ndarray:
        """d/dz of a face field evaluated at centres (..., Nz)."""
        return (qf[..., 1:] - qf[..., :-1]) / self.dzc

    def ddz_c2f(self, qc: np.ndarray, bottom=None, top=None) -> np.ndarray:
        """d/dz of a centre field evaluated at faces (..., Nz+1).  Boundary
        face values are ``bottom``/``top`` if given, else zero."""
        Nz = self.Nz
        df = np.zeros(qc.shape[:-1] + (Nz + 1,), dtype=qc.dtype)
        df[..., 1:-1] = (qc[..., 1:] - qc[..., :-1]) / self.dzf[1:-1]
        if bottom is not None:
            df[..., 0] = bottom
        if top is not None:
            df[..., -1] = top
        return df

    def ddz_c(self, qc: np.ndarray, dqdz_bottom=0.0, dqdz_top=0.0) -> np.ndarray:
        """d/dz of a centre field at centres: average of the two adjacent face
        gradients; boundary face gradients are supplied (Neumann data)."""
        df = self.ddz_c2f(qc, bottom=dqdz_bottom, top=dqdz_top)
        return 0.5 * (df[..., 1:] + df[..., :-1])

    def ddz_f(self, qf: np.ndarray) -> np.ndarray:
        """d/dz of a face field at faces (central difference at interior faces,
        one-sided at the two boundary faces)."""
        Nz = self.Nz
        d = np.empty_like(qf)
        zf = self.zf
        d[..., 1:-1] = (qf[..., 2:] - qf[..., :-2]) / (zf[2:] - zf[:-2])
        d[..., 0] = (qf[..., 1] - qf[..., 0]) / (zf[1] - zf[0])
        d[..., -1] = (qf[..., -1] - qf[..., -2]) / (zf[-1] - zf[-2])
        return d
