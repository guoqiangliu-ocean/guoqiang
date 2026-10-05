"""Initial conditions and grid-to-grid interpolation for the base flow."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .grid import Grid


def mean_profile(grid: Grid, Re_tau: float, kappa: float = 0.41, A_plus: float = 26.0) -> np.ndarray:
    """Approximate equilibrium mean velocity U(z) at the centres for the
    surface-stress-driven layer: total stress tau = 1 + z (1 at the surface,
    0 at the free-slip bottom) carried by nu + nu_t with a van Driest damped
    mixing length l = kappa d (1 - d) (1 - exp(-d+/A+)), d = -z.  Shifted so
    that the depth integral vanishes (the flow starts from rest)."""
    nu = 1.0 / Re_tau
    d = np.linspace(0.0, grid.H, 20001)
    tau = 1.0 - d / grid.H
    l = kappa * d * (1.0 - d / grid.H) * (1.0 - np.exp(-d * Re_tau / A_plus))
    # tau = (nu + l^2 |dU/dd|) |dU/dd|  ->  solve the quadratic for |dU/dd|
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(l > 1e-12, (-nu + np.sqrt(nu ** 2 + 4.0 * l ** 2 * tau)) / (2.0 * np.maximum(l, 1e-12) ** 2),
                     tau / nu)
    # U decreases with depth: U(d) = U_s - int_0^d s
    deficit = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(d))])
    U = np.interp(-grid.zc, d, -deficit)
    U -= np.sum(U * grid.dzc) / grid.H
    return U


def random_divfree_perturbation(grid: Grid, amp: float, seed: int, kmax_frac: float = 0.25,
                                depth_decay: float = 0.3):
    """Random solenoidal perturbation (u, v at centres; w at faces) that
    satisfies w = 0 at both boundaries, built from a vector potential.

    The perturbation is concentrated near the surface (e-folding depth
    ``depth_decay``) and contains horizontal modes up to ``kmax_frac`` of the
    resolved range.  Discrete divergence is removed afterwards by the solver's
    projection (``set_state``)."""
    rng = np.random.default_rng(seed)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    zc, zf = grid.zc, grid.zf
    env_c = np.exp(zc / depth_decay) * np.sin(np.pi * (zc + grid.H) / grid.H) ** 0.5
    env_f = np.exp(zf / depth_decay) * np.sin(np.pi * (zf + grid.H) / grid.H) ** 0.5
    mask = (np.abs(grid.mx) <= kmax_frac * Nx / 2).reshape(Nx, 1, 1) & \
           (grid.my <= kmax_frac * Ny / 2).reshape(1, grid.Nyh, 1)
    out = []
    for env, nz in ((env_c, Nz), (env_c, Nz), (env_f, Nz + 1)):
        a = rng.standard_normal((Nx, Ny, nz))
        ah = grid.fft(a) * mask
        a = grid.ifft(ah) * env
        out.append(a)
    u, v, w = out
    w[..., 0] = 0.0
    w[..., -1] = 0.0
    # normalise to the requested rms amplitude in the top half
    top = zc > -0.5
    rms = np.sqrt(np.mean(u[..., top] ** 2 + v[..., top] ** 2) / 2.0)
    s = amp / max(rms, 1e-30)
    return u * s, v * s, w * s


def interp_horizontal(a: np.ndarray, Nx_new: int, Ny_new: int, workers: int = 4) -> np.ndarray:
    """Spectral (zero-padding / truncation) interpolation in x and y of a
    real array (Nx, Ny, nz)."""
    Nx, Ny = a.shape[0], a.shape[1]
    ah = sfft.rfft2(a, axes=(0, 1), workers=workers)
    Nyh_new = Ny_new // 2 + 1
    out = np.zeros((Nx_new, Nyh_new) + a.shape[2:], dtype=complex)
    nx_keep = min(Nx, Nx_new) // 2
    ny_keep = min(Ny // 2, Ny_new // 2)
    out[:nx_keep, :ny_keep] = ah[:nx_keep, :ny_keep]
    out[-nx_keep + 1:, :ny_keep] = ah[-nx_keep + 1:, :ny_keep] if nx_keep > 1 else 0.0
    out *= (Nx_new * Ny_new) / (Nx * Ny)
    return sfft.irfft2(out, s=(Nx_new, Ny_new), axes=(0, 1), workers=workers)


def interp_vertical(a: np.ndarray, z_old: np.ndarray, z_new: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation along the last axis (constant extrapolation)."""
    idx = np.clip(np.searchsorted(z_old, z_new) - 1, 0, len(z_old) - 2)
    z0, z1 = z_old[idx], z_old[idx + 1]
    w1 = np.clip((z_new - z0) / (z1 - z0), 0.0, 1.0)
    return a[..., idx] * (1.0 - w1) + a[..., idx + 1] * w1


def regrid_state(u, v, w, g_old: Grid, g_new: Grid):
    """Interpolate a velocity state from one grid to another (horizontal
    spectral, vertical linear).  The result must be re-projected."""
    wk = g_new.workers
    u2 = interp_vertical(interp_horizontal(u, g_new.Nx, g_new.Ny, wk), g_old.zc, g_new.zc)
    v2 = interp_vertical(interp_horizontal(v, g_new.Nx, g_new.Ny, wk), g_old.zc, g_new.zc)
    w2 = interp_vertical(interp_horizontal(w, g_new.Nx, g_new.Ny, wk), g_old.zf, g_new.zf)
    w2[..., 0] = 0.0
    return u2, v2, w2
