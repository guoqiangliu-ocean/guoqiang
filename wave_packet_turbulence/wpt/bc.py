"""Taylor-expanded free-surface boundary conditions at the mean surface z = 0.

SPEC.md section 3; paper Appendix A, eqs. (A6)-(A8), truncated at O(alpha):

    (A6)  w_s     = d(eta u_s)/dx + d(eta v_s)/dy
    (A7)  du/dz|s = tau_x^r/nu - dw_s/dx + 2 eta_x (2 du_s/dx + dv_s/dy)
                    + eta_y (du_s/dy + dv_s/dx) - eta (w_zx|s + u_zz|s)
    (A8)  dv/dz|s = tau_y^r/nu - dw_s/dy + 2 eta_y (2 dv_s/dy + du_s/dx)
                    + eta_x (dv_s/dx + du_s/dy) - eta (w_zy|s + v_zz|s)

(A7)/(A8) are the paper's expressions solved for the vertical gradients: the paper
writes  tau_x^r/(rho nu) = (w_x + u_z) - 2 eta_x (u_x - w_z) - eta_y (u_y + v_x)
+ eta (w_zx + u_zz), and with w_z|s = -(u_x + v_y) (continuity) u_x - w_z = 2 u_x + v_y.

The O(alpha) corrections are evaluated explicitly from the current (lagged)
rotational field; all products with eta are dealiased with the 2/3 rule.

u_zz|s
------
The O(alpha) term ``eta u_zz|s`` is the first-order Taylor estimate of
``u_z(eta) - u_z(0)``.  Four approximations are available (``uzz_method``):

* ``'wave_scale'`` (default): curvature of the quadratic through the top centre and
  the centres closest to the depths L and 2L, ``L = max(2 dzc[-1], max|eta|)``,
  i.e. the mean curvature over the depth range swept by the surface.  For a smooth
  field its error is O(L) = O(a), so the error of ``eta u_zz`` is O(alpha^2), the
  order already neglected by (A6)-(A8).  It is the only variant found to be stable
  with the wave fields of the paper (see below).
* ``'cubic'``: second derivative at z = 0 of the cubic through the four top centres
  (second-order accurate in dz; used for the convergence test);
* ``'quadratic'``: same with the three top centres (first order);
* ``'top_cell'``: the SPEC.md formula ``[g - (q[-1]-q[-2])/dzf[-2]] / dzc[-1]`` with
  ``g = g_prev['dudz_s']`` (lagged surface gradient), first order.

Stability.  The continuous problem ``u_t + W u_z = nu u_zz`` with the Robin-type
condition ``u_z + eta u_zz = R`` at z = 0 has, for eta < 0, the mode
``exp(-z/eta)`` with growth rate ``nu/eta^2 + W/eta`` (unbounded as eta -> 0): the
first-order Taylor condition is ill-posed at scales below |eta|.  On the reduced and
paper grids ``a0/dzc[-1]`` = 4.2 and 7.5, so grid-scale stencils excite it:
``'top_cell'`` amplifies the lagged gradient by ``eta/dzc[-1]`` per step, and the
top-cell stencils ``'cubic'``/``'quadratic'`` feed back through the wave advection
``wphi du/dz`` at the top centre with a per-step factor ~ dt |wphi eta| c/dz^2 >> 1.
``'wave_scale'`` reduces that factor to ~ dt |wphi| / (2 max|eta|) ~ 0.07.
"""
from __future__ import annotations

import numpy as np

from .grid import Grid
from .params import Params

__all__ = ["surface_w", "surface_bc", "uzz_weights", "wave_scale_stencil"]


def _fft2(grid: Grid, a: np.ndarray) -> np.ndarray:
    """rfft2 over (x, y) of a stack (Nx, Ny, m)."""
    return grid.fft(a)


def _ifft2(grid: Grid, ah: np.ndarray) -> np.ndarray:
    return grid.ifft(ah)


def surface_w(u_s: np.ndarray, v_s: np.ndarray, eta: np.ndarray, grid: Grid) -> np.ndarray:
    """Eq. (A6): w_s = d(eta u_s)/dx + d(eta v_s)/dy, pseudo-spectral and dealiased.

    ``u_s, v_s``: physical (Nx, Ny); ``eta``: (Nx, 1) (or (Nx, Ny)).
    Returns physical (Nx, Ny) with zero horizontal mean (exactly zero (0,0) mode)."""
    Nx, Ny = grid.Nx, grid.Ny
    eta = np.asarray(eta)
    P = np.empty((Nx, Ny, 2))
    P[..., 0] = eta * u_s
    P[..., 1] = eta * v_s
    Ph = _fft2(grid, P)
    wh = 1j * grid.kx[..., 0] * Ph[..., 0] + 1j * grid.ky[..., 0] * Ph[..., 1]
    wh *= grid.dealias[..., 0]
    wh[0, 0] = 0.0
    return _ifft2(grid, wh[..., None])[..., 0]


def uzz_weights(grid: Grid, npts: int = 4) -> np.ndarray:
    """Weights c (npts,) such that sum_i c_i q[Nz-1-i] approximates d2q/dz2 at
    z = 0 (exact for polynomials of degree npts-1)."""
    z = grid.zc[::-1][:npts]                 # top centre first
    V = np.vander(z, npts, increasing=True).T  # V[m, i] = z_i^m
    rhs = np.zeros(npts)
    rhs[2] = 2.0
    return np.linalg.solve(V, rhs)


def wave_scale_stencil(grid: Grid, eta) -> tuple:
    """Centre indices (k1, k2, k3) for the 'wave_scale' u_zz estimate: the top centre
    and the centres closest to depths L and 2L, L = max(2 dzc[-1], max|eta|)."""
    zc = grid.zc
    Nz = grid.Nz
    L = max(2.0 * grid.dzc[-1], float(np.max(np.abs(eta))) if np.size(eta) else 0.0)
    k1 = Nz - 1
    k2 = int(np.argmin(np.abs(zc + L)))
    k2 = min(k2, k1 - 1)
    k3 = int(np.argmin(np.abs(zc + 2.0 * L)))
    k3 = max(min(k3, k2 - 1), 0)
    if k2 <= k3:          # very coarse grid: fall back to the top three centres
        k2, k3 = k1 - 1, k1 - 2
    return k1, k2, k3


def _uzz(q: np.ndarray, grid: Grid, method: str, g=None, eta=None) -> np.ndarray:
    if method == "wave_scale":
        k1, k2, k3 = wave_scale_stencil(grid, eta)
        z = grid.zc
        d12 = (q[..., k1] - q[..., k2]) / (z[k1] - z[k2])
        d23 = (q[..., k2] - q[..., k3]) / (z[k2] - z[k3])
        return 2.0 * (d12 - d23) / (z[k1] - z[k3])
    if method == "cubic" or method == "quadratic":
        n = 4 if method == "cubic" else 3
        c = uzz_weights(grid, n)
        out = c[0] * q[..., -1]
        for i in range(1, n):
            out = out + c[i] * q[..., -1 - i]
        return out
    if method == "top_cell":
        s = (q[..., -1] - q[..., -2]) / grid.dzf[-2]
        if g is None:
            g = s
        return (g - s) / grid.dzc[-1]
    raise ValueError(f"unknown uzz_method {method!r}")


def surface_bc(u: np.ndarray, v: np.ndarray, w_top_prev: np.ndarray, grid: Grid, wf,
               p: Params, g_prev: dict | None = None, tau_x: float | None = None,
               tau_y: float = 0.0, uzz_method: str = "wave_scale") -> dict:
    """Surface boundary data from the rotational field (SPEC.md section 3).

    Parameters
    ----------
    u, v : physical centre fields (Nx, Ny, Nz)
    w_top_prev : current surface vertical velocity w[..., Nz] (Nx, Ny); used for dw_s/dx, dw_s/dy
    wf : WaveFields (uses eta, eta_x, eta_y, irrot_stress_x)
    g_prev : previous result of surface_bc (only used by ``uzz_method='top_cell'``)
    tau_x, tau_y : leading-order rotational surface stress / nu without the irrotational
        part; default ``tau_x = p.Re_tau`` (tau_0 = 1).  The LES passes 0 when its
        ``surface_stress_on`` switch is off.

    Returns dict with physical (Nx, Ny) arrays 'u_s', 'v_s', 'dudz_s', 'dvdz_s', 'w_s',
    'uzz_s', 'vzz_s'.
    """
    Nx, Ny = grid.Nx, grid.Ny
    h_t = grid.dzf[-1]
    if tau_x is None:
        tau_x = p.Re_tau
    eta = np.asarray(wf.eta, dtype=float)
    eta_x = np.asarray(wf.eta_x, dtype=float)
    eta_y = np.asarray(wf.eta_y, dtype=float)
    mask = grid.dealias[..., 0]                               # (Nx, Nyh)
    kx = grid.kx                                              # (Nx, 1, 1)
    ky = grid.ky

    # (1) provisional surface values by linear extrapolation of the top two centres
    su = (u[..., -1] - u[..., -2]) / grid.dzf[-2]
    sv = (v[..., -1] - v[..., -2]) / grid.dzf[-2]
    us0 = u[..., -1] + h_t * su
    vs0 = v[..., -1] + h_t * sv

    # (2) horizontal derivatives, spectral
    S = np.empty((Nx, Ny, 3))
    S[..., 0] = us0
    S[..., 1] = vs0
    S[..., 2] = w_top_prev
    Sh = _fft2(grid, S) * mask[..., None]                     # (Nx, Nyh, 3)
    Dx = 1j * kx * Sh
    Dy = 1j * ky * Sh
    divh = Dx[..., 0] + Dy[..., 1]                            # spectrum of u_x + v_y
    Dh = np.empty(Sh.shape[:2] + (8,), dtype=complex)
    Dh[..., 0] = Dx[..., 0]                                   # us_x
    Dh[..., 1] = Dy[..., 0]                                   # us_y
    Dh[..., 2] = Dx[..., 1]                                   # vs_x
    Dh[..., 3] = Dy[..., 1]                                   # vs_y
    Dh[..., 4] = -1j * kx[..., 0] * divh                      # w_zx|s
    Dh[..., 5] = -1j * ky[..., 0] * divh                      # w_zy|s
    Dh[..., 6] = Dx[..., 2]                                   # dw_s/dx (lagged w_s)
    Dh[..., 7] = Dy[..., 2]                                   # dw_s/dy
    D = _ifft2(grid, Dh)
    us_x, us_y, vs_x, vs_y, wzx, wzy, ws_x, ws_y = (D[..., i] for i in range(8))

    # (3) second vertical derivatives at the surface
    g_u = None if g_prev is None else g_prev.get("dudz_s")
    g_v = None if g_prev is None else g_prev.get("dvdz_s")
    uzz = _uzz(u, grid, uzz_method, g_u, eta)
    vzz = _uzz(v, grid, uzz_method, g_v, eta)

    # (4) O(alpha) corrections of (A7), (A8), dealiased products
    C = np.empty((Nx, Ny, 2))
    C[..., 0] = (2.0 * eta_x * (2.0 * us_x + vs_y) + eta_y * (us_y + vs_x)
                 - eta * (wzx + uzz))
    C[..., 1] = (2.0 * eta_y * (2.0 * vs_y + us_x) + eta_x * (vs_x + us_y)
                 - eta * (wzy + vzz))
    Ch = _fft2(grid, C) * mask[..., None]
    C = _ifft2(grid, Ch)

    irr = np.asarray(wf.irrot_stress_x, dtype=float)
    dudz = (tau_x - irr - ws_x) + C[..., 0]
    dvdz = (tau_y - ws_y) + C[..., 1]
    dudz = np.ascontiguousarray(np.broadcast_to(dudz, (Nx, Ny)))
    dvdz = np.ascontiguousarray(np.broadcast_to(dvdz, (Nx, Ny)))

    # (5) surface values with the new gradient
    u_s = u[..., -1] + h_t * dudz
    v_s = v[..., -1] + h_t * dvdz
    # (6) kinematic condition (A6)
    w_s = surface_w(u_s, v_s, eta, grid)
    return {"u_s": u_s, "v_s": v_s, "dudz_s": dudz, "dvdz_s": dvdz, "w_s": w_s,
            "uzz_s": uzz, "vzz_s": vzz}
