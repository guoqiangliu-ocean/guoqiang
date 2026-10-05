"""LES solver for the rotational velocity u (paper eq. 2.5-2.6), SPEC.md section 4.

Discretisation
--------------
* x, y: Fourier pseudo-spectral, 2/3-rule dealiasing; every prognostic spectral
  field (uh, vh, wh, ph) is kept truncated.
* z: staggered second-order finite differences (u, v, p at centres, w at faces).
* time: AB2 (Euler on the first step and after ``set_state``/``load`` without
  history) for the explicit terms, Crank-Nicolson for the implicit vertical
  operator ``A q = nu q_zz - [implicit_wave_vadv] wphi q_z``, pressure-correction
  projection (Kim & Moin 1985) onto the discretely divergence-free space.

Implementation notes (see also the module docstrings of grid.py / bc.py)
-------------------------------------------------------------------------
* AB2 supports a variable step: ``E_AB = (1 + r/2) E^n - (r/2) E^{n-1}``,
  ``r = dt / dt_prev``.
* The implicit vertical systems are solved column-wise in physical space by a
  vectorised Thomas algorithm.  Because the wave field is independent of y the
  tridiagonal matrices depend only on (x, z): they are factorised on (Nx, 1, Nz)
  and the sweeps are vectorised over (Nx, Ny).
* Surface Neumann data of u, v in CN: A^n uses ``dudz_s^n`` (the stored bc of the
  current state) and A^{n+1} uses ``dudz_s^{n+1}`` from ``surface_bc`` (lagged in its
  O(alpha) corrections).  SPEC.md 4.1.3 uses ``dudz_s^{n+1}`` in both, which makes the
  boundary forcing first order in time; with ``self.cn_flux_both_new = True`` the
  SPEC variant is used.
* Pressure: for every (kx, ky) a tridiagonal Neumann problem; the (0,0) mode is
  integrated directly (phi[0] = 0, then dzc-weighted mean removed) and its w set to 0.
* After the step ``bc['u_s'], bc['v_s']`` are refreshed with the new u, v and the
  new surface gradient (``w_s`` stays the Dirichlet value used in the projection).
* ``bc_extrapolate`` (default False = SPEC.md 4.1.1, lagged): ``bc_np1`` is computed
  from u^n, v^n, w_s^n, which makes the scheme first order in time whenever eta != 0
  (the rotational surface velocity is modulated at the wave frequency).  With
  ``bc_extrapolate=True`` the inputs of ``surface_bc`` are linearly extrapolated to
  t^{n+1} (``q^n + r (q^n - q^{n-1})``), which restores second order in time
  (tests/test_les.py::test_time_order_with_waves).
"""
from __future__ import annotations

import numpy as np

from .bc import surface_bc
from .grid import Grid
from .params import Params
from .wavefields import WaveFields, zero_wave_fields

__all__ = ["LESSolver", "thomas_factor", "thomas_solve"]


# ---------------------------------------------------------------------------
# vectorised Thomas algorithm (tridiagonal systems along the last axis)
def thomas_factor(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Factorise tridiagonal systems  a[k] x[k-1] + b[k] x[k] + c[k] x[k+1] = d[k]
    (a[..., 0] and c[..., -1] are ignored).  Returns (a, cp, inv) for thomas_solve."""
    a_in = np.asarray(a)
    a, b, c = np.broadcast_arrays(a, b, c)
    n = b.shape[-1]
    cp = np.empty(b.shape, dtype=np.result_type(b, c))
    inv = np.empty_like(cp)
    inv[..., 0] = 1.0 / b[..., 0]
    cp[..., 0] = c[..., 0] * inv[..., 0]
    for k in range(1, n):
        inv[..., k] = 1.0 / (b[..., k] - a[..., k] * cp[..., k - 1])
        cp[..., k] = c[..., k] * inv[..., k]
    return a_in, cp, inv


def thomas_solve(fac, d: np.ndarray) -> np.ndarray:
    """Solve in place (d is overwritten with the solution and returned).
    ``fac`` arrays broadcast against d along the leading axes."""
    a, cp, inv = fac
    n = d.shape[-1]
    d[..., 0] *= inv[..., 0]
    for k in range(1, n):
        d[..., k] -= a[..., k] * d[..., k - 1]
        d[..., k] *= inv[..., k]
    for k in range(n - 2, -1, -1):
        d[..., k] -= cp[..., k] * d[..., k + 1]
    return d


def _is_zero_wave(wf: WaveFields) -> bool:
    return not (np.any(wf.uphi_c) or np.any(wf.wphi_c) or np.any(wf.uphi_f)
                or np.any(wf.wphi_f))


class LESSolver:
    """Rotational-velocity LES (SPEC.md section 4)."""

    def __init__(self, p: Params, grid: Grid, sgs=None, surface_stress_on: bool = True,
                 mean_pgrad_on: bool = True, uzz_method: str = "wave_scale",
                 bc_extrapolate: bool = False):
        self.params = p
        self.grid = grid
        self.sgs = sgs
        self.surface_stress_on = surface_stress_on
        self.mean_pgrad_on = mean_pgrad_on
        self.uzz_method = uzz_method
        self.bc_extrapolate = bc_extrapolate
        self.cn_flux_both_new = False      # True -> SPEC 4.1.3 variant (dudz_s^{n+1} in A^n too)
        self.nu = p.nu
        g = grid
        Nx, Ny, Nz, Nyh = g.Nx, g.Ny, g.Nz, g.Nyh
        self.ikx = 1j * g.kx
        self.iky = 1j * g.ky
        self.mask = g.dealias
        self.nk2 = -self.nu * g.k2                         # horizontal viscous symbol

        # state
        self.uh = np.zeros((Nx, Nyh, Nz), complex)
        self.vh = np.zeros((Nx, Nyh, Nz), complex)
        self.wh = np.zeros((Nx, Nyh, Nz + 1), complex)
        self.ph = np.zeros((Nx, Nyh, Nz), complex)
        self.u = np.zeros((Nx, Ny, Nz))
        self.v = np.zeros((Nx, Ny, Nz))
        self.w = np.zeros((Nx, Ny, Nz + 1))
        self.p = np.zeros((Nx, Ny, Nz))
        self.t = 0.0
        self.nstep = 0
        self.nut = None
        self.bc = None
        self._bc_wave_t = None          # wave time used for self.bc (None: zero waves)
        self._hist = None               # (Exh, Eyh, Ezh) of the previous step
        self._dt_prev = None
        self._prev_surf = None          # (u, v, w_top) of the previous step (bc_extrapolate)
        self._fac_cache = {}

        self._init_poisson()
        self._bc_zero_wave_state()

    # ------------------------------------------------------------------
    def _tau_x(self) -> float:
        return self.params.Re_tau if self.surface_stress_on else 0.0

    def _bc_zero_wave_state(self):
        wf0 = zero_wave_fields(self.grid, self.t)
        self.bc = surface_bc(self.u, self.v, self.w[..., -1], self.grid, wf0, self.params,
                             tau_x=self._tau_x(), uzz_method=self.uzz_method)
        self._bc_wave_t = None

    def _init_poisson(self):
        g = self.grid
        Nz = g.Nz
        lo = np.zeros(Nz)
        up = np.zeros(Nz)
        lo[1:] = 1.0 / (g.dzf[1:-1] * g.dzc[1:])
        up[:-1] = 1.0 / (g.dzf[1:-1] * g.dzc[:-1])
        k2 = np.array(g.k2, dtype=float).copy()          # (Nx, Nyh, 1)
        k2[0, 0, 0] = 1.0                                # dummy, (0,0) solved separately
        b = -(lo + up) - k2                              # (Nx, Nyh, Nz)
        self._pfac = thomas_factor(lo, b, up)

    # ------------------------------------------------------------------
    # spectral helpers
    def _fft(self, a):
        h = self.grid.fft(a)
        h *= self.mask
        return h

    def _ifft(self, ah):
        return self.grid.ifft(ah)

    def _refresh_physical(self):
        g = self.grid
        Nz = g.Nz
        # one batched inverse transform for u, v, w, p
        A = np.concatenate([self.uh, self.vh, self.wh, self.ph], axis=-1)
        R = g.ifft(A)
        self.u = np.ascontiguousarray(R[..., :Nz])
        self.v = np.ascontiguousarray(R[..., Nz:2 * Nz])
        self.w = np.ascontiguousarray(R[..., 2 * Nz:3 * Nz + 1])
        self.p = np.ascontiguousarray(R[..., 3 * Nz + 1:])

    # ------------------------------------------------------------------
    def _project(self, uh, vh, wh, dt):
        """Pressure-correction projection in place; returns phi (spectral)."""
        g = self.grid
        dzc, dzf = g.dzc, g.dzf
        div = self.ikx * uh + self.iky * vh + (wh[..., 1:] - wh[..., :-1]) / dzc
        rhs = div / dt
        phi = thomas_solve(self._pfac, rhs)
        # (0,0) mode: horizontal-mean w must vanish identically
        W = wh[0, 0].copy()
        W[0] = 0.0
        W[-1] = 0.0
        Phi = np.zeros(g.Nz, complex)
        Phi[1:] = np.cumsum(W[1:-1] * dzf[1:-1] / dt)
        Phi -= np.sum(Phi * dzc) / np.sum(dzc)
        phi[0, 0] = Phi
        phi *= self.mask
        uh -= dt * self.ikx * phi
        vh -= dt * self.iky * phi
        wh[..., 1:-1] -= dt * (phi[..., 1:] - phi[..., :-1]) / dzf[1:-1]
        wh[0, 0] = 0.0
        return phi

    # ------------------------------------------------------------------
    def set_state(self, u, v, w, t: float = 0.0, wf: WaveFields | None = None) -> None:
        """Set the velocity (physical), dealias, enforce w = 0 at the bottom and
        w = w_s (eq. A6 with ``wf.eta``; 0 without waves) at the surface, project to a
        discretely divergence-free field, compute ``bc`` and reset the AB2 history."""
        g = self.grid
        self.t = float(t)
        uh = self._fft(np.asarray(u, dtype=float))
        vh = self._fft(np.asarray(v, dtype=float))
        wh = self._fft(np.asarray(w, dtype=float))
        wh[..., 0] = 0.0
        u_p, v_p = self._ifft(uh), self._ifft(vh)
        w_top = self._ifft(wh[..., -1:])[..., 0]
        wf_use = wf if wf is not None else zero_wave_fields(g, t)
        bc = surface_bc(u_p, v_p, w_top, g, wf_use, self.params, tau_x=self._tau_x(),
                        uzz_method=self.uzz_method)
        wh[..., -1] = self._fft(bc["w_s"][..., None])[..., 0]
        self._project(uh, vh, wh, 1.0)
        self.uh, self.vh, self.wh = uh, vh, wh
        self.ph = np.zeros_like(uh)
        self._refresh_physical()
        # surface data of the projected state; dw_s/dx from the surface w actually set
        bc = surface_bc(self.u, self.v, self.w[..., -1], g, wf_use, self.params,
                        tau_x=self._tau_x(), uzz_method=self.uzz_method)
        bc["w_s"] = self.w[..., -1].copy()
        self.bc = bc
        self._bc_wave_t = None if wf is None else float(wf.t)
        self._hist = None
        self._dt_prev = None
        self._prev_surf = None
        self.nut = None

    # ------------------------------------------------------------------
    def _explicit(self, bc, wf: WaveFields, wave_active: bool):
        """Explicit right-hand sides E^n (spectral, truncated); E_z at boundary faces = 0."""
        g = self.grid
        p = self.params
        Nz = g.Nz
        u, v, w = self.u, self.v, self.w
        uf = g.c2f(u, bottom=u[..., 0], top=bc["u_s"])
        vf = g.c2f(v, bottom=v[..., 0], top=bc["v_s"])
        wc = g.f2c(w)

        # physical parts
        ex = -g.ddz_f2c(w * uf)
        ey = -g.ddz_f2c(w * vf)
        ez = np.zeros_like(w)
        wc2 = wc * wc
        ez[..., 1:-1] = -(wc2[..., 1:] - wc2[..., :-1]) / g.dzf[1:-1]

        if wave_active:
            D = self._ifft(np.concatenate([self.ikx * self.uh, self.ikx * self.vh,
                                           self.ikx * self.wh], axis=-1))
            dudx = D[..., :Nz]
            dvdx = D[..., Nz:2 * Nz]
            dwdx = D[..., 2 * Nz:]
            ex -= (wf.uphi_c * dudx + u * wf.duphi_dx_c + wc * wf.duphi_dz_c)
            ey -= wf.uphi_c * dvdx
            ez -= (wf.uphi_f * dwdx + uf * wf.dwphi_dx_f + w * wf.dwphi_dz_f)
            if not p.implicit_wave_vadv:
                ex -= wf.wphi_c * g.ddz_c(u, 0.0, bc["dudz_s"])
                ey -= wf.wphi_c * g.ddz_c(v, 0.0, bc["dvdz_s"])
                ez -= wf.wphi_f * g.ddz_f(w)

        if self.sgs is not None:
            r = self.sgs.compute(u, v, w, bc)
            ex -= r["div_x"]
            ey -= r["div_y"]
            ez[..., 1:-1] -= r["div_z"][..., 1:-1]
            self.nut = r.get("nut")

        # forward transforms: products for horizontal fluxes + physical parts
        Pc = np.concatenate([u * u, u * v, v * v, ex, ey], axis=-1)
        Pf = np.concatenate([uf * w, vf * w, ez], axis=-1)
        Pch = g.fft(Pc)
        Pfh = g.fft(Pf)
        uuh, uvh, vvh = Pch[..., :Nz], Pch[..., Nz:2 * Nz], Pch[..., 2 * Nz:3 * Nz]
        Exh = Pch[..., 3 * Nz:4 * Nz]
        Eyh = Pch[..., 4 * Nz:]
        n1 = Nz + 1
        uwh, vwh, Ezh = Pfh[..., :n1], Pfh[..., n1:2 * n1], Pfh[..., 2 * n1:]

        Exh = Exh - self.ikx * uuh - self.iky * uvh + self.nk2 * self.uh
        Eyh = Eyh - self.ikx * uvh - self.iky * vvh + self.nk2 * self.vh
        Ezh = Ezh - self.ikx * uwh - self.iky * vwh + self.nk2 * self.wh
        if self.mean_pgrad_on:
            Exh[0, 0, :] += -1.0 * g.Nx * g.Ny          # f = (-1, 0, 0)
        Exh *= self.mask
        Eyh *= self.mask
        Ezh *= self.mask
        Ezh[..., 0] = 0.0
        Ezh[..., -1] = 0.0
        return Exh, Eyh, Ezh

    # ------------------------------------------------------------------
    # implicit vertical operators
    def _centre_coeffs(self, wphi_c):
        """A q = lo q[k-1] + di q[k] + up q[k+1] (+ top * g at k = Nz-1)."""
        g = self.grid
        nu = self.nu
        Nz = g.Nz
        al = nu / g.dzc                                   # (Nz,)
        hw = 0.0 if wphi_c is None else 0.5 * wphi_c      # (Nx,1,Nz) or scalar
        shape = np.broadcast_shapes(np.shape(hw), (Nz,))
        lo = np.zeros(shape)
        up = np.zeros(shape)
        hwk = np.broadcast_to(hw, shape)
        lo[..., 1:] = (al[1:] + hwk[..., 1:]) / g.dzf[1:-1]
        up[..., :-1] = (al[:-1] - hwk[..., :-1]) / g.dzf[1:-1]
        di = -(lo + up)
        top = al[-1] - hwk[..., -1]                       # coefficient of the surface gradient
        return lo, di, up, top

    def _face_coeffs(self, wphi_f):
        """Operator at interior faces k = 1..Nz-1 (arrays of length Nz-1)."""
        g = self.grid
        nu = self.nu
        Nz = g.Nz
        be = nu / g.dzf[1:-1]
        h2 = g.zf[2:] - g.zf[:-2]
        wk = 0.0 if wphi_f is None else wphi_f[..., 1:-1] / h2
        shape = np.broadcast_shapes(np.shape(wk), (Nz - 1,))
        wk = np.broadcast_to(wk, shape)
        up = be / g.dzc[1:] - wk
        lo = be / g.dzc[:-1] + wk
        di = np.broadcast_to(-be * (1.0 / g.dzc[1:] + 1.0 / g.dzc[:-1]), shape)
        return lo, di, up

    @staticmethod
    def _apply3(lo, di, up, q):
        """lo q[k-1] + di q[k] + up q[k+1] along the last axis (q fully given; the
        boundary rows use lo[0] = up[-1] = 0 for centre operators)."""
        out = di * q
        out[..., 1:] += lo[..., 1:] * q[..., :-1]
        out[..., :-1] += up[..., :-1] * q[..., 1:]
        return out

    def _factor(self, lo, di, up, dt, key=None):
        if key is not None and key in self._fac_cache:
            return self._fac_cache[key]
        h = 0.5 * dt
        fac = thomas_factor(-h * lo, 1.0 - h * di, -h * up)
        if key is not None:
            if len(self._fac_cache) > 8:
                self._fac_cache.clear()
            self._fac_cache[key] = fac
        return fac

    # ------------------------------------------------------------------
    def step(self, dt: float, wf_n: WaveFields, wf_np1: WaveFields) -> None:
        g = self.grid
        p = self.params
        Nz = g.Nz
        dt = float(dt)
        h = 0.5 * dt
        tau_x = self._tau_x()

        wave_n = not _is_zero_wave(wf_n)
        wave_np1 = not _is_zero_wave(wf_np1)
        wave_active = wave_n or wave_np1

        # bc of the current state must correspond to wf_n
        bc_n = self.bc
        stale = (self._bc_wave_t is None and wave_n) or (
            self._bc_wave_t is not None and abs(self._bc_wave_t - wf_n.t) > 1e-14 * max(1.0, abs(wf_n.t)))
        if stale:
            bc_n = surface_bc(self.u, self.v, self.w[..., -1], g, wf_n, p, g_prev=self.bc,
                              tau_x=tau_x, uzz_method=self.uzz_method)
        # 1. surface bc at n+1 (lagged corrections, or extrapolated inputs)
        ub, vb, wtb = self.u, self.v, self.w[..., -1]
        if self.bc_extrapolate and self._prev_surf is not None and self._dt_prev:
            r = dt / self._dt_prev
            u0, v0, wt0 = self._prev_surf
            ub = self.u + r * (self.u - u0)
            vb = self.v + r * (self.v - v0)
            wtb = wtb + r * (wtb - wt0)
        bc_np1 = surface_bc(ub, vb, wtb, g, wf_np1, p, g_prev=bc_n,
                            tau_x=tau_x, uzz_method=self.uzz_method)
        prev_surf = (self.u, self.v, self.w[..., -1].copy()) if self.bc_extrapolate else None

        # 2. explicit terms
        Exh, Eyh, Ezh = self._explicit(bc_n, wf_n, wave_active)
        if self._hist is None:
            Ax, Ay, Az = Exh, Eyh, Ezh
        else:
            r = dt / self._dt_prev
            c1, c0 = 1.0 + 0.5 * r, -0.5 * r
            Ox, Oy, Oz = self._hist
            Ax = c1 * Exh + c0 * Ox
            Ay = c1 * Eyh + c0 * Oy
            Az = c1 * Ezh + c0 * Oz
        Ephys = self._ifft(np.concatenate([Ax, Ay, Az], axis=-1))
        Eu = Ephys[..., :Nz]
        Ev = Ephys[..., Nz:2 * Nz]
        Ew = Ephys[..., 2 * Nz:]

        # 3. implicit vertical operator (CN), physical space
        impl_w = p.implicit_wave_vadv and wave_active
        wc_n = wf_n.wphi_c if impl_w else None
        wc_1 = wf_np1.wphi_c if impl_w else None
        wfn = wf_n.wphi_f if impl_w else None
        wf1 = wf_np1.wphi_f if impl_w else None

        lo0, di0, up0, top0 = self._centre_coeffs(wc_n)
        if impl_w:
            lo1, di1, up1, top1 = self._centre_coeffs(wc_1)
            fac_c = self._factor(lo1, di1, up1, dt)
        else:
            lo1, di1, up1, top1 = lo0, di0, up0, top0
            fac_c = self._factor(lo1, di1, up1, dt, key=("c", dt))

        gu1, gv1 = bc_np1["dudz_s"], bc_np1["dvdz_s"]
        if self.cn_flux_both_new:
            gu0, gv0 = gu1, gv1
        else:
            gu0, gv0 = bc_n["dudz_s"], bc_n["dvdz_s"]

        rhs = np.empty((2,) + self.u.shape)
        for i, (q, E, g0, g1) in enumerate(((self.u, Eu, gu0, gu1), (self.v, Ev, gv0, gv1))):
            r_ = q + dt * E + h * self._apply3(lo0, di0, up0, q)
            r_[..., -1] += h * (top0 * g0 + top1 * g1)
            rhs[i] = r_
        thomas_solve(fac_c, rhs)
        us, vs = rhs[0], rhs[1]

        flo0, fdi0, fup0 = self._face_coeffs(wfn)
        if impl_w:
            flo1, fdi1, fup1 = self._face_coeffs(wf1)
            fac_f = self._factor(flo1, fdi1, fup1, dt)
        else:
            flo1, fdi1, fup1 = flo0, fdi0, fup0
            fac_f = self._factor(flo1, fdi1, fup1, dt, key=("f", dt))
        w = self.w
        ws_new = bc_np1["w_s"]
        Aw = fdi0 * w[..., 1:-1] + flo0 * w[..., :-2] + fup0 * w[..., 2:]
        rw = w[..., 1:-1] + dt * Ew[..., 1:-1] + h * Aw
        rw[..., -1] += h * fup1[..., -1] * ws_new
        thomas_solve(fac_f, rw)
        wstar = np.empty_like(w)
        wstar[..., 0] = 0.0
        wstar[..., 1:-1] = rw
        wstar[..., -1] = ws_new

        # 4. to spectral, truncate, project
        Sh = self._fft(np.concatenate([us, vs, wstar], axis=-1))
        uh = Sh[..., :Nz]
        vh = Sh[..., Nz:2 * Nz]
        wh = Sh[..., 2 * Nz:]
        wh[..., 0] = 0.0
        wh[..., -1] = self._fft(ws_new[..., None])[..., 0]
        wh[0, 0, -1] = 0.0
        phi = self._project(uh, vh, wh, dt)
        self.uh = np.ascontiguousarray(uh)
        self.vh = np.ascontiguousarray(vh)
        self.wh = np.ascontiguousarray(wh)
        self.ph = phi

        # 5. physical fields, history, time
        self._refresh_physical()
        self._hist = (Exh, Eyh, Ezh)
        self._dt_prev = dt
        self._prev_surf = prev_surf
        self.t += dt
        self.nstep += 1
        h_t = g.dzf[-1]
        bc_np1["u_s"] = self.u[..., -1] + h_t * bc_np1["dudz_s"]
        bc_np1["v_s"] = self.v[..., -1] + h_t * bc_np1["dvdz_s"]
        self.bc = bc_np1
        self._bc_wave_t = float(wf_np1.t) if wave_np1 else None

    # ------------------------------------------------------------------
    def divergence(self) -> np.ndarray:
        """Discrete divergence at the centres (physical)."""
        g = self.grid
        d = self.ikx * self.uh + self.iky * self.vh + (self.wh[..., 1:] - self.wh[..., :-1]) / g.dzc
        return self._ifft(d)

    def cfl(self, dt: float, wf: WaveFields | None = None) -> dict:
        """Courant numbers max|u| dt/dx, max|v| dt/dy, max|w| dt/dz (rotational and,
        if ``wf`` is given, total velocity u + u_phi) and the vertical viscous number."""
        g = self.grid
        dzw = np.minimum(np.concatenate([[g.dzc[0]], g.dzc]), np.concatenate([g.dzc, [g.dzc[-1]]]))
        out = {
            "x": float(np.abs(self.u).max() * dt / g.dx),
            "y": float(np.abs(self.v).max() * dt / g.dy),
            "z": float((np.abs(self.w) / dzw).max() * dt),
            "visc_z": float(self.nu * dt / g.dzc.min() ** 2),
        }
        if wf is not None:
            out["x_total"] = float(np.abs(self.u + wf.uphi_c).max() * dt / g.dx)
            out["z_total"] = float((np.abs(self.w + wf.wphi_f) / dzw).max() * dt)
            out["x_wave"] = float(np.abs(wf.uphi_c).max() * dt / g.dx)
            out["z_wave"] = float((np.abs(wf.wphi_f) / dzw).max() * dt)
        out["max"] = max(v for k, v in out.items() if k != "visc_z")
        return out

    # ------------------------------------------------------------------
    def save(self, path) -> None:
        d = dict(u=self.u, v=self.v, w=self.w, p=self.p, t=self.t, nstep=self.nstep,
                 have_hist=self._hist is not None,
                 dt_prev=-1.0 if self._dt_prev is None else self._dt_prev,
                 bc_wave_t=np.nan if self._bc_wave_t is None else self._bc_wave_t,
                 Nx=self.grid.Nx, Ny=self.grid.Ny, Nz=self.grid.Nz)
        if self._hist is not None:
            d["Exh"], d["Eyh"], d["Ezh"] = self._hist
        if self._prev_surf is not None:
            d["prev_u"], d["prev_v"], d["prev_wtop"] = self._prev_surf
        for k, v in self.bc.items():
            d["bc_" + k] = v
        np.savez(path, **d)

    def load(self, path) -> None:
        z = np.load(path)
        g = self.grid
        if (int(z["Nx"]), int(z["Ny"]), int(z["Nz"])) != (g.Nx, g.Ny, g.Nz):
            raise ValueError("grid mismatch in " + str(path))
        self.uh = self._fft(z["u"])
        self.vh = self._fft(z["v"])
        self.wh = self._fft(z["w"])
        self.ph = self._fft(z["p"])
        self.wh[..., 0] = 0.0
        self._refresh_physical()
        self.t = float(z["t"])
        self.nstep = int(z["nstep"])
        self.bc = {k[3:]: np.array(z[k]) for k in z.files if k.startswith("bc_") and k != "bc_wave_t"}
        bwt = float(z["bc_wave_t"])
        self._bc_wave_t = None if np.isnan(bwt) else bwt
        if bool(z["have_hist"]):
            self._hist = (np.array(z["Exh"]), np.array(z["Eyh"]), np.array(z["Ezh"]))
            self._dt_prev = float(z["dt_prev"])
        else:
            self._hist = None
            self._dt_prev = None
        if "prev_u" in z.files:
            self._prev_surf = (np.array(z["prev_u"]), np.array(z["prev_v"]), np.array(z["prev_wtop"]))
        else:
            self._prev_surf = None
        self.nut = None
