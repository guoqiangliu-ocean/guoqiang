"""Subgrid-scale (SGS) models for the rotational velocity (SPEC.md section 5).

Two eddy-viscosity models are provided:

* :class:`DynamicSmagorinsky` -- Smagorinsky model whose coefficient
  ``Cs^2(z)`` is computed by the dynamic procedure of Germano et al. (1991)
  with the least-squares contraction of Lilly (1992), averaged over horizontal
  planes and clipped at zero;
* :class:`Smagorinsky` -- constant ``Cs = p.cs_const`` with van Driest damping
  ``Cs(z) = cs_const * (1 - exp(-d+/25))``, ``d+ = d u_*/nu`` with ``d = -z``
  the distance from the mean surface and the nominal ``u_* = 1``.

Common interface::

    model = DynamicSmagorinsky(grid, p)
    out = model.compute(u, v, w, bc)
    out['div_x'], out['div_y']   (Nx, Ny, Nz)    d_j tau_ij at centres
    out['div_z']                 (Nx, Ny, Nz+1)  d_j tau_3j at faces (0 at k = 0, Nz)
    out['nut']                   (Nx, Ny, Nz)    eddy viscosity at centres
    out['cs2']                   (Nz,)           coefficient Cs^2(z)
    # extra keys (free by-products, may be used by the solver / statistics):
    out['div_x_h'], ['div_y_h'], ['div_z_h']     same, spectral and dealiased
    out['tau13_f'], out['tau23_f'] (Nx, Ny, Nz+1) face shear stresses used in
                                                  the divergence (0 at k = 0, Nz)

The solver must *add* ``-div`` to the right-hand side of the momentum
equations (``-d_j tau_ij`` in eq. 2.5).

Options (attributes): ``truncate_input`` (default True; set False when u, v, w
are already dealiased solver fields -> saves three inverse FFTs),
``update_interval`` (default 1; recompute the dynamic Cs^2(z) only every n-th
call and reuse it in between).  ``make_sgs(grid, p)`` builds the model named
by ``p.sgs``.

Discretisation
--------------
``tau_ij = -2 nu_t S_ij`` is built from the *rotational* velocity only (the wave
velocity is irrotational, potential and fully resolved by HOS; its strain
does not belong to the unresolved turbulence and would otherwise generate a
spurious, wave-phase-locked eddy viscosity of order Cs^2 Delta^2 a omega k).
The trace of ``S_ij`` is the discrete divergence of the solver and therefore
vanishes to round-off, so no explicit trace removal is applied to ``tau``.

* inputs are first truncated with the 2/3 mask (no-op for solver fields);
* horizontal derivatives are spectral; ``du/dz, dv/dz`` at centres by
  ``grid.ddz_c`` with the Neumann data ``bc['dudz_s'], bc['dvdz_s']`` at the
  surface and 0 at the bottom; ``dw/dz`` by ``grid.ddz_f2c``; ``dw/dx, dw/dy``
  at centres by ``grid.f2c`` of the spectral face derivatives;
* ``Delta = (dx dy dzc)^(1/3)`` per level;
* every product (stress components, test-filtered products) is formed in
  physical space, transformed and truncated (2/3 rule);
* divergence in staggered form:
  ``div_x = d_x tau_11 + d_y tau_12 + (tau_13f[k+1] - tau_13f[k]) / dzc[k]``,
  ``div_y`` likewise, ``div_z[k] = d_x tau_13f + d_y tau_23f +
  (tau_33[k] - tau_33[k-1]) / dzf[k]`` at interior faces;
  ``tau_13f = -nu_t,f ((du/dz)_f + (dw/dx)_f)`` with ``nu_t,f`` linearly
  interpolated to the faces; ``tau_13f = tau_23f = 0`` at the bottom and
  surface faces (the imposed surface stress is carried by the resolved
  viscous flux), hence the SGS term conserves the horizontal-mean momentum
  exactly.

Dynamic procedure
-----------------
Test filter: sharp spectral cut-off at half of the resolved (dealiased)
horizontal wavenumbers, i.e. keep ``|mx| < Nx/6`` and ``my < Ny/6``;
``Delta_hat / Delta = 2^(2/3)`` (filter width doubled in x and y only).
``L_ij = hat(u_i u_j) - hat(u_i) hat(u_j)`` (with w interpolated to centres),
``M_ij = 2 Delta^2 [hat(|S| S_ij) - (Delta_hat/Delta)^2 |hat S| hat(S_ij)]``,
both made deviatoric, ``Cs^2(z) = max(0, <L_ij M_ij>_xy / <M_ij M_ij>_xy)``;
levels whose ``<M_ij M_ij>`` is at round-off level (relative to
``(2 Delta^2 max|S|^2)^2``) get ``Cs^2 = 0``.  ``nu_t = Cs^2 Delta^2 |S|``,
``|S| = sqrt(2 S_ij S_ij)``.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .grid import Grid
from .params import Params

# index pairs of the 6 independent components of a symmetric tensor
_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
_WEIGHTS = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)   # multiplicity in a full contraction


def _plane_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """sum over (x, y) of a*b for (Nx, Ny, Nz) arrays -> (Nz,)."""
    return np.einsum("ijk,ijk->k", a, b, optimize=False)


class DynamicSmagorinsky:
    """Dynamic Smagorinsky model, plane-averaged coefficient (SPEC.md section 5)."""

    def __init__(self, grid: Grid, p: Params):
        self.grid = grid
        self.p = p
        g = grid
        Nx, Ny = g.Nx, g.Ny
        self.delta = (g.dx * g.dy * g.dzc) ** (1.0 / 3.0)      # (Nz,)
        self.delta2 = self.delta ** 2
        self.filter_ratio = 2.0 ** (2.0 / 3.0)                  # Delta_hat / Delta
        self.mask = g.dealias                                  # (Nx, Nyh, 1) bool
        self.mask2d = g.dealias[:, :, 0]
        tm = (np.abs(g.mx) < Nx / 6.0).reshape(Nx, 1, 1) & \
             (g.my < Ny / 6.0).reshape(1, g.Nyh, 1)
        self.test_mask = tm & self.mask
        self.test_mask2d = self.test_mask[:, :, 0]
        self.nyt = int(np.count_nonzero(self.test_mask[0, :, 0]))   # my < Ny/6
        self.tmx = self.test_mask[:, :1, :]                          # (Nx, 1, 1) x part
        self._bufs = {}
        self.ikx = 1j * g.kx                                   # (Nx, 1, 1)
        self.iky = 1j * g.ky                                   # (1, Nyh, 1)
        # relative round-off guard for <M M> (see _coefficient)
        self.guard = 1e-24
        # False: trust that u, v, w are already dealiased (solver fields) and
        # skip three inverse transforms
        self.truncate_input = True
        # recompute the dynamic coefficient every `update_interval` calls
        # (1 = every call, the default); in between the last Cs^2(z) is reused
        self.update_interval = 1
        self.ncalls = 0
        self._cs2 = None
        self.last = None

    # ------------------------------------------------------------------
    # small helpers
    def _fft_t(self, a: np.ndarray) -> np.ndarray:
        ah = self.grid.fft(a)
        ah *= self.mask
        return ah

    def _surface_data(self, val) -> np.ndarray:
        """Broadcast a Neumann datum to (Nx, Ny) and truncate it (2/3 rule)."""
        g = self.grid
        a = np.broadcast_to(np.asarray(val, dtype=float), (g.Nx, g.Ny))
        ah = sfft.rfft2(a, axes=(0, 1), workers=g.workers)
        ah *= self.mask2d
        return ah

    def _irfft2d(self, ah: np.ndarray) -> np.ndarray:
        g = self.grid
        return sfft.irfft2(ah, s=(g.Nx, g.Ny), axes=(0, 1), workers=g.workers)

    # ------------------------------------------------------------------
    def _ddz_c(self, qc: np.ndarray, top: np.ndarray) -> np.ndarray:
        """grid.ddz_c(qc, 0, top) with fewer temporaries."""
        g = self.grid
        df = np.empty(qc.shape[:-1] + (g.Nz + 1,), dtype=qc.dtype)
        np.subtract(qc[..., 1:], qc[..., :-1], out=df[..., 1:-1])
        df[..., 1:-1] /= g.dzf[1:-1]
        df[..., 0] = 0.0
        df[..., -1] = top
        out = df[..., 1:] + df[..., :-1]
        out *= 0.5
        return out

    def _resolved(self, u, v, w, bc) -> dict:
        """Dealiased velocities, strain-rate tensor and |S| at centres."""
        g = self.grid
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        w = np.asarray(w, dtype=float)
        uh = self._fft_t(u)
        vh = self._fft_t(v)
        wh = self._fft_t(w)                      # faces
        dush = self._surface_data(bc.get("dudz_s", 0.0))
        dvsh = self._surface_data(bc.get("dvdz_s", 0.0))
        dus = self._irfft2d(dush)
        dvs = self._irfft2d(dvsh)

        if self.truncate_input:
            ud, vd, wd = g.ifft(uh), g.ifft(vh), g.ifft(wh)
        else:
            ud, vd, wd = u, v, w
        wx_f = g.ifft(self.ikx * wh)             # faces
        wy_f = g.ifft(self.iky * wh)

        S = [None] * 6
        S[0] = g.ifft(self.ikx * uh)
        S[1] = g.ifft(self.iky * vh)
        S[2] = g.ddz_f2c(wd)
        S[3] = g.ifft(0.5 * (self.iky * uh + self.ikx * vh))
        S[4] = self._ddz_c(ud, dus)
        S[4] += g.f2c(wx_f)
        S[4] *= 0.5
        S[5] = self._ddz_c(vd, dvs)
        S[5] += g.f2c(wy_f)
        S[5] *= 0.5
        Smag = self._norm(S)
        return dict(uh=uh, vh=vh, wh=wh, dush=dush, dvsh=dvsh,
                    ud=ud, vd=vd, wd=wd, wx_f=wx_f, wy_f=wy_f, S=S, Smag=Smag)

    @staticmethod
    def _norm(S) -> np.ndarray:
        """|S| = sqrt(2 S_ij S_ij)."""
        acc = S[0] * S[0]
        acc += S[1] * S[1]
        acc += S[2] * S[2]
        off = S[3] * S[3]
        off += S[4] * S[4]
        off += S[5] * S[5]
        acc += 2.0 * off
        acc *= 2.0
        np.sqrt(acc, out=acc)
        return acc

    # ------------------------------------------------------------------
    # test-filter transforms, pruned to the retained band: only the columns
    # my < Ny/6 are transformed in x (identical to rfft2/irfft2 + mask).
    def _tf_fwd(self, a: np.ndarray) -> np.ndarray:
        """Physical (Nx, Ny, nz) -> test-band spectrum (Nx, nyt, nz)."""
        w = self.grid.workers
        ay = sfft.rfft(a, axis=1, workers=w)[:, :self.nyt]
        ah = sfft.fft(ay, axis=0, workers=w, overwrite_x=True)
        ah *= self.tmx
        return ah

    def _tf_inv(self, ah: np.ndarray) -> np.ndarray:
        """Test-band spectrum (Nx, nyt, nz) (x-masked) -> physical (Nx, Ny, nz)."""
        g = self.grid
        nz = ah.shape[-1]
        buf = self._bufs.get(nz)
        if buf is None:
            buf = np.zeros((g.Nx, g.Nyh, nz), dtype=complex)
            self._bufs[nz] = buf
        buf[:, :self.nyt] = sfft.ifft(ah, axis=0, workers=g.workers)
        return sfft.irfft(buf, n=g.Ny, axis=1, workers=g.workers)

    def _tband(self, ah: np.ndarray) -> np.ndarray:
        """Restrict a full (Nx, Nyh, nz) spectrum to the test band."""
        return ah[:, :self.nyt] * self.tmx

    def _test_filter(self, a: np.ndarray) -> np.ndarray:
        return self._tf_inv(self._tf_fwd(a))

    def _coefficient(self, R: dict) -> np.ndarray:
        """Plane-averaged dynamic coefficient Cs^2(z) (Germano/Lilly)."""
        g = self.grid
        nyt = self.nyt
        ikx, iky = self.ikx, self.iky[:, :nyt]
        uh = self._tband(R["uh"])
        vh = self._tband(R["vh"])
        wh = self._tband(R["wh"])               # faces
        whc = g.f2c(wh)                          # w at centres

        # resolved and test-filtered velocities at centres
        # L_ij is invariant under a plane-uniform translation (the test filter
        # preserves plane means), so the plane means are removed before the
        # products: avoids cancellation of O(U^2) terms (mean current)
        vel = []
        for q in (R["ud"], R["vd"], g.f2c(R["wd"])):
            q = q - q.mean(axis=(0, 1))
            vel.append(q)
        velf = []
        for qh in (uh, vh, whc):
            qh = qh.copy()
            qh[0, 0, :] = 0.0
            velf.append(self._tf_inv(qh))

        # test-filtered strain rate: the filter commutes with the vertical FD,
        # so everything is formed on the (small) test-band spectra and only
        # the final components are transformed back
        tmx2 = self.tmx[:, :, 0]
        dushf = R["dush"][:, :nyt] * tmx2
        dvshf = R["dvsh"][:, :nyt] * tmx2
        Sf = [None] * 6
        Sf[0] = self._tf_inv(ikx * uh)
        Sf[1] = self._tf_inv(iky * vh)
        Sf[2] = self._tf_inv(g.ddz_f2c(wh))
        Sf[3] = self._tf_inv(0.5 * (iky * uh + ikx * vh))
        a = self._ddz_c(uh, dushf)
        a += ikx * whc
        a *= 0.5
        Sf[4] = self._tf_inv(a)
        a = self._ddz_c(vh, dvshf)
        a += iky * whc
        a *= 0.5
        Sf[5] = self._tf_inv(a)
        rSfmag = self._norm(Sf)
        rSfmag *= self.filter_ratio ** 2

        S, Smag = R["S"], R["Smag"]
        twod2 = 2.0 * self.delta2                # (Nz,)
        num = np.zeros(g.Nz)
        den = np.zeros(g.Nz)
        trL = np.zeros_like(Smag)
        trM = np.zeros_like(Smag)
        tmp = np.empty_like(Smag)
        for n, (i, j) in enumerate(_PAIRS):
            np.multiply(vel[i], vel[j], out=tmp)
            L = self._test_filter(tmp)
            np.multiply(velf[i], velf[j], out=tmp)
            L -= tmp
            np.multiply(Smag, S[n], out=tmp)
            M = self._test_filter(tmp)
            np.multiply(rSfmag, Sf[n], out=tmp)
            M -= tmp
            wgt = _WEIGHTS[n]
            num += wgt * _plane_dot(L, M)
            den += wgt * _plane_dot(M, M)
            if i == j:
                trL += L
                trM += M
        # deviatoric parts: A^d_ij B^d_ij = A_ij B_ij - trA trB / 3
        num -= _plane_dot(trL, trM) / 3.0
        den -= _plane_dot(trM, trM) / 3.0
        # M_ij = 2 Delta^2 [...]: apply the level-dependent factor here
        num *= twod2
        den *= twod2 ** 2

        # round-off guard: M ~ 2 Delta^2 Sref^2 with Sref = max |S| (a
        # Galilean-invariant strain scale; a uniform translation must not
        # switch the model off)
        Sref = float(Smag.max())
        npts = g.Nx * g.Ny
        thresh = self.guard * npts * (twod2 * Sref ** 2) ** 2
        ok = (den > thresh) & np.isfinite(num) & np.isfinite(den) & (den > 0.0)
        cs2 = np.zeros(g.Nz)
        cs2[ok] = num[ok] / den[ok]
        np.maximum(cs2, 0.0, out=cs2)
        self.last_num, self.last_den = num, den
        return cs2

    # ------------------------------------------------------------------
    def _divergence(self, R: dict, nut: np.ndarray) -> dict:
        """d_j tau_ij with tau_ij = -2 nu_t S_ij in staggered conservative form."""
        g = self.grid
        S = R["S"]
        out = {}
        # centre stresses -> dealiased spectra
        m2nut = -2.0 * nut
        tmp = np.empty_like(m2nut)
        th = []
        for n in (0, 1, 2, 3):                   # tau_11, tau_22, tau_33, tau_12
            np.multiply(m2nut, S[n], out=tmp)
            th.append(self._fft_t(tmp))
        t11h, t22h, t33h, t12h = th
        # face shear stresses tau_13f = -nu_t,f (du/dz + dw/dx)_f, 0 at k = 0, Nz
        mnutf = np.zeros(nut.shape[:-1] + (g.Nz + 1,))
        mnutf[..., 1:-1] = g.wlo[1:-1] * nut[..., :-1]
        mnutf[..., 1:-1] += g.whi[1:-1] * nut[..., 1:]
        mnutf *= -1.0
        taus = []
        for qd, dq_f in ((R["ud"], R["wx_f"]), (R["vd"], R["wy_f"])):
            t = np.empty_like(mnutf)
            np.subtract(qd[..., 1:], qd[..., :-1], out=t[..., 1:-1])
            t[..., 1:-1] /= g.dzf[1:-1]
            t[..., 0] = 0.0
            t[..., -1] = 0.0
            t += dq_f
            t *= mnutf                           # boundary faces: mnutf = 0
            taus.append(t)
        tau13f, tau23f = taus
        t13h = self._fft_t(tau13f)
        t23h = self._fft_t(tau23f)

        dxh = self.ikx * t11h
        dxh += self.iky * t12h
        dxh += g.ddz_f2c(t13h)
        dyh = self.ikx * t12h
        dyh += self.iky * t22h
        dyh += g.ddz_f2c(t23h)
        dzh = self.ikx * t13h
        dzh += self.iky * t23h
        dzh[..., 1:-1] += (t33h[..., 1:] - t33h[..., :-1]) / g.dzf[1:-1]
        dzh[..., 0] = 0.0
        dzh[..., -1] = 0.0
        out["div_x_h"], out["div_y_h"], out["div_z_h"] = dxh, dyh, dzh
        out["div_x"] = g.ifft(dxh)
        out["div_y"] = g.ifft(dyh)
        out["div_z"] = g.ifft(dzh)
        out["tau13_f"], out["tau23_f"] = tau13f, tau23f
        return out

    # ------------------------------------------------------------------
    def eddy_viscosity(self, R: dict, cs2: np.ndarray) -> np.ndarray:
        return (cs2 * self.delta2) * R["Smag"]

    def compute(self, u, v, w, bc: dict) -> dict:
        """SGS stress divergence for the rotational velocity (u, v at centres,
        w at faces) with surface Neumann data ``bc['dudz_s'], bc['dvdz_s']``."""
        R = self._resolved(u, v, w, bc if bc is not None else {})
        if self._cs2 is None or self.ncalls % max(1, int(self.update_interval)) == 0:
            self._cs2 = self._coefficient(R)
        self.ncalls += 1
        cs2 = self._cs2.copy()
        nut = self.eddy_viscosity(R, cs2)
        out = self._divergence(R, nut)
        out["nut"] = nut
        out["cs2"] = cs2
        self.last = out
        return out

    def divergence_from_nut(self, u, v, w, bc: dict, nut) -> dict:
        """Divergence of ``-2 nut S_ij`` for a prescribed eddy viscosity
        (array broadcastable to (Nx, Ny, Nz)); used for consistency tests."""
        g = self.grid
        R = self._resolved(u, v, w, bc)
        nut = np.broadcast_to(np.asarray(nut, dtype=float), (g.Nx, g.Ny, g.Nz))
        out = self._divergence(R, np.ascontiguousarray(nut))
        out["nut"] = nut
        return out

    def strain_rate(self, u, v, w, bc: dict):
        """Return (S, |S|) with S the list [S11, S22, S33, S12, S13, S23] at centres."""
        R = self._resolved(u, v, w, bc)
        return R["S"], R["Smag"]

    def stress_tensor(self, u, v, w, bc: dict) -> np.ndarray:
        """Full 3x3 SGS stress at centres, shape (3, 3, Nx, Ny, Nz), built from
        the velocity-gradient tensor as -nu_t (A_ij + A_ji) (diagnostic)."""
        g = self.grid
        R = self._resolved(u, v, w, bc)
        cs2 = self._coefficient(R)
        nut = self.eddy_viscosity(R, cs2)
        uh, vh, wh = R["uh"], R["vh"], R["wh"]
        dus = self._irfft2d(R["dush"])
        dvs = self._irfft2d(R["dvsh"])
        A = np.empty((3, 3) + nut.shape)
        A[0, 0] = g.ifft(self.ikx * uh)
        A[0, 1] = g.ifft(self.iky * uh)
        A[0, 2] = g.ddz_c(R["ud"], 0.0, dus)
        A[1, 0] = g.ifft(self.ikx * vh)
        A[1, 1] = g.ifft(self.iky * vh)
        A[1, 2] = g.ddz_c(R["vd"], 0.0, dvs)
        A[2, 0] = g.f2c(R["wx_f"])
        A[2, 1] = g.f2c(R["wy_f"])
        A[2, 2] = g.ddz_f2c(R["wd"])
        return -nut * (A + A.transpose(1, 0, 2, 3, 4))


class Smagorinsky(DynamicSmagorinsky):
    """Constant-coefficient Smagorinsky model with van Driest damping towards
    the surface: Cs(z) = cs_const (1 - exp(-d+/25)), d+ = (-z) u_*/nu, u_* = 1."""

    A_plus = 25.0

    def __init__(self, grid: Grid, p: Params):
        super().__init__(grid, p)
        dplus = (-grid.zc) / p.nu        # u_* = 1 (nondimensional)
        cs = p.cs_const * (1.0 - np.exp(-dplus / self.A_plus))
        self.cs2_profile = cs ** 2

    def _coefficient(self, R: dict) -> np.ndarray:
        return self.cs2_profile.copy()


def make_sgs(grid: Grid, p: Params):
    """Factory from ``p.sgs`` ('dynamic', 'smagorinsky' or 'none')."""
    kind = (p.sgs or "none").lower()
    if kind == "dynamic":
        return DynamicSmagorinsky(grid, p)
    if kind == "smagorinsky":
        return Smagorinsky(grid, p)
    if kind == "none":
        return None
    raise ValueError(f"unknown sgs model {p.sgs!r}")
