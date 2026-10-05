"""High-order spectral (HOS) solver for 2-D deep-water gravity waves.

Paper: Xuan, Deng & Shen (2024), section 2.1, eqs (2.11), (2.13), (2.17)-(2.19);
method of Dommermuth & Yue (1987) and West et al. (1987).  See SPEC.md section 2.

Unknowns: surface elevation ``eta(x, t)`` and surface potential
``phis(x, t) = phi(x, eta(x, t), t)`` on N equispaced points of the periodic
interval [0, Lx).  Zakharov form of the free-surface conditions:

    eta_t  = -eta_x phis_x + (1 + eta_x^2) W
    phis_t = -g eta - 0.5 phis_x^2 + 0.5 (1 + eta_x^2) W^2

with W = phi_z at z = eta.  The potential is expanded as
phi = sum_{m=1}^M phi^(m), each phi^(m) = sum_k phihat^(m)_k e^{|k| z} e^{ikx}
(deep water), with the Dirichlet data obtained by Taylor expansion about z = 0:

    phi^(1)|_0 = phis,
    phi^(m)|_0 = - sum_{l=1}^{m-1} eta^l / l! * d^l phi^(m-l)/dz^l |_0,
    W          = sum_{m=1}^M sum_{l=0}^{M-m} eta^l / l! * d^{l+1} phi^(m)/dz^{l+1} |_0.

Order-consistent truncation (West et al. 1987)
-----------------------------------------------
W is split by order, W = sum_n W_n with
W_n = sum_{m+l=n} eta^l/l! d^{l+1}phi^(m)/dz^{l+1}.  The evolution equations are
truncated consistently at order M (amplitude^M):

    eta_t  = -eta_x phis_x + sum_{n<=M} W_n + eta_x^2 sum_{n<=M-2} W_n
    phis_t = -g eta - 0.5 phis_x^2 + 0.5 sum_{n1+n2<=M} W_n1 W_n2
             + 0.5 eta_x^2 sum_{n1+n2<=M-2} W_n1 W_n2

(for M = 3 the last term vanishes).  This is the standard HOS of order M; it
differs from writing (1 + eta_x^2) W with the full W only by terms of order
>= M + 1.

Dealiasing (standard (M+1)/2 zero-padding rule)
-----------------------------------------------
All spectral fields keep the modes 0 <= m < N/2 (the Nyquist mode is always
zero).  Every nonlinear product is evaluated on a zero-padded grid of
Np >= (M+1) N / 2 points.  With the order-consistent truncation every term
above, and every source term of the phi^(m) recursion, is a product of at most
M band-limited factors (each phi^(m) is truncated back to |m| < N/2 after it
is formed), so a product of p <= M factors has modes up to p N/2 and its
aliases on the padded grid fall at |m| >= Np - p N/2 >= N/2, i.e. outside the
retained band: the right-hand side is *exactly* alias-free (verified in the
tests by comparing with a twice finer grid).  Results are transformed back and
truncated to |m| < N/2.

An optional exponential low-pass filter on eta and phis after every RK4 step
(``filter_frac``, off by default) is available for very long or very steep
runs; it is not used anywhere by default.

Time integration: classical RK4 on the spectral coefficients.  The solver is
stepped with the LES time step (dt ~ 1.4e-5, i.e. omega dt ~ 0.13 for the
carrier and ~0.4 at the HOS Nyquist for the reduced preset, well inside the
RK4 stability limit 2.83 on the imaginary axis).

Spectral convention: ``numpy.fft.rfft`` (unnormalised forward), wavenumbers
``k = 2 pi m / Lx``, m = 0..N/2.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.fft as sfft

from .params import Params


class HOS:
    """Deep-water HOS solver of order M = p.hos_order on N = p.hos_N points.

    Extra keyword arguments (not used by the drivers):

    order : override p.hos_order
    N     : override p.hos_N
    filter_frac : None (default, no filter) or f in (0, 1): after every RK4
        step multiply the spectra of eta and phis by
        sigma(m) = exp(-36 ((m - mc)/(N/2 - mc))^4) for m > mc = f N/2.
    """

    def __init__(self, p: Params, order: int | None = None, N: int | None = None,
                 filter_frac: float | None = None):
        self.p = p
        self.N = int(p.hos_N if N is None else N)
        self.M = int(p.hos_order if order is None else order)
        if self.N % 2 or self.N < 4:
            raise ValueError("HOS: N must be even and >= 4")
        if self.M < 1:
            raise ValueError("HOS: order must be >= 1")
        self.Lx = float(p.Lx)
        self.g = float(p.g)
        N = self.N
        self.nk = N // 2 + 1
        self.x = np.arange(N) * (self.Lx / N)
        self.k = 2.0 * math.pi / self.Lx * np.arange(self.nk)       # >= 0
        # retained band: 0 <= m < N/2 (Nyquist zeroed)
        self.keep = np.ones(self.nk, dtype=bool)
        self.keep[-1] = False
        self._ik = 1j * self.k * self.keep
        self._kk = self.k * self.keep                                # |k| (truncated)
        # zero-padded grid for products: Np >= (M+1) N / 2, even
        Np = max(N, ((self.M + 1) * N + 1) // 2)
        Np = sfft.next_fast_len(Np, real=True)
        if Np % 2:
            Np += 1
        self.Np = Np
        self.nkp = Np // 2 + 1
        # filter
        self.filter_frac = filter_frac
        if filter_frac is not None:
            m = np.arange(self.nk, dtype=float)
            mc = filter_frac * N / 2
            s = np.clip((m - mc) / (N / 2 - mc), 0.0, None)
            self._filter = np.exp(-36.0 * s ** 4) * self.keep
        else:
            self._filter = None

        self.t = 0.0
        self.eta = np.zeros(N)
        self.phis = np.zeros(N)
        # powers |k|^j (j = 0..M) of the truncated wavenumber, rows
        self._kpow = np.vstack([self._kk ** j for j in range(self.M + 1)])
        self._fact = [math.factorial(l) for l in range(self.M + 1)]
        self._modes_cache = None     # (eta_bytes_id, modes)

    # ------------------------------------------------------------------
    # transforms
    def _fwd(self, a: np.ndarray) -> np.ndarray:
        """Physical (N,) -> truncated rfft coefficients (N/2+1,)."""
        ah = sfft.rfft(a)
        ah[~self.keep] = 0.0
        return ah

    def _inv(self, ah: np.ndarray) -> np.ndarray:
        return sfft.irfft(ah, n=self.N)

    def _to_pad(self, ah: np.ndarray) -> np.ndarray:
        """Spectral (..., N/2+1) -> physical values on the padded grid (..., Np)."""
        buf = np.zeros(ah.shape[:-1] + (self.nkp,), dtype=complex)
        buf[..., :self.nk] = ah
        buf *= self.Np / self.N
        return sfft.irfft(buf, n=self.Np, axis=-1)

    def _from_pad(self, a: np.ndarray) -> np.ndarray:
        """Physical on padded grid (..., Np) -> truncated spectral (..., N/2+1)
        in the rfft convention of the N-point grid."""
        ah = sfft.rfft(a, axis=-1)[..., :self.nk]
        ah *= (self.N / self.Np) * self.keep
        return ah

    # ------------------------------------------------------------------
    # initial conditions
    def init_packet(self) -> None:
        """Gaussian wave packet, eqs (2.17)-(2.19); the distance x - x0 is
        wrapped periodically to [-Lx/2, Lx/2)."""
        p = self.p
        k0, om0 = p.k0, p.omega0
        d = np.mod(self.x - p.x0 + 0.5 * self.Lx, self.Lx) - 0.5 * self.Lx
        A0 = p.a0 * np.exp(-d ** 2 / (2.0 * p.chi ** 2))
        self.eta = self._inv(self._fwd(A0 * np.cos(k0 * self.x)))
        self.phis = self._inv(self._fwd(A0 * (om0 / k0) * np.sin(k0 * self.x)))
        self.t = 0.0

    def _check_k(self, k: float) -> float:
        m = k * self.Lx / (2.0 * math.pi)
        if abs(m - round(m)) > 1e-8 * max(1.0, abs(m)) or not (0 < round(m) < self.N // 2):
            raise ValueError(f"wavenumber k={k} is not a resolved periodic mode (m={m})")
        return 2.0 * math.pi * round(m) / self.Lx

    def init_monochromatic(self, a: float, k: float) -> None:
        """Linear progressive wave eta = a cos(kx), phis = (a omega/k) sin(kx),
        omega = sqrt(g k), travelling in +x."""
        k = self._check_k(k)
        om = math.sqrt(self.g * k)
        self.eta = a * np.cos(k * self.x)
        self.phis = a * om / k * np.sin(k * self.x)
        self.t = 0.0

    def init_stokes(self, a: float, k: float) -> float:
        """Third-order deep-water Stokes wave with fundamental amplitude a:
        eta = a [cos th + (ka/2) cos 2th + (3/8)(ka)^2 cos 3th],
        phi = (a omega/k) e^{kz} sin th (exact to third order),
        phis = phi(x, eta(x)), omega = sqrt(g k) (1 + (ka)^2/2).
        Returns omega."""
        k = self._check_k(k)
        eps = k * a
        om = math.sqrt(self.g * k) * (1.0 + 0.5 * eps ** 2)
        th = k * self.x
        eta = a * (np.cos(th) + 0.5 * eps * np.cos(2 * th) + 0.375 * eps ** 2 * np.cos(3 * th))
        self.eta = self._inv(self._fwd(eta))
        self.phis = self._inv(self._fwd(a * om / k * np.exp(k * eta) * np.sin(th)))
        self.t = 0.0
        return om

    # ------------------------------------------------------------------
    # HOS core
    def _modes(self, eh: np.ndarray, ph: np.ndarray):
        """HOS modal expansion.  Returns (phihat list m = 1..M, E (padded eta),
        D dict (m) -> padded rows d^j phi^(m)/dz^j, j = 1..M-m+1)."""
        M = self.M
        kp = self._kpow
        fact = self._fact
        # batch 1: eta, eta_x (unused here) are done by the caller; here phi^(1)
        E = self._to_pad(eh)
        Epow = [None, E]
        for l in range(2, M):
            Epow.append(Epow[-1] * E)
        phih = [None, ph]
        D = {}
        D[1] = self._to_pad(kp[1:M + 1] * ph)                 # j = 1..M
        for m in range(2, M + 1):
            src = Epow[1] * D[m - 1][0]                       # l = 1: eta * d phi^(m-1)/dz
            for l in range(2, m):
                src = src + Epow[l] * (D[m - l][l - 1] / fact[l])
            phm = -self._from_pad(src)
            phih.append(phm)
            D[m] = self._to_pad(kp[1:M - m + 2] * phm)         # j = 1..M-m+1
        return phih, E, Epow, D

    def _rhs_hat(self, eh: np.ndarray, ph: np.ndarray):
        """Spectral right-hand side (deta_dt_hat, dphis_dt_hat)."""
        M = self.M
        g = self.g
        ik = self._ik
        # linear parts (exact in spectral space)
        deh = self._kk * ph
        dph = -g * eh
        if M == 1:
            return deh, dph * self.keep
        phih, E, Epow, D = self._modes(eh, ph)
        fact = self._fact
        ExPx = self._to_pad(np.vstack([ik * eh, ik * ph]))
        Ex, Px = ExPx[0], ExPx[1]
        # W_n, n = 1..M  (n = m + l)
        W = [None]
        for n in range(1, M + 1):
            acc = None
            for m in range(1, n + 1):
                l = n - m
                term = D[m][l]                                  # d^{l+1} phi^(m)
                if l >= 1:
                    term = Epow[l] * term if l == 1 else Epow[l] * (term / fact[l])
                acc = term if acc is None else acc + term
            W.append(acc)
        # eta_t nonlinear part (W_1 is linear and already in deh)
        nl_e = -Ex * Px
        for n in range(2, M + 1):
            nl_e = nl_e + W[n]
        if M >= 3:
            Ex2 = Ex * Ex
            Wlow = W[1]
            for n in range(2, M - 1):
                Wlow = Wlow + W[n]
            nl_e = nl_e + Ex2 * Wlow
        # phis_t nonlinear part
        nl_p = -0.5 * Px * Px
        WW = None
        for n1 in range(1, M):
            for n2 in range(1, M - n1 + 1):
                t_ = W[n1] * W[n2]
                WW = t_ if WW is None else WW + t_
        nl_p = nl_p + 0.5 * WW
        if M >= 4:
            WWl = None
            for n1 in range(1, M - 2):
                for n2 in range(1, M - 1 - n1):
                    t_ = W[n1] * W[n2]
                    WWl = t_ if WWl is None else WWl + t_
            nl_p = nl_p + 0.5 * Ex2 * WWl
        nl = self._from_pad(np.vstack([nl_e, nl_p]))
        return deh + nl[0], dph * self.keep + nl[1]

    # ------------------------------------------------------------------
    # public API
    def rhs(self, eta: np.ndarray, phis: np.ndarray):
        """Physical right-hand side (deta_dt, dphis_dt) of the HOS equations."""
        de, dp = self._rhs_hat(self._fwd(np.asarray(eta, float)),
                               self._fwd(np.asarray(phis, float)))
        return self._inv(de), self._inv(dp)

    def step(self, dt: float) -> None:
        """Advance eta, phis and t by one classical RK4 step."""
        eh = self._fwd(self.eta)
        ph = self._fwd(self.phis)
        f = self._rhs_hat
        k1e, k1p = f(eh, ph)
        h = 0.5 * dt
        k2e, k2p = f(eh + h * k1e, ph + h * k1p)
        k3e, k3p = f(eh + h * k2e, ph + h * k2p)
        k4e, k4p = f(eh + dt * k3e, ph + dt * k3p)
        s = dt / 6.0
        eh = eh + s * (k1e + 2.0 * (k2e + k3e) + k4e)
        ph = ph + s * (k1p + 2.0 * (k2p + k3p) + k4p)
        if self._filter is not None:
            eh *= self._filter
            ph *= self._filter
        self.eta = self._inv(eh)
        self.phis = self._inv(ph)
        self.t += dt

    def phi_modes(self) -> np.ndarray:
        """Sum over m of the HOS modal amplitudes phihat^(m) (rfft convention
        on the N-point grid, unnormalised) such that, for z <= 0,
        phi(x, z) = irfft(phi_modes * exp(|k| z), n=N)."""
        eh = self._fwd(self.eta)
        ph = self._fwd(self.phis)
        if self.M == 1:
            return ph.copy()
        phih = self._modes(eh, ph)[0]
        out = phih[1].copy()
        for m in range(2, self.M + 1):
            out += phih[m]
        return out

    def surface_W(self) -> np.ndarray:
        """Vertical velocity at the surface, W = phi_z(x, eta) (order M)."""
        eh = self._fwd(self.eta)
        ph = self._fwd(self.phis)
        phih, E, Epow, D = self._modes(eh, ph) if self.M > 1 else ([None, ph], None, None, None)
        if self.M == 1:
            return self._inv(self._kk * ph)
        acc = 0.0
        for m in range(1, self.M + 1):
            for l in range(0, self.M - m + 1):
                term = D[m][l]
                if l >= 1:
                    term = Epow[l] * term / self._fact[l]
                acc = acc + term
        return self._inv(self._from_pad(acc))

    def energy(self, kind: str = "full") -> float:
        """Wave energy per unit length (density rho = 1).

        kind='full'   : (1/Lx) int [0.5 g eta^2 + 0.5 phis eta_t] dx, with eta_t
                        from the HOS right-hand side (kinetic energy
                        0.5 int phis dphi/dn ds = 0.5 int phis eta_t dx);
        kind='linear' : (1/Lx) int [0.5 g eta^2 + 0.5 phis W_lin] dx,
                        W_lin = irfft(|k| phis_hat).
        Integrals are exact (spectral) for band-limited data."""
        eh = self._fwd(self.eta)
        ph = self._fwd(self.phis)
        if kind == "full":
            de, _ = self._rhs_hat(eh, ph)
        elif kind == "linear":
            de = self._kk * ph
        else:
            raise ValueError(kind)
        # Parseval: mean(a b) = (1/N^2) [a0 b0 + 2 sum_{m>=1} Re(a_m conj b_m)]
        def pmean(a, b):
            s = (a[0] * np.conj(b[0])).real + 2.0 * np.sum((a[1:] * np.conj(b[1:])).real)
            return s / self.N ** 2
        return 0.5 * self.g * pmean(eh, eh) + 0.5 * pmean(ph, de)
