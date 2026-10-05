"""Physical and numerical parameters.

All quantities are nondimensionalised with the boundary-layer depth H, the
surface friction velocity u_* = sqrt(tau_0/rho) and the water density rho, so
H = u_* = rho = 1.  Time is in units of H/u_*.

Paper: Xuan, Deng & Shen (2024), J. Fluid Mech. 999, A45, section 2.2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace


@dataclass
class Params:
    # ---- physical parameters (paper section 2.2) ----
    Re_tau: float = 2000.0      # u_* H / nu
    Fr: float = 3.74e-4         # u_* / sqrt(g H)
    k0H: float = 12.0           # carrier wavenumber times H
    alpha: float = 0.12         # steepness a0 k0 (W12, W09, W06 -> 0.12, 0.09, 0.06)
    eps_bw: float = 0.13        # bandwidth parameter 1/(k0 chi)
    x0_frac: float = 0.25       # initial packet centre x0 = x0_frac * Lx (not given in paper)
    include_irrot_stress: bool = True   # keep t.sigma_{u_phi}.n in tau^r (eq. 2.15)
    wave_on: bool = True        # False -> base flow without waves

    # ---- domain and grid ----
    Lx: float = 6.0 * math.pi
    Ly: float = 2.0 * math.pi
    H: float = 1.0
    Nx: int = 512
    Ny: int = 256
    Nz: int = 112
    z_stretch: float = 2.0      # tanh clustering of vertical faces towards z = 0

    # ---- wave (HOS) solver ----
    hos_order: int = 3          # nonlinear order M of the HOS expansion
    hos_N: int = 1024           # HOS collocation points in x (must be a multiple of Nx)

    # ---- time integration ----
    dt: float = 1.0e-5
    implicit_wave_vadv: bool = True   # CN for w_phi d/dz (vertical wave advection)

    # ---- subgrid-scale model: 'dynamic', 'smagorinsky' or 'none' ----
    sgs: str = "dynamic"
    cs_const: float = 0.1       # used only when sgs == 'smagorinsky'

    # ---- statistics (paper section 3) ----
    t1: float = 0.01            # start of packet-following average
    t2: float = 0.04            # end of packet-following average

    # ---- misc ----
    workers: int = 4            # threads for scipy.fft

    # ------------------------------------------------------------------
    @property
    def nu(self) -> float:
        return 1.0 / self.Re_tau

    @property
    def g(self) -> float:
        return 1.0 / self.Fr ** 2

    @property
    def k0(self) -> float:
        return self.k0H / self.H

    @property
    def omega0(self) -> float:
        return math.sqrt(self.g * self.k0)

    @property
    def c0(self) -> float:
        return self.omega0 / self.k0

    @property
    def cg(self) -> float:
        return 0.5 * self.c0

    @property
    def a0(self) -> float:
        return self.alpha / self.k0

    @property
    def chi(self) -> float:
        return 1.0 / (self.eps_bw * self.k0)

    @property
    def T0(self) -> float:
        return 2.0 * math.pi / self.omega0

    @property
    def x0(self) -> float:
        return self.x0_frac * self.Lx

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("nu", "g", "k0", "omega0", "c0", "cg", "a0", "chi", "T0", "x0"):
            d[k] = getattr(self, k)
        return d

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)


CASES = {"W12": 0.12, "W09": 0.09, "W06": 0.06}


def paper(**kw) -> Params:
    """Exact configuration of the paper (512 x 256 x 112, Re_tau = 2000)."""
    return Params(**kw)


def reduced(**kw) -> Params:
    """Reduced configuration that fits a 4-core workstation.

    Re_tau is lowered to 500 so that the near-surface viscous layer can be
    resolved with 48 stretched vertical cells; the domain and all wave
    parameters (k0 H, Fr, alpha, eps) are those of the paper.
    """
    base = dict(Re_tau=500.0, Nx=192, Ny=128, Nz=48, z_stretch=2.2,
                hos_N=768, dt=1.4e-5)
    base.update(kw)
    return Params(**base)


def tiny(**kw) -> Params:
    """Very small configuration for smoke tests."""
    base = dict(Re_tau=300.0, Lx=2.0 * math.pi, Ly=math.pi, Nx=96, Ny=32,
                Nz=24, z_stretch=1.5, hos_N=192, dt=2.0e-5)
    base.update(kw)
    return Params(**base)
