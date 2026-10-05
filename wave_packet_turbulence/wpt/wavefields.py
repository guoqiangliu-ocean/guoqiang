"""Container for the irrotational (wave) quantities sampled on the LES grid.

The imposed wave packet is two-dimensional (independent of y), so every field
has a singleton y axis and broadcasts against (Nx, Ny, nz) arrays.

Notation (paper eq. 2.3-2.4): u_phi = grad(phi), so
    uphi = phi_x,  wphi = phi_z,
    duphi_dx = phi_xx,  duphi_dz = phi_xz = dwphi_dx,  dwphi_dz = phi_zz = -phi_xx.
All fields are already dealiased in x (only |mx| < Nx/3 retained).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WaveFields:
    t: float
    # surface quantities on the LES x grid, shape (Nx, 1)
    eta: np.ndarray
    eta_x: np.ndarray
    # 2 * phi_xz at z = 0, shape (Nx, 1): the irrotational viscous stress
    # t_x . sigma_{u_phi} . n / (rho nu) to O(alpha)   (paper eq. 2.15)
    irrot_stress_x: np.ndarray
    # wave velocity and gradients at cell centres, shape (Nx, 1, Nz)
    uphi_c: np.ndarray
    wphi_c: np.ndarray
    duphi_dx_c: np.ndarray
    duphi_dz_c: np.ndarray
    dwphi_dx_c: np.ndarray
    dwphi_dz_c: np.ndarray
    # wave velocity and gradients at cell faces, shape (Nx, 1, Nz + 1)
    uphi_f: np.ndarray
    wphi_f: np.ndarray
    duphi_dx_f: np.ndarray
    duphi_dz_f: np.ndarray
    dwphi_dx_f: np.ndarray
    dwphi_dz_f: np.ndarray

    @property
    def eta_y(self) -> np.ndarray:
        return np.zeros_like(self.eta)


def zero_wave_fields(grid, t: float = 0.0) -> WaveFields:
    Nx, Nz = grid.Nx, grid.Nz
    s = np.zeros((Nx, 1))
    c = np.zeros((Nx, 1, Nz))
    f = np.zeros((Nx, 1, Nz + 1))
    return WaveFields(t=t, eta=s, eta_x=s.copy(), irrot_stress_x=s.copy(),
                      uphi_c=c, wphi_c=c.copy(), duphi_dx_c=c.copy(),
                      duphi_dz_c=c.copy(), dwphi_dx_c=c.copy(), dwphi_dz_c=c.copy(),
                      uphi_f=f, wphi_f=f.copy(), duphi_dx_f=f.copy(),
                      duphi_dz_f=f.copy(), dwphi_dx_f=f.copy(), dwphi_dz_f=f.copy())
