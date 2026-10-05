# Wave packet over shear-driven turbulence — reproduction

Python reproduction of

> A. Xuan, B.-Q. Deng & L. Shen (2024) *Effect of an incoming Gaussian wave packet on
> underlying turbulence*. J. Fluid Mech. **999**, A45. doi:10.1017/jfm.2024.724

The method follows section 2 of the paper:

* Helmholtz-type decomposition `U = grad(phi) + u` (eq. 2.3): the irrotational wave part is
  computed by a high-order spectral (HOS) solver (`wpt/hos.py`), the rotational
  turbulence by an LES of eq. (2.5)-(2.6) in a rectangular box (`wpt/les.py`);
* wave-turbulence coupling through `-u_phi.grad(u) - u.grad(u_phi)`;
* free-surface boundary conditions transferred from `z = eta` to `z = 0` by a first-order
  Taylor expansion (Appendix A, eqs. A6-A8; `wpt/bc.py`);
* Fourier pseudo-spectral in x, y, second-order staggered finite differences in z,
  AB2 + fractional-step projection, dynamic Smagorinsky SGS model (`wpt/sgs.py`);
* ensemble + spanwise averaging and packet-following averaging (section 3), Reynolds
  stresses, enstrophy, spanwise spectra and the spectral budget of Appendix D
  (`wpt/stats.py`).

`SPEC.md` is the detailed implementation specification (equations, discretisation,
interfaces, tests).

## Layout

```
wpt/params.py      parameters and presets (paper / reduced / tiny)
wpt/grid.py        grid, FFTs, staggered vertical operators
wpt/hos.py         HOS wave solver (deep water, order M)
wpt/coupling.py    wave fields on the LES grid
wpt/bc.py          Taylor-expanded surface boundary conditions (A6-A8)
wpt/les.py         LES solver for the rotational velocity
wpt/sgs.py         dynamic / constant Smagorinsky
wpt/stats.py       packet-following statistics, spectra, spectral budget
wpt/initial.py     initial conditions and regridding
scripts/run_base.py    base-flow spin-up and ensemble snapshots
scripts/run_packet.py  wave-packet runs (one case, several members)
scripts/combine.py     merge members -> data/<case>/results.npz
scripts/plot_figures.py figures 2-17 and tables 1-2
tests/             unit and verification tests (python3 -m pytest -q tests)
```

## Running

```bash
pip install numpy scipy matplotlib pytest
python3 -m pytest -q tests
python3 scripts/run_base.py --preset reduced                 # spin-up + 8 snapshots
python3 scripts/run_packet.py --preset reduced --case W12 --members 0 1 2 3 4 5 6 7
python3 scripts/run_packet.py --preset reduced --case W09 --members 0 1 2 3
python3 scripts/run_packet.py --preset reduced --case W06 --members 0 1 2 3
python3 scripts/combine.py --preset reduced
python3 scripts/plot_figures.py
```

`--preset paper` reproduces the exact configuration of the paper
(512 x 256 x 112, Re_tau = 2000, 30 members), which needs an HPC system.

## Parameters

| quantity | paper | `reduced` preset |
|---|---|---|
| Re_tau = u_* H / nu | 2000 | 500 |
| domain Lx x Ly x H | 6 pi H x 2 pi H x H | same |
| grid | 512 x 256 x 112 | 192 x 128 x 48 |
| k0 H, Fr, eps | 12, 3.74e-4, 0.13 | same |
| alpha = a0 k0 | 0.12, 0.09, 0.06 | same |
| ensemble members | 30 | 8 (W12), 4 (W09, W06) |
| averaging window | t in [0.01, 0.04] H/u_* | same |

## Deviations from the paper

See SPEC.md section 8.  In short: lower Re_tau and resolution, fewer members, implicit
treatment of the vertical wave advection, SGS stress built from the rotational velocity,
packet initial position `x0 = Lx/4` (not stated in the paper).
