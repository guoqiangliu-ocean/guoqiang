#!/usr/bin/env python3
"""Spin up the shear-driven base turbulence (no waves) and write ensemble
initial snapshots (paper section 2.2: "we first simulate the base flow without
waves ... ensemble simulations ... using different initial instants").

Stages
  1. coarse spin-up on a grid with Nx/cf, Ny/cf (same vertical grid);
  2. spectral regridding to the target grid and further spin-up;
  3. ``--n-snap`` snapshots separated by ``--snap-interval`` (H/u_*), with
     base-flow statistics accumulated in between.

Every stage writes a checkpoint so the run can be resumed with --resume.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from common import ROOT, Logger, max_rate, preset, save_json  # noqa: F401
from wpt.grid import Grid
from wpt.initial import mean_profile, random_divfree_perturbation, regrid_state
from wpt.les import LESSolver
from wpt.sgs import DynamicSmagorinsky, Smagorinsky
from wpt.stats import BaseAccumulator
from wpt.wavefields import zero_wave_fields


def make_solver(p):
    g = Grid(p)
    if p.sgs == "dynamic":
        sgs = DynamicSmagorinsky(g, p)
        sgs.truncate_input = False
    elif p.sgs == "smagorinsky":
        sgs = Smagorinsky(g, p)
        sgs.truncate_input = False
    else:
        sgs = None
    return g, LESSolver(p, g, sgs)


def choose_dt(solver, g, cfl, dt_max):
    r = max_rate(solver.u, solver.v, solver.w, g)
    return min(dt_max, cfl / max(r, 1e-12))


def advance(solver, g, t_end, cfl, dt_max, log, tag, ckpt=None, ckpt_every=2000,
            on_step=None):
    """Integrate to t_end with a fixed dt that is re-chosen (with an AB2
    restart) only when the CFL number drifts outside [0.5 cfl, 1.3 cfl]."""
    wf = zero_wave_fields(g)
    dt = choose_dt(solver, g, cfl, dt_max)
    solver.set_state(solver.u, solver.v, solver.w, t=solver.t)
    n = 0
    while solver.t < t_end - 1e-12:
        dt_step = min(dt, t_end - solver.t)
        if dt_step < dt:
            solver.set_state(solver.u, solver.v, solver.w, t=solver.t)
        solver.step(dt_step, wf, wf)
        n += 1
        if on_step is not None:
            on_step(solver)
        if n % 50 == 0:
            c = max_rate(solver.u, solver.v, solver.w, g) * dt
            if not np.isfinite(c):
                raise FloatingPointError(f"{tag}: non-finite velocity at t={solver.t}")
            if c > 1.3 * cfl or (c < 0.5 * cfl and dt < dt_max):
                dt = choose_dt(solver, g, cfl, dt_max)
                solver.set_state(solver.u, solver.v, solver.w, t=solver.t)
        if n % 500 == 0:
            U = solver.u.mean(axis=(0, 1))
            urms = solver.u.std(axis=(0, 1))
            wrms = g.f2c(solver.w).std(axis=(0, 1))
            div = float(np.abs(solver.divergence()).max())
            log(f"{tag} n={n} t={solver.t:.4f} dt={dt:.2e} Us={U[-1]:.2f} "
                f"u'max={urms.max():.3f} w'max={wrms.max():.3f} div={div:.1e}")
        if ckpt is not None and n % ckpt_every == 0:
            solver.save(ckpt)
    if ckpt is not None:
        solver.save(ckpt)
    return solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="reduced")
    ap.add_argument("--coarse-factor", type=int, default=2)
    ap.add_argument("--t-coarse", type=float, default=15.0)
    ap.add_argument("--t-fine", type=float, default=4.0)
    ap.add_argument("--n-snap", type=int, default=8)
    ap.add_argument("--snap-interval", type=float, default=1.0)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--dt-max", type=float, default=2.0e-3)
    ap.add_argument("--pert-amp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "base"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    log = Logger(os.path.join(a.out, "run_base.log"))
    p = preset(a.preset, wave_on=False, workers=a.workers)
    save_json(os.path.join(a.out, "params.json"), p.to_dict())
    log(f"preset={a.preset} Re_tau={p.Re_tau} grid={p.Nx}x{p.Ny}x{p.Nz}")

    ck_coarse = os.path.join(a.out, "ckpt_coarse.npz")
    ck_fine = os.path.join(a.out, "ckpt_fine.npz")

    # ---------------- stage 1: coarse spin-up ----------------
    pc = p.with_(Nx=p.Nx // a.coarse_factor, Ny=p.Ny // a.coarse_factor)
    gc, sc = make_solver(pc)
    if a.resume and os.path.exists(ck_fine):
        log("fine checkpoint found, skipping coarse stage")
    else:
        if a.resume and os.path.exists(ck_coarse):
            sc.load(ck_coarse)
            log(f"resumed coarse stage at t={sc.t:.3f}")
        else:
            U = mean_profile(gc, pc.Re_tau)
            du, dv, dw = random_divfree_perturbation(gc, a.pert_amp, a.seed)
            sc.set_state(U[None, None, :] + du, dv, dw, t=0.0)
        advance(sc, gc, a.t_coarse, a.cfl, a.dt_max, log, "coarse", ckpt=ck_coarse)

    # ---------------- stage 2: fine spin-up ----------------
    g, s = make_solver(p)
    if a.resume and os.path.exists(ck_fine):
        s.load(ck_fine)
        log(f"resumed fine stage at t={s.t:.3f}")
    else:
        u, v, w = regrid_state(sc.u, sc.v, sc.w, gc, g)
        s.set_state(u, v, w, t=sc.t)
    t_fine_end = a.t_coarse + a.t_fine
    advance(s, g, t_fine_end, a.cfl, a.dt_max, log, "fine", ckpt=ck_fine)

    # ---------------- stage 3: snapshots + base statistics ----------------
    acc = BaseAccumulator(g, p)
    every = [0]

    def on_step(sol):
        every[0] += 1
        if every[0] % 10 == 0:
            acc.add_sample(sol)

    for i in range(a.n_snap):
        fn = os.path.join(a.out, f"snap_{i:03d}.npz")
        if a.resume and os.path.exists(fn):
            continue
        t_target = t_fine_end + (i + 1) * a.snap_interval
        advance(s, g, t_target, a.cfl, a.dt_max, log, f"snap{i}", ckpt=ck_fine, on_step=on_step)
        np.savez(fn, u=s.u, v=s.v, w=s.w, t=s.t)
        log(f"wrote {fn} at t={s.t:.3f}")
        acc.save(os.path.join(a.out, "base_acc.npz"))
    res = acc.results()
    np.savez(os.path.join(a.out, "base_stats.npz"), **res)
    log("done")


if __name__ == "__main__":
    main()
