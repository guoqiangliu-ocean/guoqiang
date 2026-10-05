#!/usr/bin/env python3
"""Impose the Gaussian wave packet on base-turbulence snapshots and
accumulate the packet-following statistics (paper sections 2.2 and 3).

Each ensemble member starts from ``data/base/snap_XXX.npz`` (the wave field is
identical for every member) and is integrated to ``t2``; samples are taken at
identical instants in ``[t1, t2]`` and stored in a per-member accumulator.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from common import ROOT, Logger, max_rate, preset, save_json
from wpt.coupling import wave_fields_from_hos
from wpt.grid import Grid
from wpt.hos import HOS
from wpt.les import LESSolver
from wpt.params import CASES
from wpt.sgs import DynamicSmagorinsky, Smagorinsky
from wpt.stats import PacketAccumulator


def sample_schedule(p, dt, n_samples):
    """Integer step indices in [t1, t2] at an (almost) uniform interval."""
    n1 = int(math.ceil(p.t1 / dt - 1e-9))
    n2 = int(math.floor(p.t2 / dt + 1e-9))
    steps = np.unique(np.round(np.linspace(n1, n2, n_samples)).astype(int))
    return steps, steps * dt


def run_member(p, case, member, base_dir, out_dir, n_samples, budget, log):
    g = Grid(p)
    if p.sgs == "dynamic":
        sgs = DynamicSmagorinsky(g, p)
        sgs.truncate_input = False
    elif p.sgs == "smagorinsky":
        sgs = Smagorinsky(g, p)
        sgs.truncate_input = False
    else:
        sgs = None
    solver = LESSolver(p, g, sgs)
    snap = np.load(os.path.join(base_dir, f"snap_{member:03d}.npz"))
    solver.set_state(snap["u"], snap["v"], snap["w"], t=0.0)

    hos = HOS(p)
    hos.init_packet()
    wf_n = wave_fields_from_hos(hos, g, p)

    dt = p.dt
    steps, times = sample_schedule(p, dt, n_samples)
    acc = PacketAccumulator(g, p, times, budget=budget)
    nsteps = int(steps[-1])
    isamp = 0
    phase_step = (p.omega0 * (times[1] - times[0]) / 2.0 / (2 * math.pi)) % 1.0 if len(times) > 1 else 0
    log(f"{case} m={member}: dt={dt:.3e} ({p.T0 / dt:.1f} steps/period) nsteps={nsteps} "
        f"samples={len(times)} carrier-phase step={phase_step:.3f} cycle")
    for n in range(1, nsteps + 1):
        hos.step(dt)
        wf_np1 = wave_fields_from_hos(hos, g, p)
        solver.step(dt, wf_n, wf_np1)
        wf_n = wf_np1
        if isamp < len(steps) and n == steps[isamp]:
            acc.add_sample(isamp, solver, wf_n)
            isamp += 1
        if n % 200 == 0 or n == nsteps:
            c = max_rate(solver.u, solver.v, solver.w, g) * dt
            if not np.isfinite(c):
                raise FloatingPointError(f"non-finite velocity at step {n}")
            div = float(np.abs(solver.divergence()).max())
            log(f"{case} m={member} n={n}/{nsteps} t={solver.t:.5f} cfl_rot={c:.3f} "
                f"div={div:.1e} eta_max={np.abs(hos.eta).max() * p.k0:.4f}(k0)")
    path = os.path.join(out_dir, f"acc_member_{member:03d}.npz")
    acc.save(path)
    log(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="reduced")
    ap.add_argument("--case", default="W12", choices=sorted(CASES))
    ap.add_argument("--members", type=int, nargs="+", default=[0])
    ap.add_argument("--base-dir", default=os.path.join(ROOT, "data", "base"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--n-samples", type=int, default=61)
    ap.add_argument("--no-budget", action="store_true")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    kw = dict(alpha=CASES[a.case], wave_on=True, workers=a.workers)
    if a.dt is not None:
        kw["dt"] = a.dt
    p = preset(a.preset, **kw)
    out_dir = os.path.join(a.out, a.case)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "params.json"), p.to_dict())
    log = Logger(os.path.join(out_dir, "run_packet.log"))
    for m in a.members:
        if a.skip_existing and os.path.exists(os.path.join(out_dir, f"acc_member_{m:03d}.npz")):
            log(f"member {m} exists, skipping")
            continue
        run_member(p, a.case, m, a.base_dir, out_dir, a.n_samples, not a.no_budget, log)


if __name__ == "__main__":
    main()
