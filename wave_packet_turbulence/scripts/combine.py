#!/usr/bin/env python3
"""Merge per-member accumulators of a case and write data/<case>/results.npz."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from common import ROOT, preset
from wpt.grid import Grid
from wpt.params import CASES
from wpt.stats import PacketAccumulator, packet_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="reduced")
    ap.add_argument("--cases", nargs="+", default=sorted(CASES))
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--window", type=float, default=3.5)
    a = ap.parse_args()
    for case in a.cases:
        d = os.path.join(a.data, case)
        files = sorted(glob.glob(os.path.join(d, "acc_member_*.npz")))
        if not files:
            print(f"{case}: no accumulators, skipped")
            continue
        p = preset(a.preset, alpha=CASES[case], wave_on=True)
        g = Grid(p)
        acc = PacketAccumulator.load(files[0])
        for f in files[1:]:
            acc.merge(PacketAccumulator.load(f))
        res = packet_results(acc, g, p, window=a.window)
        np.savez(os.path.join(d, "results.npz"), **res)
        with open(os.path.join(d, "members.json"), "w") as fh:
            json.dump([os.path.basename(f) for f in files], fh)
        print(f"{case}: merged {len(files)} members -> {os.path.join(d, 'results.npz')}")


if __name__ == "__main__":
    main()
