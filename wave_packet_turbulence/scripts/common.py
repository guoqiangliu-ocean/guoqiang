"""Helpers shared by the driver scripts."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wpt import params as P  # noqa: E402


def preset(name: str, **kw):
    return {"paper": P.paper, "reduced": P.reduced, "tiny": P.tiny}[name](**kw)


def max_rate(u, v, w, grid):
    """Largest advective rate |u|/dx + |v|/dy + |w|/dz (1/time)."""
    wc = 0.5 * (np.abs(w[..., 1:]) + np.abs(w[..., :-1]))
    r = np.abs(u) / grid.dx + np.abs(v) / grid.dy + wc / grid.dzc
    return float(r.max())


class Logger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", buffering=1)
        self.t0 = time.time()

    def __call__(self, msg: str):
        line = f"[{time.time() - self.t0:9.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
