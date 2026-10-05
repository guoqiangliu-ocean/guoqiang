"""Tests of the Taylor-expanded surface boundary conditions (SPEC.md section 3, tests iii-iv)."""
import math

import numpy as np
import pytest

from wpt.bc import surface_bc, surface_w, uzz_weights
from wpt.grid import Grid
from wpt.params import Params
from wpt.wavefields import WaveFields, zero_wave_fields


def make_params(**kw):
    base = dict(Re_tau=300.0, Lx=2 * math.pi, Ly=2 * math.pi, Nx=16, Ny=16, Nz=16,
                z_stretch=1.5, workers=1)
    base.update(kw)
    return Params(**base)


def surface_wave_fields(g, eta, eta_x, irr):
    wf = zero_wave_fields(g)
    wf.eta, wf.eta_x, wf.irrot_stress_x = eta, eta_x, irr
    return wf


class Manufactured:
    """u = u0 + G z + Q z^2/2 + R z^3/6 + P z^4/24 (same for v) with G, Gv chosen as the
    right-hand sides of (A7), (A8) evaluated by hand, so that the exact surface gradient
    satisfies the boundary conditions and every output has a closed form."""

    def __init__(self, g, p, a=0.05):
        X = g.x.reshape(-1, 1)
        Y = g.y.reshape(1, -1)
        self.g, self.p = g, p
        self.eta = a * np.cos(X) + 0.5 * a * np.sin(2 * X)
        self.eta_x = -a * np.sin(X) + a * np.cos(2 * X)
        self.irr = 0.7 * np.cos(X)
        # surface values and their derivatives
        u0 = 1.0 + 0.5 * np.cos(X + Y) + 0.3 * np.sin(2 * Y)
        u0x = -0.5 * np.sin(X + Y)
        u0y = -0.5 * np.sin(X + Y) + 0.6 * np.cos(2 * Y)
        u0xx = -0.5 * np.cos(X + Y)
        u0xy = -0.5 * np.cos(X + Y)
        v0 = 0.4 * np.sin(X) * np.cos(Y) + 0.2
        v0x = 0.4 * np.cos(X) * np.cos(Y)
        v0y = -0.4 * np.sin(X) * np.sin(Y)
        v0xy = -0.4 * np.cos(X) * np.sin(Y)
        v0yy = -0.4 * np.sin(X) * np.cos(Y)
        self.W = 0.3 * np.sin(X - Y)                  # w_top_prev
        Wx = 0.3 * np.cos(X - Y)
        Wy = -0.3 * np.cos(X - Y)
        Q = 2.0 + np.cos(Y - X)
        Qv = np.sin(X + Y)
        wzx = -(u0xx + v0xy)
        wzy = -(u0xy + v0yy)
        eta, eta_x = self.eta, self.eta_x
        # (A7), (A8) with eta_y = 0, tau_x/nu = Re_tau - irr, tau_y = 0
        self.G = p.Re_tau - self.irr - Wx + 2 * eta_x * (2 * u0x + v0y) - eta * (wzx + Q)
        self.Gv = -Wy + eta_x * (v0x + u0y) - eta * (wzy + Qv)
        self.u0, self.v0, self.Q, self.Qv = u0, v0, Q, Qv
        self.R, self.Rv = 5.0 * np.cos(X) + 0 * Y, 3.0 * np.sin(Y) + 0 * X
        self.P = 7.0
        # exact (A6)
        self.ws = eta_x * u0 + eta * u0x + eta * v0y

    def fields(self):
        Z = self.g.zc.reshape(1, 1, -1)

        def build(q0, G, Q, R):
            return (q0[..., None] + G[..., None] * Z + Q[..., None] * Z ** 2 / 2
                    + R[..., None] * Z ** 3 / 6 + self.P * Z ** 4 / 24)

        return build(self.u0, self.G, self.Q, self.R), build(self.v0, self.Gv, self.Qv, self.Rv)


def _errors(Nz, method="cubic", stretch=1.5):
    p = make_params(Nz=Nz, z_stretch=stretch)
    g = Grid(p)
    m = Manufactured(g, p)
    u, v = m.fields()
    wf = surface_wave_fields(g, m.eta, m.eta_x, m.irr)
    g_prev = {"dudz_s": m.G + 0 * m.W, "dvdz_s": m.Gv + 0 * m.W}
    bc = surface_bc(u, v, m.W, g, wf, p, g_prev=g_prev, uzz_method=method)
    return np.array([
        np.abs(bc["dudz_s"] - m.G).max() / np.abs(m.G).max(),
        np.abs(bc["dvdz_s"] - m.Gv).max(),
        np.abs(bc["u_s"] - m.u0).max(),
        np.abs(bc["v_s"] - m.v0).max(),
        np.abs(bc["w_s"] - m.ws).max(),
        np.abs(bc["uzz_s"] - m.Q).max(),
    ])


@pytest.mark.parametrize("stretch", [0.0, 1.5])
def test_bc_manufactured_second_order(stretch):
    E = np.array([_errors(Nz, stretch=stretch) for Nz in (16, 32, 64, 128)])
    orders = np.log2(E[:-1] / E[1:])
    print("errors (dudz, dvdz, u_s, v_s, w_s, uzz):\n", E, "\norders:\n", orders)
    assert np.all(E[-1] < 1e-3)
    assert np.all(orders[-2:] > 1.8)


def test_bc_top_cell_variant_first_order():
    E = np.array([_errors(Nz, method="top_cell") for Nz in (16, 32, 64, 128)])
    orders = np.log2(E[:-1] / E[1:])
    print("top_cell errors:\n", E, "\norders:\n", orders)
    assert np.all(orders[-1, [0, 1, 5]] > 0.8)


def test_bc_eta_zero():
    p = make_params(Nz=12)
    g = Grid(p)
    rng = np.random.default_rng(0)
    u = rng.standard_normal((g.Nx, g.Ny, g.Nz))
    v = rng.standard_normal((g.Nx, g.Ny, g.Nz))
    wf = zero_wave_fields(g)
    bc = surface_bc(u, v, np.zeros((g.Nx, g.Ny)), g, wf, p)
    assert np.abs(bc["dudz_s"] - p.Re_tau).max() < 1e-12 * p.Re_tau
    assert np.abs(bc["dvdz_s"]).max() < 1e-12
    assert np.abs(bc["w_s"]).max() == 0.0
    h = g.dzf[-1]
    assert np.allclose(bc["u_s"], u[..., -1] + h * p.Re_tau)
    assert np.allclose(bc["v_s"], v[..., -1])
    # irrotational stress enters with a minus sign (eq. 2.15)
    irr = 0.3 * np.cos(2 * g.x).reshape(-1, 1)
    wf2 = surface_wave_fields(g, 0 * irr, 0 * irr, irr)
    bc2 = surface_bc(u, v, np.zeros((g.Nx, g.Ny)), g, wf2, p, tau_x=0.0)
    assert np.allclose(bc2["dudz_s"], -irr * np.ones((1, g.Ny)), atol=1e-12)


def test_surface_w():
    p = make_params()
    g = Grid(p)
    X = g.x.reshape(-1, 1)
    Y = g.y.reshape(1, -1)
    eta = 0.1 * np.cos(2 * X)
    us = np.sin(X + 2 * Y) + 3.0
    vs = np.cos(Y) * np.sin(X)
    ws = surface_w(us, vs, eta, g)
    exact = (-0.2 * np.sin(2 * X) * us + eta * np.cos(X + 2 * Y)
             + eta * (-np.sin(Y) * np.sin(X)))
    assert np.abs(ws - exact).max() < 1e-12
    assert abs(ws.mean()) < 1e-15


def test_uzz_weights_exact_for_cubic():
    g = Grid(make_params(Nz=10, z_stretch=2.0))
    c = uzz_weights(g, 4)
    z = g.zc[::-1][:4]
    for poly, d2 in ((lambda z: z ** 3 + 2 * z ** 2, 4.0), (lambda z: 1 + z, 0.0)):
        assert abs(np.dot(c, poly(z)) - d2) < 1e-9 * np.abs(c).sum()


def test_top_cell_feedback_unstable_when_eta_exceeds_dz():
    """Documents the SPEC.md 'top_cell' u_zz problem: iterating the lagged surface
    gradient amplifies it by ~ eta/dzc[-1] per call, while the default is independent of g_prev."""
    p = make_params(Nz=16, z_stretch=2.0)
    g = Grid(p)
    eta = np.full((g.Nx, 1), 3.0 * g.dzc[-1])
    wf = surface_wave_fields(g, eta, 0 * eta, 0 * eta)
    u = np.zeros((g.Nx, g.Ny, g.Nz))
    v = np.zeros_like(u)
    w0 = np.zeros((g.Nx, g.Ny))
    out = {}
    for method in ("top_cell", "cubic"):
        bc = None
        for _ in range(6):
            bc = surface_bc(u, v, w0, g, wf, p, g_prev=bc, uzz_method=method)
        out[method] = np.abs(bc["dudz_s"]).max()
    assert out["top_cell"] > 50 * p.Re_tau
    assert abs(out["cubic"] - p.Re_tau) < 1e-9 * p.Re_tau


def test_wave_scale_uzz():
    """Default u_zz estimate: exact for profiles quadratic in z, O(L) otherwise,
    stencil depth L = max(2 dzc[-1], max|eta|)."""
    from wpt.bc import wave_scale_stencil
    p = make_params(Nz=48, z_stretch=2.2)
    g = Grid(p)
    m = Manufactured(g, p)
    wf = surface_wave_fields(g, m.eta, m.eta_x, m.irr)
    Z = g.zc.reshape(1, 1, -1)
    uq = m.u0[..., None] + m.G[..., None] * Z + 0.5 * m.Q[..., None] * Z ** 2
    bc = surface_bc(uq, 0 * uq, m.W, g, wf, p)
    assert np.abs(bc["uzz_s"] - m.Q).max() < 1e-8 * np.abs(m.Q).max()
    k1, k2, k3 = wave_scale_stencil(g, m.eta)
    L = np.abs(m.eta).max()
    assert k1 > k2 > k3 and abs(g.zc[k2] + L) < g.dzc[k2] and abs(g.zc[k3] + 2 * L) < g.dzc[k3]
    u, v = m.fields()
    bc = surface_bc(u, v, m.W, g, wf, p)
    # third-order term R z^3/6: curvature error ~ |R| * O(L)
    err = np.abs(bc["uzz_s"] - m.Q).max()
    assert err < 2.0 * np.abs(m.R).max() * 3 * L
    # small eta: falls back to (at least) a 2 dz stencil
    k1, k2, k3 = wave_scale_stencil(g, 0 * m.eta)
    assert k2 == g.Nz - 2 and k3 <= g.Nz - 3
