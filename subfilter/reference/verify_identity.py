"""Part IV of surface_term_two_routes.pdf: the identity
   rho <u'.(uS x w')> + uS.F^R = -rho uS.grad k_t ,  F^R = -rho div<u'u'>,  k_t = <|u'|^2>/2.
Every step is checked pointwise on a general solenoidal field u = curl A and a general uS(x,y,z);
the averaging step is checked on a two-member ensemble.      python3 verify_identity.py"""
import sympy as sp
x, y, z = sp.symbols("x y z", real=True); X = (x, y, z); rho = sp.Symbol("rho", positive=True)
def curl(F): return sp.Matrix([sp.diff(F[2], y)-sp.diff(F[1], z), sp.diff(F[0], z)-sp.diff(F[2], x), sp.diff(F[1], x)-sp.diff(F[0], y)])
def div(F): return sum(sp.diff(F[i], X[i]) for i in range(3))
def grad(f): return sp.Matrix([sp.diff(f, v) for v in X])
def adv(a, b): return sp.Matrix([sum(a[j]*sp.diff(b[i], X[j]) for j in range(3)) for i in range(3)])   # (a.grad) b
ok = lambda e: sp.simplify(sp.expand(e)) == 0

A = sp.Matrix([sp.Function("A%d" % i)(*X) for i in range(3)]); u = curl(A); w = curl(u)          # solenoidal u'
uS = sp.Matrix([sp.Function("S%d" % i)(*X) for i in range(3)])                                   # general uS
print("div u' = 0 :", ok(div(u)))

# Step 1: b x (curl b) = grad(|b|^2/2) - (b.grad) b  for ANY b (use a non-solenoidal b too)
b = sp.Matrix([sp.Function("B%d" % i)(*X) for i in range(3)])
print("Step 1  b x curl b - [grad|b|^2/2 - (b.grad)b] = 0 :", all(ok(e) for e in (b.cross(curl(b)) - (grad(b.dot(b)/2) - adv(b, b)))))
lhs = u.dot(uS.cross(w))
print("Step 1  u'.(uS x w') - uS.[(u'.grad)u' - grad|u'|^2/2] = 0 :", ok(lhs - uS.dot(adv(u, u) - grad(u.dot(u)/2))))

# Step 2: (u'.grad)u'_i = d_j(u'_i u'_j)  for solenoidal u'
divuu = sp.Matrix([sum(sp.diff(u[i]*u[j], X[j]) for j in range(3)) for i in range(3)])
print("Step 2  (u'.grad)u' - div(u'u') = 0 :", all(ok(e) for e in (adv(u, u) - divuu)))
print("Step 2  u'.(uS x w') - [uS_i d_j(u'_i u'_j) - uS.grad|u'|^2/2] = 0 :", ok(lhs - (uS.dot(divuu) - uS.dot(grad(u.dot(u)/2)))))

# Step 4: the three-term split and the symmetric/antisymmetric cancellation, pointwise
I = u.dot(grad(uS.dot(u))); II = -u.dot(adv(uS, u)); III = -u.dot(adv(u, uS))
print("Step 4  u'.(uS x w') = I + II + III :", ok(lhs - (I + II + III)))
Iflux = sum(sp.diff(u[i]*uS.dot(u), X[i]) for i in range(3))
print("Step 4  I = d_i(u'_i uS.u') :", ok(I - Iflux))
sym_anti = sum(u[i]*u[j]*(sp.diff(uS[j], X[i]) - sp.diff(uS[i], X[j])) for i in range(3) for j in range(3))
print("Step 4  u'_i u'_j (d_i uS_j - d_j uS_i) = 0 :", ok(sym_anti))
stokes_prod = -sum(u[i]*u[j]*sp.diff(uS[i], X[j]) for i in range(3) for j in range(3))
print("Step 4  u'.(uS x w') = Stokes production + d_i(u'_i uS.u') - uS.grad|u'|^2/2 :", ok(lhs - (stokes_prod + Iflux - uS.dot(grad(u.dot(u)/2)))))

# Step 3: the average. Two-member ensemble u1, u2 (both solenoidal), <X> = (X1 + X2)/2, uS deterministic.
A2 = sp.Matrix([sp.Function("C%d" % i)(*X) for i in range(3)]); u2 = curl(A2)
avg = lambda f1, f2: (f1 + f2)/2
Rij = sp.Matrix(3, 3, lambda i, j: avg(u[i]*u[j], u2[i]*u2[j]))                     # <u'_i u'_j>
FR = -rho*sp.Matrix([sum(sp.diff(Rij[i, j], X[j]) for j in range(3)) for i in range(3)])
kt = avg(u.dot(u), u2.dot(u2))/2
lhs_av = rho*avg(u.dot(uS.cross(w)), u2.dot(uS.cross(curl(u2))))
print("Step 3  rho<u'.(uS x w')> + uS.F^R + rho uS.grad k_t = 0 :", ok(lhs_av + uS.dot(FR) + rho*uS.dot(grad(kt))))
print("        <u'_i u'_j> symmetric :", Rij == Rij.T)

# Step 5: Doppler weight  rho N k 2k e^{2kz} = rho u^S
a, k, g = sp.symbols("a k g", positive=True); sigma = sp.sqrt(g*k); N = g*a**2/(2*sigma)
print("Step 5  rho N k 2k e^{2kz} - rho a^2 sigma k e^{2kz} = 0 :", ok(rho*N*k*2*k*sp.exp(2*k*z) - rho*a**2*sigma*k*sp.exp(2*k*z)))

# Step 6: horizontally homogeneous depth integral: -int uS tau_z dz = P_S - W_S (pointwise form of the integration by parts)
us = sp.Function("us")(z); tau = sp.Function("tau")(z)
print("Step 6  -uS tau_z = tau uS_z - d(uS tau)/dz :", ok(-us*sp.diff(tau, z) - (tau*sp.diff(us, z) - sp.diff(us*tau, z))))
# Step 6: for horizontal homogeneity the right side vanishes: uS horizontal, grad k_t vertical
ktz = sp.Function("kt")(z); uSh = sp.Matrix([sp.Function("usx")(z), sp.Function("usy")(z), 0])
print("Step 6  uS.grad k_t = 0 for uS horizontal, k_t = k_t(z) :", ok(uSh.dot(grad(ktz))))

# Scope item 3: for a Reynolds average <u'> = 0 and <U X> = U <X>, so the Leonard and cross terms of the filtered stress vanish
Ui = sp.Matrix([sp.Function("U%d" % i)(*X) for i in range(3)])
Cij = sp.Matrix(3, 3, lambda i, j: Ui[i]*avg(u[j], -u[j]) + avg(u[i], -u[i])*Ui[j])     # ensemble {u', -u'} has <u'> = 0
print("Scope 3 cross term C_ij = U_i<u'_j> + <u'_i>U_j = 0 when <u'> = 0 :", all(ok(e) for e in Cij))

# Section 8: the carried mean stress has zero action (pseudomomentum) projection.
#   X^l = -tau'(zbar) (xi_{z,z}, xi_{z,x}) / rho ;  forcing of p_x = -<xi_{j,x} X^l_j>
th, zz = sp.symbols("theta z", real=True); a, k = sp.symbols("a k", positive=True)
tauz = sp.Function("taubar")(zz); tp = sp.diff(tauz, zz)
xi_x = -a*sp.exp(k*zz)*sp.sin(th); xi_z = a*sp.exp(k*zz)*sp.cos(th)
avth = lambda f: sp.integrate(sp.expand(f), (th, 0, 2*sp.pi))/(2*sp.pi)
Xl_x = -tp*sp.diff(xi_z, zz)/rho; Xl_z = -tp*(k*sp.diff(xi_z, th))/rho          # d/dx = k d/dtheta
px = -rho*avth(Xl_x*(k*sp.diff(xi_x, th))); pz = -rho*avth(Xl_z*(k*sp.diff(xi_z, th)))
print("Sec 8   x-part:", sp.simplify(px), ";  z-part:", sp.simplify(pz), ";  sum = 0 :", ok(px + pz))

# Scope item 3, numerically: tau_sgs = <uu> - UU = L + C + R with L = <UU> - UU, C = <Uu'> + <u'U>, R = <u'u'>,
# for a non-idempotent (Gaussian) filter and for a sharp spectral cut-off; neither makes L or C vanish.
import numpy as np
rng = np.random.default_rng(1); n = 256; kx = np.fft.fftfreq(n, d=1.0/n)
spec = (rng.standard_normal(n) + 1j*rng.standard_normal(n)) / (1.0 + np.abs(kx))**1.2; spec[0] = 0
u1 = np.real(np.fft.ifft(spec)) * n
def make_filter(kind, kc=24.0):
    if kind == "gauss": G = np.exp(-(np.pi*kx/kc)**2/6.0)
    else: G = (np.abs(kx) <= kc).astype(float)
    return lambda f: np.real(np.fft.ifft(G*np.fft.fft(f)))
for kind in ("gauss", "sharp"):
    F = make_filter(kind); U = F(u1); up = u1 - U
    tau = F(u1*u1) - U*U
    L = F(U*U) - U*U; C = F(U*up) + F(up*U); Rr = F(up*up)
    Lg = F(U*U) - F(U)*F(U)                          # the Germano form, inconsistent with this C and R unless the filter is idempotent
    print("Scope 3 [%s]  |tau - (L+C+R)|max = %.1e ;  |tau - (Lg+C+R)|max = %.1e ;  |L|max/|tau|max = %.2f ;  |C|max/|tau|max = %.2f ;  |<u'>|max = %.1e"
          % (kind, np.abs(tau-(L+C+Rr)).max(), np.abs(tau-(Lg+C+Rr)).max(), np.abs(L).max()/np.abs(tau).max(), np.abs(C).max()/np.abs(tau).max(), np.abs(F(up)).max()))

# Section 8, normal stresses: carried R_xx(zbar), R_zz(zbar) give forces -R_xx' xi_{z,x} (x) and -d_z(xi_z R_zz') (z);
# their pseudomomentum projections -<xi_{x,x} X_x> and -<xi_{z,x} X_z> vanish (quadrature).
Rxx = sp.Function("Rxx")(zz); Rzz = sp.Function("Rzz")(zz)
Xn_x = -sp.diff(Rxx, zz)*(k*sp.diff(xi_z, th))/rho
Xn_z = -sp.diff(xi_z*sp.diff(Rzz, zz), zz)/rho
pn_x = -rho*avth(Xn_x*(k*sp.diff(xi_x, th))); pn_z = -rho*avth(Xn_z*(k*sp.diff(xi_z, th)))
print("Sec 8   normal stresses: x-projection =", sp.simplify(pn_x), "; z-projection =", sp.simplify(pn_z))

# Step 5, finite depth: Doppler weight (Kirby & Chen 1989) 2k cosh 2k(z+h)/sinh 2kh is proportional to the Stokes drift
# u^S = a^2 sigma k cosh 2k(z+h) / (2 sinh^2 kh), and integrates to 1.
h = sp.Symbol("h", positive=True)
W = 2*k*sp.cosh(2*k*(zz+h))/sp.sinh(2*k*h)
print("Step 5  finite depth: int_{-h}^0 weight dz =", sp.simplify(sp.integrate(W, (zz, -h, 0))),
      "; weight / Stokes profile independent of z :", sp.simplify(sp.diff(W/sp.cosh(2*k*(zz+h)), zz)) == 0)
