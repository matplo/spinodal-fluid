"""Finite-temperature two-phase EoS construction (WP2b), vectorised.  Units: MeV, fm.

Hadron: nucleons + antinucleons (g = 4) + pions (g = 3) + mean field w(rho);
plasma: massless gluons + 2-flavour quarks with bag B = 300 MeV/fm^3 (as in the reference prototype).
The construction itself (X branch, quintic join, matching densities) is documented in docs/eos_finite_T.md.

This file is the reviewed construction module with the following additions (no change to the construction or its parameters):
  * ``QUADRATURE``: "legacy" reproduces the original fixed-panel momentum quadrature and the original ``mu0_of`` bit for bit;
    "refined" (default) adds panel break points at the Fermi edge, at E = mu0 + j T (|j| <= 30) and at E = m + j T, uses a safeguarded
    Newton iteration on ln n in ``mu0_of`` (legacy Newton diverges for T <~ 1 MeV), and breaks the pion panels at E = m_pi + j T.
  * ``hadron0``, ``plasma0``, ``TwoPhase.f0``: the exact T = 0 limit of the construction.
  * ``TwoPhase.f_low``: for 0 < T < T_LOW, f = f0(rho) - alpha(rho) T^2 / 2 (Sommerfeld form), alpha from f(T = T_LOW).
  * ``TwoPhase.fs``: f, mu and s = -df/dT (analytic in the X and plasma regions, central difference with step H_S in the quintic region).
"""
import numpy as np
from numpy.polynomial import polynomial as P

hc = 197.3269804; mN = 940.0; mpi = 140.0; gN = 4.0; gpi = 3.0; rhos = 0.153; Bag = 300.0
eps_s = mN * rhos
# mean-field constants: K = 300 MeV and saturation at rho_s (same as reference prototype)
def _ekin0(rho):
    kF = (6*np.pi**2*rho/gN)**(1/3)*hc
    return gN/(16*np.pi**2)*(kF*np.sqrt(kF**2+mN**2)*(2*kF**2+mN**2) - mN**4*np.arcsinh(kF/mN))/hc**3
_h = 1e-4
_EA = lambda r: _ekin0(r)/r - mN
_d1 = (_EA(rhos+_h) - _EA(rhos-_h))/(2*_h)
_d2 = (_EA(rhos+_h) - 2*_EA(rhos) + _EA(rhos-_h))/_h**2
KN_ = 9*rhos**2*_d2
Bw = (300.0 - KN_)/18.0
Aw = rhos*_d1 + 2*Bw
def w(rho):   c = rho/rhos; return (-Aw*c + Bw*c*c)*rho
def dw(rho):  c = rho/rhos; return -2*Aw*c + 3*Bw*c*c
def d2w(rho): c = rho/rhos; return (-2*Aw + 6*Bw*c)/rhos

_xg, _wg = np.polynomial.legendre.leggauss(64)
def _kgrid(kmax, npan):
    e = np.linspace(0, kmax, npan+1)
    k = np.concatenate([0.5*(b-a)*_xg+0.5*(b+a) for a, b in zip(e[:-1], e[1:])])
    wk = np.concatenate([0.5*(b-a)*_wg for a, b in zip(e[:-1], e[1:])])
    return k, wk
KN, WN = _kgrid(9000.0, 40); EN = np.sqrt(KN**2 + mN**2)
KP, WP = _kgrid(9000.0, 40); EP = np.sqrt(KP**2 + mpi**2)

QUADRATURE = "refined"          # "refined" | "legacy"
def set_quadrature(mode):
    global QUADRATURE
    assert mode in ("refined", "legacy"); QUADRATURE = mode

_JF = np.array([-30., -20., -12., -6., -3., 0., 3., 6., 12., 20., 30.])      # Fermi band, units of T
_JT = np.array([1., 2., 4., 8., 16., 32., 64.])                              # thermal tail from threshold, units of T
_KFIX = np.linspace(0, 9000.0, 41)
def _panels(edges):
    """edges (m, ne) sorted rows -> nodes and weights (m, (ne-1)*64); zero-width panels carry zero weight."""
    a, b = edges[:, :-1, None], edges[:, 1:, None]
    k = 0.5*(b - a)*_xg + 0.5*(b + a); wk = 0.5*(b - a)*_wg
    return k.reshape(len(edges), -1), wk.reshape(len(edges), -1)
def _kofE(E, m):  return np.sqrt(np.maximum(E*E - m*m, 0.0))
def _nucleon_nodes(T, mu0):
    """mu0 (m,) -> k, w, E of shape (m, N) with breaks at E = mu0 + j T (Fermi edge), E = mN + j T and the fixed panel edges."""
    E1 = mu0[:, None] + T*_JF[None, :]; E2 = mN + T*_JT[None, :] + 0*mu0[:, None]
    kb = np.concatenate([_kofE(E1, mN), _kofE(E2, mN), np.broadcast_to(_KFIX, (len(mu0), len(_KFIX)))], axis=1)
    kb = np.sort(np.clip(kb, 0.0, 9000.0), axis=1)
    k, wk = _panels(kb); return k, wk, np.sqrt(k*k + mN*mN)

def nucleon_gc_refined(T, mu0):
    mu0 = np.atleast_1d(np.asarray(mu0, float)); K, W, E = _nucleon_nodes(T, mu0); m = mu0[:, None]
    xp = (E - m)/T; xm = (E + m)/T
    fp = 0.5*(1 - np.tanh(xp/2)); fm = 0.5*(1 - np.tanh(xm/2))
    lp = np.logaddexp(0, -xp); lm = np.logaddexp(0, -xm)
    pre = gN/(2*np.pi**2)/hc**3; K2 = K*K*W
    n = pre*np.sum((fp - fm)*K2, axis=1); p = pre*T*np.sum((lp + lm)*K2, axis=1)
    e = pre*np.sum((fp + fm)*E*K2, axis=1); chi = pre/T*np.sum((fp*(1 - fp) + fm*(1 - fm))*K2, axis=1)
    return n, p, (e + p - mu0*n)/T, chi

def _pion_nodes(T):
    kb = np.sort(np.clip(np.concatenate([_kofE(mpi + T*_JT, mpi), _KFIX]), 0.0, 9000.0))
    k, wk = _panels(kb[None, :]); return k[0], wk[0]
def pion_refined(T):
    k, wk = _pion_nodes(T); E = np.sqrt(k*k + mpi**2); x = E/T; pre = gpi/(2*np.pi**2)/hc**3
    p = pre*T*np.sum(-np.log1p(-np.exp(-x))*k*k*wk); e = pre*np.sum(E/np.expm1(x)*k*k*wk)
    return p, (e + p)/T

def mu0_of_refined(T, rho):
    """invert n(T, mu0) = rho: safeguarded Newton iteration on ln n with a bracketing interval (rho >= 0)."""
    rho = np.atleast_1d(np.asarray(rho, float)); out = np.zeros_like(rho); pos = rho > 0
    if not pos.any(): return out
    r = rho[pos]; lo = np.zeros_like(r); hi = np.full_like(r, 3000.0)
    g = np.linspace(0.0, 3000.0, 241); ng = np.maximum(nucleon_gc_refined(T, g)[0], 1e-300)
    m = np.clip(np.interp(np.log(r), np.log(ng), g), 1.0, 2999.0)
    for it in range(80):
        n, _, _, chi = nucleon_gc_refined(T, m)
        below = n < r; lo = np.where(below, m, lo); hi = np.where(below, hi, m)
        dm = -(np.log(np.maximum(n, 1e-300)) - np.log(r))*n/np.maximum(chi, 1e-300)
        mn = m + dm; bad = ~np.isfinite(mn) | (mn <= lo) | (mn >= hi); mn = np.where(bad, 0.5*(lo + hi), mn)
        done = (np.abs(mn - m) <= 1e-13*(1 + m)) | (hi - lo < 1e-13)
        m = mn
        if done.all(): break
    out[pos] = m; return out

def nucleon_gc(T, mu0):
    """n, p, s, chi=dn/dmu0 for nucleons+antinucleons at (T, mu0[]); T > 0."""
    if QUADRATURE == "refined": return nucleon_gc_refined(T, mu0)
    mu0 = np.atleast_1d(np.asarray(mu0, float))[:, None]
    xp = (EN-mu0)/T; xm = (EN+mu0)/T
    fp = 0.5*(1-np.tanh(xp/2)); fm = 0.5*(1-np.tanh(xm/2))
    lp = np.logaddexp(0, -xp); lm = np.logaddexp(0, -xm)
    pre = gN/(2*np.pi**2)/hc**3; K2 = KN**2
    n = pre*((fp-fm)*K2) @ WN
    p = pre*T*((lp+lm)*K2) @ WN
    e = pre*((fp+fm)*EN*K2) @ WN
    chi = pre/T*((fp*(1-fp) + fm*(1-fm))*K2) @ WN
    return n, p, (e + p - mu0[:, 0]*n)/T, chi

def pion(T):
    if QUADRATURE == "refined": return pion_refined(T)
    x = EP/T; pre = gpi/(2*np.pi**2)/hc**3
    p = pre*T*(-np.log1p(-np.exp(-x))*KP**2) @ WP
    e = pre*(EP/np.expm1(x)*KP**2) @ WP
    return p, (e + p)/T

def mu0_of(T, rho):
    """invert n(T,mu0)=rho (vectorised over rho >= 0)."""
    if QUADRATURE == "refined": return mu0_of_refined(T, rho)
    rho = np.atleast_1d(np.asarray(rho, float))
    grid = np.linspace(0.0, 3000.0, 1201)
    ng = nucleon_gc(T, grid)[0]
    m = np.interp(rho, ng, grid)
    for _ in range(8):
        n, _, _, chi = nucleon_gc(T, m)
        m = np.clip(m - (n - rho)/chi, 0.0, None)
    return m

def hadron(T, rho):
    """canonical hadron phase at (T, rho[]): dict f, mu, p, s, d2f (=dmu/drho)."""
    rho = np.atleast_1d(np.asarray(rho, float))
    m0 = mu0_of(T, rho)
    n, pN, sN, chi = nucleon_gc(T, m0)
    ppi, spi = pion(T)
    return dict(f=m0*rho - pN - ppi + w(rho), mu=m0 + dw(rho), p=pN + ppi + rho*dw(rho) - w(rho),
                s=sN + spi, d2f=1.0/chi + d2w(rho))

# ---- plasma (grand canonical in mu) ----
def pQ(T, mu):  return (16*np.pi**2/90*T**4 + 12*(7*np.pi**2/360*T**4 + (mu/3)**2*T**2/12 + (mu/3)**4/(24*np.pi**2)))/hc**3 - Bag
def nQ(T, mu):  return (2/9*mu*T*T + 2/(81*np.pi**2)*mu**3)/hc**3
def sQ(T, mu):  return (4*16*np.pi**2/90*T**3 + 12*(4*7*np.pi**2/360*T**3 + (mu/3)**2*T/6))/hc**3
def chiQ(T, mu): return (2/9*T*T + 6/(81*np.pi**2)*mu**2)/hc**3
def muQ_of(T, rho):
    rho = np.atleast_1d(np.asarray(rho, float))
    # solve cubic a3 mu^3 + a1 mu - rho = 0 by Newton from a safe start
    m = np.cbrt(rho*hc**3*81*np.pi**2/2) + 0.0
    m = np.maximum(m, 1e-9)
    for _ in range(60):
        m = m - (nQ(T, m) - rho)/chiQ(T, m)
        m = np.maximum(m, 0.0)
    return m
def plasma(T, rho):
    rho = np.atleast_1d(np.asarray(rho, float)); m = muQ_of(T, rho)
    return dict(f=m*rho - pQ(T, m), mu=m, p=pQ(T, m), s=sQ(T, m), d2f=1.0/chiQ(T, m))

# ---- low-density branch: Legendre transform of the smooth maximum of p_H, p_Q ----
from scipy.interpolate import CubicSpline
def xbranch(T, rho, delta, ddelta, rho_i_max=10.0*rhos, n=400):
    """X branch at (T, rho[]) for smoothing delta(T) [MeV/fm^3] and its T-derivative.
    p_X(T,mu) = (p_H + p_Q + sqrt((p_H-p_Q)^2 + delta^2))/2 (convex in (T,mu) if delta>=0 convex).
    Returns dict f, mu, p, s, d2f.  Requires f_H convex in rho (T above the nuclear liquid-gas region)."""
    rho = np.atleast_1d(np.asarray(rho, float))
    if delta <= 1e-6:
        return hadron(T, rho)
    def at_rhoi(ri):
        H = hadron(T, ri); mu = H['mu']
        pq, nq, sq, cq = pQ(T, mu), nQ(T, mu), sQ(T, mu), chiQ(T, mu)
        D = H['p'] - pq; R = np.sqrt(D*D + delta*delta); u = D/R
        chih = 1.0/H['d2f']; dr = ri - nq
        return dict(mu=mu, p=0.5*(H['p'] + pq + R), rho=0.5*(ri + nq + u*dr),
                    s=0.5*(H['s'] + sq + u*(H['s'] - sq) + delta*ddelta/R),
                    chi=0.5*(chih + cq) + 0.5*u*(chih - cq) + 0.5*delta**2*dr**2/R**3)
    ri = np.concatenate([[0.0], np.geomspace(1e-6*rhos, rho_i_max, n-1)])
    # If f_H is concave at small rho (thermal baryon pairs at T >~ 260 MeV), the grand-canonical
    # p_H(mu) is the Legendre transform of the convex hull: a symmetric Maxwell plateau at mu = 0
    # between -rho* and +rho*, with mu_H(rho*) = 0.  Start the parametrisation at rho*.
    muH0 = hadron(T, ri)['mu']; r0 = 0.0
    if np.any(muH0[1:] < 0):
        j = np.where(muH0 < 0)[0].max()
        from scipy.optimize import brentq as _bq
        r0 = _bq(lambda x: hadron(T, np.array([x]))['mu'][0], ri[j], ri[j+1], xtol=1e-14)
        ri = np.concatenate([[r0], np.geomspace(r0*(1+1e-6), rho_i_max, n-1)])
    G = at_rhoi(ri)
    if np.any(np.diff(G['rho']) <= 0): raise RuntimeError("rho_X not monotone in rho_i (T=%g)" % T)
    if rho.max() > G['rho'][-1]: raise RuntimeError("rho_i_max too small at T=%g" % T)
    plateau = rho < G['rho'][0]
    if plateau.any():                       # mu = 0 plateau: f flat, s from the mu = 0 state
        X0 = at_rhoi(np.array([r0]))
        out = xbranch_core = None
        rest = ~plateau
        res = dict(f=np.full(rho.shape, -X0['p'][0]), mu=np.zeros(rho.shape), p=np.full(rho.shape, X0['p'][0]),
                   s=np.full(rho.shape, X0['s'][0]), d2f=np.zeros(rho.shape), rho_check=rho.copy())
        if rest.any():
            R_ = xbranch(T, rho[rest], delta, ddelta, rho_i_max, n)
            for k in res: res[k][rest] = R_[k]
        return res
    ri_t = CubicSpline(G['rho'], ri)(rho)
    for _ in range(3):                      # Newton polish on rho_i so that rho_X(rho_i) = rho exactly
        X = at_rhoi(ri_t)
        dri = (at_rhoi(ri_t*(1+1e-7)+1e-12)['rho'] - X['rho'])/(ri_t*1e-7+1e-12)
        ri_t = np.maximum(ri_t - (X['rho'] - rho)/dri, 0.0)
    X = at_rhoi(ri_t)
    return dict(f=X['mu']*rho - X['p'], mu=X['mu'], p=X['p'], s=X['s'], d2f=1.0/X['chi'], rho_check=X['rho'])

# ---- full construction ----
def cstep(x):
    """C-infinity step: 0 for x<=0, 1 for x>=1."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    with np.errstate(divide='ignore', over='ignore'):
        e0 = np.where(x > 0, np.exp(-1/np.where(x > 0, x, 1)), 0.0)
        e1 = np.where(x < 1, np.exp(-1/np.where(x < 1, 1-x, 1)), 0.0)
    return e0/(e0+e1)

class Model:
    """two-phase EoS: f = X branch (rho<=a), quintic (a<rho<b), plasma (rho>=b) at each T."""
    def __init__(self, coex_T, coex_H, coex_Q, A_f=0.5, B_f=4.0, T1=136.0, T2=150.0,
                 xH=0.9455, xQ=1.0047, k_delta=2.0, T_delta=135.0, w_delta=5.0):
        self.CH = CubicSpline(coex_T, coex_H); self.CQ = CubicSpline(coex_T, coex_Q)
        self.p = dict(A_f=A_f, B_f=B_f, T1=T1, T2=T2, xH=xH, xQ=xQ, k_delta=k_delta, T_delta=T_delta, w_delta=w_delta)
    def delta(self, T):
        p = self.p; x = (T - p['T_delta'])/p['w_delta']
        return p['k_delta']*p['w_delta']*np.logaddexp(0, x), p['k_delta']/(1+np.exp(-x))
    def ab(self, T):
        p = self.p; lam = float(cstep((T - p['T1'])/(p['T2'] - p['T1'])))
        if lam >= 1.0: return p['A_f']*rhos, p['B_f']*rhos
        a0 = p['xH']*float(self.CH(T)); b0 = p['xQ']*float(self.CQ(T))
        return (1-lam)*a0 + lam*p['A_f']*rhos, (1-lam)*b0 + lam*p['B_f']*rhos
    def coeffs(self, T):
        a, b = self.ab(T); d, dd = self.delta(T)
        X = xbranch(T, np.array([a]), d, dd); Q = plasma(T, np.array([b])); L = b - a
        M = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0],[1,L,L**2,L**3,L**4,L**5],
                      [0,1,2*L,3*L**2,4*L**3,5*L**4],[0,0,2,6*L,12*L**2,20*L**3]], float)
        c = np.linalg.solve(M, [X['f'][0], X['mu'][0], X['d2f'][0], Q['f'][0], Q['mu'][0], Q['d2f'][0]])
        return a, b, c
    def f(self, T, rho, C=None):
        rho = np.atleast_1d(np.asarray(rho, float)); a, b, c = C if C is not None else self.coeffs(T)
        out = np.empty_like(rho); mu = np.empty_like(rho)
        lo, hi = rho <= a, rho >= b; mid = ~(lo | hi)
        if lo.any():
            d, dd = self.delta(T); X = xbranch(T, rho[lo], d, dd); out[lo] = X['f']; mu[lo] = X['mu']
        if hi.any():
            Q = plasma(T, rho[hi]); out[hi] = Q['f']; mu[hi] = Q['mu']
        if mid.any():
            out[mid] = P.polyval(rho[mid]-a, c); mu[mid] = P.polyval(rho[mid]-a, P.polyder(c))
        return out, mu
    def min_f2(self, T, n=2001):
        a, b, c = self.coeffs(T); x = np.linspace(0, b-a, n); f2 = P.polyval(x, P.polyder(c, 2))
        return f2.min(), (a + x[f2.argmin()])/rhos

# ---- final WP2b construction (smooth-max floors on the matching densities) ----
DEFAULT_PARAMS = dict(xH=0.9455, xQ=1.0047, A_f=0.5, B_f=3.5875, eps=0.3, u_a=0.45, Tc_ref=142.5,
                      Tcut=(146.0, 155.0), k_delta=2.0, T_delta=135.0, w_delta=5.0)
class TwoPhase(Model):
    """Matching densities (units rho_s):
         a(T) = smax( xH (1 - u_a (T/Tc_ref)^4) rhoH_coex(T) c(T), A_f ),
         b(T) = smax( xQ rhoQ_coex(T) c(T), B_f ),
       smax(u,v) = (u+v+sqrt((u-v)^2+eps^2))/2, c(T) = 1 - cstep((T-Tcut0)/(Tcut1-Tcut0)) removes the
       pure-phase coexistence curves smoothly before they end at T_max.  B_f fixes the critical point."""
    def __init__(self, coex_file, **kw):
        z = np.load(coex_file); prm = dict(DEFAULT_PARAMS); prm.update(kw)
        super().__init__(z['T'], z['rhoH'], z['rhoQ'], A_f=prm['A_f'], B_f=prm['B_f'], xH=prm['xH'], xQ=prm['xQ'],
                         k_delta=prm['k_delta'], T_delta=prm['T_delta'], w_delta=prm['w_delta'])
        self.p.update(prm); self.Tmax_coex = float(z['T'][-1])
    def ab(self, T):
        p = self.p; c = 1.0 - float(cstep((T - p['Tcut'][0])/(p['Tcut'][1] - p['Tcut'][0])))
        Tq = min(T, self.Tmax_coex - 0.1)
        x = p['xH']*(1 - p['u_a']*(T/p['Tc_ref'])**4)*float(self.CH(Tq))/rhos*c
        y = p['xQ']*float(self.CQ(Tq))/rhos*c
        sm = lambda u, v: 0.5*(u + v + np.sqrt((u - v)**2 + p['eps']**2))
        return sm(x, p['A_f'])*rhos, sm(y, p['B_f'])*rhos


# ---- WP2b additions: exact T = 0 limit, Sommerfeld form at very low T, entropy density ----
T_LOW = 0.5      # MeV: below this temperature f = f0(rho) - alpha(rho) T^2 / 2
H_S = 0.05       # MeV: central-difference step for s = -df/dT in the quintic region

def hadron0(rho):
    """T = 0 hadron phase (degenerate nucleons, no pions): dict f, mu, p, s, d2f (= dmu/drho)."""
    rho = np.atleast_1d(np.asarray(rho, float)); r = np.maximum(rho, 1e-300)
    kF = (6*np.pi**2*r/gN)**(1/3)*hc; E_F = np.sqrt(kF**2 + mN**2)
    f = _ekin0(r) + w(r); mu = E_F + dw(r)
    return dict(f=np.where(rho > 0, f, 0.0), mu=np.where(rho > 0, mu, mN), p=rho*mu - f, s=np.zeros_like(rho),
                d2f=kF**2/(3*r*E_F) + d2w(r))

def plasma0(rho):
    """T = 0 plasma phase: dict f, mu, p, s, d2f."""
    rho = np.atleast_1d(np.asarray(rho, float)); m = np.cbrt(rho*hc**3*81*np.pi**2/2)
    return dict(f=m*rho - pQ(0.0, m), mu=m, p=pQ(0.0, m), s=sQ(0.0, m)*0, d2f=1.0/chiQ(0.0, np.maximum(m, 1e-300)))

def _quintic(a, b, X, Q):
    L = b - a
    M = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,2,0,0,0],[1,L,L**2,L**3,L**4,L**5],
                  [0,1,2*L,3*L**2,4*L**3,5*L**4],[0,0,2,6*L,12*L**2,20*L**3]], float)
    return np.linalg.solve(M, [X['f'][0], X['mu'][0], X['d2f'][0], Q['f'][0], Q['mu'][0], Q['d2f'][0]])

def _f0(self, rho):
    """f(0, rho), mu(0, rho) of the construction (X branch -> hadron0 because delta(0) ~ 1e-11 MeV/fm^3)."""
    rho = np.atleast_1d(np.asarray(rho, float)); a, b = self.ab(0.0)
    c = _quintic(a, b, hadron0(np.array([a])), plasma0(np.array([b])))
    out = np.empty_like(rho); mu = np.empty_like(rho)
    lo, hi = rho <= a, rho >= b; mid = ~(lo | hi)
    if lo.any(): H = hadron0(rho[lo]); out[lo], mu[lo] = H['f'], H['mu']
    if hi.any(): Q = plasma0(rho[hi]); out[hi], mu[hi] = Q['f'], Q['mu']
    if mid.any(): out[mid] = P.polyval(rho[mid]-a, c); mu[mid] = P.polyval(rho[mid]-a, P.polyder(c))
    return out, mu
Model.f0 = _f0

def _fs(self, T, rho):
    """f, mu, s = -df/dT at (T, rho[]).  T = 0: exact limit (s = 0).  0 < T < T_LOW: f = f0 - alpha T^2/2, mu likewise (T^2 scaling),
    s = alpha T.  T >= T_LOW: analytic s in the X (rho <= a) and plasma (rho >= b) regions, central difference (step H_S) in a < rho < b."""
    rho = np.atleast_1d(np.asarray(rho, float))
    f0, mu0 = self.f0(rho)
    if T <= 0.0: return dict(f=f0, mu=mu0, s=np.zeros_like(rho))
    if T < T_LOW:
        r = self.fs(T_LOW, rho); q = (T/T_LOW)**2
        alpha = -2.0*(r['f'] - f0)/T_LOW**2
        return dict(f=f0 - 0.5*alpha*T*T, mu=mu0 + (r['mu'] - mu0)*q, s=alpha*T)
    a, b, c = self.coeffs(T); out = dict(f=np.empty_like(rho), mu=np.empty_like(rho), s=np.empty_like(rho))
    lo, hi = rho <= a, rho >= b; mid = ~(lo | hi)
    if lo.any():
        d, dd = self.delta(T); X = xbranch(T, rho[lo], d, dd)
        out['f'][lo], out['mu'][lo], out['s'][lo] = X['f'], X['mu'], X['s']
    if hi.any():
        Q = plasma(T, rho[hi]); out['f'][hi], out['mu'][hi], out['s'][hi] = Q['f'], Q['mu'], Q['s']
    if mid.any():
        out['f'][mid] = P.polyval(rho[mid]-a, c); out['mu'][mid] = P.polyval(rho[mid]-a, P.polyder(c))
        fp = self.f(T + H_S, rho[mid])[0]; fm = self.f(T - H_S, rho[mid])[0]
        out['s'][mid] = -(fp - fm)/(2*H_S)
    return out
Model.fs = _fs
