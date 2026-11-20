#!/usr/bin/env python3
"""Reference prototype for the two-phase equation of state of
Randrup, PRC 82, 034902 (arXiv:1007.1448, App. A) and Steinheimer & Randrup (arXiv:1209.2462).

Scope: pure phases (pi+N+Nbar gas with interaction w(rho); bag-model QGP), their phase crossing,
the T=0 quintic spline, interface tension, collision kinematics.  NOT in scope: the temperature
dependence of the spline matching densities (unspecified in the papers; see plan, WP2b).
Units: MeV, fm.  Writes eos_reference_values.json next to this file.
"""
import json, os, warnings
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve
from scipy.special import zeta
warnings.filterwarnings("ignore")
hc = 197.3269804; mN = 940.0; mpi = 140.0; gN = 4.0; gpi = 3.0; rhos = 0.153; Bag = 300.0
eps_s = mN * rhos

# ---------------- hadronic phase ----------------
def kF(rho): return (6 * np.pi**2 * rho / gN) ** (1 / 3) * hc
def ekin0(rho):
    k = kF(rho); E = np.hypot(k, mN)
    return gN / (16 * np.pi**2) * (k * E * (2 * k * k + mN * mN) - mN**4 * np.arcsinh(k / mN)) / hc**3
def EA_kin(rho): return ekin0(rho) / rho - mN
_h = 1e-4
_d1 = (EA_kin(rhos + _h) - EA_kin(rhos - _h)) / (2 * _h)
_d2 = (EA_kin(rhos + _h) - 2 * EA_kin(rhos) + EA_kin(rhos - _h)) / _h**2
KN = 9 * rhos**2 * _d2                 # Fermi-motion part of K = 9 rho^2 d^2(E/A)/d rho^2
Bw = (300.0 - KN) / 18.0               # K = K_N + 18 B = 300 MeV
Aw = rhos * _d1 + 2 * Bw               # saturation: d(E/A)/d rho = 0 at rho_s
def w(rho): c = rho / rhos; return (-Aw * c + Bw * c * c) * rho
def dw(rho): c = rho / rhos; return -2 * Aw * c + 3 * Bw * c * c
def p_pi(T):
    if T < 1e-6: return 0.0
    f = lambda e: np.sqrt(e * e - mpi * mpi) * e * np.log1p(-np.exp(-e / T))
    return -gpi * T / (2 * np.pi**2) * quad(f, mpi, mpi + 60 * T, limit=200)[0] / hc**3
def pN(T, mu0):
    if T < 1e-6:
        if mu0 <= mN: return 0.0
        k = np.sqrt(mu0**2 - mN**2)
        return gN / (48 * np.pi**2) * (k * mu0 * (2 * k * k - 3 * mN * mN) + 3 * mN**4 * np.arcsinh(k / mN)) / hc**3
    f = lambda e: np.sqrt(e * e - mN * mN) * e * (np.logaddexp(0, -(e - mu0) / T) + np.logaddexp(0, -(e + mu0) / T))
    up = max(mu0, mN) + 60 * T
    return gN * T / (2 * np.pi**2) * quad(f, mN, up, limit=300, points=[min(max(mu0, mN), up)])[0] / hc**3
def rhoN(T, mu0):
    if T < 1e-6: return 0.0 if mu0 <= mN else gN * (mu0**2 - mN**2) ** 1.5 / (6 * np.pi**2) / hc**3
    f = lambda e: np.sqrt(e * e - mN * mN) * e * (1 / (np.exp((e - mu0) / T) + 1) - 1 / (np.exp((e + mu0) / T) + 1))
    up = max(mu0, mN) + 60 * T
    return gN / (2 * np.pi**2) * quad(f, mN, up, limit=300, points=[min(max(mu0, mN), up)])[0] / hc**3
def mu0_of(T, rho): return brentq(lambda m: rhoN(T, m) - rho, 0.0, 6000.0, xtol=1e-9)
def fH(T, rho): m0 = mu0_of(T, rho); return m0 * rho - pN(T, m0) - p_pi(T) + w(rho)
def muH(T, rho): return mu0_of(T, rho) + dw(rho)
def pH(T, rho): m0 = mu0_of(T, rho); return pN(T, m0) + p_pi(T) + rho * dw(rho) - w(rho)

# ---------------- deconfined phase ----------------
def pQ(T, mu): return (16 * np.pi**2 / 90 * T**4 + 12 * (7 * np.pi**2 / 360 * T**4 + (mu / 3) ** 2 * T**2 / 12 + (mu / 3) ** 4 / (24 * np.pi**2))) / hc**3 - Bag
def rhoQ(T, mu): return (2 / 9 * mu * T * T + 2 / (81 * np.pi**2) * mu**3) / hc**3
def muQ(T, rho): return brentq(lambda m: rhoQ(T, m) - rho, 0, 1e4, xtol=1e-9)
def fQ(T, rho): m = muQ(T, rho); return m * rho - pQ(T, m)

def pure_coex(T, guess):
    F = lambda x: [muH(T, x[0]) - muQ(T, x[1]), pH(T, x[0]) - pQ(T, muQ(T, x[1]))]
    return fsolve(F, guess, xtol=1e-10)
def coex_T(T):
    g = (4.44 * rhos, 10.45 * rhos)
    for Tm in np.linspace(0, T, max(2, int(T / 4) + 1)): g = pure_coex(Tm, g)
    return g

# ---------------- quintic spline in rho at fixed T ----------------
def build(T, a, b):
    hh = 1e-3 * rhos
    d2H = (muH(T, a + hh) - muH(T, a - hh)) / (2 * hh)
    mq = muQ(T, b); d2Q = 1.0 / ((2 / 9 * T * T + 6 / (81 * np.pi**2) * mq**2) / hc**3)
    L = b - a
    M = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 2, 0, 0, 0],
                  [1, L, L**2, L**3, L**4, L**5], [0, 1, 2 * L, 3 * L**2, 4 * L**3, 5 * L**4], [0, 0, 2, 6 * L, 12 * L**2, 20 * L**3]], float)
    c = np.linalg.solve(M, [fH(T, a), muH(T, a), d2H, fQ(T, b), mq, d2Q])
    return dict(T=T, a=a, b=b, c=c)
def f_all(S, rho, der=0):
    T = S['T']
    if rho <= S['a']: return fH(T, rho) if der == 0 else muH(T, rho)
    if rho >= S['b']: return fQ(T, rho) if der == 0 else muQ(T, rho)
    c = S['c']
    for _ in range(der): c = P.polyder(c)
    return P.polyval(rho - S['a'], c)
def min_f2(S, n=1501):
    xs = np.linspace(S['a'], S['b'], n); return P.polyval(xs - S['a'], P.polyder(S['c'], 2)).min()
def analyse(S):
    xs = np.linspace(S['a'], S['b'], 2001); f2 = P.polyval(xs - S['a'], P.polyder(S['c'], 2))
    r = dict(min_f2=float(f2.min()))
    if f2.min() < 0:
        idx = np.where(np.diff(np.sign(f2)) != 0)[0]
        mu = lambda x: f_all(S, x, 1); pr = lambda x: x * mu(x) - f_all(S, x, 0)
        r1, r2 = fsolve(lambda x: [mu(x[0]) - mu(x[1]), pr(x[0]) - pr(x[1])],
                        (0.5 * (S['a'] + xs[idx[0]]), xs[idx[-1]] + 0.3 * (S['b'] - xs[idx[-1]])), xtol=1e-11)
        cs = np.linspace(r1, r2, 4001)
        df = np.array([f_all(S, x, 0) - (f_all(S, r1, 0) + (x - r1) * mu(r1)) for x in cs])
        r.update(rhoA=xs[idx[0]] / rhos, rhoB=xs[idx[-1]] / rhos, rho1=r1 / rhos, rho2=r2 / rhos, mu_coex=mu(r1), p_coex=pr(r1),
                 dfmax=df.max(), sigma_over_a=np.trapezoid(np.sqrt(2 * eps_s * np.clip(df, 0, None)), cs / rhos))
    return {k: float(v) for k, v in r.items()}

if __name__ == "__main__":
    out = {"units": "MeV, fm", "constants": dict(hbarc=hc, mN=mN, mpi=mpi, gN=gN, gpi=gpi, rho_s=rhos, eps_s=eps_s, bag=Bag)}
    EF = kF(rhos) ** 2 / (2 * mN)
    out["hadron_interaction"] = dict(A=Aw, B=Bw, K_N_relativistic=KN, K_N_paper_NR=-1.2 * EF, E_over_A_saturation=EA_kin(rhos) - Aw + Bw,
                                     A_NR_variant=0.4 * EF + 2 * (300 + 1.2 * EF) / 18, B_NR_variant=(300 + 1.2 * EF) / 18,
                                     p_at_saturation_T0=pH(0.0, rhos))
    Tmax = brentq(lambda T: p_pi(T) + pN(T, 0.0) - pQ(T, 0.0), 120, 200)
    out["T_max"] = dict(prototype=Tmax, paper=156.4, p_pi_at_156p4=p_pi(156.4), p_NNbar_at_156p4=pN(156.4, 0.0), p_Q_at_156p4=pQ(156.4, 0.0))
    pc = {}
    for T in (0.0, 40.0, 70.0, 100.0, 120.0, 130.0, 142.5, 150.0):
        g = coex_T(T); pc[str(T)] = dict(rho_H=g[0] / rhos, rho_Q=g[1] / rhos, mu=muH(T, g[0]), p=pH(T, g[0]))
    out["pure_phase_coexistence"] = pc
    S0 = build(0.0, 4.2 * rhos, 10.5 * rhos); R0 = analyse(S0)
    R0["matching_densities_rho_s"] = [4.2, 10.5]
    R0["sigma0_MeV_fm2"] = {str(a): R0["sigma_over_a"] * a for a in (0.01, 0.033, 0.05)}
    out["T0_spline"] = R0
    # temperature dependence experiments (multiplicative widening u about the pure-phase coexistence densities)
    xH0, yQ0 = 4.2 / pc["0.0"]["rho_H"], 10.5 / pc["0.0"]["rho_Q"]
    us = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    scan = {}
    for T in (100.0, 120.0, 130.0, 138.0, 142.5, 146.0, 150.0, 153.0, 156.0):
        g = coex_T(T); scan[str(T)] = [float(min_f2(build(T, xH0 * (1 - u) * g[0], yQ0 * (1 + u) * g[1]))) for u in us]
    g = coex_T(142.5)
    uc = brentq(lambda u: min_f2(build(142.5, xH0 * (1 - u) * g[0], yQ0 * (1 + u) * g[1])), 0.0, 0.3)
    S = build(142.5, xH0 * (1 - uc) * g[0], yQ0 * (1 + uc) * g[1]); xs = np.linspace(S['a'], S['b'], 3001)
    out["T_dependence_experiments"] = dict(u_grid=us, min_f2_by_T=scan, u_c_at_142p5=uc,
        rho_c_over_rho_s=float(xs[P.polyval(xs - S['a'], P.polyder(S['c'], 2)).argmin()] / rhos),
        matching_at_Tc_rho_s=[S['a'] / rhos, S['b'] / rhos])
    # fixed (temperature-independent) matching densities a, b in units of rho_s
    Tfix = (142.5, 146.0, 150.0, 154.0, 157.0, 165.0, 180.0, 200.0)
    out["fixed_matching_experiments"] = dict(T=list(Tfix), min_f2={"%.2f-%.2f" % (a, b): [float(min_f2(build(T, a * rhos, b * rhos), 1201)) for T in Tfix]
        for a, b in ((1.5, 4.5), (1.5, 6.0), (1.0, 5.0), (1.0, 7.0), (0.5, 5.0), (0.5, 8.0), (2.0, 7.0))})
    # T_max under variations of the inputs
    def Tmax_var(mpi_=140.0, nuc=True, bag=300.0):
        def ppi(T):
            f = lambda e: np.sqrt(e * e - mpi_**2) * e * np.log1p(-np.exp(-e / T))
            return -gpi * T / (2 * np.pi**2) * quad(f, mpi_, mpi_ + 60 * T, limit=200)[0] / hc**3
        return brentq(lambda T: ppi(T) + (pN(T, 0.0) if nuc else 0.0) - (pQ(T, 0.0) + Bag - bag), 120, 200)
    out["T_max"]["variants"] = dict(no_nucleons=Tmax_var(nuc=False), mpi_138=Tmax_var(mpi_=138.0), bag_220MeV_4=Tmax_var(bag=220.0**4 / hc**3))
    # development EoS "cold_thermal": p0 = p_c(rho) + (G-1)(eps - eps_c(rho)), eps_c = f_{T=0}; analytic sound speed and growth-rate maximum
    def ct_point(c, eth_es, G=4.0 / 3.0):
        rho = c * rhos; f = f_all(S0, rho, 0); mu = f_all(S0, rho, 1); f2 = P.polyval(rho - S0['a'], P.polyder(S0['c'], 2))
        pc = rho * mu - f; eth = eth_es * eps_s; h = f + pc + G * eth
        vs2 = (G - 1) + (rho / h) * (rho * f2 - (G - 1) * mu)
        r = dict(rho_over_rho_s=c, eth_over_eps_s=eth_es, eps_over_eps_s=(f + eth) / eps_s, p0=pc + (G - 1) * eth, h=h, vs2=vs2)
        if vs2 < 0:
            for a in (0.033, 0.05):
                D = a * a * (eps_s / h) * c * c
                r["a_%g" % a] = dict(k_opt=float(np.sqrt(-vs2 / (2 * D))), gamma_max=float(-vs2 / (2 * np.sqrt(D))), k_cut=float(np.sqrt(-vs2 / D)))
        return r
    out["cold_thermal_points"] = [ct_point(c, e) for c in (5.6, 5.8, 6.0, 6.5, 7.0, 8.0, 8.4) for e in (0.0, 0.5, 2.0)]
    # kinematics (fixed target, per nucleon kinetic energy Elab in GeV)
    kin = {}
    for E in (1, 2, 3, 4, 5, 6, 10):
        s = 2 * mN**2 + 2 * mN * (E * 1000 + mN); g = np.sqrt(s) / (2 * mN); bg = np.sqrt(g * g - 1)
        kin[str(E)] = dict(sqrt_sNN_GeV=np.sqrt(s) / 1000, gamma_cm=g, rho_streaming_over_rho_s=2 * g, eps_streaming_over_eps_s=2 * g * g,
                           t_init_fm_R7p11=2 * 1.2 * 208 ** (1 / 3) / bg, t_init_fm_R6p62=2 * 6.62 / bg)
    out["kinematics"] = kin
    c0 = (34 * zeta(3) / np.pi**2) ** (1 / 3) / (4 * np.pi)
    g0 = 1 / 1.38; k0 = 2.0; vs = np.sqrt(2) * g0 / k0
    out["linear_response_1007_1448"] = dict(c0=float(c0), fig2_ideal_gamma0=g0, fig2_ideal_k0=k0,
        implied_abs_vs=vs, implied_C_rho2_over_h_fm2=vs * vs / (2 * k0 * k0))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eos_reference_values.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
