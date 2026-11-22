#!/usr/bin/env python3
"""Validation of tools/eos/finite_T/eos2b.py against (a) the reviewed construction module it was derived from and (b) adaptive quadrature.

  python validate_eos2b.py --input /path/to/original_eos2b.py [--json out.json]

(a) 50 random (T, rho) points (T = 5..300 MeV, rho = 0.02..16 rho_s, fixed seed): f, mu, s of eos2b.TwoPhase.fs against the original module
    (X branch and plasma: analytic s of the original xbranch / plasma; quintic region: 5-point central difference h = 0.05 MeV of the original f).
    Also checks that QUADRATURE = "legacy" reproduces the original hadron() bit for bit.
(b) nucleon n, p, s and pion p, s at T = 0.5, 1, 2, 5, 10 MeV, rho = 0.1 ... 5 rho_s against scipy.integrate.quad (epsrel 1e-13, break points at the
    Fermi edge); the same for the original fixed-panel quadrature.
(c) the Sommerfeld form below T_LOW: continuity and deviation from the direct evaluation at T = 1, 2 MeV.
"""
import argparse, importlib.util, json, os, sys, warnings
import numpy as np
from scipy.integrate import quad

here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import eos2b as E
warnings.filterwarnings("ignore")
COEX = os.path.join(here, "..", "..", "..", "data", "eos", "finite_T", "coex_pure.npz")


def load_original(path):
    spec = importlib.util.spec_from_file_location("eos2b_original", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def nucleon_ref(T, mu0):
    hc, mN, gN = E.hc, E.mN, E.gN; pre = gN/(2*np.pi**2)/hc**3; kF = np.sqrt(max(mu0**2 - mN**2, 0.0)); ef = lambda k: np.sqrt(k*k + mN*mN)
    pts = sorted(p for p in (kF*(1 + d) for d in (-.5, -.2, -.1, -.05, -.02, -.01, 0, .01, .02, .05, .1, .2, .5)) if p > 0)
    up = np.sqrt(max(mu0 + 60*T, mN + 60*T)**2 - mN**2)
    I = lambda g: quad(g, 0, up, points=pts, epsabs=0, epsrel=1e-13, limit=2000)[0]
    fp = lambda k: 1/(np.exp(min((ef(k) - mu0)/T, 700)) + 1); fm = lambda k: 1/(np.exp(min((ef(k) + mu0)/T, 700)) + 1)
    n = pre*I(lambda k: k*k*(fp(k) - fm(k)))
    p = pre*T*I(lambda k: k*k*(np.logaddexp(0, -(ef(k) - mu0)/T) + np.logaddexp(0, -(ef(k) + mu0)/T)))
    e = pre*I(lambda k: k*k*ef(k)*(fp(k) + fm(k)))
    return n, p, (e + p - mu0*n)/T


def pion_ref(T):
    hc, mpi, gpi = E.hc, E.mpi, E.gpi; pre = gpi/(2*np.pi**2)/hc**3; ef = lambda k: np.sqrt(k*k + mpi*mpi); up = np.sqrt((mpi + 80*T)**2 - mpi**2)
    p = pre*T*quad(lambda k: -k*k*np.log1p(-np.exp(-ef(k)/T)), 0, up, epsabs=0, epsrel=1e-13, limit=2000)[0]
    e = pre*quad(lambda k: k*k*ef(k)/np.expm1(ef(k)/T), 0, up, epsabs=0, epsrel=1e-13, limit=2000)[0]
    return p, (e + p)/T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True); ap.add_argument("--json"); a = ap.parse_args()
    O = load_original(a.input); out = {}
    M = E.TwoPhase(COEX); MO = O.TwoPhase(COEX)
    # (a) legacy bit-for-bit
    E.set_quadrature("legacy"); rng = np.random.default_rng(1); worst = 0.0
    for _ in range(10):
        T = rng.uniform(20, 200); rho = rng.uniform(0.02, 2, 5); h1, h2 = E.hadron(T, rho), O.hadron(T, rho)
        worst = max(worst, max(float(np.max(np.abs(h1[k] - h2[k]))) for k in ("f", "mu", "s", "p", "d2f")))
    out["legacy_vs_original_max_abs_diff_hadron"] = worst
    # (a) 50 random points, refined quadrature
    E.set_quadrature("refined"); rng = np.random.default_rng(7); Ts = rng.uniform(5, 300, 50); Rs = rng.uniform(0.02, 16, 50)*E.rhos
    rows = []
    for T, r in zip(Ts, Rs):
        g = M.fs(T, [r]); aa, bb = MO.ab(T); rr = np.array([r])
        if r <= aa:
            d, dd = MO.delta(T); X = O.xbranch(T, rr, d, dd); ref = dict(f=X["f"][0], mu=X["mu"][0], s=X["s"][0]); reg = "X"
        elif r >= bb:
            Q = O.plasma(T, rr); ref = dict(f=Q["f"][0], mu=Q["mu"][0], s=Q["s"][0]); reg = "plasma"
        else:
            f_, mu_ = MO.f(T, rr); fv = {k: MO.f(T + k*.05, rr)[0][0] for k in (-2, -1, 1, 2)}; s_ = -(8*(fv[1] - fv[-1]) - (fv[2] - fv[-2]))/(12*.05); ref = dict(f=f_[0], mu=mu_[0], s=s_); reg = "quintic"
        rows.append((reg,) + tuple(abs(g[k][0]/ref[k] - 1) for k in ("f", "mu", "s")))
    out["random50"] = {reg: dict(n=sum(1 for x in rows if x[0] == reg), max_rel_f=max(x[1] for x in rows if x[0] == reg), max_rel_mu=max(x[2] for x in rows if x[0] == reg),
                                 max_rel_s=max(x[3] for x in rows if x[0] == reg)) for reg in ("X", "quintic", "plasma") if any(x[0] == reg for x in rows)}
    out["random50_max_rel_all"] = max(max(x[1:]) for x in rows)
    # (b) quadrature
    q = {}
    for T in (0.5, 1, 2, 5, 10):
        rho = np.array([0.1, 0.2, 0.5, 1, 2, 3, 4, 5])*E.rhos
        E.set_quadrature("refined"); m0 = E.mu0_of(T, rho); n, p, s, _ = E.nucleon_gc(T, m0)
        ref = np.array([nucleon_ref(T, m) for m in m0])
        err_new = float(np.max(np.abs(np.array([n, p, s]).T/ref - 1))); rec = float(np.max(np.abs(n/rho - 1)))
        pp, sp = E.pion(T); rp, rs = pion_ref(T); err_pi = float(max(abs(pp/rp - 1), abs(sp/rs - 1)))
        E.set_quadrature("legacy"); nl, pl, sl, _ = E.nucleon_gc(T, m0); err_old = float(np.max(np.abs(np.array([nl, pl, sl]).T/ref - 1)))
        ppo, spo = E.pion(T); err_pi_old = float(max(abs(ppo/rp - 1), abs(spo/rs - 1)))
        q[str(T)] = dict(refined_nucleon_max_rel_err_n_p_s=err_new, rho_recovery=rec, refined_pion=err_pi, original_nucleon=err_old, original_pion=err_pi_old)
    E.set_quadrature("refined"); out["quadrature"] = q
    # (c) Sommerfeld
    y = np.array([0.02, 0.1, 0.5, 1, 2, 5, 10, 16]); rho = y*E.rhos
    g0, g5, g1, g2 = (M.fs(T, rho) for T in (0.0, 0.5, 1.0, 2.0)); alpha = -2*(g5["f"] - g0["f"])/0.25
    out["sommerfeld"] = dict(y=y.tolist(),
        s_jump_at_0p5=(g5["s"]/(0.5*alpha) - 1).tolist(),
        f_dev_T1=((g0["f"] - 0.5*alpha - g1["f"])/(g0["f"] - g1["f"])).tolist(), f_dev_T2=((g0["f"] - 2*alpha - g2["f"])/(g0["f"] - g2["f"])).tolist())
    print(json.dumps(out, indent=1))
    if a.json: json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
