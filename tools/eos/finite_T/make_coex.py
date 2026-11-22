#!/usr/bin/env python3
"""Regenerate the pure-phase coexistence curves (data/eos/finite_T/coex_pure.npz).

For each T the densities rhoH (hadron phase) and rhoQ (plasma phase) with equal chemical potential and equal pressure,
    mu_H(T, rhoH) = mu_Q(T, rhoQ),   p_H(T, rhoH) = p_Q(T, rhoQ),
are found by continuation in T (starting from the T = 0 pair near 4.44 / 10.45 rho_s) with scipy.optimize.fsolve.  The phases are the
pure phases of eos2b (hadron: eos2b.hadron, plasma: eos2b.plasma; T = 0: eos2b.hadron0 / plasma0).
Output: npz with T [MeV] = 0, 0.5, ..., 157, rhoH, rhoQ [fm^-3].   Units MeV, fm.

    python make_coex.py --out data/eos/finite_T/coex_pure.npz [--compare reference.npz] [--quadrature refined|legacy]
"""
import argparse, os, sys, warnings
import numpy as np
from scipy.optimize import fsolve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eos2b as E

warnings.filterwarnings("ignore")


def pure_pair(T, guess):
    """(rhoH, rhoQ) in rho_s units at temperature T, from `guess`."""
    def F(x):
        if T <= 0.0:
            H, Q = E.hadron0(np.array([x[0]*E.rhos])), E.plasma0(np.array([x[1]*E.rhos]))
        else:
            H, Q = E.hadron(T, np.array([x[0]*E.rhos])), E.plasma(T, np.array([x[1]*E.rhos]))
        return [H['mu'][0] - Q['mu'][0], H['p'][0] - Q['p'][0]]
    sol, info, ier, msg = fsolve(F, guess, xtol=1e-12, full_output=True)
    r = F(sol)    # accept a converged iteration, or a stalled one whose residuals are at the noise level (mu in MeV, p in MeV/fm^3)
    if ier != 1 and not (abs(r[0]) < 1e-7 and abs(r[1]) < 1e-9):
        raise RuntimeError(f"fsolve failed at T={T}: {msg}; residuals {r}")
    return sol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="npz to compare with (T, rhoH, rhoQ)")
    ap.add_argument("--quadrature", default="refined", choices=["refined", "legacy"])
    ap.add_argument("--Tmax", type=float, default=157.0)
    a = ap.parse_args(argv)
    E.set_quadrature(a.quadrature)
    T = np.arange(0.0, a.Tmax + 1e-9, 0.5)
    g = (4.44, 10.45); rh = []; rq = []
    for t in T:
        g = pure_pair(float(t), g); rh.append(g[0]*E.rhos); rq.append(g[1]*E.rhos)
    rh, rq = np.array(rh), np.array(rq)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, T=T, rhoH=rh, rhoQ=rq)
    print(f"wrote {a.out}: {len(T)} points, T = {T[0]}..{T[-1]}; rhoH(0), rhoQ(0) [rho_s] = {rh[0]/E.rhos:.8f}, {rq[0]/E.rhos:.8f}")
    if a.compare:
        z = np.load(a.compare)
        dH = np.abs(rh/z["rhoH"] - 1); dQ = np.abs(rq/z["rhoQ"] - 1)
        print(f"vs {os.path.basename(a.compare)}: max rel diff rhoH {dH.max():.3e} (T = {T[dH.argmax()]}), rhoQ {dQ.max():.3e} (T = {T[dQ.argmax()]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
