#!/usr/bin/env python3
"""Finite-temperature two-phase equation-of-state tables of spinodal-fluid and their Maxwell partner (work package WP2b).

Stages (each writes into --work, default data/eos/generated):
  canonical   f, mu, s = -df/dT of the construction (eos2b.TwoPhase.fs) on a non-uniform T grid for every lattice rho
              (y = rho/rho_s = 0 ... 16, step dy), parallel over T                                 -> canonical_two_phase.npz
  maxwell     for every T below the critical point: lower convex hull of f over rho >= rho_s (fine-grid hull, then the Maxwell
              conditions mu(r1) = mu(r2), p(r1) = p(r2) solved with fsolve); f linear between r1 and r2, mu = mu_coex,
              s = linear interpolation of s(r1), s(r2) (Gibbs mixture)                         -> canonical_maxwell.npz
  tables      invert eps(T) at fixed rho on the lattice x = eps/eps_s and write the WP2c table pairs (writer: write_table.py):
              eos_two_phase, eos_maxwell (full lattice) and the reduced test pair (step 0.2 in x, 0.08 in y)
  all         the three stages

Inversion: for every rho column f(T) is interpolated between the T nodes by the cubic Hermite polynomial with f' = -s (nodes carry f and s);
eps_H(T) = f_H - T f_H' is solved for the lattice value by bisection inside the bracketing interval (the node values eps(T_i) = f_i + T_i s_i
are checked to be strictly increasing); mu is interpolated with PCHIP; s = -f_H'; p0 = mu rho - f_H.  Below the cold curve every field is
continued linearly in eps through the cold value at eps_min and the first lattice value above the curve (finite, smooth, never read).
Units inside: MeV, fm; tables in GeV, fm.   Requires numpy, scipy.
"""
import argparse, json, multiprocessing as mp, os, sys, time, warnings
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, ".."))
import eos2b as E
from write_table import write_table

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(here, "..", "..", ".."))
COEX = os.path.join(ROOT, "data", "eos", "finite_T", "coex_pure.npz")
RHO_S, M_N, EPS_S = E.rhos, E.mN, E.eps_s        # fm^-3, MeV, MeV/fm^3


def t_grid():
    """Non-uniform T grid [MeV]: dense near the transition (130-160 MeV), top above the T where eps(T, rho = 0) = 40 eps_s (about 244 MeV)."""
    parts = [np.array([0.0, 0.25]), np.arange(0.5, 10.0, 0.5), np.arange(10.0, 100.0, 1.0), np.arange(100.0, 130.0, 0.5),
             np.arange(130.0, 160.0, 0.1), np.arange(160.0, 200.0, 0.5), np.arange(200.0, 250.01, 1.0)]
    return np.round(np.concatenate(parts), 6)


_M = None
def _model():
    global _M
    if _M is None:
        _M = E.TwoPhase(COEX)
    return _M


def _canon_row(args):
    T, rho = args
    g = _model().fs(float(T), rho)
    return g["f"], g["mu"], g["s"]


def stage_canonical(work, dy, nproc, Tg=None):
    Tg = t_grid() if Tg is None else Tg
    y = np.arange(int(round(16.0/dy)) + 1)*dy; rho = y*RHO_S
    t0 = time.time()
    with mp.get_context("fork").Pool(nproc) as pool:
        R = pool.map(_canon_row, [(T, rho) for T in Tg], chunksize=1)
    F, MU, S = (np.array([r[i] for r in R]) for i in range(3))
    np.savez(os.path.join(work, "canonical_two_phase.npz"), T=Tg, y=y, f=F, mu=MU, s=S)
    print(f"canonical: {len(Tg)} T values x {len(y)} rho values in {time.time()-t0:.0f} s")


# ----------------------------------------------------------------------------------------- Maxwell construction
def _hull_edges(x, f):
    """vertex indices of the lower convex hull of the points (x, f), x increasing."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2 and (f[h[-1]] - f[h[-2]])*(x[i] - x[h[-2]]) >= (f[i] - f[h[-2]])*(x[h[-1]] - x[h[-2]]):
            h.pop()
        h.append(i)
    return np.array(h)


def maxwell_pair(M, T, fine_step=0.01, min_gap=3):
    """(r1, r2) in rho_s units of the Maxwell construction over rho >= rho_s at temperature T, or None if f is convex there."""
    y = np.arange(1.0, 16.0 + 1e-9, fine_step); C = M.coeffs(T) if T >= E.T_LOW else None
    def fmu(v):        # f, mu at (T, rho); T < T_LOW: exact T = 0 limit / Sommerfeld form of eos2b.fs
        if C is None:
            g = M.fs(T, v); return g["f"], g["mu"]
        return M.f(T, v, C)
    f, mu = fmu(y*RHO_S)
    h = _hull_edges(y, f); gaps = np.diff(h)
    if gaps.max() < min_gap:
        return None
    k = gaps.argmax(); i1, i2 = h[k], h[k + 1]
    def F(v):
        ff, mm = fmu(np.array(v)*RHO_S)
        return [mm[0] - mm[1], (v[0]*RHO_S*mm[0] - ff[0]) - (v[1]*RHO_S*mm[1] - ff[1])]
    guess = (y[i1], y[i2])
    sol, info, ier, msg = fsolve(F, guess, xtol=1e-12, full_output=True)
    ok = ier == 1 or (abs(F(sol)[0]) < 1e-7 and abs(F(sol)[1]) < 1e-9)
    if ok and abs(sol[0] - guess[0]) < 3*fine_step and abs(sol[1] - guess[1]) < 3*fine_step and sol[0] < sol[1]:
        return float(sol[0]), float(sol[1]), True
    return float(guess[0]), float(guess[1]), False


def _maxwell_row(args):
    T, y = args
    M = _model(); out = dict(r1=np.nan, r2=np.nan, ok=False)
    pr = maxwell_pair(M, float(T)) if T < 150.0 else None
    if pr is None:
        return out
    r1, r2, ok = pr
    g = M.fs(float(T), np.array([r1, r2])*RHO_S)
    out.update(r1=r1, r2=r2, ok=ok, f1=g["f"][0], f2=g["f"][1], mu=g["mu"][0], mu2=g["mu"][1], s1=g["s"][0], s2=g["s"][1])
    return out


def stage_maxwell(work, nproc):
    z = np.load(os.path.join(work, "canonical_two_phase.npz")); Tg, y, F, MU, S = z["T"], z["y"], z["f"], z["mu"], z["s"]
    t0 = time.time()
    with mp.get_context("fork").Pool(nproc) as pool:
        R = pool.map(_maxwell_row, [(T, y) for T in Tg], chunksize=1)
    FM, MM, SM = F.copy(), MU.copy(), S.copy()
    hull = np.full((len(Tg), 8), np.nan); nfall = 0
    for i, r in enumerate(R):
        if not np.isfinite(r["r1"]):
            continue
        if not r["ok"]:
            nfall += 1
        r1, r2 = r["r1"], r["r2"]; sel = (y > r1) & (y < r2) & (y >= 1.0)
        lam = (y[sel] - r1)/(r2 - r1)
        FM[i, sel] = r["f1"] + r["mu"]*(y[sel] - r1)*RHO_S
        MM[i, sel] = r["mu"]
        SM[i, sel] = (1 - lam)*r["s1"] + lam*r["s2"]
        hull[i] = [r1, r2, r["mu"], r["mu"]*r1*RHO_S - r["f1"], r["f1"], r["f2"], r["s1"], r["s2"]]
    np.savez(os.path.join(work, "canonical_maxwell.npz"), T=Tg, y=y, f=FM, mu=MM, s=SM, hull=hull,
             hull_columns=np.array(["rho1_rhos", "rho2_rhos", "mu_coex_MeV", "p_coex_MeV_fm3", "f1", "f2", "s1", "s2"]))
    n = int(np.isfinite(hull[:, 0]).sum())
    Tc_hi = Tg[np.isfinite(hull[:, 0])].max() if n else float("nan")
    print(f"maxwell: {n} T rows with a hull ({nfall} fall back to the fine-grid hull vertices), highest T with hull {Tc_hi}; {time.time()-t0:.0f} s")


# ----------------------------------------------------------------------------------------- inversion
def _hermite(Ti, h, fi, f1, di, d1, T):
    """cubic Hermite value and derivative at T in [Ti, Ti + h]; di, d1 = df/dT at the ends."""
    t = (T - Ti)/h; t2, t3 = t*t, t*t*t
    h00, h10, h01, h11 = 2*t3 - 3*t2 + 1, t3 - 2*t2 + t, -2*t3 + 3*t2, t3 - t2
    d00, d10, d01, d11 = 6*t2 - 6*t, 3*t2 - 4*t + 1, -6*t2 + 6*t, 3*t2 - 2*t
    return h00*fi + h10*h*di + h01*f1 + h11*h*d1, (d00*fi + d01*f1)/h + d10*di + d11*d1


def invert_column(Tg, f, s, mu, rho, target):
    """For one rho: T, f, s, mu (and p) at the energy densities `target` (MeV/fm^3, all > eps_min and <= eps(T_max))."""
    eps = f + Tg*s
    i = np.clip(np.searchsorted(eps, target) - 1, 0, len(Tg) - 2)
    Ti, h = Tg[i], Tg[i + 1] - Tg[i]
    fi, f1, di, d1 = f[i], f[i + 1], -s[i], -s[i + 1]
    lo, hi = Ti.copy(), Ti + h
    for _ in range(60):
        T = 0.5*(lo + hi); fh, dfh = _hermite(Ti, h, fi, f1, di, d1, T)
        up = fh - T*dfh < target
        lo = np.where(up, T, lo); hi = np.where(up, hi, T)
    T = 0.5*(lo + hi); fh, dfh = _hermite(Ti, h, fi, f1, di, d1, T)
    muT = PchipInterpolator(Tg, mu)(T)
    return T, fh, -dfh, muT, muT*rho - fh


def build_arrays(z, dx, nx):
    """z: canonical npz contents; returns p0, T, mu, s (nx, ny) in GeV/fm units and eps_min (ny) plus diagnostics."""
    Tg, y, F, MU, S = z["T"], z["y"], z["f"], z["mu"], z["s"]; ny = len(y)
    x = dx*np.arange(nx); tgt = x*EPS_S
    P0, TT, MM, SS = (np.zeros((nx, ny)) for _ in range(4)); emin = np.zeros(ny); nonmono = 0; ntop = 0
    for j in range(ny):
        f, s, mu = F[:, j], S[:, j], MU[:, j]; rho = y[j]*RHO_S; eps = f + Tg*s
        if np.any(np.diff(eps) <= 0):
            nonmono += 1
        e0 = eps[0]; emin[j] = e0
        ab = tgt > e0
        if ab.any() and tgt[ab].max() > eps[-1]:
            ntop += 1
        T, fh, sh, muh, ph = invert_column(Tg, f, s, mu, rho, np.clip(tgt[ab], e0, eps[-1]))
        TT[ab, j], SS[ab, j], MM[ab, j], P0[ab, j] = T, sh, muh, ph
        # continuation below the cold curve: line through (eps_min, cold value) and the first node above the curve
        be = np.where(~ab)[0]
        if len(be) and ab.any():
            i0 = np.where(ab)[0][0]; cold = dict(T=0.0, s=0.0, mu=mu[0], p=mu[0]*rho - f[0])
            for arr, key in ((TT, "T"), (SS, "s"), (MM, "mu"), (P0, "p")):
                slope = (arr[i0, j] - cold[key])/(tgt[i0] - e0)
                arr[be, j] = cold[key] + slope*(tgt[be] - e0)
        elif len(be):
            for arr, key, v in ((TT, "T", 0.0), (SS, "s", 0.0), (MM, "mu", mu[0]), (P0, "p", mu[0]*rho - f[0])):
                arr[be, j] = v
    G = 1e-3
    return P0*G, TT*G, MM*G, SS, emin*G, dict(nonmonotone_columns=nonmono, columns_above_Tmax=ntop)


def stage_tables(work, outdir, dx, dy, tdx, tdy, only_test=False):
    os.makedirs(outdir, exist_ok=True); meta = {}
    for name, can in (("eos_two_phase", "canonical_two_phase.npz"), ("eos_maxwell", "canonical_maxwell.npz")):
        z = dict(np.load(os.path.join(work, can)))
        desc = {"eos_two_phase": "Finite-temperature two-phase equation of state (X branch + quintic Hermite join + plasma, smooth-maximum floors; "
                                 "docs/eos_finite_T.md), p0, T, mu, s on the (eps/eps_s, rho/rho_s) lattice.",
                "eos_maxwell": "Maxwell partner of eos_two_phase: for T < T_c the free energy at rho >= rho_s is replaced by its lower convex hull "
                               "(linear between rho1(T) and rho2(T), mu = mu_coex, s = Gibbs mixture); identical elsewhere (docs/eos_finite_T.md)."}[name]
        tasks = [(False, dx, dy)] if only_test else [(False, dx, dy), (True, tdx, tdy)]
        for test, ddx, ddy in tasks:
            stride = int(round(ddy/(z["y"][1] - z["y"][0]))); zz = dict(z); 
            for k in ("y", "f", "mu", "s"):
                zz[k] = z[k][..., ::stride] if k != "y" else z[k][::stride]
            nx = int(round(40.0/ddx)) + 1
            t0 = time.time(); P0, TT, MM, SS, emin, diag = build_arrays(zz, ddx, nx)
            stem = os.path.join(outdir if not test else os.path.join(ROOT, "data", "eos", "test"), name + ("_test" if test else ""))
            os.makedirs(os.path.dirname(stem), exist_ok=True)
            h = write_table(stem, P0, TT, MM, SS, emin, x0=0.0, dx=ddx, y0=0.0, dy=ddy, rho_s=RHO_S, m_N=M_N/1000.0,
                            description=desc + (" REDUCED TEST LATTICE for CI (step %g in x, %g in y)." % (ddx, ddy) if test else ""),
                            extra={"tool": "tools/eos/finite_T/make_finite_T_table.py", "construction": "smoothmax_floor",
                                   "T_grid_points": int(len(z["T"])), "T_max_MeV": float(z["T"][-1])})
            meta[os.path.basename(stem)] = dict(diag, sha256=h["sha256"], nx=nx, ny=len(zz["y"]), seconds=round(time.time() - t0, 1))
            print(stem, meta[os.path.basename(stem)])
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["canonical", "maxwell", "tables", "all"])
    ap.add_argument("--work", default=os.path.join(ROOT, "data", "eos", "generated"))
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--dx", type=float, default=0.05); ap.add_argument("--dy", type=float, default=0.02)
    ap.add_argument("--test-dx", type=float, default=0.2); ap.add_argument("--test-dy", type=float, default=0.08)
    a = ap.parse_args(argv); os.makedirs(a.work, exist_ok=True)
    if a.stage in ("canonical", "all"): stage_canonical(a.work, a.dy, a.nproc)
    if a.stage in ("maxwell", "all"): stage_maxwell(a.work, a.nproc)
    if a.stage in ("tables", "all"): stage_tables(a.work, a.work, a.dx, a.dy, a.test_dx, a.test_dy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
