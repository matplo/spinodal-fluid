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
are checked to be strictly increasing); mu by the cubic Hermite polynomial with dmu/dT = -ds/drho; s = -f_H'; p0 = mu rho - f_H.  Below the cold curve every field is
continued linearly in eps through the cold value at eps_min and the first lattice value above the curve (finite, smooth, never read).
Units inside: MeV, fm; tables in GeV, fm.   Requires numpy, scipy.
"""
import argparse, json, multiprocessing as mp, os, sys, time, warnings
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, fsolve

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, ".."))
import eos2b as E
from write_table import write_table

warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(here, "..", "..", ".."))
COEX = os.path.join(ROOT, "data", "eos", "finite_T", "coex_pure.npz")
RHO_S, M_N, EPS_S = E.rhos, E.mN, E.eps_s        # fm^-3, MeV, MeV/fm^3


def t_grid():
    """Non-uniform T grid [MeV], refined until the change of the tables under halving of the grid is small (docs/eos_finite_T.md):
    steps 0.05 (0-4), 0.1 (4-10), 0.25 (10-100), 0.0625 (100-130), 0.025 (130-142), 0.01 (142-143) and 0.0025 (142.3-142.6), 0.05 (143-160), 0.125 (160-200), 1.0 (200-250);
    the top is above the T where eps(T, rho = 0) = 40 eps_s (about 244 MeV)."""
    parts = [np.arange(0.0, 4.0, 0.05), np.arange(4.0, 10.0, 0.1), np.arange(10.0, 100.0, 0.25), np.arange(100.0, 130.0, 0.0625),
             np.arange(130.0, 142.0, 0.025), np.arange(142.0, 143.0, 0.01), np.arange(142.3, 142.6, 0.0025), np.arange(143.0, 160.0, 0.05), np.arange(160.0, 200.0, 0.125),
             np.arange(200.0, 250.01, 1.0)]
    return np.unique(np.round(np.concatenate(parts), 6))


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
    """f, mu, s on the T grid for every lattice rho; rows already present in canonical_two_phase.npz are kept (incremental)."""
    Tg = t_grid() if Tg is None else np.asarray(Tg, float)
    y = np.arange(int(round(16.0/dy)) + 1)*dy; rho = y*RHO_S
    path = os.path.join(work, "canonical_two_phase.npz"); old = dict(np.load(path)) if os.path.exists(path) else None
    have = set(np.round(old["T"], 6)) if old is not None else set()
    new = np.array([T for T in Tg if round(float(T), 6) not in have])
    t0 = time.time()
    if len(new):
        with mp.get_context("fork").Pool(nproc) as pool:
            R = pool.map(_canon_row, [(T, rho) for T in new], chunksize=1)
        F, MU, S = (np.array([r[i] for r in R]) for i in range(3))
        if old is not None:
            T_all = np.concatenate([old["T"], new]); F, MU, S = (np.concatenate([old[k], a_]) for k, a_ in (("f", F), ("mu", MU), ("s", S)))
        else:
            T_all = new
        o = np.argsort(T_all); np.savez(path, T=T_all[o], y=y, f=F[o], mu=MU[o], s=S[o])
    print(f"canonical: {len(new)} new T rows x {len(y)} rho values in {time.time()-t0:.0f} s (total {len(have) + len(new)})")


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
    """Maxwell canonical arrays; rows already present in canonical_maxwell.npz are kept (incremental)."""
    z = np.load(os.path.join(work, "canonical_two_phase.npz")); Tg, y, F, MU, S = z["T"], z["y"], z["f"], z["mu"], z["s"]
    path = os.path.join(work, "canonical_maxwell.npz"); old = dict(np.load(path)) if os.path.exists(path) else None
    row_of = {round(float(T), 6): i for i, T in enumerate(old["T"])} if old is not None else {}
    new_idx = [i for i, T in enumerate(Tg) if round(float(T), 6) not in row_of]
    t0 = time.time()
    with mp.get_context("fork").Pool(nproc) as pool:
        R = pool.map(_maxwell_row, [(Tg[i], y) for i in new_idx], chunksize=1) if new_idx else []
    FM, MM, SM = F.copy(), MU.copy(), S.copy(); hull = np.full((len(Tg), 8), np.nan); nfall = 0
    for i, T in enumerate(Tg):
        k = row_of.get(round(float(T), 6))
        if k is not None:
            FM[i], MM[i], SM[i], hull[i] = old["f"][k], old["mu"][k], old["s"][k], old["hull"][k]
    for i, r in zip(new_idx, R):
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
    np.savez(path, T=Tg, y=y, f=FM, mu=MM, s=SM, hull=hull,
             hull_columns=np.array(["rho1_rhos", "rho2_rhos", "mu_coex_MeV", "p_coex_MeV_fm3", "f1", "f2", "s1", "s2"]))
    n = int(np.isfinite(hull[:, 0]).sum())
    Tc_hi = Tg[np.isfinite(hull[:, 0])].max() if n else float("nan")
    print(f"maxwell: {len(new_idx)} new rows; {n} T rows with a hull ({nfall} new fall back to the fine-grid hull vertices), highest T with hull {Tc_hi}; {time.time()-t0:.0f} s")


# ----------------------------------------------------------------------------------------- inversion
def _hermite(Ti, h, fi, f1, di, d1, T):
    """cubic Hermite value and derivative at T in [Ti, Ti + h]; di, d1 = df/dT at the ends."""
    t = (T - Ti)/h; t2, t3 = t*t, t*t*t
    h00, h10, h01, h11 = 2*t3 - 3*t2 + 1, t3 - 2*t2 + t, -2*t3 + 3*t2, t3 - t2
    d00, d10, d01, d11 = 6*t2 - 6*t, 3*t2 - 4*t + 1, -6*t2 + 6*t, 3*t2 - 2*t
    return h00*fi + h10*h*di + h01*f1 + h11*h*d1, (d00*fi + d01*f1)/h + d10*di + d11*d1


def dmu_dT_two_phase(z):
    """d mu / d T = -d s / d rho on the canonical rows: 4th-order difference along rho (central inside, one-sided at the ends); 0 at rho = 0;
    spline derivative of mu(T) for rho < 0.5 rho_s."""
    s = z["s"]; h = (z["y"][1] - z["y"][0])*RHO_S; d = np.empty_like(s)
    d[:, 2:-2] = (-s[:, 4:] + 8*s[:, 3:-1] - 8*s[:, 1:-3] + s[:, :-4])/(12*h)
    d[:, 1] = (-3*s[:, 0] - 10*s[:, 1] + 18*s[:, 2] - 6*s[:, 3] + s[:, 4])/(12*h)
    d[:, -2] = (3*s[:, -1] + 10*s[:, -2] - 18*s[:, -3] + 6*s[:, -4] - s[:, -5])/(12*h)
    d[:, -1] = (25*s[:, -1] - 48*s[:, -2] + 36*s[:, -3] - 16*s[:, -4] + 3*s[:, -5])/(12*h)
    d = -d; d[:, 0] = 0.0
    # rho < 0.5 rho_s: s(rho) varies on the scale of the thermal baryon-pair density (less than the lattice step at high T), no kinks occur there;
    # use the derivative of the cubic spline of mu(T) of the column instead
    for j in np.where((z["y"] < 0.5) & (z["y"] > 0))[0]:
        d[:, j] = CubicSpline(z["T"], z["mu"][:, j]).derivative()(z["T"])
    return d


def dmu_dT_maxwell(z_m, dm_two):
    """Same for the Maxwell table: exact Gibbs-mixture value -(s2 - s1)/(r2 - r1) inside the hull, the two-phase value outside."""
    dm = dm_two.copy(); hull, y = z_m["hull"], z_m["y"]
    for i in range(len(z_m["T"])):
        if np.isfinite(hull[i, 0]):
            sel = (y > hull[i, 0]) & (y < hull[i, 1]) & (y >= 1.0)
            dm[i, sel] = -(hull[i, 7] - hull[i, 6])/((hull[i, 1] - hull[i, 0])*RHO_S)
    return dm


# ----------------------------------------------------------------------------------------- kink nodes
# f is C2 across the matching densities a(T), b(T) (and f_M is C1 across the hull edges r1(T), r2(T)), so at fixed rho the T dependence
# has kinks at the temperatures where rho = a(T), b(T), r1(T), r2(T).  Cubic Hermite interpolation in a cell that contains a kink is only
# O(h^2) accurate; the kink temperatures are therefore added as extra nodes of the column of that rho and the model is evaluated exactly
# there (f, mu, s, and dmu/dT = -ds/drho by a 5-point difference in rho).
DY_FD = 1e-3      # rho_s: step of the 5-point difference in rho for dmu/dT at the kink nodes


def _kink_two_phase(args):
    T, y = args
    M = _model(); r5 = (y + DY_FD*np.arange(-2, 3))*RHO_S; g = M.fs(float(T), r5); sv = g["s"]
    dmu = -(8*(sv[3] - sv[1]) - (sv[4] - sv[0]))/(12*DY_FD*RHO_S)
    return dict(f=g["f"][2], mu=g["mu"][2], s=sv[2], dmuL=dmu, dmuR=dmu)


def _kink_hull(args):
    T, y, edge, inside_left = args
    M = _model(); pr = maxwell_pair(M, float(T))
    if pr is None:
        return None
    r1, r2, _ = pr; g = M.fs(float(T), np.array([r1, r2])*RHO_S)
    lam = (y - r1)/(r2 - r1)
    f = g["f"][0] + g["mu"][0]*(y - r1)*RHO_S; s = (1 - lam)*g["s"][0] + lam*g["s"][1]
    d_in = -(g["s"][1] - g["s"][0])/((r2 - r1)*RHO_S)
    r5 = (y + DY_FD*np.arange(-2, 3))*RHO_S; sv = M.fs(float(T), r5)["s"]; d_out = -(8*(sv[3] - sv[1]) - (sv[4] - sv[0]))/(12*DY_FD*RHO_S)
    return dict(f=f, mu=g["mu"][0], s=s, dmuL=d_in if inside_left else d_out, dmuR=d_out if inside_left else d_in)


def kink_lists(M, y, hull_T=None, r1=None, r2=None, Tmax=250.0):
    """(T, j) pairs of the matching-density kinks (kind 'a','b') and, if hull rows are given, of the hull edges (kind 'e1','e2')."""
    Tf = np.arange(E.T_LOW + 1e-3, Tmax + 1e-9, 0.02); AB = np.array([M.ab(T) for T in Tf])/RHO_S; out = []
    for col, kind in ((0, "a"), (1, "b")):
        v = AB[:, col]
        for j in range(1, len(y)):
            sg = np.sign(v - y[j])
            for i in np.where(sg[1:]*sg[:-1] < 0)[0]:
                Tk = brentq(lambda T: M.ab(T)[col]/RHO_S - y[j], Tf[i], Tf[i + 1], xtol=1e-12); out.append((Tk, j, kind))
    hl = []
    if hull_T is not None:
        s1, s2 = CubicSpline(hull_T, r1), CubicSpline(hull_T, r2); Th = np.arange(hull_T[0] + 1e-3, hull_T[-1], 0.02)
        for sp, kind in ((s1, "e1"), (s2, "e2")):
            v = sp(Th)
            for j in range(1, len(y)):
                if y[j] < 1.0: continue
                sg = np.sign(v - y[j])
                for i in np.where(sg[1:]*sg[:-1] < 0)[0]:
                    Tk = brentq(lambda T: float(sp(T)) - y[j], Th[i], Th[i + 1], xtol=1e-12)
                    inside_left = bool((s1(Tk - 0.005) < y[j] < s2(Tk - 0.005)))
                    hl.append((Tk, j, kind, inside_left))
    return out, hl


def stage_kinks(work, nproc):
    z = np.load(os.path.join(work, "canonical_two_phase.npz")); y = z["y"]; zm = np.load(os.path.join(work, "canonical_maxwell.npz"))
    ok = np.isfinite(zm["hull"][:, 0]); hT, r1, r2 = zm["T"][ok], zm["hull"][ok, 0], zm["hull"][ok, 1]
    M = _model(); ab, hl = kink_lists(M, y, hT, r1, r2); t0 = time.time()
    with mp.get_context("fork").Pool(nproc) as pool:
        R_ab = pool.map(_kink_two_phase, [(T, y[j]) for T, j, kind in ab], chunksize=1)
        R_hl = pool.map(_kink_hull, [(T, y[j], kind, il) for T, j, kind, il in hl], chunksize=1)
    sp1, sp2 = CubicSpline(hT, r1), CubicSpline(hT, r2)
    def pack(rows, items):
        keys = ("f", "mu", "s", "dmuL", "dmuR"); good = [(it, r) for it, r in zip(items, rows) if r is not None]
        return dict(T=np.array([it[0] for it, r in good]), j=np.array([it[1] for it, r in good], int), **{k: np.array([r[k] for it, r in good]) for k in keys})
    np.savez(os.path.join(work, "kinks_two_phase.npz"), **pack(R_ab, ab))
    # Maxwell: matching-density kinks that lie outside the hull, plus the hull-edge nodes
    keep = [i for i, (T, j, kind) in enumerate(ab) if not (T <= hT[-1] and sp1(T) < y[j] < sp2(T) and y[j] >= 1.0)]
    mx_items = [ab[i] for i in keep] + [h[:3] for h in hl]; mx_rows = [R_ab[i] for i in keep] + R_hl
    np.savez(os.path.join(work, "kinks_maxwell.npz"), **pack(mx_rows, mx_items))
    print(f"kinks: {len(ab)} matching-density kinks, {len(hl)} hull-edge kinks ({sum(r is None for r in R_hl)} without hull); {time.time()-t0:.0f} s")


def _kink_columns(work, name, stride):
    """per-column kink dicts of the table lattice (column jj of the subset <-> j = jj*stride of the full lattice)."""
    k = dict(np.load(os.path.join(work, "kinks_" + name + ".npz"))); cols = {}
    for jj in np.unique(k["j"]):
        if jj % stride: continue
        sel = k["j"] == jj; cols[int(jj)//stride] = {q: k[q][sel] for q in ("T", "f", "mu", "s", "dmuL", "dmuR")}
    return cols


# ----------------------------------------------------------------------------------------- inversion
def _hermite(Ti, h, fi, f1, di, d1, T):
    """cubic Hermite value and derivative at T in [Ti, Ti + h]; di, d1 = df/dT at the ends."""
    t = (T - Ti)/h; t2, t3 = t*t, t*t*t
    h00, h10, h01, h11 = 2*t3 - 3*t2 + 1, t3 - 2*t2 + t, -2*t3 + 3*t2, t3 - t2
    d00, d10, d01, d11 = 6*t2 - 6*t, 3*t2 - 4*t + 1, -6*t2 + 6*t, 3*t2 - 2*t
    return h00*fi + h10*h*di + h01*f1 + h11*h*d1, (d00*fi + d01*f1)/h + d10*di + d11*d1


def column_nodes(Tg, f, s, mu, dmu, kn):
    """node arrays of one column: the global T rows plus the exactly evaluated kink nodes (dmu from the left / right of each node) and the
    kink mask."""
    if kn is None or len(kn["T"]) == 0:
        return Tg, f, s, mu, dmu, dmu, np.zeros(len(Tg), bool)
    keep = np.min(np.abs(kn["T"][:, None] - Tg[None, :]), axis=1) > 1e-7
    T = np.concatenate([Tg, kn["T"][keep]]); o = np.argsort(T, kind="stable"); cat = lambda a, b: np.concatenate([a, b[keep]])[o]
    km = np.concatenate([np.zeros(len(Tg), bool), np.ones(int(keep.sum()), bool)])[o]
    out = [T[o], cat(f, kn["f"]), cat(s, kn["s"]), cat(mu, kn["mu"]), cat(dmu, kn["dmuL"]), cat(dmu, kn["dmuR"]), km]
    while True:       # a kink node whose T was located with finite accuracy (hull edges close to T_c) may break the monotonicity of eps(T): drop it
        d = np.diff(out[1] + out[0]*out[2]); bad = np.where(d <= 0)[0]
        drop = [i for i in np.r_[bad, bad + 1] if out[6][i]]
        if not drop:
            return tuple(out)
        m = np.ones(len(out[0]), bool); m[drop[0]] = False; out = [a_[m] for a_ in out]


def segment_dmu(T, mu, dL, dR, km):
    """dmu/dT at the nodes of a column from cubic splines of mu(T) on the segments between kink nodes, clamped at the segment ends with the
    given derivatives (exact one-sided values at kink nodes, the rho-difference value at the ends of the column); returns (left, right)."""
    n = len(T); idx = np.unique(np.r_[0, np.where(km)[0], n - 1]); dl, dr = dL.copy(), dR.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a < 2:
            continue
        d = CubicSpline(T[a:b + 1], mu[a:b + 1], bc_type=((1, dR[a]), (1, dL[b]))).derivative()(T[a + 1:b]); dr[a + 1:b] = d; dl[a + 1:b] = d
    return dl, dr


def invert_column(Tg, f, s, mu, dmuL, dmuR, rho, target):
    """For one rho: T, f, s, mu and p at the energy densities `target` (MeV/fm^3, all > eps_min and <= eps(T_max)); f and mu by cubic Hermite
    interpolation in T (derivatives -s and -ds/drho)."""
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
    muT, _ = _hermite(Ti, h, mu[i], mu[i + 1], dmuR[i], dmuL[i + 1], T)
    return T, fh, -dfh, muT, muT*rho - fh


def build_arrays(z, dx, nx, dm, kinks=None, cont="linear"):   # cont: "linear" (used), "quadratic", "flat" (tested alternatives, see docs/eos_finite_T.md)
    """z: canonical npz contents (rows T, lattice y), dm: dmu/dT on the canonical rows, kinks: {column: node dict} or None;
    returns p0, T, mu, s (nx, ny) in GeV/fm units and eps_min (ny) plus diagnostics."""
    Tg0, y, F, MU, S = z["T"], z["y"], z["f"], z["mu"], z["s"]; ny = len(y)
    x = dx*np.arange(nx); tgt = x*EPS_S
    P0, TT, MM, SS = (np.zeros((nx, ny)) for _ in range(4)); emin = np.zeros(ny); nonmono = 0; ntop = 0
    for j in range(ny):
        Tg, f, s, mu, dmuL, dmuR, km = column_nodes(Tg0, F[:, j], S[:, j], MU[:, j], dm[:, j], (kinks or {}).get(j))
        dmuL, dmuR = segment_dmu(Tg, mu, dmuL, dmuR, km); rho = y[j]*RHO_S; eps = f + Tg*s
        if np.any(np.diff(eps) <= 0):
            nonmono += 1
        e0 = eps[0]; emin[j] = e0
        ab = tgt > e0
        if ab.any() and tgt[ab].max() > eps[-1]:
            ntop += 1
        T, fh, sh, muh, ph = invert_column(Tg, f, s, mu, dmuL, dmuR, rho, np.clip(tgt[ab], e0, eps[-1]))
        TT[ab, j], SS[ab, j], MM[ab, j], P0[ab, j] = T, sh, muh, ph
        # continuation below the cold curve: line through (eps_min, cold value) and the first node above the curve
        be = np.where(~ab)[0]
        if len(be) and ab.any():
            i0 = np.where(ab)[0][0]; cold = dict(T=0.0, s=0.0, mu=mu[0], p=mu[0]*rho - f[0])
            for arr, key in ((TT, "T"), (SS, "s"), (MM, "mu"), (P0, "p")):
                slope = (arr[i0, j] - cold[key])/(tgt[i0] - e0)
                if cont == "quadratic" and i0 + 1 < nx:     # parabola through the cold value and the first two nodes above the curve
                    d0, d1 = tgt[i0] - e0, tgt[i0 + 1] - e0; v0, v1 = arr[i0, j] - cold[key], arr[i0 + 1, j] - cold[key]
                    c2 = (v1/d1 - v0/d0)/(d1 - d0); c1 = v0/d0 - c2*d0; dd = tgt[be] - e0
                    arr[be, j] = cold[key] + c1*dd + c2*dd*dd
                elif cont == "flat":                        # cold value at every node below the curve
                    arr[be, j] = cold[key]
                else:
                    arr[be, j] = cold[key] + slope*(tgt[be] - e0)
        elif len(be):
            for arr, key, v in ((TT, "T", 0.0), (SS, "s", 0.0), (MM, "mu", mu[0]), (P0, "p", mu[0]*rho - f[0])):
                arr[be, j] = v
    G = 1e-3
    return P0*G, TT*G, MM*G, SS, emin*G, dict(nonmonotone_columns=nonmono, columns_above_Tmax=ntop)


# ----------------------------------------------------------------------------------------- exact refinement of flagged nodes
def exact_state(M, T, rho, mode):
    """f, mu, s of the construction (mode 'two_phase') or of its Maxwell partner (mode 'maxwell', hull evaluated at this T) at (T, rho); MeV, fm."""
    if mode == "maxwell" and T < 150.0:
        pr = maxwell_pair(M, float(T))
        if pr is not None and pr[0]*RHO_S < rho < pr[1]*RHO_S:
            g = M.fs(float(T), np.array([pr[0], pr[1]])*RHO_S); lam = (rho/RHO_S - pr[0])/(pr[1] - pr[0])
            return g["f"][0] + g["mu"][0]*(rho - pr[0]*RHO_S), g["mu"][0], (1 - lam)*g["s"][0] + lam*g["s"][1]
    g = M.fs(float(T), np.array([rho])); return g["f"][0], g["mu"][0], g["s"][0]


def _refine_node(args):
    """secant iteration for eps_exact(T) = target starting from the interpolated T; returns the exact T, p, mu, s of the node."""
    mode, rho, target, T0, slope = args; M = _model()
    def ev(T):
        f, mu, s = exact_state(M, T, rho, mode); return f + T*s - target, (f, mu, s)
    Ta = T0; ea, _ = ev(Ta); Tb = Ta - ea/slope; eb, st = ev(Tb)
    for _ in range(8):
        if abs(eb) <= 1e-11*target or eb == ea:
            break
        Tn = Tb - eb*(Tb - Ta)/(eb - ea); Ta, ea = Tb, eb; Tb = Tn; eb, st = ev(Tb)
    f, mu, s = st
    return dict(T=Tb, p=mu*rho - f, mu=mu, s=s, resid=eb/target)


def halving_change(zz, dm, kc, A, dx, nx):
    """relative change of (p0, T, mu, s) when every second T row is dropped (kink nodes kept); p0 and mu relative to max(|value|, 1 MeV)."""
    idx = np.r_[np.arange(0, len(zz["T"]) - 1, 2), len(zz["T"]) - 1]
    zh = {q: (zz[q][idx] if q in ("T", "f", "mu", "s") else zz[q]) for q in zz}
    B = build_arrays(zh, dx, nx, dm[idx], kc)[:4]; out = []
    for q, floor in ((0, 1e-3), (1, 0.0), (2, 1e-3), (3, 0.0)):
        out.append(np.abs(A[q] - B[q])/np.maximum(np.abs(A[q]), floor if floor else 1e-300))
    return out


def refine_flagged(zz, dm, kc, arrs, dx, nx, name, nproc, thr=1e-6):
    """Nodes on or above the cold curve whose p0, T, mu or s change by more than thr under halving of the T grid (the T-interpolation error is
    then not negligible) are replaced by the exact model values at the exactly inverted T (secant iteration on eps_exact(T) = eps)."""
    P0, TT, MM, SS, emin = arrs; r = halving_change(zz, dm, kc, (P0, TT, MM, SS), dx, nx)
    tgt = dx*np.arange(nx)*EPS_S*1e-3; ab = tgt[:, None] >= emin[None, :]
    names = ("p0", "T", "mu", "s"); before = {n: float(np.where(ab, q, 0).max()) for n, q in zip(names, r)}
    flag = ab & ((r[0] > thr) | (r[1] > thr) | (r[2] > thr) | (r[3] > thr)); nodes = np.argwhere(flag); y = zz["y"]; args = []
    for i, j in nodes:
        i0, i1 = max(i - 1, 0), min(i + 1, nx - 1); slope = (i1 - i0)*dx*EPS_S/max((TT[i1, j] - TT[i0, j])*1e3, 1e-9)
        args.append(("maxwell" if name == "eos_maxwell" else "two_phase", y[j]*RHO_S, tgt[i]*1e3, TT[i, j]*1e3, slope))
    if len(args):
        with mp.get_context("fork").Pool(nproc) as pool:
            res = pool.map(_refine_node, args, chunksize=1)
        for (i, j), e in zip(nodes, res):
            P0[i, j], TT[i, j], MM[i, j], SS[i, j] = e["p"]*1e-3, e["T"]*1e-3, e["mu"]*1e-3, e["s"]
        worst = float(max(abs(e["resid"]) for e in res))
    else:
        worst = 0.0
    rest = {n: float(np.where(ab & ~flag, q, 0).max()) for n, q in zip(names, r)}
    return dict(threshold=thr, halving_change_max_before=before, nodes_refined=int(len(nodes)), nodes_above_cold_curve=int(ab.sum()),
                halving_change_max_unrefined_nodes=rest, max_energy_residual_of_refined=worst)


def stage_tables(work, outdir, dx, dy, tdx, tdy, only_test=False, use_kinks=True, refine=True, nproc=16):
    os.makedirs(outdir, exist_ok=True); meta = {}
    z2 = dict(np.load(os.path.join(work, "canonical_two_phase.npz"))); dm2 = dmu_dT_two_phase(z2)
    for name, can in (("eos_two_phase", "canonical_two_phase.npz"), ("eos_maxwell", "canonical_maxwell.npz")):
        z = dict(np.load(os.path.join(work, can))); dm = dm2 if name == "eos_two_phase" else dmu_dT_maxwell(z, dm2)
        desc = {"eos_two_phase": "Finite-temperature two-phase equation of state (X branch + quintic Hermite join + plasma, smooth-maximum floors; "
                                 "docs/eos_finite_T.md), p0, T, mu, s on the (eps/eps_s, rho/rho_s) lattice.",
                "eos_maxwell": "Maxwell partner of eos_two_phase: for T < T_c the free energy at rho >= rho_s is replaced by its lower convex hull "
                               "(linear between rho1(T) and rho2(T), mu = mu_coex, s = Gibbs mixture); identical elsewhere (docs/eos_finite_T.md)."}[name]
        tasks = [(False, dx, dy)] if only_test else [(False, dx, dy), (True, tdx, tdy)]
        for test, ddx, ddy in tasks:
            stride = int(round(ddy/(z["y"][1] - z["y"][0]))); zz = dict(z)
            for k in ("y", "f", "mu", "s"):
                zz[k] = z[k][..., ::stride] if k != "y" else z[k][::stride]
            nx = int(round(40.0/ddx)) + 1
            kc = _kink_columns(work, name[4:], stride) if use_kinks else None
            t0 = time.time(); dms = dm[:, ::stride]; P0, TT, MM, SS, emin, diag = build_arrays(zz, ddx, nx, dms, kc)
            if refine:
                diag["refinement"] = refine_flagged(zz, dms, kc, (P0, TT, MM, SS, emin), ddx, nx, name, nproc)
                print(name, "test" if test else "full", diag["refinement"])
            stem = os.path.join(outdir if not test else os.path.join(ROOT, "data", "eos", "test"), name + ("_test" if test else ""))
            os.makedirs(os.path.dirname(stem), exist_ok=True)
            h = write_table(stem, P0, TT, MM, SS, emin, x0=0.0, dx=ddx, y0=0.0, dy=ddy, rho_s=RHO_S, m_N=M_N/1000.0,
                            description=desc + (" REDUCED TEST LATTICE for CI (step %g in x, %g in y)." % (ddx, ddy) if test else ""),
                            extra={"tool": "tools/eos/finite_T/make_finite_T_table.py", "construction": "smoothmax_floor",
                                   "T_grid_points": int(len(z["T"])), "T_max_MeV": float(z["T"][-1]), "kink_nodes": bool(use_kinks),
                                   "exact_refinement": diag.get("refinement")})
            meta[os.path.basename(stem)] = dict(diag, sha256=h["sha256"], nx=nx, ny=len(zz["y"]), seconds=round(time.time() - t0, 1))
            print(stem, meta[os.path.basename(stem)])
    return meta


def write_reference(testdir=None):
    """data/eos/test/finite_T_reference.json: 10 lattice points (5 per test table) and 3 v_s^2 sign points for tests/test_eos_finite_T.cpp."""
    import check_tables as C
    testdir = testdir or os.path.join(ROOT, "data", "eos", "test")
    T2, MX = C.Tab(os.path.join(testdir, "eos_two_phase_test")), C.Tab(os.path.join(testdir, "eos_maxwell_test"))
    rng = np.random.default_rng(12345); pts = []
    for name, t in (("eos_two_phase_test", T2), ("eos_maxwell_test", MX)):
        ix_ok = [(i, j) for i in range(2, t.nx - 2) for j in range(2, t.ny - 2) if i*t.dx*C.EPS_S > t.emin[j] + 0.5 and t.F["T"][i, j] > 0.001]
        for k in rng.choice(len(ix_ok), 5, replace=False):
            i, j = ix_ok[k]
            pts.append(dict(table=name, ix=int(i), iy=int(j), eps_GeV_fm3=i*t.dx*C.EPS_S, rho_fm3=j*t.dy*C.RHO_S, p0=float(t.F["p0"][i, j]),
                            T=float(t.F["T"][i, j]), mu=float(t.F["mu"][i, j]), s=float(t.F["s"][i, j])))
    def vs(t, eps, rho): return float(t.vs2_h(np.array(eps), np.array(rho))[0])
    # spinodal: most negative v_s^2 at a point halfway between lattice nodes in the two-phase table
    V = np.array([[vs(T2, (i + .5)*T2.dx*C.EPS_S, (j + .5)*T2.dy*C.RHO_S) if (i + .5)*T2.dx*C.EPS_S > T2.emin[j + 1] + 0.1 else 0.0
                   for j in range(T2.ny - 1)] for i in range(T2.nx - 1)])
    i, j = np.unravel_index(V.argmin(), V.shape)
    sp = dict(label="spinodal", table="eos_two_phase_test", eps_GeV_fm3=(i + .5)*T2.dx*C.EPS_S, rho_fm3=(j + .5)*T2.dy*C.RHO_S, sign=-1)
    j2 = int(round(2.0/T2.dy)); i2 = int(round(5.0/T2.dx)); out = dict(label="outside", table="eos_two_phase_test", eps_GeV_fm3=(i2 + .5)*T2.dx*C.EPS_S, rho_fm3=(j2 + .5)*T2.dy*C.RHO_S, sign=+1)
    # Maxwell: rho = 6 rho_s, node-centred eps where T is close to 70 MeV (inside the coexistence interval)
    jm = int(round(6.0/MX.dy)); col = MX.F["T"][:, jm]*1000; im = int(np.argmin(np.abs(col - 70.0))); mxp = dict(label="Maxwell", table="eos_maxwell_test", eps_GeV_fm3=(im + .5)*MX.dx*C.EPS_S, rho_fm3=(jm + .5)*MX.dy*C.RHO_S, sign=+1)
    sp_pts = []
    for q in (sp, out, mxp):
        q["vs2_python"] = vs(T2 if q["table"] == "eos_two_phase_test" else MX, q["eps_GeV_fm3"], q["rho_fm3"]); sp_pts.append(q)
    assert sp_pts[0]["vs2_python"] < -0.01 and sp_pts[1]["vs2_python"] > 0.01 and sp_pts[2]["vs2_python"] > 0, sp_pts
    with open(os.path.join(testdir, "finite_T_reference.json"), "w") as f:
        json.dump(dict(description="Reference values written by tools/eos/finite_T/make_finite_T_table.py (write_reference); units GeV, fm; "
                       "v_s^2 from the Python port of the bicubic reader (tools/eos/finite_T/check_tables.py)", lattice_points=pts, vs2_points=sp_pts), f, indent=1)
    print("wrote finite_T_reference.json:", [round(q["vs2_python"], 5) for q in sp_pts])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["canonical", "maxwell", "kinks", "tables", "reference", "all"])
    ap.add_argument("--work", default=os.path.join(ROOT, "data", "eos", "generated"))
    ap.add_argument("--nproc", type=int, default=16)
    ap.add_argument("--dx", type=float, default=0.05); ap.add_argument("--dy", type=float, default=0.02)
    ap.add_argument("--test-dx", type=float, default=0.2); ap.add_argument("--test-dy", type=float, default=0.08)
    a = ap.parse_args(argv); os.makedirs(a.work, exist_ok=True)
    if a.stage in ("canonical", "all"): stage_canonical(a.work, a.dy, a.nproc)
    if a.stage in ("maxwell", "all"): stage_maxwell(a.work, a.nproc)
    if a.stage in ("kinks", "all"): stage_kinks(a.work, a.nproc)
    if a.stage in ("tables", "all"): stage_tables(a.work, a.work, a.dx, a.dy, a.test_dx, a.test_dy, nproc=a.nproc); write_reference()
    if a.stage == "reference": write_reference()
    return 0


if __name__ == "__main__":
    sys.exit(main())
