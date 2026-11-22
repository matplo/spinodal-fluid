#!/usr/bin/env python3
"""Acceptance checks C1-C7 of the finite-temperature tables (docs/validation.md, WP2b).

  python check_tables.py [--work data/eos/generated] [--json results.json] [--nproc 16]

Reads eos_two_phase / eos_maxwell (.json/.bin) and canonical_*.npz from --work.  v_s^2 uses a Python port of the bicubic Hermite interpolation of
include/spf/eos/table.hpp (centred-difference node derivatives, tensor-product Catmull-Rom).  Units GeV, fm in the tables; MeV inside the model.
"""
import argparse, json, multiprocessing as mp, os, sys, warnings
import numpy as np

here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, ".."))
import eos2b as E
from write_table import read_table, read_cold_curve, cold_hermite
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(here, "..", "..", ".."))
RHO_S, M_N = 0.153, 0.940; EPS_S = RHO_S*M_N


# ------------------------------------------------------------------ Python port of EosTable (bicubic)
class Tab:
    def __init__(self, stem):
        h, a = read_table(stem); L = h["lattice"]; self.h = h
        self.nx, self.ny, self.dx, self.dy = L["nx"], L["ny"], L["dx"], L["dy"]
        self.F = {k: a[k] for k in ("p0", "T", "mu", "s")}; self.emin = a["eps_min"]
        self.D = {k: self._derivs(v) for k, v in self.F.items()}
        self.em_d = np.gradient(self.emin)   # d1 of reader: central, one-sided at the ends

    @staticmethod
    def _derivs(F):
        fx = np.gradient(F, axis=0); fy = np.gradient(F, axis=1); fxy = np.gradient(fy, axis=0); return F, fx, fy, fxy

    def _locate(self, eps, rho):
        eps, rho = np.broadcast_arrays(np.asarray(eps, float), np.asarray(rho, float))
        snap = lambda f: np.where(np.abs(f - np.floor(f + .5)) < 1e-10, np.floor(f + .5), f)
        fy = snap(rho/RHO_S/self.dy); fy = np.clip(fy, 0, self.ny - 1); j = np.minimum(fy.astype(int), self.ny - 2); u = fy - j
        t2, t3 = u*u, u*u*u
        h00, h10, h01, h11 = 2*t3 - 3*t2 + 1, t3 - 2*t2 + u, -2*t3 + 3*t2, t3 - t2
        d00, d10, d01, d11 = 6*t2 - 6*u, 3*t2 - 4*u + 1, -6*t2 + 6*u, 3*t2 - 2*u
        E_, D_ = self.emin, self.em_d
        em = h00*E_[j] + h10*D_[j] + h01*E_[j + 1] + h11*D_[j + 1]; dem = d00*E_[j] + d10*D_[j] + d01*E_[j + 1] + d11*D_[j + 1]
        x = eps/EPS_S; xm = em/EPS_S; cl = x < xm; x = np.where(cl, xm, x)
        fx = snap(x/self.dx); fx = np.clip(fx, 0, self.nx - 1); i = np.minimum(fx.astype(int), self.nx - 2); t = fx - i
        return dict(i=i, j=j, t=t, u=u, cl=cl, slope=dem/(self.dy*RHO_S), eps=eps, rho=rho)

    def _eval(self, key, L):
        F, fx_, fy_, fxy_ = self.D[key]; t, u = L["t"], L["u"]
        t2, t3, u2, u3 = t*t, t*t*t, u*u, u*u*u
        ht0 = [2*t3 - 3*t2 + 1, -2*t3 + 3*t2]; ht1 = [t3 - 2*t2 + t, t3 - t2]; dt0 = [6*t2 - 6*t, -6*t2 + 6*t]; dt1 = [3*t2 - 4*t + 1, 3*t2 - 2*t]
        hu0 = [2*u3 - 3*u2 + 1, -2*u3 + 3*u2]; hu1 = [u3 - 2*u2 + u, u3 - u2]; du0 = [6*u2 - 6*u, -6*u2 + 6*u]; du1 = [3*u2 - 4*u + 1, 3*u2 - 2*u]
        v = vx = vy = 0.0
        for a in (0, 1):
            for b in (0, 1):
                i, j = L["i"] + a, L["j"] + b; f, fx, fy, fxy = F[i, j], fx_[i, j], fy_[i, j], fxy_[i, j]
                v = v + f*ht0[a]*hu0[b] + fx*ht1[a]*hu0[b] + fy*ht0[a]*hu1[b] + fxy*ht1[a]*hu1[b]
                vx = vx + f*dt0[a]*hu0[b] + fx*dt1[a]*hu0[b] + fy*dt0[a]*hu1[b] + fxy*dt1[a]*hu1[b]
                vy = vy + f*ht0[a]*du0[b] + fx*ht1[a]*du0[b] + fy*ht0[a]*du1[b] + fxy*ht1[a]*du1[b]
        return v, vx, vy

    def field(self, key, eps, rho):
        L = self._locate(eps, rho); return self._eval(key, L)[0]

    def vs2_h(self, eps, rho):
        L = self._locate(eps, rho); v, vx, vy = self._eval("p0", L); px = vx/(self.dx*EPS_S)
        a = np.where(L["cl"], 0.0, px); b = vy/(self.dy*RHO_S) + np.where(L["cl"], px*L["slope"], 0.0); h = L["eps"] + v
        return np.where(h > 0, a + L["rho"]/np.where(h > 0, h, 1)*b, a), h

    def eps_of_T(self, T, rho, n=70):
        """eps (GeV/fm^3) at which the interpolated table has temperature T (GeV) at rho; bisection (T(eps) is monotone above the cold curve)."""
        rho = np.asarray(rho, float); lo = np.interp(rho/RHO_S, np.arange(self.ny)*self.dy, self.emin) + 1e-9; hi = np.full_like(rho, self.nx*self.dx*EPS_S*0.999)
        for _ in range(n):
            m = 0.5*(lo + hi); up = self.field("T", m, rho) < T; lo = np.where(up, m, lo); hi = np.where(up, hi, m)
        return 0.5*(lo + hi)


# ------------------------------------------------------------------ model evaluation for C2 (worker)
_M = None
def _model():
    global _M
    if _M is None: _M = E.TwoPhase(os.path.join(ROOT, "data", "eos", "finite_T", "coex_pure.npz"))
    return _M


def _f_at(T, rho, mode):
    """f(T, rho) of the construction (mode 'two_phase') or of its Maxwell partner (mode 'maxwell'); MeV, fm."""
    M = _model(); r = np.array([rho])
    if mode == "maxwell" and T < 150.0:
        import make_finite_T_table as Mk
        pr = Mk.maxwell_pair(M, float(T))
        if pr is not None and pr[0]*E.rhos < rho < pr[1]*E.rhos:
            f1, mu1 = M.f(T, np.array([pr[0]*E.rhos]), M.coeffs(T)); return f1[0] + mu1[0]*(rho - pr[0]*E.rhos)
    return M.f(T, r)[0][0]


def _exact_state(args):
    """f, mu, s of the construction (or of the Maxwell partner) evaluated exactly at (T, rho); returns eps, p, mu (GeV, fm) and s."""
    T, rho, mode = args; M = _model(); r = np.array([rho]); res = None
    if mode == "maxwell" and T < 150.0:
        import make_finite_T_table as Mk
        pr = Mk.maxwell_pair(M, float(T))
        if pr is not None and pr[0]*E.rhos < rho < pr[1]*E.rhos:
            g = M.fs(T, np.array([pr[0], pr[1]])*E.rhos); lam = (rho/E.rhos - pr[0])/(pr[1] - pr[0])
            f = g["f"][0] + g["mu"][0]*(rho - pr[0]*E.rhos); mu = g["mu"][0]; s_ = (1 - lam)*g["s"][0] + lam*g["s"][1]; res = (f, mu, s_)
    if res is None:
        g = M.fs(T, r); res = (g["f"][0], g["mu"][0], g["s"][0])
    f, mu, s_ = res
    return dict(eps=(f + T*s_)*1e-3, p=(mu*rho - f)*1e-3, mu=mu*1e-3, s=s_)


def _fd_s(args):
    T, rho, h, mode = args
    f = [_f_at(T + k*h, rho, mode) for k in (-2, -1, 1, 2)]
    return -(f[0] - 8*f[1] + 8*f[2] - f[3])/(12*h)       # 5-point stencil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.path.join(ROOT, "data", "eos", "generated")); ap.add_argument("--json"); ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args(); R = {}
    tabs = {k: Tab(os.path.join(a.work, k)) for k in ("eos_two_phase", "eos_maxwell")}
    can = {"eos_two_phase": dict(np.load(os.path.join(a.work, "canonical_two_phase.npz"))), "eos_maxwell": dict(np.load(os.path.join(a.work, "canonical_maxwell.npz")))}
    tp, mx = tabs["eos_two_phase"], tabs["eos_maxwell"]
    x = np.arange(tp.nx)*tp.dx; y = np.arange(tp.ny)*tp.dy
    ab = (x[:, None]*EPS_S >= tp.emin[None, :])      # nodes on or above the cold curve (thermodynamic nodes)
    R["lattice"] = dict(nx=tp.nx, ny=tp.ny, nodes_above_cold_curve=int(ab.sum()), nodes_total=int(ab.size))

    # C1
    for k, t in tabs.items():
        eps = x[:, None]*EPS_S + 0*y[None, :]; rho = y[None, :]*RHO_S + 0*x[:, None]
        res = t.F["T"]*t.F["s"] - eps + t.F["mu"]*rho - t.F["p0"]
        r = np.abs(res)/np.maximum(np.abs(t.F["p0"]), 1e-3); abt = (x[:, None]*EPS_S >= t.emin[None, :])
        R.setdefault("C1", {})[k] = dict(max_ratio=float(r[abt].max()), max_abs_residual_GeV_fm3=float(np.abs(res)[abt].max()))
    # C2
    rng = np.random.default_rng(2026); out = {}
    for k, t in tabs.items():
        ab_t = np.arange(t.nx)[:, None]*t.dx*EPS_S >= t.emin[None, :]
        cand = np.argwhere(ab_t & (t.F["T"] > 0.002) & (t.F["T"] < 0.250) & (x[:, None] > 0.1) & (np.arange(t.nx)[:, None] < t.nx - 3)
                           & (np.arange(t.ny)[None, :] > 1) & (np.arange(t.ny)[None, :] < t.ny - 2))
        sel = cand[rng.choice(len(cand), 200, replace=False)]
        args = [(t.F["T"][i, j]*1000, y[j]*RHO_S, 0.1, "maxwell" if k == "eos_maxwell" else "two_phase") for i, j in sel]
        with mp.get_context("fork").Pool(a.nproc) as pool: fd = np.array(pool.map(_fd_s, args, chunksize=2))
        tab_s = np.array([t.F["s"][i, j] for i, j in sel]); rel = np.abs(tab_s/fd - 1)
        out[k] = dict(max_rel=float(rel.max()), median_rel=float(np.median(rel)), argmax=dict(T_MeV=float(args[rel.argmax()][0]), y=float(args[rel.argmax()][1]/RHO_S)),
                      T_min_MeV=float(min(q[0] for q in args)), T_max_MeV=float(max(q[0] for q in args)))
    R["C2"] = out
    # C2b: table values against the exact evaluation of the construction at the inverted T (random nodes and the nodes with the largest
    # change under halving of the T grid), eps_exact(T*) against the lattice eps, p0 and mu relative to max(|value|, 1 MeV)
    import make_finite_T_table as Mk
    dm2 = Mk.dmu_dT_two_phase(can["eos_two_phase"]); R["C2b"] = {}; R["C8"] = {}
    for k, t in tabs.items():
        zc = can[k]; kc = Mk._kink_columns(a.work, k[4:], 1) if os.path.exists(os.path.join(a.work, "kinks_" + k[4:] + ".npz")) else None
        dmk = dm2 if k == "eos_two_phase" else Mk.dmu_dT_maxwell(zc, dm2)
        idx = np.r_[np.arange(0, len(zc["T"]) - 1, 2), len(zc["T"]) - 1]; zh = {q: (zc[q][idx] if q in ("T", "f", "mu", "s") else zc[q]) for q in zc}
        A_ = Mk.build_arrays(zh, tp.dx, tp.nx, dmk[idx], kc)       # halved T grid (kink nodes kept)
        abt = (np.arange(t.nx)[:, None]*t.dx*EPS_S >= t.emin[None, :]); out8 = {}
        for q, name, floor in ((0, "p0", 1e-3), (1, "T", 0.0), (2, "mu", 1e-3), (3, "s", 0.0)):
            ref = t.F[name if name != "p0" else "p0"]; d = np.abs(ref - A_[q]); sc = np.maximum(np.abs(ref), floor if floor else 1e-300); r_ = np.where(abt, d/sc, 0.0)
            i_, j_ = np.unravel_index(r_.argmax(), r_.shape)
            out8[name] = dict(max_rel=float(r_.max()), n_above_1e6=int((r_ > 1e-6).sum()), at=dict(x=float(i_*t.dx), y=float(j_*t.dy), T_MeV=float(t.F["T"][i_, j_]*1000)))
        R["C8"][k] = dict(final_table_vs_halved_unrefined_grid=out8, refinement_report=t.h.get("generator", {}).get("exact_refinement"))
        rng2 = np.random.default_rng(77); cand = np.argwhere(abt & (t.F["T"] > 0.002) & (t.F["T"] < 0.249)); sel = cand[rng2.choice(len(cand), 150, replace=False)]
        # stress nodes: largest p0 change under halving
        ref = t.F["p0"]; r_ = np.where(abt, np.abs(ref - A_[0])/np.maximum(np.abs(ref), 1e-3), 0.0); top = np.argsort(r_.ravel())[-24:]
        stress = np.array([np.unravel_index(q, r_.shape) for q in top]); allsel = np.vstack([sel, stress])
        args = [(t.F["T"][i, j]*1000, y[j]*E.rhos, "maxwell" if k == "eos_maxwell" else "two_phase") for i, j in allsel]
        with mp.get_context("fork").Pool(a.nproc) as pool: ex = pool.map(_exact_state, args, chunksize=1)
        def stats(ids):
            e = [ex[q] for q in ids]; nodes = [tuple(allsel[q]) for q in ids]
            et = np.array([i*t.dx*EPS_S for i, j in nodes]); er = np.array([abs(v["eps"] - w)/w for v, w in zip(e, et)])
            pe = np.array([abs(v["p"] - t.F["p0"][i, j])/max(abs(v["p"]), 1e-3) for v, (i, j) in zip(e, nodes)])
            me = np.array([abs(v["mu"] - t.F["mu"][i, j])/max(abs(v["mu"]), 1e-3) for v, (i, j) in zip(e, nodes)])
            se = np.array([abs(v["s"] - t.F["s"][i, j])/abs(v["s"]) for v, (i, j) in zip(e, nodes)])
            return dict(n=len(ids), eps_rel=float(er.max()), p0_rel=float(pe.max()), mu_rel=float(me.max()), s_rel=float(se.max()))
        R["C2b"][k] = dict(random=stats(range(len(sel))), stress=stats(range(len(sel), len(allsel))))
    # C3
    for k, t in tabs.items():
        c = can[k]; eps_node = c["f"] + c["T"][:, None]*c["s"]; d_eps = np.diff(eps_node, axis=0)
        dT = np.diff(t.F["T"], axis=0); ds = np.diff(t.F["s"], axis=0); abt = (x[:, None]*EPS_S >= t.emin[None, :]); both = abt[1:] & abt[:-1]
        R.setdefault("C3", {})[k] = dict(canonical_nodes_nonincreasing_eps_of_T=int((d_eps <= 0).sum()), min_d_eps_node=float(d_eps.min()),
            table_nodes_dT_deps_le0=int(((dT <= 0) & both).sum()), table_nodes_ds_deps_le0=int(((ds <= 0) & both).sum()),
            table_min_dT_deps=float((dT[both]/ (tp.dx*EPS_S)).min()), table_min_ds_deps=float((ds[both]/(tp.dx*EPS_S)).min()))
    # C4
    cm = can["eos_two_phase"]; Tg = cm["T"]; fpp = np.gradient(cm["mu"], cm["y"]*RHO_S, axis=1)   # df''/drho^2 = dmu/drho [MeV fm^3] on the canonical lattice
    def vs2_nodes(t):
        eps = np.broadcast_to(x[:, None]*EPS_S, (t.nx, t.ny)); rho = np.broadcast_to(y[None, :]*RHO_S, (t.nx, t.ny))
        v, _ = t.vs2_h(eps, rho); return v
    V = {k: vs2_nodes(t) for k, t in tabs.items()}
    # isothermal-spinodal mask at the table nodes (T from the node, f'' interpolated along T in the column)
    mask = np.zeros((tp.nx, tp.ny), bool)
    for j in range(tp.ny):
        col = np.interp(tp.F["T"][:, j]*1000, Tg, fpp[:, j]); mask[:, j] = (col < 0) & (y[j] >= 1.0)
    from scipy.ndimage import binary_dilation
    near = binary_dilation(mask, iterations=2)
    vt = V["eos_two_phase"]; hi = y >= 1.0
    abm = (x[:, None]*EPS_S >= mx.emin[None, :]); vm = V["eos_maxwell"]; first = np.zeros_like(abm)
    for jj in range(mx.ny):
        w_ = np.where(abm[:, jj])[0]
        if len(w_): first[w_[0], jj] = True
    negm = (vm < 0) & abm & hi[None, :]
    mxstat = dict(min_rho_ge_rhos=float(vm[abm & hi[None, :]].min()), max=float(vm[abm].max()), neg_nodes_rho_ge_rhos=int(negm.sum()),
                  neg_nodes_first_node_above_cold_curve=int((negm & first).sum()), neg_nodes_elsewhere=int((negm & ~first).sum()),
                  min_excluding_first_node=float(vm[abm & hi[None, :] & ~first].min()), min_all=float(vm[abm].min()))
    neg = (vt < 0) & ab; outside = neg & (~near) & hi[None, :]
    pos_in = ((vt > 0) & ab & mask)
    R["C4"] = dict(two_phase=dict(min=float(vt[ab].min()), max=float(vt[ab].max()), neg_nodes=int(neg.sum()), neg_nodes_rho_ge_rhos=int((neg & hi[None, :]).sum()),
                       neg_outside_spinodal_mask_rho_ge_rhos=int(outside.sum()), nodes_in_iso_spinodal=int((mask & ab).sum()), nodes_in_iso_spinodal_with_vs2_gt0=int(pos_in.sum()),
                       neg_nodes_rho_lt_rhos=int((neg & ~hi[None, :]).sum()), min_rho_lt_rhos=float(vt[:, ~hi][ab[:, ~hi]].min()), min_rho_ge_rhos=float(vt[:, hi][ab[:, hi]].min())),
                   maxwell=mxstat)
    # spinodal boundaries along isotherms
    fine = np.arange(1.5, 16.0, 0.01); bnd = {}
    for T in (0, 50, 100, 130, 140):
        rho = fine*RHO_S
        if T == 0: eps = tp.emin[0]*0 + np.interp(fine, np.arange(tp.ny)*tp.dy, tp.emin) + 1e-6
        else: eps = tp.eps_of_T(T/1000., rho)
        vs, _ = tp.vs2_h(eps, rho); cr = lambda v: [float(fine[i] - v[i]*(fine[i+1] - fine[i])/(v[i+1] - v[i])) for i in np.where(np.diff(np.sign(v)) != 0)[0]]
        entry = dict(isentropic_vs2_zero=cr(vs))
        if T > 0:
            mu = tp.field("mu", eps, rho); entry["isothermal_dmu_drho_zero_table"] = cr(np.gradient(mu, rho))
        i = int(np.where(np.isclose(Tg, T))[0][0]); entry["isothermal_model"] = [float(c) for c in
            [cm["y"][k] - fpp[i, k]*(cm["y"][k+1] - cm["y"][k])/(fpp[i, k+1] - fpp[i, k]) for k in np.where(np.diff(np.sign(fpp[i]))[:] != 0)[0] if cm["y"][k] >= 1.0]]
        bnd[str(T)] = entry
    R["C4"]["spinodal_boundaries_rho_over_rhos"] = bnd
    # isentropic spinodal (v_s^2 < 0) from the canonical (T, rho) lattice, rho >= 1.5 rho_s
    rr = cm["y"]*RHO_S; Pc = cm["mu"]*rr - cm["f"]; Ec = cm["f"] + Tg[:, None]*cm["s"]
    dTc = lambda A: np.gradient(A, Tg, axis=0); dRc = lambda A: np.gradient(A, rr, axis=1)
    with np.errstate(all="ignore"):
        VSc = dTc(Pc)/dTc(Ec) + (rr[None, :]/(Ec + Pc))*(dRc(Pc) - dTc(Pc)*dRc(Ec)/dTc(Ec))
    negT = np.array([(VSc[i, cm["y"] >= 1.5] < 0).any() for i in range(len(Tg))]); iv = []
    for i in range(len(Tg)):
        if negT[i] and (i == 0 or not negT[i - 1]): a0 = Tg[i]
        if negT[i] and (i == len(Tg) - 1 or not negT[i + 1]): iv.append([float(a0), float(Tg[i])])
    R["C4"]["isentropic_spinodal_T_intervals_MeV_canonical"] = iv
    # C5
    d_rho, ec, mc = read_cold_curve(os.path.join(ROOT, "data", "eos", "cold_curve.dat")); ecold, muc, pc = cold_hermite(y*RHO_S, d_rho, ec, mc)
    diff = tp.emin - ecold; k = int(np.abs(diff).argmax())
    R["C5"] = dict(max_abs_diff_GeV_fm3=float(np.abs(diff).max()), at_y=float(y[k]), eps_min=float(tp.emin[k]), cold_curve=float(ecold[k]), rel=float(abs(diff[k])/ecold[k]),
                   max_abs_diff_over_eps_s=float(np.abs(diff).max()/EPS_S), max_rel_diff_y_ge_1=float((np.abs(diff)/np.maximum(ecold, 1e-12))[y >= 1].max()),
                   maxwell_emin_identical=bool(np.array_equal(tp.emin, mx.emin)))
    # C6: critical point from the table: highest T with min_rho vs2 < 0 along the isotherm; same from dmu/drho
    fine = np.arange(1.5, 5.0, 0.005); Ts = np.arange(140.0, 145.0, 0.05); mv, mi, rv, ri = [], [], [], []
    for T in Ts:
        rho = fine*RHO_S; eps = tp.eps_of_T(T/1000., rho); vs, _ = tp.vs2_h(eps, rho); mu = tp.field("mu", eps, rho)
        mv.append(vs.min()); rv.append(fine[vs.argmin()]); d = np.gradient(mu, rho); mi.append(d.min()); ri.append(fine[d.argmin()])
    mv, mi = np.array(mv), np.array(mi)
    def tc(m):
        i = np.where((m[:-1] < 0) & (m[1:] >= 0))[0]
        return float(Ts[i[-1]] - m[i[-1]]*(Ts[i[-1]+1] - Ts[i[-1]])/(m[i[-1]+1] - m[i[-1]])) if len(i) else float("nan")
    Tc_v, Tc_i = tc(mv), tc(mi); jv = int(np.argmin(np.abs(Ts - Tc_v)))
    R["C6"] = dict(Tc_from_vs2_MeV=Tc_v, Tc_from_isothermal_MeV=Tc_i, rho_c_at_Tc_vs2=float(rv[jv]), rho_c_isothermal=float(ri[int(np.argmin(np.abs(Ts - Tc_i)))]),
                   model_Tc_reference=142.50, model_rho_c_reference=2.94, min_vs2_by_T={f"{t:.2f}": float(m) for t, m in zip(Ts[::10], mv[::10])},
                   canonical_Tc_isothermal=float("nan"))
    # canonical critical point: highest grid T with f'' < 0 at some rho in [1.5, 6]
    sel = (cm["y"] >= 1.5) & (cm["y"] <= 6); mn = fpp[:, sel].min(axis=1); iT = np.where(mn < 0)[0].max()
    R["C6"]["canonical_highest_T_with_negative_fpp_MeV"] = float(Tg[iT]); R["C6"]["canonical_next_T_MeV"] = float(Tg[iT + 1]); R["C6"]["canonical_rho_c"] = float(cm["y"][sel][fpp[iT, sel].argmin()])
    # C7
    j = int(round(5.8/tp.dy)); eps7 = tp.emin[j] + 2*EPS_S; rho7 = 5.8*RHO_S
    vs7, h7 = tp.vs2_h(np.array(eps7), np.array(rho7)); R["C7"] = dict(eps_GeV_fm3=float(eps7), eps_over_eps_s=float(eps7/EPS_S), vs2=float(vs7), h_GeV_fm3=float(h7),
        T_MeV=float(tp.field("T", np.array(eps7), np.array(rho7))*1000), p_GeV_fm3=float(tp.field("p0", np.array(eps7), np.array(rho7))), mu_GeV=float(tp.field("mu", np.array(eps7), np.array(rho7))),
        s_fm3=float(tp.field("s", np.array(eps7), np.array(rho7))), reference_cold_thermal=dict(vs2=-0.05428, h=1.98840))
    print(json.dumps(R, indent=1))
    if a.json: json.dump(R, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
