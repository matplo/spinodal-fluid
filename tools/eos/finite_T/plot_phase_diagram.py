#!/usr/bin/env python3
"""Phase-diagram figures and interface tension of the finite-T two-phase equation of state (docs/figures/).

  python plot_phase_diagram.py [--work data/eos/generated] [--out docs/figures] [--check results.json] [--nproc 16]

  eos_phase_T_rho.png    (i)   (rho/rho_s, T): pure-phase coexistence, two-phase coexistence (Maxwell construction), isothermal spinodal, critical point
  eos_phase_eps_rho.png  (ii)  the same in (rho/rho_s, eps/eps_s) with the cold curve (compare Fig. 1 of arXiv:1209.2462)
  eos_sigma_T.png        (iii) sigma_T(T) = a Int sqrt(2 eps_s Delta f) d(rho/rho_s) over the coexistence interval, a = 0.01, 0.033, 0.05 fm
                               (compare Fig. 2 of arXiv:1209.2462); values are also written to sigma_T.json in --work
"""
import argparse, json, multiprocessing as mp, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import eos2b as E
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(here, "..", "..", ".."))
RHOS, EPS_S = E.rhos, E.eps_s
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7, "xtick.direction": "out",
                     "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 150})
C_COEX, C_SPIN, C_PURE, C_CP, C_ISEN = "#1f4e9c", "#c46a00", "#7a7a7a", "#b00020", "#2a8a3c"
_M = None


def _sigma_row(args):
    global _M
    if _M is None: _M = E.TwoPhase(os.path.join(ROOT, "data", "eos", "finite_T", "coex_pure.npz"))
    T, r1, r2 = args
    y = np.linspace(r1, r2, 4001); rho = y*RHOS
    if T < E.T_LOW: g = _M.fs(T, rho); f, mu = g["f"], g["mu"]
    else: f, mu = _M.f(T, rho, _M.coeffs(T))
    df = f - (f[0] + mu[0]*(rho - rho[0]))       # common tangent through (r1, f(r1)) with slope mu_coex = mu(r1)
    df = np.clip(df, 0, None)
    return float(np.trapezoid(np.sqrt(2*EPS_S*df), y))       # sigma/a in MeV/fm^3


def spinodal_roots(y, v, ymin=1.0):
    k = np.where((np.sign(v[:-1]) != np.sign(v[1:])) & (y[:-1] >= ymin))[0]
    return [y[i] - v[i]*(y[i+1] - y[i])/(v[i+1] - v[i]) for i in k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=os.path.join(ROOT, "data", "eos", "generated")); ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    ap.add_argument("--check"); ap.add_argument("--nproc", type=int, default=16); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = np.load(os.path.join(a.work, "canonical_maxwell.npz")); t = np.load(os.path.join(a.work, "canonical_two_phase.npz"))
    Tg, y = t["T"], t["y"]; hull = c["hull"]; has = np.isfinite(hull[:, 0])
    fpp = np.gradient(t["mu"], y*RHOS, axis=1); eps_node = t["f"] + Tg[:, None]*t["s"]
    pure = np.load(os.path.join(ROOT, "data", "eos", "finite_T", "coex_pure.npz"))
    Tc, rhoc = 142.5, 2.94
    if a.check and os.path.exists(a.check):
        C6 = json.load(open(a.check))["C6"]; Tc, rhoc = C6["Tc_from_isothermal_MeV"], C6["rho_c_isothermal"]
    # spinodals (isothermal) from the canonical lattice
    sp_lo, sp_hi, sp_T, sp_elo, sp_ehi = [], [], [], [], []
    for i, T in enumerate(Tg):
        r = spinodal_roots(y, fpp[i])
        if len(r) >= 2:
            sp_T.append(T); sp_lo.append(r[0]); sp_hi.append(r[-1])
            sp_elo.append(np.interp(r[0], y, eps_node[i])/EPS_S); sp_ehi.append(np.interp(r[-1], y, eps_node[i])/EPS_S)
    sp_T = np.array(sp_T)
    Tcn = Tg[has]; r1, r2 = hull[has, 0], hull[has, 1]; e1 = (hull[has, 4] + Tcn*hull[has, 6])/EPS_S; e2 = (hull[has, 5] + Tcn*hull[has, 7])/EPS_S
    # isentropic spinodal (v_s^2 = 0 at fixed s/rho) from the canonical lattice; digitised boundaries of arXiv:1007.1448 Fig. 5
    rr = y*RHOS; Pc = t["mu"]*rr - t["f"]
    dTc = lambda A_: np.gradient(A_, Tg, axis=0); dRc = lambda A_: np.gradient(A_, rr, axis=1)
    with np.errstate(all="ignore"):
        VS = dTc(Pc)/dTc(eps_node) + (rr[None, :]/(eps_node + Pc))*(dRc(Pc) - dTc(Pc)*dRc(eps_node)/dTc(eps_node))
    VS = np.where((y[None, :] >= 1.5) & np.isfinite(VS), VS, np.nan)
    paper = {}
    for nm in ("coexistence", "isothermal_spinodal", "isentropic_spinodal"):
        rows = [l.strip().split(",") for l in open(os.path.join(ROOT, "tools", "data", f"fig5_boundary_{nm}.csv")) if not l.startswith("#")][1:]
        paper[nm] = np.array([[float(q) for q in r] for r in rows if len(r) == 2])
    # (i)
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    cs = ax.contour(y, Tg, VS, levels=[0.0], colors=[C_ISEN], linewidths=1.4)
    ax.plot([], [], color=C_ISEN, lw=1.4, label=r"isentropic spinodal ($v_s^2 = 0$), this work")
    for nm, ls_ in (("coexistence", "-"), ("isothermal_spinodal", "--"), ("isentropic_spinodal", ":")):
        ax.plot(paper[nm][:, 0], paper[nm][:, 1], color="k", ls=ls_, lw=0.9, label="arXiv:1007.1448 Fig. 5: " + nm.replace("_", " "))
    ax.plot(pure["rhoH"]/RHOS, pure["T"], color=C_PURE, ls="--", lw=1); ax.plot(pure["rhoQ"]/RHOS, pure["T"], color=C_PURE, ls="--", lw=1, label="pure-phase coexistence")
    ax.plot(r1, Tcn, color=C_COEX, lw=1.6); ax.plot(r2, Tcn, color=C_COEX, lw=1.6, label="two-phase coexistence")
    ax.plot(sp_lo, sp_T, color=C_SPIN, lw=1.2); ax.plot(sp_hi, sp_T, color=C_SPIN, lw=1.2, label="isothermal spinodal")
    ax.plot([rhoc], [Tc], "o", color=C_CP, ms=5, label=f"critical point ({Tc:.1f} MeV, {rhoc:.2f}" + r" $\rho_s$)")
    ax.set_xlim(0, 12); ax.set_ylim(0, 165); ax.set_xlabel(r"$\rho/\rho_s$"); ax.set_ylabel("T (MeV)"); ax.legend(loc="upper right", fontsize=5.5)
    ax.set_title("Coexistence and spinodal region close at one critical point", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "eos_phase_T_rho.png")); plt.close(fig)
    # (ii)
    emin = np.interp(y, y, eps_node[0])/EPS_S
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(y, emin, color="k", lw=1, label="cold curve (T = 0)")
    ax.plot(r1, e1, color=C_COEX, lw=1.6); ax.plot(r2, e2, color=C_COEX, lw=1.6, label="two-phase coexistence")
    ax.plot(sp_lo, sp_elo, color=C_SPIN, lw=1.2); ax.plot(sp_hi, sp_ehi, color=C_SPIN, lw=1.2, label="isothermal spinodal")
    i = int(np.argmin(np.abs(Tg - Tc))); ec = np.interp(rhoc, y, eps_node[i])/EPS_S
    ax.plot([rhoc], [ec], "o", color=C_CP, ms=5, label="critical point")
    ax.set_xlim(0, 12); ax.set_ylim(0, 30); ax.set_xlabel(r"$\rho/\rho_s$"); ax.set_ylabel(r"$\varepsilon/\varepsilon_s$"); ax.legend(loc="upper left")
    ax.set_title("Same phase boundaries in the hydrodynamic variables", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "eos_phase_eps_rho.png")); plt.close(fig)
    # (iii)
    sel = np.where(has & (np.isclose(Tg % 1, 0) | (Tg >= 130)))[0]
    with mp.get_context("fork").Pool(a.nproc) as pool:
        sig = pool.map(_sigma_row, [(Tg[i], hull[i, 0], hull[i, 1]) for i in sel], chunksize=1)
    sig = np.array(sig); Ts = Tg[sel]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for av, ls in ((0.05, ":"), (0.033, "-"), (0.01, "--")):
        ax.plot(Ts, sig*av, ls=ls, color=C_COEX, lw=1.6 if av == 0.033 else 1.0, label=f"a = {av} fm")
    ax.set_xlabel("T (MeV)"); ax.set_ylabel(r"$\sigma_T$ (MeV/fm$^2$)"); ax.set_xlim(0, Tc + 3); ax.set_ylim(0, None); ax.legend(loc="upper right")
    ax.set_title("Interface tension falls to zero at the critical point", loc="left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "eos_sigma_T.png")); plt.close(fig)
    rep = {f"{T:g}": dict(sigma_over_a_MeV_fm3=float(s_), sigma_a0p033_MeV_fm2=float(s_*0.033)) for T, s_ in zip(Ts, sig) if T in (0, 50, 100, 130, 140)}
    json.dump(dict(reported=rep, T=Ts.tolist(), sigma_over_a=sig.tolist()), open(os.path.join(a.work, "sigma_T.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
