#!/usr/bin/env python3
"""Figures of WP4, generated from the CSV files written by spf_growth (no values are typed in here).

usage: plot_growth.py growth.csv growth_series.csv [growth_stable_bg.csv] outdir
Writes growth_rates.png, omega2_stable.png and growth_series.png into outdir.
The analytic curves are the analytic_omega2 column of each (dx, a) set, fitted as A k^2 + B k^4 (exact form of the linear theory).
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)


def analytic_curve(rows, kgrid):
    k = rows["k_fm"]; w2 = rows["analytic_omega2"]
    M = np.vstack([k**2, k**4]).T
    A, B = np.linalg.lstsq(M, w2, rcond=None)[0]
    return A * kgrid**2 + B * kgrid**4


def sel(d, dx, a):
    return d[np.isclose(d["dx_fm"], dx) & np.isclose(d["a_fm"], a)]


def main():
    args = sys.argv[1:]
    outdir = args[-1]
    csvs = args[:-1]
    growth = load(csvs[0]); series = load(csvs[1])
    stable = load(csvs[2]) if len(csvs) > 2 else None
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    cols = {0.0: "k", 0.01: "tab:green", 0.033: "tab:blue", 0.05: "tab:red"}
    for ax, dx, avals in ((axes[0], 0.2, [0.0, 0.01, 0.033, 0.05]), (axes[1], 0.1, [0.033])):
        for a in avals:
            r = sel(growth, dx, a)
            if len(r) == 0:
                continue
            kg = np.linspace(0.3, r["k_fm"].max(), 600)
            w2 = analytic_curve(r, kg)
            ax.plot(kg, np.sqrt(np.where(w2 < 0, -w2, np.nan)), "-", color=cols[a], lw=1.2, label=f"analytic a={a}")
            u = r[(r["kind"] == "unstable") & (r["analytic_gamma"] > 0.05)]
            ax.plot(u["k_fm"], u["gamma_fit"], "o", ms=4, color=cols[a], mfc="none", label=f"measured a={a}")
        ax.set_xlabel("k [fm$^{-1}$]"); ax.set_title(f"dx = {dx} fm"); ax.set_xlim(0, 6.5 if dx == 0.2 else 7)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel(r"$\gamma$ [c/fm]"); axes[0].set_ylim(0, 1.8)
    fig.suptitle(r"Growth rates, cold-thermal two-phase EoS, $\rho_0=5.8\rho_s$, $\epsilon_0=\epsilon_c+2\epsilon_s$ (modes with analytic $\gamma>0.05$)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "growth_rates.png"), dpi=140); plt.close(fig)

    nax = 2 if stable is not None else 1
    fig, axes = plt.subplots(1, nax, figsize=(6 * nax, 4.6), squeeze=False)
    panels = [(axes[0, 0], growth, r"spinodal background ($\rho_0=5.8\rho_s$), stable modes")]
    if stable is not None:
        panels.append((axes[0, 1], stable, r"stable background ($\rho_0=2\rho_s$)"))
    for ax, d, title in panels:
        for a, c in ((0.033, "tab:blue"), (0.05, "tab:red")):
            for dx, mk in ((0.2, "s"), (0.1, "o")):
                r = sel(d, dx, a)
                r = r[r["kind"] == "stable"]
                if len(r):
                    ax.plot(r["k_fm"], r["omega_fit"]**2, mk, color=c, ms=4, mfc="none", label=f"measured a={a}, dx={dx}")
            r = sel(d, 0.1, a)
            if len(r):
                kg = np.linspace(r["k_fm"].min(), r["k_fm"].max(), 400)
                w2 = analytic_curve(r, kg)
                ax.plot(kg, np.where(w2 > 0, w2, np.nan), "-", color=c, lw=1, label=f"analytic a={a}")
        ax.set_xlabel("k [fm$^{-1}$]"); ax.set_ylabel(r"$\omega^2$ [fm$^{-2}$]"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "omega2_stable.png"), dpi=140); plt.close(fig)

    s_all = series[np.isclose(series["a_fm"], 0.033) & np.isclose(series["dx_fm"], 0.2)]
    ms = sorted(set(int(m) for m in s_all["m"]))
    fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 4)); axes = np.atleast_1d(axes)
    for ax, m in zip(axes, ms):
        s = s_all[s_all["m"] == m]
        t, A = s["t_fm"], s["A_abs"]
        row = growth[np.isclose(growth["dx_fm"], 0.2) & np.isclose(growth["a_fm"], 0.033) & (growth["m"] == m)][0]
        g, A0 = row["gamma_fit"], row["A0_fit_over_A0"]
        ax.semilogy(t, A / A[0], "k.", ms=3, label="measured |A|/A(0)")
        ax.semilogy(t, A0 * np.cosh(g * t), "r-", lw=1, label=rf"fit $A_0\cosh(\gamma t)$, $\gamma$={g:.3f}")
        ax.semilogy(t, np.cosh(row["analytic_gamma"] * t), "b--", lw=1, label=rf"analytic $\gamma$={row['analytic_gamma']:.3f}")
        ax.axhspan(1.05, 3.0, color="0.9", zorder=0)
        ax.set_xlabel("t [fm/c]"); ax.set_title(f"k = {row['k_fm']:.3f} fm$^{{-1}}$ (m = {m})"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("a = 0.033 fm, dx = 0.2 fm; grey band: fit window (1.05 to 3 times A(0))")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "growth_series.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
