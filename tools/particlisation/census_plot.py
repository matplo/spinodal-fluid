#!/usr/bin/env python3
"""Figures and a Markdown table of the surface census (census.py output).

  census_plot.py --census <dir with census.json and census_hist.npz> --ab tools/particlisation/data/ab_two_phase.json --out <dir>

Figures: per run, the surface elements in (rho / rho_s, T) weighted by the energy flux and by the baryon flux (all runs in one figure per weight,
with the matching densities a(T), b(T) of the two-phase construction), and the T and rho projections; table census_table.md.  Needs matplotlib.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True)
    ap.add_argument("--ab", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.census, "census.json")) as f:
        C = json.load(f)
    H = np.load(os.path.join(a.census, "census_hist.npz"))
    with open(a.ab) as f:
        ab = json.load(f)
    re, te = H["rho_edges"], H["T_edges"]
    labels = list(C["runs"].keys())
    for w, wname in (("E", "energy flux $T^{0\\mu}d\\Sigma_\\mu$"), ("B", "baryon flux $N^\\mu d\\Sigma_\\mu$")):
        n = len(labels)
        fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False, constrained_layout=True)
        for ax, lab in zip(axs[0], labels):
            h = H[lab + "__surface_" + w]
            tot = h.sum()
            img = np.where(h > 0, h / tot / (np.diff(re)[:, None] * np.diff(te)[None, :]), np.nan)
            m = ax.pcolormesh(re, te, img.T, shading="flat", norm=matplotlib.colors.LogNorm())
            ax.plot(ab["a_rho_s"], ab["T_MeV"], "k-", lw=0.9, label="a(T) (two-phase matching density)")
            ax.plot(ab["b_rho_s"], ab["T_MeV"], "k--", lw=0.9, label="b(T) (two-phase matching density)")
            R = C["runs"][lab]
            ax.set_title("%s\nspline fraction %.3f" % (lab, R["pooled_surface_spline_frac_" + w]), fontsize=9)
            ax.set_xlabel(r"$\rho/\rho_s$")
            ax.set_ylabel("T [MeV]")
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 200)
            ax.legend(loc="lower right", fontsize=7, framealpha=0.6)
            fig.colorbar(m, ax=ax, label="fraction per ($\\rho_s$ MeV)")
        fig.suptitle("surface elements weighted by the " + wname, fontsize=10)
        fig.savefig(os.path.join(a.out, "census_rho_T_%s.png" % w), dpi=140)
        plt.close(fig)
    # projections
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for lab in labels:
        for w, ls in (("E", "-"), ("B", "--")):
            h = H[lab + "__surface_" + w]
            axs[0].step(re[:-1], h.sum(1) / h.sum() / np.diff(re), where="post", ls=ls, label="%s, %s flux" % (lab, "energy" if w == "E" else "baryon"))
            axs[1].step(te[:-1], h.sum(0) / h.sum() / np.diff(te), where="post", ls=ls, label="%s, %s flux" % (lab, "energy" if w == "E" else "baryon"))
    axs[0].set_xlabel(r"$\rho/\rho_s$"); axs[0].set_ylabel("flux fraction per $\\rho_s$"); axs[0].set_xlim(0, 8)
    axs[1].set_xlabel("T [MeV]"); axs[1].set_ylabel("flux fraction per MeV"); axs[1].set_xlim(0, 200)
    axs[1].legend(fontsize=6, loc="upper left")
    fig.savefig(os.path.join(a.out, "census_projections.png"), dpi=140)
    plt.close(fig)
    # table
    lines = ["| run | events | elements per event (initial slice) | file per event [MB] | fluid end after t_init [fm/c] | closure E | closure B | "
             "spline fraction, surface, E / B flux | spline fraction, initial slice, E / B flux | inflow share of surface E | wall fluid / surface per event [s] |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab in labels:
        R = C["runs"][lab]
        S = R["stats"]
        ev = R["events"]
        infl = np.mean([x["surface_E_neg"] / x["surface_E_py"] for x in ev])
        lines.append("| %s | %d | %.3g +- %.2g (%.3g) | %.1f +- %.1f | %.2f +- %.2f | %.2e +- %.1e | %.2e +- %.1e | %.3f / %.3f | %.3f / %.3f | %.3f | %.1f / %.1f |" % (
            lab, S["n_elements"]["n"], S["n_elements"]["mean"], S["n_elements"]["std"], S["n_initial_slice"]["mean"], S["file_MB"]["mean"],
            S["file_MB"]["std"], S["end_time_after_init_fm"]["mean"], S["end_time_after_init_fm"]["std"], S["closure_E"]["mean"], S["closure_E"]["std"],
            S["closure_B"]["mean"], S["closure_B"]["std"], R["pooled_surface_spline_frac_E"], R["pooled_surface_spline_frac_B"],
            R["pooled_initial_spline_frac_E"], R["pooled_initial_spline_frac_B"], infl, S["wall_evolve_s"]["mean"], S["wall_surface_s"]["mean"]))
    with open(os.path.join(a.out, "census_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
