#!/usr/bin/env python3
"""Moment enhancement versus t - t_init for every energy: one panel per energy, two-phase solid and Maxwell dashed, N = 2 ... 7 (mean, s.e. band).

  plot_energy_grid.py --agg-root . --energies 1 2 3 4 5 6 --out moments_vs_energy.png [--title 'finite-T table']
"""
import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {2: "k", 3: "tab:green", 4: "tab:blue", 5: "tab:purple", 6: "tab:pink", 7: "tab:red"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agg-root", default=".")
    ap.add_argument("--energies", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="moments_vs_energy.png")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    n = len(a.energies)
    nc = 3
    nr = (n + nc - 1) // nc
    fig, axs = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.3 * nr), sharex=True, squeeze=False)
    for ax, E in zip(axs.flat, a.energies):
        for lab, ls in (("two_phase", "-"), ("maxwell", "--")):
            rows = list(csv.reader(open(os.path.join(a.agg_root, "agg_e%d" % E, lab, "ensemble_enhancement.csv"))))
            d = np.array([[float(x) for x in r] for r in rows[1:]])
            for i, N in enumerate(range(2, 8)):
                m, se = d[:, 1 + 2 * i], d[:, 2 + 2 * i]
                ax.plot(d[:, 0], m, ls, color=COL[N], lw=1.2, label="N=%d" % N if lab == "two_phase" else None)
                ax.fill_between(d[:, 0], m - se, m + se, color=COL[N], alpha=0.15, lw=0)
        ax.set_yscale("log")
        ax.set_title("%d A GeV" % E, fontsize=9)
        ax.set_xlabel(r"$t - t_{\rm init}$ [fm/c]", fontsize=8)
    axs.flat[0].legend(fontsize=6, ncol=2)
    for ax in axs[:, 0]:
        ax.set_ylabel("enhancement", fontsize=8)
    fig.suptitle("%s: two-phase (solid) vs Maxwell (dashed), mean +- s.e." % a.title, fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140)


if __name__ == "__main__":
    main()
