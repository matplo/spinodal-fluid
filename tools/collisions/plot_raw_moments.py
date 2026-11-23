#!/usr/bin/env python3
"""Normalised density moments <rho^N>/<rho>^N themselves (not divided by their initial value) versus t - t_init for two-phase and Maxwell runs, over the
digitised Fig. 3 of arXiv:1209.2462 (mean over 200 UrQMD events there). Mean over the events of the run directories.

  plot_raw_moments.py --tp runs/two_phase --mx runs/maxwell --paper tools/data --out plots/raw_moments.png
"""
import argparse
import csv
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mean_moments(d):
    curves = []
    for f in sorted(glob.glob(os.path.join(d, "event_*", "observables.csv"))):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        curves.append(np.array([[float(r["t_fm"])] + [float(r["mom%d" % n]) for n in range(2, 8)] for r in rows]))
    L = min(len(c) for c in curves)
    a = np.array([c[:L] for c in curves])
    return a[0, :, 0], a[:, :, 1:].mean(0), len(curves)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tp", required=True)
    ap.add_argument("--mx", required=True)
    ap.add_argument("--paper", default="")
    ap.add_argument("--out", default="raw_moments.png")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    col = {2: "k", 3: "tab:green", 4: "tab:blue", 5: "tab:purple", 6: "tab:pink", 7: "tab:red"}
    for d, ls in ((a.tp, "-"), (a.mx, "--")):
        t, m, n = mean_moments(d)
        for i, N in enumerate(range(2, 8)):
            ax.plot(t, m[:, i], ls, color=col[N], lw=1.4, label="N=%d" % N if ls == "-" else None)
    if a.paper:
        for N in range(2, 8):
            for kind in ("two_phase", "maxwell"):
                f = os.path.join(a.paper, "p1209_fig3_N%d_%s.csv" % (N, kind))
                if os.path.exists(f):
                    p = np.genfromtxt(f, delimiter=",", skip_header=3)
                    ax.plot(p[:, 0], p[:, 1], ":", color="gray", lw=0.9, label="arXiv:1209.2462 Fig. 3" if (N, kind) == (7, "two_phase") else None)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t - t_{\rm init}$ [fm/c]")
    ax.set_ylabel(r"$\langle\rho^N\rangle/\langle\rho\rangle^N$")
    ax.set_title("%s: two-phase (solid), Maxwell (dashed), %d events; dotted: paper" % (a.title, n), fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    main()
