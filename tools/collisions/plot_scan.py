#!/usr/bin/env python3
"""Mean maximum enhancement versus E_lab (Stage B): from aggregate.py output directories of several energies, with the digitised curves of
arXiv:1209.2462 Fig. 4 and arXiv:1302.2956 Fig. 6.

  plot_scan.py --out plots --paper tools/data --point 3.0:two_phase=agg_e3/two_phase --point 3.0:maxwell=agg_e3/maxwell ...
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="plots")
    ap.add_argument("--paper", default="")
    ap.add_argument("--tag", default="scan")
    ap.add_argument("--point", action="append", required=True, help="E_lab:LABEL=aggregate_dir/LABEL")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pts = {}
    for p in a.point:
        left, d = p.split("=", 1)
        e, label = left.split(":")
        s = json.load(open(os.path.join(d, "ensemble_summary.json")))
        pts.setdefault(label, []).append((float(e), {n: (s["max_enhancement"][n]["mean"], s["max_enhancement"][n]["stderr"]) for n in ("N5", "N7")}))
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, n, pn in zip(axs, ("N7", "N5"), (7, 5)):
        for label, c, mk in (("two_phase", "tab:red", "o"), ("maxwell", "tab:blue", "s")):
            if label in pts:
                v = sorted(pts[label])
                ax.errorbar([x[0] for x in v], [x[1][n][0] for x in v], [x[1][n][1] for x in v], color=c, marker=mk, ms=4, label=label)
        if a.paper:
            for kind, ls in (("two_phase", ":"), ("maxwell", ":")):
                f = os.path.join(a.paper, "p1209_fig4_N%d_%s.csv" % (pn, kind))
                if os.path.exists(f):
                    d = np.genfromtxt(f, delimiter=",", skip_header=3)
                    ax.plot(d[:, 0], d[:, 1], ls, marker="x", color="k", lw=0.8, label="arXiv:1209.2462 Fig. 4" if kind == "two_phase" else None)
        ax.set_yscale("log")
        ax.set_xlabel(r"$E_{\rm lab}$ [A GeV]")
        ax.set_ylabel("mean maximum enhancement, %s" % n)
        ax.legend(fontsize=7)
    fig.savefig(os.path.join(a.out, a.tag + "_enhancement_vs_E.png"), dpi=150)
    json.dump({k: sorted(v) for k, v in pts.items()}, open(os.path.join(a.out, a.tag + "_points.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
