#!/usr/bin/env python3
"""Plots of a two-phase versus Maxwell comparison from the output of aggregate.py, with the digitised curves of the papers.

  plot_stage.py --agg agg --tp two_phase --mx maxwell --out plots --tag cold_thermal [--paper tools/data] [--figures-title '3 A GeV']

Figures (<out>/<tag>_*.png) and numbers (<out>/<tag>_numbers.json):
  moments       enhancement mom_N(t) / mom_N(t_init) for N = 2 ... 7 versus t - t_init: mean with standard-error band; two-phase solid, Maxwell dashed;
                dotted black: arXiv:1302.2956 Fig. 5 (N = 2, 5, 7; two-phase and Maxwell) when --paper is given
  maxdist       distribution over events of the maximum enhancement for N = 5 and 7 (two-phase and Maxwell)
  unstable      baryon fraction in cells with v_s^2 < 0 (the isentropic spinodal of the EoS) versus time, mean over events
  clumps        clump-size distribution (probability versus baryon number) at the time of the maximum of the mean N = 7 enhancement of the
                two-phase run, two-phase minus Maxwell, and the digitised difference of arXiv:1302.2956 Fig. 7
Every figure carries the git description of the code and the configuration file of the run in its footer.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

NS = range(2, 8)
COL = {2: "k", 3: "tab:green", 4: "tab:blue", 5: "tab:purple", 6: "tab:pink", 7: "tab:red"}


def read_csv(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], np.array([[float(x) for x in r] for r in rows[1:]])


def paper(path, name):
    p = os.path.join(path, name)
    if not os.path.exists(p):
        return None
    return np.genfromtxt(p, delimiter=",", skip_header=3)


def footer(fig, prov):
    code = (prov.get("code") or {})
    fig.text(0.01, 0.005, "code %s | %s" % (code.get("git", "?"), os.path.basename(prov.get("directory", ""))), fontsize=6, color="gray")


def events(d):
    out = []
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n, "summary.json")
        if n.startswith("event_") and os.path.exists(p):
            s = json.load(open(p))
            if s.get("status") in ("fluid_end", "t_max"):
                out.append((n, s))
    return out


def clump_hist(d, t_target):
    """summed histogram (bins 0 ... max, overflow as the last entry) over the events at the record nearest to t_target; the time used per event"""
    tot, times = None, []
    for n, _ in events(d):
        best = None
        for f in glob.glob(os.path.join(d, n, "clumps_step*.json")):
            j = json.load(open(f))
            if best is None or abs(j["time_fm"] - t_target) < abs(best["time_fm"] - t_target):
                best = j
        if best is None:
            continue
        h = np.array(best["histogram"] + [best["overflow"]], float)
        tot = h if tot is None else tot + h
        times.append(best["time_fm"])
    return tot, times


def clump_hist_window(d, t_lo, t_hi):
    """histogram summed over the events and over all clump records with t_lo <= t <= t_hi (more clumps than at a single time)"""
    tot, nrec = None, 0
    for n, _ in events(d):
        for f in glob.glob(os.path.join(d, n, "clumps_step*.json")):
            j = json.load(open(f))
            if t_lo - 1e-9 <= j["time_fm"] <= t_hi + 1e-9:
                h = np.array(j["histogram"] + [j["overflow"]], float)
                tot = h if tot is None else tot + h
                nrec += 1
    return tot, nrec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", required=True)
    ap.add_argument("--tp", default="two_phase")
    ap.add_argument("--mx", default="maxwell")
    ap.add_argument("--out", default="plots")
    ap.add_argument("--tag", default="stage")
    ap.add_argument("--paper", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--runs-root", default="", help="directory that holds the run directories (when the aggregation ran elsewhere: provenance holds the original paths)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prov = {k: json.load(open(os.path.join(a.agg, k, "provenance.json"))) for k in (a.tp, a.mx)}
    ens = {k: read_csv(os.path.join(a.agg, k, "ensemble_enhancement.csv")) for k in (a.tp, a.mx)}
    summ = {k: json.load(open(os.path.join(a.agg, k, "ensemble_summary.json"))) for k in (a.tp, a.mx)}
    if a.runs_root:
        for k in prov:
            prov[k]["directory"] = os.path.join(a.runs_root, os.path.basename(prov[k]["directory"].rstrip("/")))
    num = {"tag": a.tag}

    # ---- moments
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for k, ls in ((a.tp, "-"), (a.mx, "--")):
        hdr, d = ens[k]
        t = d[:, 0]
        for i, n in enumerate(NS):
            m, se = d[:, 1 + 2 * i], d[:, 2 + 2 * i]
            ax.plot(t, m, ls, color=COL[n], lw=1.4, label="N=%d" % n if k == a.tp else None)
            ax.fill_between(t, m - se, m + se, color=COL[n], alpha=0.2, lw=0)
    if a.paper:
        for n in (7, 5, 2):
            for kind, ls in (("two_phase", ":"), ("maxwell", ":")):
                p = paper(a.paper, "p1302_fig5_N%d_%s.csv" % (n, kind))
                if p is not None:
                    ax.plot(p[:, 0], p[:, 1], ls, color="k", lw=0.9, label="arXiv:1302.2956 Fig. 5" if (n, kind) == (7, "two_phase") else None)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t - t_{\rm init}$ [fm/c]")
    ax.set_ylabel(r"$\langle\rho^N\rangle/\langle\rho\rangle^N$ relative to $t_{\rm init}$")
    ax.set_title("%s: two-phase (solid) vs Maxwell (dashed), mean $\\pm$ s.e., %d / %d events" % (a.title or a.tag, summ[a.tp]["n_events"], summ[a.mx]["n_events"]), fontsize=8)
    ax.legend(fontsize=7, ncol=2)
    footer(fig, prov[a.tp])
    fig.savefig(os.path.join(a.out, a.tag + "_moments.png"), dpi=150)
    plt.close(fig)
    # numbers: maximum of the mean curves
    num["max_of_mean_enhancement"] = {}
    for k in (a.tp, a.mx):
        hdr, d = ens[k]
        num["max_of_mean_enhancement"][k] = {"N%d" % n: {"value": float(d[:, 1 + 2 * i].max()), "time_fm": float(d[np.argmax(d[:, 1 + 2 * i]), 0])} for i, n in enumerate(NS)}

    # ---- per-event maximum distribution
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, n in zip(axs, (5, 7)):
        for k, c in ((a.tp, "tab:red"), (a.mx, "tab:blue")):
            v = np.array(summ[k]["max_enhancement"]["N%d" % n]["per_event"])
            ax.hist(v, bins=np.logspace(np.log10(max(min(v.min(), 0.9), 0.5)), np.log10(max(v.max(), 1.5) * 1.05), 16), color=c, alpha=0.5, label="%s: mean %.2f, s.d. %.2f" % (k, v.mean(), v.std(ddof=1) if len(v) > 1 else 0))
        ax.set_xscale("log")
        ax.set_xlabel("per-event maximum enhancement, N=%d" % n)
        ax.set_ylabel("events")
        ax.legend(fontsize=7)
    for ax in axs:
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    footer(fig, prov[a.tp])
    fig.savefig(os.path.join(a.out, a.tag + "_maxdist.png"), dpi=150)
    plt.close(fig)
    num["max_enhancement_per_event"] = {k: {n: {kk: summ[k]["max_enhancement"][n][kk] for kk in ("mean", "stderr", "std", "spread_max_over_mean", "max_over_min")} for n in ("N5", "N7")} for k in (a.tp, a.mx)}
    r7 = summ[a.tp]["max_enhancement"]["N7"]["mean"] / summ[a.mx]["max_enhancement"]["N7"]["mean"]
    r5 = summ[a.tp]["max_enhancement"]["N5"]["mean"] / summ[a.mx]["max_enhancement"]["N5"]["mean"]
    num["ratio_two_phase_over_maxwell_of_mean_max"] = {"N7": r7, "N5": r5}

    # ---- unstable fraction versus time
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for k, c in ((a.tp, "tab:red"), (a.mx, "tab:blue")):
        d0 = prov[k]["directory"]
        curves = []
        for n, _ in events(d0):
            hdr, o = read_csv(os.path.join(d0, n, "observables.csv"))
            curves.append(o)
        L = min(len(o) for o in curves)
        u = np.array([o[:L, hdr.index("unstable_fraction")] for o in curves])
        t = curves[0][:L, 0]
        ax.plot(t, u.mean(0), color=c, label=k)
        ax.fill_between(t, u.mean(0) - u.std(0, ddof=1) / np.sqrt(len(u)), u.mean(0) + u.std(0, ddof=1) / np.sqrt(len(u)), color=c, alpha=0.25, lw=0)
        num.setdefault("unstable_fraction_mean", {})[k] = {"peak": float(u.mean(0).max()), "time_of_peak_fm": float(t[np.argmax(u.mean(0))])}
    ax.set_xlabel(r"$t - t_{\rm init}$ [fm/c]")
    ax.set_ylabel(r"baryon fraction with $v_s^2<0$")
    ax.legend(fontsize=7)
    footer(fig, prov[a.tp])
    fig.savefig(os.path.join(a.out, a.tag + "_unstable.png"), dpi=150)
    plt.close(fig)

    # ---- clumps at the time of the maximum of the mean N = 7 enhancement (two-phase run)
    t_star = num["max_of_mean_enhancement"][a.tp]["N7"]["time_fm"]
    htp, ttp = clump_hist(prov[a.tp]["directory"], t_star)
    hmx, tmx = clump_hist(prov[a.mx]["directory"], t_star)
    if htp is not None and hmx is not None:
        ptp, pmx = htp / max(htp.sum(), 1.0), hmx / max(hmx.sum(), 1.0)  # an ensemble without clumps has probability 0 in every bin
        diff = ptp - pmx
        k = np.arange(len(ptp) - 1)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.step(k, ptp[:-1], where="mid", color="tab:red", label="two-phase")
        ax.step(k, pmx[:-1], where="mid", color="tab:blue", label="Maxwell")
        pos = diff[:-1] > 0
        ax.plot(k[pos], diff[:-1][pos], "o-", color="k", ms=3, label="two-phase $-$ Maxwell (positive part)")
        if a.paper:
            p = paper(a.paper, "p1302_fig7_difference_dots.csv")
            if p is not None:
                bx = np.floor(p[:, 0] + 1e-9)
                keep = (p[:, 0] - bx > 0.15) & (p[:, 0] - bx < 0.85)
                pb = {int(b): float(np.median(p[(bx == b) & keep, 1])) for b in np.unique(bx) if np.any((bx == b) & keep)}
                ax.plot([b + 0.5 for b in pb], list(pb.values()), "s", mfc="none", color="tab:green", label="arXiv:1302.2956 Fig. 7 (difference; bin [k,k+1) at k+0.5)")
        # pooled over the records within +-1 fm of t_star (more clumps, same quantity)
        wtp, ntp = clump_hist_window(prov[a.tp]["directory"], t_star - 1.0, t_star + 1.0)
        wmx, nmx = clump_hist_window(prov[a.mx]["directory"], t_star - 1.0, t_star + 1.0)
        if wtp is not None and wmx is not None and (wtp.sum() > 0 or wmx.sum() > 0):
            wd = wtp / max(wtp.sum(), 1.0) - wmx / max(wmx.sum(), 1.0)
            posw = wd[:-1] > 0
            ax.plot(k[posw], wd[:-1][posw], "^--", color="gray", ms=3, label="same, pooled over t* $\\pm$ 1 fm (%d + %d clumps)" % (wtp.sum(), wmx.sum()))
            num["clumps_pooled_window"] = {"window_fm": [t_star - 1.0, t_star + 1.0], "n_clumps_tp": float(wtp.sum()), "n_clumps_mx": float(wmx.sum()),
                                           "difference_by_bin": [float(x) for x in wd[:-1]], "difference_peak_baryon_number": int(np.argmax(wd[:-1])),
                                           "difference_peak_value": float(wd[:-1].max()), "p_two_phase": [float(x) for x in wtp / max(wtp.sum(), 1.0)],
                                           "p_maxwell": [float(x) for x in wmx / max(wmx.sum(), 1.0)]}
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1)
        ax.set_xlabel("baryon number in clump (round(B))")
        ax.set_ylabel("probability")
        ax.set_title("clumps at t - t_init = %.2f fm (record nearest to the N=7 maximum); %d + %d clumps" % (np.mean(ttp), htp.sum(), hmx.sum()), fontsize=7)
        ax.legend(fontsize=6)
        footer(fig, prov[a.tp])
        fig.savefig(os.path.join(a.out, a.tag + "_clumps.png"), dpi=150)
        plt.close(fig)
        kmax = int(np.argmax(diff[:-1]))
        num["clumps"] = {"t_star_fm": t_star, "mean_record_time_tp": float(np.mean(ttp)), "mean_record_time_mx": float(np.mean(tmx)),
                         "n_clumps_tp": float(htp.sum()), "n_clumps_mx": float(hmx.sum()), "n_clumps_per_event_tp": float(htp.sum() / len(ttp)),
                         "n_clumps_per_event_mx": float(hmx.sum() / len(tmx)), "difference_peak_baryon_number": kmax,
                         "difference_peak_value": float(diff[kmax]), "difference_by_bin": [float(x) for x in diff[:-1]],
                         "p_two_phase": [float(x) for x in ptp], "p_maxwell": [float(x) for x in pmx]}
    json.dump(num, open(os.path.join(a.out, a.tag + "_numbers.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in num.items() if k != "clumps"}, indent=1)[:1500])


if __name__ == "__main__":
    sys.exit(main())
