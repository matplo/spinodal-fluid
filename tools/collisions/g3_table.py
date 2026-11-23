#!/usr/bin/env python3
"""Comparison table of Stage A style ensembles (several start times, resolutions or equations of state) in one place.

  g3_table.py --out g3_tables --paper tools/data --set 'R7.11 cold two-phase=agg_cold/two_phase@runs/cold_thermal_two_phase' ...

Each --set LABEL=AGG_DIR@RUN_DIR: AGG_DIR is a directory written by aggregate.py for one run (it holds ensemble_enhancement.csv, ensemble_summary.json,
run_summary.json), RUN_DIR the spf_collide output directory of the same run (event_*/observables.csv, summary.json).
Per set: per-event maximum enhancement N = 5, 7 (mean, standard error, s.d.), maximum of the mean curves and its time, mean fluid-end time, raw
moments N = 2 ... 7 at t_init (mean over events, with the digitised arXiv:1209.2462 Fig. 3 values at t = 0), baryon fraction with v_s^2 < 0
at t_init, 1.5 fm and the end, and (if cell_diagnostics.csv exists) the maxima over time and events of the floor share of energy and baryon number.
Writes <out>/g3_table.json and <out>/g3_table.md.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def paper_t0(paper):
    out = {}
    for n in range(2, 8):
        f = os.path.join(paper, "p1209_fig3_N%d_two_phase.csv" % n)
        if os.path.exists(f):
            out[n] = float(np.genfromtxt(f, delimiter=",", skip_header=3)[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--out", default="g3_tables")
    ap.add_argument("--paper", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p0 = paper_t0(a.paper) if a.paper else {}
    res = {}
    for spec in a.set:
        label, rest = spec.split("=", 1)
        agg, run = rest.split("@", 1)
        ens = list(csv.reader(open(os.path.join(agg, "ensemble_enhancement.csv"))))
        hdr, rows = ens[0], np.array([[float(x) for x in r] for r in ens[1:]])
        summ = json.load(open(os.path.join(agg, "ensemble_summary.json")))
        rs = json.load(open(os.path.join(agg, "run_summary.json")))
        r = {"n_events": summ["n_events"], "recovery_failures_total": rs["recovery_failures_total"],
             "baryon_leaving_max": rs["baryon_leaving_fraction_max"], "fluid_end_time_mean_fm": rs["fluid_end_time_fm_mean"],
             "wall_per_event_s": rs["wall_total_s_mean"]}
        for n in (5, 7):
            m = summ["max_enhancement"]["N%d" % n]
            r["per_event_max_N%d" % n] = {"mean": m["mean"], "stderr": m["stderr"], "std": m["std"], "per_event": m["per_event"]}
            col = 1 + 2 * (n - 2)
            i = int(np.argmax(rows[:, col]))
            r["max_of_mean_N%d" % n] = {"value": float(rows[i, col]), "time_fm": float(rows[i, 0])}
        raw, unst = [], []
        for d in sorted(glob.glob(os.path.join(run, "event_*"))):
            s = json.load(open(os.path.join(d, "summary.json")))
            if s.get("status") not in ("fluid_end", "t_max"):
                continue
            rr = list(csv.DictReader(open(os.path.join(d, "observables.csv"))))
            raw.append([float(rr[0]["mom%d" % n]) for n in range(2, 8)])
            t = np.array([float(x["t_fm"]) for x in rr])
            u = np.array([float(x["unstable_fraction"]) for x in rr])
            unst.append([u[0], u[int(np.argmin(abs(t - 1.5)))], u[-1]])
        r["raw_moments_t_init"] = {"N%d" % n: float(v) for n, v in zip(range(2, 8), np.mean(raw, 0))}
        r["unstable_fraction_t_init_1p5fm_end"] = [float(x) for x in np.mean(unst, 0)]
        fl = []
        for f in glob.glob(os.path.join(run, "event_*", "summary.json")):
            s = json.load(open(f))
            if "cell_diagnostics" in s:
                cd = s["cell_diagnostics"]
                fl.append((cd["max_E_floor_fraction"], cd["max_R_floor_fraction"], cd["max_floor_cells"], cd["vacuum_cut_removed_E_fraction_of_initial"],
                           cd["vacuum_cut_removed_R_fraction_of_initial"], cd["initial_state"]["E_vac_fraction"], cd["initial_state"]["R_vac_fraction"],
                           cd["initial_state"]["n_vac_cells"]))
        if fl:
            fl = np.array(fl)
            r["cell_diagnostics"] = {"max_E_floor_fraction_over_events": float(fl[:, 0].max()), "mean_of_event_max_E_floor_fraction": float(fl[:, 0].mean()),
                                     "max_R_floor_fraction_over_events": float(fl[:, 1].max()), "mean_of_event_max_R_floor_fraction": float(fl[:, 1].mean()),
                                     "max_floor_cells_over_events": float(fl[:, 2].max()),
                                     "vacuum_cut_removed_E_fraction_max": float(fl[:, 3].max()), "vacuum_cut_removed_R_fraction_max": float(fl[:, 4].max()),
                                     "initial_E_vac_fraction_max": float(fl[:, 5].max()), "initial_R_vac_fraction_max": float(fl[:, 6].max()),
                                     "initial_n_vac_cells_max": float(fl[:, 7].max())}
        res[label] = r
    json.dump({"paper_fig3_t0": p0, "sets": res}, open(os.path.join(a.out, "g3_table.json"), "w"), indent=1)
    L = ["| set | events | N=7 per-event max (mean +- s.e.) | N=5 | N=7 max of mean (t) | N=5 max of mean (t) | fluid end [fm] | raw moments N=2..7 at t_init | v_s^2<0 fraction t_init / 1.5 fm / end |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        L.append("| %s | %d | %.2f +- %.2f | %.2f +- %.2f | %.2f (%.2f fm) | %.2f (%.2f fm) | %.2f | %s | %s |" % (
            k, r["n_events"], r["per_event_max_N7"]["mean"], r["per_event_max_N7"]["stderr"], r["per_event_max_N5"]["mean"], r["per_event_max_N5"]["stderr"],
            r["max_of_mean_N7"]["value"], r["max_of_mean_N7"]["time_fm"], r["max_of_mean_N5"]["value"], r["max_of_mean_N5"]["time_fm"],
            r["fluid_end_time_mean_fm"] or float("nan"), ", ".join("%.3g" % v for v in r["raw_moments_t_init"].values()),
            " / ".join("%.4f" % v for v in r["unstable_fraction_t_init_1p5fm_end"])))
    if p0:
        L.append("| paper Fig. 3 (t = 0) | 200 | | | | | | %s | |" % ", ".join("%.3g" % p0[n] for n in range(2, 8)))
    open(os.path.join(a.out, "g3_table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
