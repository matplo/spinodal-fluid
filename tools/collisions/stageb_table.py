#!/usr/bin/env python3
"""Stage B table: per energy and equation of state the quantities of docs/validation.md (Stage B pilot), and the energy dependence.

  stageb_table.py --agg-root . --runs-root runs --energies 1 2 3 4 5 6 --out stageb_tables [--paper tools/data]

Expects, per energy E, the aggregate directories <agg-root>/agg_e<E>/{two_phase,maxwell} (aggregate.py) and the run directories
<runs-root>/e<E>_table_two_phase, e<E>_table_maxwell. Per energy and EoS: number of events, per-event maximum enhancement N = 5, 7 (mean,
standard error, standard deviation), maximum of the mean curves and its time, mean fluid-end time, recovery failures, largest baryon and energy
fraction leaving the box, wall time per event, number of steps. Energy dependence: the energy at which the mean per-event maximum of the two-phase
run is largest, the ratio of the largest to the smallest Maxwell mean, and the digitised paper values at the same energies. Writes
<out>/stageb_table.json and <out>/stageb_table.md.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np


def read_ens(path):
    rows = list(csv.reader(open(os.path.join(path, "ensemble_enhancement.csv"))))
    return np.array([[float(x) for x in r] for r in rows[1:]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg-root", default=".")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--energies", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="stageb_tables")
    ap.add_argument("--paper", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for E in a.energies:
        for lab in ("two_phase", "maxwell"):
            agg = os.path.join(a.agg_root, "agg_e%d" % E, lab)
            summ = json.load(open(os.path.join(agg, "ensemble_summary.json")))
            rs = json.load(open(os.path.join(agg, "run_summary.json")))
            ens = read_ens(agg)
            r = {"n_events": summ["n_events"], "failed_events": len(rs["failed_events"]), "recovery_failures": rs["recovery_failures_total"],
                 "baryon_leaving_max": rs["baryon_leaving_fraction_max"], "energy_leaving_max": rs["energy_leaving_fraction_max"],
                 "wall_per_event_s": rs["wall_total_s_mean"], "steps": rs["steps_mean"], "fluid_end_mean_fm": rs["fluid_end_time_fm_mean"],
                 "n_events_with_fluid_end": rs["n_events_with_fluid_end"]}
            for n in (5, 7):
                m = summ["max_enhancement"]["N%d" % n]
                r["N%d_per_event_max" % n] = {"mean": m["mean"], "stderr": m["stderr"], "std": m["std"]}
                c = 1 + 2 * (n - 2)
                i = int(np.argmax(ens[:, c]))
                r["N%d_max_of_mean"  % n] = {"value": float(ens[i, c]), "time_fm": float(ens[i, 0])}
            d = os.path.join(a.runs_root, "e%d_table_%s" % (E, lab))
            t0 = []
            for f in sorted(glob.glob(os.path.join(d, "event_*", "summary.json"))):
                t0.append(json.load(open(f))["t_init_fm"])
            r["t_init_fm"] = float(np.mean(t0)) if t0 else None
            res["E%d_%s" % (E, lab)] = r
    out = {"per_energy": res}
    for n in (5, 7):
        for lab in ("two_phase", "maxwell"):
            v = [(E, res["E%d_%s" % (E, lab)]["N%d_per_event_max" % n]["mean"], res["E%d_%s" % (E, lab)]["N%d_per_event_max" % n]["stderr"]) for E in a.energies]
            k = int(np.argmax([x[1] for x in v]))
            out["N%d_%s" % (n, lab)] = {"energy_of_largest_mean_GeV": v[k][0], "largest_mean": v[k][1], "largest_stderr": v[k][2],
                                        "smallest_mean": min(x[1] for x in v), "largest_over_smallest": max(x[1] for x in v) / min(x[1] for x in v),
                                        "means": {int(x[0]): x[1] for x in v}}
    if a.paper:
        pp = {}
        for n in (5, 7):
            for kind in ("two_phase", "maxwell"):
                f = os.path.join(a.paper, "p1209_fig4_N%d_%s.csv" % (n, kind))
                if os.path.exists(f):
                    d = np.genfromtxt(f, delimiter=",", skip_header=3)
                    pp["N%d_%s" % (n, kind)] = {int(E): float(d[np.argmin(abs(d[:, 0] - E)), 1]) for E in a.energies}
        out["paper_fig4"] = pp
    json.dump(out, open(os.path.join(a.out, "stageb_table.json"), "w"), indent=1)
    L = ["| E_lab [A GeV] | EoS | events | failures | N=7 per-event max: mean +- s.e. (s.d.) | N=5 per-event max | N=7 max of mean (t [fm]) | N=5 max of mean (t) | fluid end [fm] (events reaching it) | max baryon / energy leaving | wall per event [s] |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for E in a.energies:
        for lab in ("two_phase", "maxwell"):
            r = res["E%d_%s" % (E, lab)]
            L.append("| %d | %s | %d | %d | %.2f +- %.2f (%.2f) | %.2f +- %.2f (%.2f) | %.2f (%.2f) | %.2f (%.2f) | %s (%d) | %.1e / %.1e | %.1f |" % (
                E, lab, r["n_events"], r["recovery_failures"], r["N7_per_event_max"]["mean"], r["N7_per_event_max"]["stderr"], r["N7_per_event_max"]["std"],
                r["N5_per_event_max"]["mean"], r["N5_per_event_max"]["stderr"], r["N5_per_event_max"]["std"], r["N7_max_of_mean"]["value"], r["N7_max_of_mean"]["time_fm"],
                r["N5_max_of_mean"]["value"], r["N5_max_of_mean"]["time_fm"], "%.2f" % r["fluid_end_mean_fm"] if r["fluid_end_mean_fm"] is not None else "-",
                r["n_events_with_fluid_end"], r["baryon_leaving_max"], r["energy_leaving_max"], r["wall_per_event_s"]))
    open(os.path.join(a.out, "stageb_table.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    for k, v in out.items():
        if k.startswith("N"):
            print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "means"})


if __name__ == "__main__":
    main()
