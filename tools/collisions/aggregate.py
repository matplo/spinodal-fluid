#!/usr/bin/env python3
"""Aggregate the events of one or more spf_collide output directories (one directory = one equation of state / parameter set).

  aggregate.py --out agg --run two_phase=runs/e3_two_phase --run maxwell=runs/e3_maxwell

For each --run LABEL=DIR: the events with status fluid_end or t_max (failed events are listed, not used) are passed to tools/obs_ensemble.py
(mean and standard error of the enhancement of N = 2 ... 7 versus t - t_init; per-event maximum enhancement). Events of unequal length (runs that
stop at the fluid end) are cut to the shortest event: the events are given to obs_ensemble in order of increasing length with --interp.
Written to <out>/<LABEL>/: ensemble_enhancement.csv, ensemble_summary.json (obs_ensemble), events_summary.csv (one row per event: the fields of
summary.json), run_summary.json (sums and means over the events: recovery failures, floor activations, baryon / energy leaving the box,
wall time, fluid end time, unstable fraction), provenance.json (configuration echo of the first event, git description of the code,
the directories). Standard library only.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ENS = os.path.join(HERE, "..", "obs_ensemble.py")
FIELDS = ["event", "status", "e_lab_gev", "t_init_fm", "steps", "time_after_init_fm", "fluid_end_time_fm", "recovery_failures", "floor_activations",
          "limiter_cells_run", "baryon_leaving_fraction", "energy_leaving_fraction", "wall_evolve_s", "wall_observe_s", "wall_total_s",
          "wall_per_step_s", "eps_max_initial_eps_s", "eps_max_peak_eps_s", "unstable_fraction_peak", "unstable_fraction_peak_time_fm",
          "unstable_fraction_final", "peak_rss_MB"]


def load_events(d):
    ev = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name, "summary.json")
        if name.startswith("event_") and os.path.exists(p):
            ev.append((name, json.load(open(p))))
    return ev


def nrows(path):
    with open(os.path.join(path, "observables.csv")) as f:
        return sum(1 for _ in f) - 1


def mean(v):
    v = [x for x in v if x is not None]
    return statistics.fmean(v) if v else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="agg")
    ap.add_argument("--run", action="append", required=True, help="LABEL=DIR")
    a = ap.parse_args()
    for spec in a.run:
        label, d = spec.split("=", 1)
        out = os.path.join(a.out, label)
        os.makedirs(out, exist_ok=True)
        ev = load_events(d)
        good = [(n, s) for n, s in ev if s.get("status") in ("fluid_end", "t_max")]
        bad = [(n, s.get("status"), s.get("message", "")) for n, s in ev if s.get("status") not in ("fluid_end", "t_max")]
        with open(os.path.join(out, "events_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            for n, s in ev:
                w.writerow([s.get(k, "") if s.get(k) is not None else "" for k in FIELDS])
        if good:
            order = sorted(good, key=lambda t: nrows(os.path.join(d, t[0])))
            dirs = [os.path.join(d, n) for n, _ in order]
            subprocess.check_call([sys.executable, ENS, "--out", out, "--interp"] + dirs)
        runsum = {
            "label": label, "directory": os.path.abspath(d), "n_events_found": len(ev), "n_events_used": len(good), "failed_events": bad,
            "recovery_failures_total": sum(s["recovery_failures"] for _, s in good),
            "floor_activations_total": sum(s["floor_activations"] for _, s in good),
            "floor_activations_per_event_mean": mean([s["floor_activations"] for _, s in good]),
            "limiter_cells_total": sum(s["limiter_cells_run"] for _, s in good),
            "baryon_leaving_fraction_max": max((s["baryon_leaving_fraction"] for _, s in good), default=None),
            "baryon_leaving_fraction_mean": mean([s["baryon_leaving_fraction"] for _, s in good]),
            "energy_leaving_fraction_max": max((s["energy_leaving_fraction"] for _, s in good), default=None),
            "energy_leaving_fraction_mean": mean([s["energy_leaving_fraction"] for _, s in good]),
            "wall_total_s_mean": mean([s["wall_total_s"] for _, s in good]),
            "wall_evolve_s_mean": mean([s["wall_evolve_s"] for _, s in good]),
            "wall_per_step_s_mean": mean([s["wall_per_step_s"] for _, s in good]),
            "steps_mean": mean([s["steps"] for _, s in good]),
            "fluid_end_time_fm_mean": mean([s["fluid_end_time_fm"] for _, s in good]),
            "n_events_with_fluid_end": sum(1 for _, s in good if s["fluid_end_time_fm"] is not None),
            "unstable_fraction_peak_mean": mean([s["unstable_fraction_peak"] for _, s in good]),
        }
        json.dump(runsum, open(os.path.join(out, "run_summary.json"), "w"), indent=1)
        prov = {"label": label, "directory": os.path.abspath(d)}
        cu = os.path.join(d, "config_used.json")
        if os.path.exists(cu):
            prov["config_used"] = json.load(open(cu))
        if good:
            prov["code"] = good[0][1].get("code")
        json.dump(prov, open(os.path.join(out, "provenance.json"), "w"), indent=1)
        print("%s: %d events used, %d failed; recovery failures %d" % (label, len(good), len(bad), runsum["recovery_failures_total"]))


if __name__ == "__main__":
    sys.exit(main())
