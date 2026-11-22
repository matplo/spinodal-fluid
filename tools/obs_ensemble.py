#!/usr/bin/env python3
"""Ensemble statistics of the density-moment enhancement over a set of event output directories (WP6).

usage: obs_ensemble.py [--out DIR] [--interp] event_dir [event_dir ...]

Each event directory holds observables.csv written by ObservableRecorder or spf_observe (columns t_fm, A, mom2 ... mom7, ...).
Enhancement of moment N: E_N(t) = mom_N(t) / mom_N(t_init), t_init = time of the first row of the file.
Written to DIR (default "ensemble_out"):
  ensemble_enhancement.csv  t - t_init, then for N = 2..7: mean and standard error of E_N over the events
  ensemble_summary.json     per N: mean and standard error of the maximum enhancement over time (std / sqrt(n), sample std with n - 1),
                            event-to-event spread max/mean (largest event maximum divided by the mean maximum, as quoted in arXiv:1209.2462),
                            max/min, and the per-event maxima
All events must have the same time grid (to 1e-9 fm) unless --interp is given: then every event is linearly interpolated onto the time grid of
the first event (events shorter than the first one are an error). Only the Python standard library is used.
"""
import argparse
import bisect
import csv
import json
import math
import os
import sys

NS = range(2, 8)


def read_events(path):
    with open(os.path.join(path, "observables.csv"), newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    if not rows:
        raise SystemExit("no rows in " + path)
    t = [float(r["t_fm"]) for r in rows]
    m = {n: [float(r["mom%d" % n]) for r in rows] for n in NS}
    return t, m


def interp(t_new, t, y):
    out = []
    for x in t_new:
        if x < t[0] - 1e-9 or x > t[-1] + 1e-9:
            raise SystemExit("--interp: time %g outside the range of an event" % x)
        i = min(max(bisect.bisect_right(t, x) - 1, 0), len(t) - 2)
        w = (x - t[i]) / (t[i + 1] - t[i])
        out.append((1 - w) * y[i] + w * y[i + 1])
    return out


def mean_stderr(v):
    n = len(v)
    mu = sum(v) / n
    if n < 2:
        return mu, float("nan"), float("nan")
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1))
    return mu, sd / math.sqrt(n), sd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ensemble_out")
    ap.add_argument("--interp", action="store_true")
    ap.add_argument("events", nargs="+")
    a = ap.parse_args()

    data = [read_events(d) for d in a.events]
    t_ref = [x - data[0][0][0] for x in data[0][0]]
    enh = {n: [] for n in NS}  # [event][time]
    for t, m in data:
        tt = [x - t[0] for x in t]
        same = len(tt) == len(t_ref) and all(abs(x - y) <= 1e-9 for x, y in zip(tt, t_ref))
        if not same and not a.interp:
            raise SystemExit("time grids differ; use --interp")
        for n in NS:
            e = [x / m[n][0] for x in m[n]]  # a zero reference moment gives a ZeroDivisionError: a broken event
            enh[n].append(e if same else interp(t_ref, tt, e))

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ensemble_enhancement.csv"), "w") as f:
        f.write("t_minus_tinit_fm," + ",".join("N%d_mean,N%d_stderr" % (n, n) for n in NS) + "\n")
        for i, t in enumerate(t_ref):
            cols = []
            for n in NS:
                mu, se, _ = mean_stderr([e[i] for e in enh[n]])
                cols += ["%.17g" % mu, "%.17g" % se]
            f.write("%.17g," % t + ",".join(cols) + "\n")

    summary = {"n_events": len(a.events), "events": a.events, "interpolated": a.interp, "max_enhancement": {}}
    for n in NS:
        mx = [max(e) for e in enh[n]]
        mu, se, sd = mean_stderr(mx)
        summary["max_enhancement"]["N%d" % n] = {
            "mean": mu, "stderr": se, "std": sd,
            "spread_max_over_mean": max(mx) / mu if mu != 0 else float("nan"),
            "max_over_min": max(mx) / min(mx) if min(mx) != 0 else float("nan"),
            "per_event": mx,
        }
    with open(os.path.join(a.out, "ensemble_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print("obs_ensemble: %d events -> %s" % (len(a.events), a.out))


if __name__ == "__main__":
    sys.exit(main())
