#!/usr/bin/env python3
"""Box size from the baryon-number marginals written by spf_collide (event_*/marginals.json, run with a generous box).

  box_study.py --out box_study.json --limit 1e-4 [--dx-ref 0.2] DIR [DIR ...]      (one DIR per energy: the spf_collide output directory)

For each event the end-of-run marginals R dV summed over slabs of x, y, z give the baryon number outside |x| < X, |y| < Y, |z| < Z; the three
outside fractions are added (an upper bound of the fraction outside the box, whose boundary baryons cannot return). The smallest symmetric
half-widths (X = Y, Z), in steps of 1 fm, for which the largest outside fraction over the events of the directory stays below --limit
are reported; the start marginals are included in the maximum (the initial state itself must fit).
"""
import argparse
import glob
import json
import os

import numpy as np


def fractions(m, dx, X, Z):
    tot = []
    for key in ("start", "end"):
        mx, my, mz = (np.array(m[key][c]) for c in "xyz")
        B = mx.sum()

        def outside(v, h):
            xs = (np.arange(len(v)) + 0.5 - 0.5 * len(v)) * dx
            return v[np.abs(xs) > h].sum() / B

        tot.append(outside(mx, X) + outside(my, X) + outside(mz, Z))
    return max(tot)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="box_study.json")
    ap.add_argument("--limit", type=float, default=1e-4)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    res = {}
    for d in a.dirs:
        evs = sorted(glob.glob(os.path.join(d, "event_*", "marginals.json")))
        ms = [json.load(open(f)) for f in evs]
        if not ms:
            continue
        dx = ms[0]["dx_fm"]
        Lmax = [0.5 * len(ms[0]["end"][c]) * dx for c in "xyz"]
        best = None
        for X in np.arange(8.0, Lmax[0] + 1e-9, 1.0):
            for Z in np.arange(8.0, Lmax[2] + 1e-9, 1.0):
                f = max(fractions(m, dx, X, Z) for m in ms)
                if f < a.limit and (best is None or X * X * Z < best[0] * best[0] * best[1]):
                    best = (X, Z, f)
        # extents of the event with the largest spread: half-width containing all but 1e-6 of the baryons at the end
        res[os.path.basename(d.rstrip("/"))] = {
            "events": len(ms), "dx_fm": dx, "generous_box_fm": [2 * x for x in Lmax],
            "min_half_width_transverse_fm": None if best is None else best[0], "min_half_width_longitudinal_fm": None if best is None else best[1],
            "max_outside_fraction_at_minimum": None if best is None else best[2],
            "box_fm": None if best is None else [2 * best[0], 2 * best[0], 2 * best[1]],
        }
        print(os.path.basename(d.rstrip("/")), res[os.path.basename(d.rstrip("/"))])
    json.dump({"limit": a.limit, "per_energy": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
