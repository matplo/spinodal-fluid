#!/usr/bin/env python3
"""Matching densities a(T), b(T) of the two-phase construction (tools/eos/finite_T/eos2b.py, class TwoPhase, method ab(T)) on a T grid, for the
surface census (census.py): an element with a(T) < rho < b(T) lies in the spline (quintic-join) region of the two-phase table.

  make_ab_table.py [--out tools/particlisation/data/ab_two_phase.json] [--tmax 300] [--dT 0.25]

Output: JSON with T_MeV, a_rho_s, b_rho_s (densities in units of rho_s = 0.153 fm^-3).  Needs numpy and scipy (eos2b).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "eos", "finite_T"))
import eos2b as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "ab_two_phase.json"))
    ap.add_argument("--tmax", type=float, default=300.0)
    ap.add_argument("--dT", type=float, default=0.25)
    a = ap.parse_args()
    M = E.TwoPhase(os.path.join(ROOT, "data", "eos", "finite_T", "coex_pure.npz"))
    T = np.round(np.arange(0.0, a.tmax + 0.5 * a.dT, a.dT), 6)
    ab = np.array([M.ab(float(t)) for t in T]) / E.rhos
    out = {"description": "matching densities a(T), b(T) of TwoPhase.ab (tools/eos/finite_T/eos2b.py), densities in units of rho_s",
           "rho_s_fm3": E.rhos, "T_MeV": T.tolist(), "a_rho_s": [round(float(x), 9) for x in ab[:, 0]],
           "b_rho_s": [round(float(x), 9) for x in ab[:, 1]]}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(a.out, len(T), "temperatures; a(0) = %.4f, b(0) = %.4f rho_s" % (ab[0, 0], ab[0, 1]))


if __name__ == "__main__":
    main()
