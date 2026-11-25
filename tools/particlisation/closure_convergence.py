#!/usr/bin/env python3
"""Closure of the switching surface versus dx (WP10a, T10a.1 convergence): runs spf_collide on the in-repo Glauber-Gaussians state (Pb+Pb, full
overlap, 3 A GeV) in a 24 fm box with the collision settings (config/collisions/common.json) for each dx and each variant, and collects the closure
(elements + inside at the end - initial on the grid) / initial from the sidecars.  Standard library only.

  closure_convergence.py --exe build/cuda-turing75/spf_collide --out conv --dx 0.4 0.2 0.1 \
      [--variant remove:particlisation.initial_slice=remove --variant keep:particlisation.initial_slice=keep] [-- key=value ...]

Default variants: remove (default of the code) and keep.  Default table: the reduced test table two_phase (data/eos/test), as in the unit test;
pass eos.table_path=... after "--" for another table.  Output: <out>/closure_convergence.json and one spf_collide directory per run.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dx", nargs="+", type=float, default=[0.4, 0.2, 0.1])
    ap.add_argument("--box", type=float, default=24.0)
    ap.add_argument("--variant", action="append", default=[], help="label:key=value[,key=value]")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    variants = a.variant or ["remove:particlisation.initial_slice=remove", "keep:particlisation.initial_slice=keep"]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for v in variants:
        label, kv = v.split(":", 1)
        for dx in a.dx:
            d = os.path.join(a.out, "%s_dx%.2f" % (label, dx))
            cmd = [a.exe, "--out", d, "--config", os.path.join(ROOT, "config", "collisions", "common.json"),
                   "ic.generator=glauber_gaussians", "ic.e_lab_gev=3", "eos.model=table_two_phase",
                   "eos.table_path=\"%s\"" % os.path.join(ROOT, "data", "eos", "test", "eos_two_phase_test"),
                   "grid.dx_fm=%g" % dx, "grid.Lx_fm=%g" % a.box, "grid.Ly_fm=%g" % a.box, "grid.Lz_fm=%g" % a.box,
                   "run.n_events=1", "ic.max_outside_fraction=1e-3"] + [x for x in kv.split(",") if x] + a.extra
            print(" ".join(cmd), flush=True)
            rc = subprocess.call(cmd)
            ev = os.path.join(d, "event_000")
            with open(os.path.join(ev, "summary.json")) as f:
                S = json.load(f)
            with open(os.path.join(ev, "event_000.surface.json")) as f:
                side = json.load(f)
            row = {"variant": label, "dx_fm": dx, "rc": rc, "status": S.get("status"), "closed": side["closed"],
                   "end_time_after_init_fm": S.get("time_after_init_fm"), "n_elements": side["n_elements"],
                   "n_initial_slice": side["n_elements_initial_slice"], "closure": side["closure"],
                   "inflow_E_fraction": side["inflow_part_of_totals"]["E_GeV"] / side["totals_initial_on_grid"]["E_GeV"],
                   "file_MB": os.path.getsize(os.path.join(ev, "event_000.surface.bin")) / 1e6,
                   "wall_evolve_s": S.get("wall_evolve_s"), "wall_surface_s": S["particlisation"]["wall_surface_s"],
                   "backend": S.get("backend"), "edge_cells": side["diagnostics"]["max_cells_above_eps_sw_on_box_edge"]}
            rows.append(row)
            print("%-8s dx %.2f: closure E %.3e B %.3e Pz %.2e, %d elements, end %.2f fm/c, wall fluid %.1f s surface %.1f s"
                  % (label, dx, row["closure"]["E"], row["closure"]["B"], row["closure"]["Pz"], row["n_elements"], row["end_time_after_init_fm"],
                     row["wall_evolve_s"], row["wall_surface_s"]), flush=True)
            os.remove(os.path.join(ev, "event_000.surface.bin"))  # the sidecar keeps the totals
    with open(os.path.join(a.out, "closure_convergence.json"), "w") as f:
        json.dump({"box_fm": a.box, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
