#!/usr/bin/env python3
"""Surface census of spf_collide runs with particlisation (WP10a): element counts, file sizes, end times, closure errors, histograms of the
elements in (rho / rho_s, T) weighted by the energy flux T^{0 mu} dSigma_mu and by the baryon flux N^mu dSigma_mu, and the flux-weighted fraction
of elements in the spline (two-phase) region a(T) < rho < b(T) of the two-phase construction (make_ab_table.py).  numpy only.

  census.py --run two_phase_sw4=runs/tp4 --run maxwell_sw4=runs/mx4 ... --table table_two_phase=data/eos/generated/eos_two_phase \
            --table table_maxwell=data/eos/generated/eos_maxwell --ab tools/particlisation/data/ab_two_phase.json --out census [--delete-bin]

Each run directory holds event_NNN/summary.json and event_NNN/event_NNN.surface.{bin,json}.  The energy flux uses p0 from a bilinear interpolation
of the table named in the sidecar (eos_model); the runs themselves use the interpolation of their configuration (bicubic in the collision files),
so the Python energy totals differ slightly from the sidecar totals; both are reported.  --delete-bin removes each .surface.bin after it was read.
Output: <out>/census.json (per run: per-event rows and statistics) and <out>/census_hist.npz (2D histograms per run).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surface_io import COL, TableP0, file_size_mb, fluxes, read_surface  # noqa: E402

RHO_EDGES = np.arange(0.0, 12.0 + 1e-9, 0.05)    # rho / rho_s
T_EDGES = np.arange(0.0, 250.0 + 1e-9, 1.0)       # MeV


def stats(xs):
    a = np.asarray(xs, float)
    if a.size == 0:
        return None
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0, "min": float(a.min()), "max": float(a.max()), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="label=directory")
    ap.add_argument("--table", action="append", required=True, help="eos_model=table stem")
    ap.add_argument("--ab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delete-bin", action="store_true")
    a = ap.parse_args()
    tables = {}
    for t in a.table:
        k, v = t.split("=", 1)
        tables[k] = TableP0(v)
    with open(a.ab) as f:
        ab = json.load(f)
    abT, abA, abB = np.array(ab["T_MeV"]), np.array(ab["a_rho_s"]), np.array(ab["b_rho_s"])
    os.makedirs(a.out, exist_ok=True)
    result = {"rho_edges_rho_s": RHO_EDGES.tolist(), "T_edges_MeV": T_EDGES.tolist(), "ab_file": os.path.basename(a.ab), "runs": {}}
    hists = {}
    for r in a.run:
        label, d = r.split("=", 1)
        rows = []
        H = {k: np.zeros((len(RHO_EDGES) - 1, len(T_EDGES) - 1)) for k in
             ("surface_E", "surface_B", "initial_E", "initial_B", "surface_N")}
        for ev in sorted(glob.glob(os.path.join(d, "event_*"))):
            name = os.path.basename(ev)
            base = os.path.join(ev, name)
            if not os.path.exists(base + ".surface.json"):
                continue
            with open(os.path.join(ev, "summary.json")) as f:
                S = json.load(f)
            recs, side = read_surface(base)
            size = file_size_mb(base)
            tab = tables[side["eos_model"]]
            if tab.sha256 and side.get("eos_table_sha256") and tab.sha256 != side["eos_table_sha256"]:
                raise SystemExit("table checksum differs from the run: %s" % base)
            p0 = tab.p0(recs[:, COL["eps"]], recs[:, COL["rho"]])
            dE, dB = fluxes(recs, p0)
            nini = side["n_elements_initial_slice"]
            rho = recs[:, COL["rho"]] / 0.153
            T = recs[:, COL["T"]] * 1e3
            a_T = np.interp(T, abT, abA)
            b_T = np.interp(T, abT, abB)
            spline = (rho > a_T) & (rho < b_T)
            row = {"event": name, "status": S.get("status"), "n_elements": int(side["n_elements"]), "n_initial_slice": int(nini),
                   "n_surface": int(side["n_elements_surface"]), "file_MB": size,
                   "end_time_after_init_fm": S.get("time_after_init_fm"), "closed": bool(side.get("closed")),
                   "closure_E": side["closure"]["E"], "closure_B": side["closure"]["B"], "closure_Pz": side["closure"]["Pz"],
                   "wall_total_s": S.get("wall_total_s"), "wall_evolve_s": S.get("wall_evolve_s"),
                   "wall_surface_s": S.get("particlisation", {}).get("wall_surface_s"), "steps": S.get("steps"),
                   "peak_rss_MB": S.get("peak_rss_MB"), "recovery_failures": S.get("recovery_failures"),
                   "max_cells_above_eps_sw_on_box_edge": side["diagnostics"]["max_cells_above_eps_sw_on_box_edge"],
                   "element_recovery_failures": side["diagnostics"].get("element_recovery_failures"),
                   "light_like_note": "elements with gamma = 1e4 are dilute light-like cells (state_from_conserved)",
                   "initial_on_grid_E": side["totals_initial_on_grid"]["E_GeV"], "initial_on_grid_B": side["totals_initial_on_grid"]["B"]}
            for part, sl in (("surface", slice(nini, None)), ("initial", slice(0, nini)), ("all", slice(None))):
                e_w, b_w, sp = dE[sl], dB[sl], spline[sl]
                row[part + "_E_py"] = float(e_w.sum())
                row[part + "_B_py"] = float(b_w.sum())
                row[part + "_E_neg"] = float(e_w[e_w < 0].sum())
                row[part + "_B_neg"] = float(b_w[b_w < 0].sum())
                row[part + "_spline_frac_E"] = float(e_w[sp].sum() / e_w.sum()) if e_w.sum() != 0 else None
                row[part + "_spline_frac_B"] = float(b_w[sp].sum() / b_w.sum()) if b_w.sum() != 0 else None
                row[part + "_spline_frac_N"] = float(sp.mean()) if sp.size else None
            row["sidecar_surface_E"] = side["totals_surface"]["E_GeV"]
            row["sidecar_initial_E"] = side["totals_initial_slice"]["E_GeV"]
            for part, sl in (("surface", slice(nini, None)), ("initial", slice(0, nini))):
                H[part + "_E"] += np.histogram2d(rho[sl], T[sl], bins=[RHO_EDGES, T_EDGES], weights=dE[sl])[0]
                H[part + "_B"] += np.histogram2d(rho[sl], T[sl], bins=[RHO_EDGES, T_EDGES], weights=dB[sl])[0]
            H["surface_N"] += np.histogram2d(rho[nini:], T[nini:], bins=[RHO_EDGES, T_EDGES])[0]
            # outside the histogram range
            row["surface_E_outside_hist"] = float(dE[nini:][(rho[nini:] >= RHO_EDGES[-1]) | (T[nini:] >= T_EDGES[-1])].sum())
            rows.append(row)
            del recs
            if a.delete_bin:
                os.remove(base + ".surface.bin")
            print("%s %s: %d elements (%d initial), %.1f MB, end %.2f fm/c, closure E %.2e B %.2e, spline fraction (surface) E %.3f B %.3f"
                  % (label, name, row["n_elements"], nini, size, row["end_time_after_init_fm"] or -1, row["closure_E"], row["closure_B"],
                     row["surface_spline_frac_E"] or -1, row["surface_spline_frac_B"] or -1), flush=True)
        keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)] if rows else []
        result["runs"][label] = {"dir": d, "events": rows, "stats": {k: stats([x[k] for x in rows if x[k] is not None]) for k in keys}}
        # pooled spline fractions from the histograms are not exact (binning); pooled fractions from the per-event sums:
        for part in ("surface", "initial", "all"):
            for w in ("E", "B"):
                num = sum((x[part + "_spline_frac_" + w] or 0.0) * x[part + "_" + w + "_py"] for x in rows)
                den = sum(x[part + "_" + w + "_py"] for x in rows)
                result["runs"][label]["pooled_%s_spline_frac_%s" % (part, w)] = num / den if den else None
        for k, v in H.items():
            hists[label + "__" + k] = v
    with open(os.path.join(a.out, "census.json"), "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(os.path.join(a.out, "census_hist.npz"), rho_edges=RHO_EDGES, T_edges=T_EDGES, **hists)
    print("wrote", os.path.join(a.out, "census.json"))


if __name__ == "__main__":
    main()
