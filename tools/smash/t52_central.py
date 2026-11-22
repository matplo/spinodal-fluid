#!/usr/bin/env python3
"""T5.2: event-averaged central (rho, eps) of the deposited SMASH states, via spf_ic (one call per event).

  t52_central.py --spf-ic build/spf_ic --runs <dir with prod_<E>/smash_data/particle_lists.oscar> --out t52_summary.json \
                 [--cold-curve data/eos/cold_curve.dat] [--energies 1 2 3 4 5 6] [key=value ...]

Central = cells within 1 fm of the box centre, rest-frame values after primitive recovery (eos.model of the call, default cold_thermal_two_phase).
Compares with the counter-streaming point (2 gamma_cm rho_s, sqrt(s_NN) gamma_cm rho_s) and with the cold curve eps_c(rho) / eps_s.
"""
import argparse, glob, json, math, os, subprocess, sys, tempfile

RHO_S, M_N = 0.153, 0.940
EPS_S = M_N * RHO_S


def load_cold(path):
    rho, eps = [], []
    for ln in open(path):
        if ln.startswith("#") or not ln.strip():
            continue
        w = ln.split()
        rho.append(float(w[0])); eps.append(float(w[1]))
    return rho, eps


def interp(x, xs, ys):
    if x <= xs[0]: return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return float("nan")


def mean_std(v):
    n = len(v); m = sum(v) / n
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spf-ic", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cold-curve", default="data/eos/cold_curve.dat")
    ap.add_argument("--energies", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cr, ce = load_cold(a.cold_curve)
    res = {}
    for E in a.energies:
        f = os.path.join(a.runs, "prod_%d" % E, "smash_data", "particle_lists.oscar")
        if not os.path.exists(f):
            print("missing", f, file=sys.stderr); continue
        nev = sum(1 for ln in open(f) if ln.startswith("# event") and " end " in ln)
        evs = []
        for ev in range(nev):
            with tempfile.TemporaryDirectory() as td:
                cmd = [a.spf_ic, "--out", td, "--no-snapshot", "ic.generator=smash_cascade", "ic.oscar_path=" + f, "ic.oscar_event=%d" % ev,
                       "ic.e_lab_gev=%g" % E, "grid.nx=160", "grid.ny=160", "grid.nz=160", "grid.dx_fm=0.2",
                       "eos.model=cold_thermal_two_phase"] + a.overrides
                r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_PROC_BIND="false"))
                if r.returncode != 0:
                    print("spf_ic failed E=%d event %d: %s" % (E, ev, r.stderr[-400:]), file=sys.stderr); continue
                evs.append(json.load(open(os.path.join(td, "ic_summary.json"))))
        if not evs: continue
        rho = [e["central_rho_over_rho_s"] for e in evs]; eps = [e["central_eps_over_eps_s"] for e in evs]
        yfrac = [e["baryon_abs_y_lt_0p5"] / e["baryon_total"] for e in evs]
        g = evs[0]["gamma_cm"]
        mr, sr = mean_std(rho); me, se = mean_std(eps)
        res[str(E)] = {
            "n_events": len(evs), "t_init_fm": evs[0]["t_init_fm"], "gamma_cm": g,
            "central_rho_over_rho_s": {"mean": mr, "std": sr, "min": min(rho), "max": max(rho)},
            "central_eps_over_eps_s": {"mean": me, "std": se, "min": min(eps), "max": max(eps)},
            "counterstreaming": {"rho_over_rho_s": 2 * g, "eps_over_eps_s": evs[0]["sqrt_snn_gev"] * g / M_N},
            "cold_curve_eps_over_eps_s_at_mean_rho": interp(mr * RHO_S, cr, ce) / EPS_S,
            "cold_curve_eps_over_eps_s_at_counterstreaming_rho": interp(2 * g * RHO_S, cr, ce) / EPS_S,
            "baryon_fraction_abs_y_lt_0p5": {"mean": mean_std(yfrac)[0], "std": mean_std(yfrac)[1]},
            "baryon_total": evs[0]["baryon_total"],
            "n_hadrons_mean": sum(e["n_hadrons"] for e in evs) / len(evs),
            "max_rho_over_rho_s_mean": sum(e["max_rho_over_rho_s"] for e in evs) / len(evs),
            "max_eps_over_eps_s_mean": sum(e["max_eps_over_eps_s"] for e in evs) / len(evs),
            "recovery_failures_total": sum(e["recovery_failures"] for e in evs),
            "limiter_cells_total": sum(e["limiter_cells"] for e in evs),
            "outside_fraction_B_max": max(abs(e["outside_fraction_B"]) for e in evs),
            "per_event": [{"rho": r_, "eps": e_, "y_frac": y_} for r_, e_, y_ in zip(rho, eps, yfrac)],
        }
        print("E=%d: n=%d rho/rho_s %.2f +- %.2f  eps/eps_s %.2f +- %.2f  (cs: %.2f, %.2f)  cold eps_c/eps_s at rho %.2f  |y|<0.5: %.3f" % (
            E, len(evs), mr, sr, me, se, 2 * g, res[str(E)]["counterstreaming"]["eps_over_eps_s"],
            res[str(E)]["cold_curve_eps_over_eps_s_at_mean_rho"], mean_std(yfrac)[0]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
