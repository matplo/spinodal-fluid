#!/usr/bin/env python3
"""G5-INT end-to-end analysis: reads the output tree of g5_endtoend.sh and writes the summary numbers (json) and the figure.

  g5_analyse.py ROOT [--json OUT.json] [--fig OUT.png] [--oscar-check]

ROOT/runs/<eos>/event_NNN/{summary.json, event_NNN.surface.json}     (fluid + surface)
ROOT/samples/<eos>/<rule>/event_NNN/{sampled.oscar, sampled.json}    (sampler)
ROOT/timing/{sampling_wall.csv}
"""
import argparse, glob, json, math, os
import numpy as np

SPECIES = ["p", "n", "pbar", "nbar", "pi+", "pi-", "pi0"]
PDG = {"p": 2212, "pi-": -211}


def read_oscar(path):
    """OSCAR2013 particle list -> list of (header_line_count_n, array[n, 11]), also the number of 'out' header counts. Strict parse."""
    events, cur, declared = [], None, None
    with open(path) as f:
        first = f.readline()
        assert first.startswith("#!OSCAR2013"), "not an OSCAR2013 file"
        for line in f:
            if line.startswith("# event") and " out " in line:
                declared = int(line.split()[-1])
                cur = []
            elif line.startswith("# event") and " end " in line:
                events.append((declared, np.array(cur, dtype=float).reshape(-1, 11)))
                cur = None
            elif line.startswith("#"):
                continue
            else:
                v = line.split()
                assert len(v) == 12, f"bad particle line: {line[:80]}"
                cur.append([float(x) for x in v[:11]])  # t x y z m p0 px py pz pdg id   (charge is v[11])
    assert cur is None, "unterminated event block"
    return events


def mean_sem(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0


def analyse(root, oscar_check=True):
    out = {"eos": {}, "root": os.path.basename(root.rstrip("/"))}
    wall = {}
    with open(os.path.join(root, "timing/sampling_wall.csv")) as f:
        next(f)
        for l in f:
            e, r, ev, w, sz = l.strip().split(",")
            wall[(e, r, int(ev))] = (float(w), int(sz))
    hist = {}
    for eos in sorted(os.listdir(os.path.join(root, "runs"))):
        evs = sorted(glob.glob(os.path.join(root, "runs", eos, "event_*")))
        o_eos = {"events": [], "rules": {}}
        fl = {}
        for d in evs:
            n = int(d[-3:])
            s = json.load(open(f"{d}/summary.json"))
            sj = json.load(open(f"{d}/event_{n:03d}.surface.json"))
            p = s["particlisation"]
            surf_s = s["wall_surface_begin_s"] + p["wall_surface_s"]
            fl[n] = {"event": n, "status": s["status"], "steps": s["steps"], "wall_total_s": s["wall_total_s"], "wall_ic_s": s["wall_ic_s"],
                     "wall_evolve_s": s["wall_evolve_s"], "wall_observe_s": s["wall_observe_s"], "wall_surface_s": surf_s,
                     "wall_fluid_s": s["wall_total_s"] - surf_s, "n_elements": sj["n_elements"], "n_initial_slice": sj["n_elements_initial_slice"],
                     "n_surface": sj["n_elements_surface"], "closure_E": sj["closure"]["E"], "closure_B": sj["closure"]["B"],
                     "surface_E_GeV": sj["totals_all_elements"]["E_GeV"], "surface_Px_GeV": sj["totals_all_elements"]["Px_GeV"],
                     "surface_Pz_GeV": sj["totals_all_elements"]["Pz_GeV"], "surface_B": sj["totals_all_elements"]["B"], "t_init_fm": sj["t_init_fm"],
                     "dx_fm": sj["dx_fm"], "closed": sj["closed"]}
        o_eos["events"] = [fl[k] for k in sorted(fl)]
        for rule in sorted(os.listdir(os.path.join(root, "samples", eos))):
            per_ev = []
            pool = {k: [] for k in SPECIES}
            Eratio, Pxd, Pzd, init_frac, Bexact, Qexact, nsamp = [], [], [], [], 0, 0, 0
            nneg, prep_s, samp_s, wall_s, mu_floor, mu_fail, mu_res, nflag_spline = [], [], [], [], [], [], [], []
            ky = {"p": [], "pi-": []}
            for d in evs:
                n = int(d[-3:])
                sd = os.path.join(root, "samples", eos, rule, f"event_{n:03d}")
                j = json.load(open(sd + "/sampled.json"))
                osc = read_oscar(sd + "/sampled.oscar") if True else None
                assert len(osc) == len(j["samples"])
                t0 = fl[n]["t_init_fm"]
                ev_counts = {k: [] for k in SPECIES}
                for (decl, arr), s in zip(osc, j["samples"]):
                    assert decl == len(arr) == s["n_particles"], (decl, len(arr), s["n_particles"])
                    for k in SPECIES:
                        ev_counts[k].append(s["counts"].get(k, 0))
                    nsamp += 1
                    Bexact += int(s["sum"]["B"] == s["targets"]["B"])
                    Qexact += int(s["sum"]["Q"] == s["targets"]["Q"])
                    Eratio.append(s["sum"]["E_GeV"] / fl[n]["surface_E_GeV"])
                    Pxd.append(s["sum"]["Px_GeV"] - sj_val(root, eos, n, "Px_GeV"))
                    Pzd.append(s["sum"]["Pz_GeV"] - sj_val(root, eos, n, "Pz_GeV"))
                    init_frac.append(float(np.mean(np.abs(arr[:, 0] - t0) < 1e-6)) if len(arr) else 0.0)
                    E, pz = arr[:, 5], arr[:, 8]
                    y = 0.5 * np.log((E + pz) / (E - pz))
                    pt = np.hypot(arr[:, 6], arr[:, 7])
                    pdg = arr[:, 9].astype(int)
                    for k, c in PDG.items():
                        ky[k].append((pt[pdg == c], y[pdg == c]))
                pr = j["preparation"]
                mu_floor.append(pr.get("mu_rule_n_at_T_floor", 0)); mu_fail.append(pr.get("mu_rule_n_solver_failed", 0))
                mu_res.append(pr.get("mu_rule_max_rel_residual", 0.0)); nflag_spline.append(pr["n_spline_region_flagged"])
                prep_s.append(j["seconds_preparation"]); samp_s.append(j["seconds_sampling"]); wall_s.append(wall[(eos, rule, n)][0])
                ev_mean = {k: float(np.mean(v)) for k, v in ev_counts.items()}
                per_ev.append({"event": n, "mean_multiplicity": ev_mean, "B_target": j["preparation"]["B_target"], "Q_target": j["preparation"]["Q_target"],
                               "seconds_preparation": j["seconds_preparation"], "seconds_sampling": j["seconds_sampling"], "wall_spf_sample_s": wall[(eos, rule, n)][0],
                               "surface_bytes": wall[(eos, rule, n)][1]})
                for k in SPECIES:
                    pool[k].extend(ev_counts[k])
            evE = [np.mean(Eratio[i * len(Eratio) // len(evs):(i + 1) * len(Eratio) // len(evs)]) for i in range(len(evs))]
            surfE = np.array([e["surface_E_GeV"] for e in o_eos["events"]])
            r = {"n_events": len(evs), "n_samples_total": nsamp,
                 "mean_multiplicity_per_sample": {k: {"mean": mean_sem(pool[k])[0], "sem": mean_sem(pool[k])[1]} for k in SPECIES},
                 "B_exact_fraction": Bexact / nsamp, "Q_exact_fraction": Qexact / nsamp,
                 "E_particles_over_E_surface": {"mean_over_samples": mean_sem(Eratio)[0], "sem": mean_sem(Eratio)[1], "min": float(np.min(Eratio)), "max": float(np.max(Eratio)),
                                                  "per_event_mean": [float(x) for x in evE]},
                 "Px_particles_minus_surface_GeV": {"mean": mean_sem(Pxd)[0], "sem": mean_sem(Pxd)[1], "rms": float(np.sqrt(np.mean(np.square(Pxd))))},
                 "Pz_particles_minus_surface_GeV": {"mean": mean_sem(Pzd)[0], "sem": mean_sem(Pzd)[1], "rms": float(np.sqrt(np.mean(np.square(Pzd))))},
                 "surface_Px_GeV_per_event": [e["surface_Px_GeV"] for e in o_eos["events"]], "surface_Pz_GeV_per_event": [e["surface_Pz_GeV"] for e in o_eos["events"]],
                 "fraction_particles_from_initial_slice": {"mean": mean_sem(init_frac)[0], "sem": mean_sem(init_frac)[1]},
                 "wall_s_per_event": {"spf_sample_total": float(np.mean(wall_s)), "preparation_incl_surface_read": float(np.mean(prep_s)), "sampling_K_samples": float(np.mean(samp_s))},
                 "mu_rule": {"n_elements_at_T_floor_per_event": mu_floor, "n_solver_failed_per_event": mu_fail, "max_rel_residual": float(np.max(mu_res)),
                             "n_spline_region_flagged_per_event": nflag_spline},
                 "per_event": per_ev}
            o_eos["rules"][rule] = r
            hist[(eos, rule)] = ky
        out["eos"][eos] = o_eos
    return out, hist


_cache = {}


def sj_val(root, eos, n, key):
    k = (root, eos, n)
    if k not in _cache:
        _cache[k] = json.load(open(f"{root}/runs/{eos}/event_{n:03d}/event_{n:03d}.surface.json"))["totals_all_elements"]
    return _cache[k][key]


def make_figure(hist, fname, nsamples_note=""):
    import matplotlib.pyplot as plt
    eos_names = {"two_phase": "two-phase EoS", "maxwell": "Maxwell EoS"}
    rule_names = {"rho_matched": r"$\mu_{\rm kin}$ from $n_N - n_{\bar N} = \rho$ (S24)", "table": r"$\mu_{\rm kin} = \mu_B - dw/d\rho$ (table)"}
    pt_edges = np.linspace(0, 2.0, 21)
    y_edges = np.linspace(-2.4, 2.4, 19)
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.2), sharey="col")
    cols = {"p": "#0072B2", "pi-": "#D55E00"}
    labels = {"p": "p", "pi-": r"$\pi^-$"}
    for i, eos in enumerate(["two_phase", "maxwell"]):
        for jr, rule in enumerate(["rho_matched", "table"]):
            ky = hist[(eos, rule)]
            for k in ["p", "pi-"]:
                ns = len(ky[k])
                pt = np.concatenate([a for a, _ in ky[k]]); y = np.concatenate([b for _, b in ky[k]])
                for ax, x, ed, ix in ((axs[i, 2 * jr], pt, pt_edges, 0), (axs[i, 2 * jr + 1], y, y_edges, 1)):
                    h, _ = np.histogram(x, ed)
                    w = np.diff(ed)
                    ax.stairs(h / ns / w, ed, color=cols[k], label=labels[k], lw=1.4)
                    c = 0.5 * (ed[1:] + ed[:-1])
                    ax.errorbar(c, h / ns / w, yerr=np.sqrt(h) / ns / w, fmt="none", ecolor=cols[k], lw=0.8, alpha=0.7)
            for ix in (0, 1):
                ax = axs[i, 2 * jr + ix]
                ax.set_yscale("log"); ax.set_ylim(0.05, 3000)
                ax.set_xlabel(r"$p_T$ (GeV/$c$)" if ix == 0 else r"$y$")
                if 2 * jr + ix in (0, 2):
                    ax.set_ylabel(r"$dN/dp_T$ (per GeV/$c$)")
                else:
                    ax.set_ylabel(r"$dN/dy$")
                ax.legend(frameon=False, loc="upper right", ncol=2)
            axs[i, 2 * jr].set_title(eos_names[eos], loc="left", fontsize=9)
    for jr, rule in enumerate(["rho_matched", "table"]):
        fig.text(0.27 + 0.47 * jr, 0.985, rule_names[rule], ha="center", va="top", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fname, dpi=200)
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default=None)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    res, hist = analyse(a.root)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if a.fig:
        make_figure(hist, a.fig)
    for eos, o in res["eos"].items():
        for rule, r in o["rules"].items():
            print(eos, rule, {k: round(v["mean"], 3) for k, v in r["mean_multiplicity_per_sample"].items()}, "E ratio", round(r["E_particles_over_E_surface"]["mean_over_samples"], 4),
                  "init", round(r["fraction_particles_from_initial_slice"]["mean"], 3), "B,Q exact", r["B_exact_fraction"], r["Q_exact_fraction"])
