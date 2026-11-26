#!/usr/bin/env python3
"""WP10e analysis of the corona-mode comparison written by corona_modes.sh.

  corona_analyse.py ROOT [--json OUT.json]

ROOT/runs/<cfg>/<eos>/event_NNN/{summary.json, event_NNN.surface.json}   fluid + surface
ROOT/samples/<cfg>/<eos>/event_NNN/{sampled.oscar, sampled.json}         sampler (K samples per event)
ROOT/census/<cfg>_<eos>/census.json                                      spline-region share (census.py)

Per configuration and EoS (mean over events, and per event): fractions of the final particles, of E and of B that come from the corona,
the initial slice, the iso-energy surface and the final slice; multiplicities; closure; wall times; the clumping-sensitive sanity observables
(variance of the proton number in |y| < 0.5 per sample, pooled within events; proton-proton pair Delta p distribution of the thermal part
normalised to mixed pairs from different samples of the same event).  numpy only.
"""
import argparse, glob, json, math, os
import numpy as np

CATS = ["corona", "initial_slice", "surface", "final_slice"]
DP_EDGES = np.linspace(0.0, 0.3, 13)       # GeV
DP_NORM_MAX = 2.0                           # normalisation range of the pair distributions [GeV]


def read_oscar(path):
    """-> list of dict(t, E, px, py, pz, pdg) arrays per event block."""
    blocks, cur = [], None
    with open(path) as f:
        f.readline()
        for line in f:
            if line.startswith("# event") and " out " in line:
                cur = []
            elif line.startswith("# event") and " end " in line:
                a = np.array(cur, float).reshape(-1, 12)
                blocks.append({"t": a[:, 0], "E": a[:, 5], "px": a[:, 6], "py": a[:, 7], "pz": a[:, 8], "pdg": a[:, 9].astype(int)})
                cur = None
            elif line.startswith("#"):
                continue
            else:
                cur.append([float(x) for x in line.split()])
    return blocks


def baryon(pdg):
    return np.isin(pdg, [2212, 2112]).astype(int) - np.isin(pdg, [-2212, -2112]).astype(int)


def rapidity(E, pz):
    return 0.5 * np.log((E + pz) / np.maximum(E - pz, 1e-300))


def pair_hist(P, Q=None):
    """Histogram of |p1 - p2| for pairs inside P (i < j) or between P and Q; P, Q arrays (n, 3)."""
    if P.shape[0] == 0 or (Q is not None and Q.shape[0] == 0):
        return np.zeros(len(DP_EDGES) - 1), 0.0
    if Q is None:
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)[np.triu_indices(P.shape[0], 1)]
    else:
        d = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2).ravel()
    return np.histogram(d, bins=DP_EDGES)[0].astype(float), float(np.count_nonzero(d < DP_NORM_MAX))


def analyse_event(rd, sd):
    s = json.load(open(os.path.join(rd, "summary.json")))
    side = json.load(open(glob.glob(os.path.join(rd, "event_*.surface.json"))[0]))
    sj = json.load(open(os.path.join(sd, "sampled.json")))
    blocks = read_oscar(os.path.join(sd, "sampled.oscar"))
    t_init = side["t_init_fm"]
    t_fin = side.get("final_slice_info", {}).get("time_fm", None)
    co = side.get("corona", {})
    has_corona = bool(co.get("applied", False))
    row = {"event": s["event"], "status": s["status"], "closed": side["closed"], "time_after_init_fm": s["time_after_init_fm"],
           "closure_E": side["closure"]["E"], "closure_B": side["closure"]["B"],
           "n_elements": side["n_elements"], "n_initial": side["n_elements_initial_slice"], "n_final": side.get("n_elements_final_slice", 0),
           "wall_fluid_s": s["wall_evolve_s"], "wall_surface_s": s["particlisation"]["wall_surface_s"],  # begin + steps + final slice
           "wall_final_slice_s": side.get("wall_final_slice_s", 0.0), "wall_ic_s": s["wall_ic_s"], "wall_corona_s": s.get("wall_corona_s", 0.0),
           "wall_sampling_s": sj["seconds_sampling"], "wall_sampling_prep_s": sj["seconds_preparation"], "wall_total_s": s["wall_total_s"]}
    # flux totals of the file
    E_cor = co["corona"]["E_GeV"] if has_corona else 0.0
    B_cor = co["corona"]["B"] if has_corona else 0.0
    parts = {"corona": (E_cor, B_cor), "initial_slice": (side["totals_initial_slice"]["E_GeV"], side["totals_initial_slice"]["B"]),
             "surface": (side["totals_surface"]["E_GeV"], side["totals_surface"]["B"]),
             "final_slice": (side.get("totals_final_slice", {}).get("E_GeV", 0.0), side.get("totals_final_slice", {}).get("B", 0.0))}
    Et = sum(v[0] for v in parts.values()); Bt = sum(v[1] for v in parts.values())
    row["flux_fraction_E"] = {k: v[0] / Et for k, v in parts.items()}
    row["flux_fraction_B"] = {k: v[1] / Bt for k, v in parts.items()}
    E_all = (co["corona"]["E_GeV"] + co["core"]["E_GeV"]) if has_corona else side["totals_initial_on_grid"]["E_GeV"]
    B_all = (co["corona"]["B"] + co["core"]["B"]) if has_corona else side["totals_initial_on_grid"]["B"]
    inside = side["totals_inside_fluid_at_end"]
    row["chain_dev_E"] = (E_cor + side["totals_all_elements"]["E_GeV"] + inside["E_GeV"] - E_all) / E_all
    row["chain_dev_B"] = (B_cor + side["totals_all_elements"]["B"] + inside["B"] - B_all) / B_all
    if has_corona:
        row["corona"] = {"n_corona": co["n_corona"], "n_total": co["n_particles_total"], "E_GeV": E_cor, "B": B_cor, "species_pdg": co["corona_species_pdg"],
                         "core_share_E": co["core"]["E_GeV"] / E_all}
    # share of the thin Gaussian tail below eps_sw in the core deposition (initial slice of the core_corona mode)
    if has_corona:
        row["core_tail_slice_E_over_core_deposit"] = side["totals_initial_slice"]["E_GeV"] / co["deposited_core_E_GeV"]
        row["core_tail_slice_B_over_core_deposit"] = side["totals_initial_slice"]["B"] / co["deposited_core_B"]
    # samples
    cnt = {k: [] for k in ("N", "p", "n", "pip", "pim", "pi0")}
    fr_n = {c: [] for c in CATS}; fr_E = {c: [] for c in CATS}; fr_B = {c: [] for c in CATS}
    Np_mid, thermal_p, hist_same, hist_ncount = [], [], np.zeros(len(DP_EDGES) - 1), 0.0
    for b, sm in zip(blocks, sj["samples"]):
        n = len(b["t"])
        cor = np.zeros(n, bool)
        if "corona" in sm:
            cor[sm["corona"]["first_index"]:sm["corona"]["end_index"]] = True
        tt = b["t"]
        ini = (~cor) & (np.abs(tt - t_init) < 1e-8)
        fin = (~cor) & (t_fin is not None) & (np.abs(tt - (t_fin if t_fin is not None else 0.0)) < 1e-8) & ~ini
        sur = ~(cor | ini | fin)
        masks = {"corona": cor, "initial_slice": ini, "surface": sur, "final_slice": fin}
        B = baryon(b["pdg"]).astype(float)
        for c in CATS:
            m = masks[c]
            fr_n[c].append(m.sum() / n); fr_E[c].append(b["E"][m].sum() / b["E"].sum())
            fr_B[c].append(B[m].sum() / B.sum() if B.sum() != 0 else 0.0)
        pdg = b["pdg"]
        cnt["N"].append(n); cnt["p"].append(int(np.sum(pdg == 2212))); cnt["n"].append(int(np.sum(pdg == 2112)))
        cnt["pip"].append(int(np.sum(pdg == 211))); cnt["pim"].append(int(np.sum(pdg == -211))); cnt["pi0"].append(int(np.sum(pdg == 111)))
        y = rapidity(b["E"], b["pz"])
        isp = (pdg == 2212)
        Np_mid.append(int(np.sum(isp & (np.abs(y) < 0.5))))
        thermal_p.append(np.stack([b["px"], b["py"], b["pz"]], axis=1)[isp & ~cor & (np.abs(y) < 0.5)])
    K = len(blocks)
    row["K"] = K
    row["mean_counts"] = {k: float(np.mean(v)) for k, v in cnt.items()}
    row["particle_fraction"] = {c: float(np.mean(fr_n[c])) for c in CATS}
    row["energy_fraction_sampled"] = {c: float(np.mean(fr_E[c])) for c in CATS}
    row["baryon_fraction_sampled"] = {c: float(np.mean(fr_B[c])) for c in CATS}
    row["Np_mid"] = Np_mid
    row["sampled_B_event_totals"] = [sm.get("event_totals", sm["sum"])["B"] for sm in sj["samples"]]
    row["sampled_B_targets"] = [sm["targets"]["B"] for sm in sj["samples"]]
    row["spline_flagged"] = sj["preparation"].get("n_spline_flagged", None)
    # pair distributions (same sample, mixed: all pairs of different samples)
    same = np.zeros(len(DP_EDGES) - 1); mixed = np.zeros(len(DP_EDGES) - 1); ns = nm = 0.0
    for i in range(K):
        h, c = pair_hist(thermal_p[i]); same += h; ns += c
        for j in range(i + 1, K):
            h, c = pair_hist(thermal_p[i], thermal_p[j]); mixed += h; nm += c
    row["pp_same"] = same.tolist(); row["pp_mixed"] = mixed.tolist(); row["pp_norm_same"] = ns; row["pp_norm_mixed"] = nm
    return row


def pooled(rows):
    """Pooled within-event scaled variance of N_p(|y| < 0.5) and the pair-correlation ratio over the events."""
    num = den = 0.0
    for r in rows:
        v = np.array(r["Np_mid"], float)
        num += v.var(ddof=1) * (len(v) - 1); den += v.mean() * (len(v) - 1)
    same = np.sum([r["pp_same"] for r in rows], axis=0); mixed = np.sum([r["pp_mixed"] for r in rows], axis=0)
    ns = sum(r["pp_norm_same"] for r in rows); nm = sum(r["pp_norm_mixed"] for r in rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = (same / ns) / (mixed / nm)
        err = C * np.sqrt(1.0 / np.maximum(same, 1) + 1.0 / np.maximum(mixed, 1))
    return {"omega_p_mid": num / den, "pp_correlation": C.tolist(), "pp_correlation_err_poisson": err.tolist(), "pp_same_pairs": same.tolist(),
            "pp_mixed_pairs": mixed.tolist(), "dp_edges_GeV": DP_EDGES.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {"configs": {}}
    for cfgdir in sorted(glob.glob(os.path.join(a.root, "runs", "*"))):
        cfg = os.path.basename(cfgdir)
        out["configs"][cfg] = {}
        for eosdir in sorted(glob.glob(os.path.join(cfgdir, "*"))):
            eos = os.path.basename(eosdir)
            rows = []
            for ev in sorted(glob.glob(os.path.join(eosdir, "event_*"))):
                name = os.path.basename(ev)
                rows.append(analyse_event(ev, os.path.join(a.root, "samples", cfg, eos, name)))
            mean = lambda f: float(np.mean([f(r) for r in rows]))
            agg = {"events": rows, "n_events": len(rows),
                   "particle_fraction": {c: mean(lambda r: r["particle_fraction"][c]) for c in CATS},
                   "energy_fraction_sampled": {c: mean(lambda r: r["energy_fraction_sampled"][c]) for c in CATS},
                   "baryon_fraction_sampled": {c: mean(lambda r: r["baryon_fraction_sampled"][c]) for c in CATS},
                   "flux_fraction_E": {c: mean(lambda r: r["flux_fraction_E"][c]) for c in CATS},
                   "flux_fraction_B": {c: mean(lambda r: r["flux_fraction_B"][c]) for c in CATS},
                   "mean_counts": {k: mean(lambda r: r["mean_counts"][k]) for k in rows[0]["mean_counts"]},
                   "closure_E": mean(lambda r: r["closure_E"]), "closure_B": mean(lambda r: r["closure_B"]),
                   "chain_dev_E": mean(lambda r: r["chain_dev_E"]), "chain_dev_B": mean(lambda r: r["chain_dev_B"]),
                   "wall_per_event_s": {k: mean(lambda r: r[k]) for k in ("wall_fluid_s", "wall_surface_s", "wall_final_slice_s", "wall_ic_s", "wall_corona_s",
                                                                            "wall_sampling_s", "wall_sampling_prep_s", "wall_total_s")}}
            agg.update(pooled(rows))
            cj = os.path.join(a.root, "census", cfg + "_" + eos, "census.json")
            if os.path.exists(cj):
                agg["census"] = json.load(open(cj))["runs"]
                for v in agg["census"].values():
                    v.pop("events", None) if isinstance(v, dict) else None
            out["configs"][cfg][eos] = agg
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    for cfg, d in out["configs"].items():
        for eos, g in d.items():
            pf = g["particle_fraction"]
            print(f"{cfg:12s} {eos:9s} particles: " + " ".join(f"{c} {pf[c]:.3f}" for c in CATS) + f"  closure E {g['closure_E']:.2e} B {g['closure_B']:.2e}  omega_p {g['omega_p_mid']:.3f}")


if __name__ == "__main__":
    main()
