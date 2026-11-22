#!/usr/bin/env python3
"""Time convention of SMASH's collider modus, measured from a run made with `run_smash.py --measure`.

  time_convention.py <measure dir with smash_data/particle_lists.oscar and full_event_history.oscar> [--out result.json]

Reports per event (i) the time of the first nucleon-nucleon collision, (ii) the time at which the baryon density in a central
cylinder (r_perp < 3 fm, |z| < 1.5 fm; computational frame = centre-of-velocity frame) is maximal (smoothed over 5 output times of 0.5 fm/c),
(iii) the time of 50 % of the nucleon-nucleon collisions that occur before the end. Requires numpy.
"""
import argparse, json, math, os
import numpy as np

RHO_S = 0.153


def is_baryon(pdg):
    a = abs(pdg)
    return (a // 1000) % 10 > 0 and (a // 100) % 10 > 0 and (a // 10) % 10 > 0


def read_blocks(path):
    ev, blocks, cur = None, [], None
    for ln in open(path):
        if ln.startswith("#"):
            w = ln[1:].split()
            if w[:1] == ["event"]:
                ev = int(w[1])
                kind = "in" if "in" in w[2:5] else ("out" if "out" in w[2:5] else None)
                if kind:
                    cur = [ev, kind, []]; blocks.append(cur)
                else:
                    cur = None
            continue
        if cur is None:
            continue
        v = ln.split()
        cur[2].append([float(x) for x in v[:9]] + [int(v[9])])
    return [(e, k, np.array(p)) for e, k, p in blocks]


def nn_collision_times(path):
    """Times of the NN collisions per event. The event of an interaction is given by the next "# event N ... end" line."""
    lines = open(path).read().split("\n")
    i, pending, out = 0, [], {}
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# event") and " end " in ln:
            out.setdefault(int(ln.split()[2]), []).extend(pending); pending = []
        elif ln.startswith("# interaction"):
            w = ln.split(); nin, nout = int(w[3]), int(w[5])
            ps = [lines[i + 1 + k].split() for k in range(nin)]
            if nin == 2 and all(abs(int(p[9])) in (2212, 2112) for p in ps):
                pending.append(float(ps[0][0]))
            i += nin + nout
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = os.path.join(a.dir, "smash_data")
    blocks = read_blocks(os.path.join(d, "particle_lists.oscar"))
    coll = nn_collision_times(os.path.join(d, "full_event_history.oscar"))
    res = []
    for ev in sorted({b[0] for b in blocks}):
        ts, rho = [], []
        for e, k, p in blocks:
            if e != ev or k != "out":
                continue
            b = np.array([(1 if x > 0 else -1) if is_baryon(int(x)) else 0 for x in p[:, 9]])
            sel = (np.hypot(p[:, 1], p[:, 2]) < 3.0) & (np.abs(p[:, 3]) < 1.5)
            ts.append(p[0, 0]); rho.append(b[sel].sum() / (math.pi * 9.0 * 3.0) / RHO_S)
        ts, rho = np.array(ts), np.array(rho)
        sm = np.convolve(rho, np.ones(5) / 5, "same")
        k = int(np.argmax(sm[2:-2])) + 2
        ct = np.array(sorted(coll.get(ev, [])))
        res.append({"event": ev, "first_nn_collision_fm": float(ct.min()), "median_nn_collision_fm": float(np.median(ct)),
                    "t_max_central_density_fm": float(ts[k]), "max_central_density_over_rho_s_smoothed": float(sm[k]),
                    "n_nn_collisions": int(len(ct))})
    summ = {k: {"mean": float(np.mean([r[k] for r in res])), "std": float(np.std([r[k] for r in res], ddof=1))}
            for k in ("first_nn_collision_fm", "median_nn_collision_fm", "t_max_central_density_fm", "max_central_density_over_rho_s_smoothed")}
    out = {"per_event": res, "summary": summ}
    print(json.dumps(summ, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
