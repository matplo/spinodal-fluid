#!/usr/bin/env python3
"""S2: initial-state variants at 3 A GeV. Usage: s2_analyse.py --out X.json LABEL=TPDIR:MXDIR ..."""
import json, csv, math, sys, argparse
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='s2_numbers.json'); ap.add_argument('--n', type=int, default=50)
ap.add_argument('sets', nargs='+'); a = ap.parse_args()
def se(x):
    x = np.asarray(x, float); return float(x.std(ddof=1)/math.sqrt(len(x)))
def load(d, n):
    out = []
    for ev in range(n):
        p = Path(d)/f'event_{ev:03d}'
        s = json.load(open(p/'summary.json'))
        rows = list(csv.DictReader(open(p/'observables.csv')))
        t = np.array([float(r['t_fm']) for r in rows]); t = t - t[0]
        m = {N: np.array([float(r[f'mom{N}']) for r in rows]) for N in range(2, 8)}
        out.append((s, t, m))
    return out
res = {}
for item in a.sets:
    lab, rest = item.split('=', 1); tp, mx = rest.split(':')
    r = {}
    for eos, d in (('two_phase', tp), ('maxwell', mx)):
        D = load(d, a.n); e = {'n_events': len(D)}
        e['raw_initial_mean'] = {f'N{N}': float(np.mean([c[2][N][0] for c in D])) for N in range(2, 8)}
        e['raw_initial_se'] = {f'N{N}': se([c[2][N][0] for c in D]) for N in range(2, 8)}
        # ratio-of-sums definition (sum <rho^N> / ...) not available; per-event mean is reported
        tg = np.arange(0, 6.0001, 0.08)
        for N in range(2, 8):
            mxv = np.array([(c[2][N]/c[2][N][0]).max() for c in D])
            mc = np.mean([np.interp(tg, c[1], c[2][N]/c[2][N][0]) for c in D], axis=0); i = int(mc.argmax())
            e[f'N{N}'] = dict(max_mean=float(mxv.mean()), max_se=se(mxv), per_event=mxv.tolist(),
                              mean_curve_max=float(mc[i]), mean_curve_max_time_fm=float(tg[i]))
        e['recovery_failures'] = int(sum(c[0]['recovery_failures'] for c in D))
        e['failed_events'] = int(sum(c[0]['status'] not in ('t_max', 'fluid_end') for c in D))
        e['wall_total_s_mean'] = float(np.mean([c[0]['wall_total_s'] for c in D]))
        e['t_init_fm_mean'] = float(np.mean([c[0]['t_init_fm'] for c in D]))
        e['eps_max_initial_eps_s_mean'] = float(np.mean([c[0]['eps_max_initial_eps_s'] for c in D]))
        e['unstable_fraction_initial_mean'] = float(np.mean([c[2][2][0]*0 + c[0].get('unstable_fraction_peak',0) for c in D]))
        e['baryon_leaving_max'] = float(max(c[0]['baryon_leaving_fraction'] for c in D))
        e['ic_limiter_cells_total'] = int(sum(c[0]['ic_limiter_cells'] for c in D))
        r[eos] = e
    r['ratio_tp_over_mx'] = {}
    for N in (5, 7):
        f = np.array(r['two_phase'][f'N{N}']['per_event']); g = np.array(r['maxwell'][f'N{N}']['per_event'])
        mf, mg = f.mean(), g.mean(); cov = np.cov(f, g, ddof=1); n = len(f)
        v = (cov[0,0]/mg**2 + mf**2*cov[1,1]/mg**4 - 2*mf*cov[0,1]/mg**3)/n
        r['ratio_tp_over_mx'][f'N{N}'] = dict(ratio_of_means=float(mf/mg), se=float(math.sqrt(max(v, 0))))
    res[lab] = r
json.dump(res, open(a.out, 'w'), indent=1)
paper = [1.47, 2.61, 5.16, 11.0, 24.9, 59.1]
print('paper init', paper)
for lab, r in res.items():
    t, m = r['two_phase'], r['maxwell']
    print(lab, 'init', [round(t['raw_initial_mean'][f'N{N}'], 2) for N in range(2, 8)],
          'N5 tp %.2f±%.2f mx %.3f±%.3f' % (t['N5']['max_mean'], t['N5']['max_se'], m['N5']['max_mean'], m['N5']['max_se']),
          'N7 tp %.2f±%.2f mx %.3f±%.3f' % (t['N7']['max_mean'], t['N7']['max_se'], m['N7']['max_mean'], m['N7']['max_se']),
          'ratioN7 %.2f' % r['ratio_tp_over_mx']['N7']['ratio_of_means'], 'fail', t['recovery_failures'], m['recovery_failures'])
