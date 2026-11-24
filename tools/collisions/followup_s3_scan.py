#!/usr/bin/env python3
"""Enhancement versus E_lab from per-event maxima (N = 5, 7), with a bootstrap over events.

  followup_s3_scan.py --runs-root RUNS --energies 1 2 3 4 5 6 --nev 200 --paper tools/data --out-prefix s3 [--boot 20000 --seed 12345]

Per energy and EoS: mean and standard error of the per-event maximum enhancement (maximum over time of <rho^N>(t)/<rho^N>(0) of one event).
Bootstrap (events resampled with replacement, independently for every energy; the two EoS are separate): distribution of the energy of the
maximum of the two-phase mean curve over the grid of simulated energies, the probability of each grid energy to be the maximum, the 68 %
interval, a three-point parabola in log E through the maximum bin and its neighbours (continuous estimate), the difference of the means at
2 and 3 A GeV, and for Maxwell the weighted slope of mean vs E and the difference of the means at 6 and 1 A GeV.
Writes <prefix>_numbers.json, <prefix>_table.md, <prefix>_enhancement_vs_E.png.
"""
import argparse, csv, json, math, os
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--runs-root', required=True); ap.add_argument('--energies', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
ap.add_argument('--nev', type=int, default=200); ap.add_argument('--paper', default=None)
ap.add_argument('--out-prefix', default='s3'); ap.add_argument('--boot', type=int, default=20000); ap.add_argument('--seed', type=int, default=12345)
ap.add_argument('--label', default=None)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
E = np.array(a.energies, float)

def maxima(d, nev):
    out = {5: [], 7: []}; fails = 0; n = 0
    for ev in range(nev):
        p = Path(d)/f'event_{ev:03d}'
        s = json.load(open(p/'summary.json'))
        if s['status'] not in ('t_max', 'fluid_end'): continue
        fails += s['recovery_failures']; n += 1
        rows = list(csv.DictReader(open(p/'observables.csv')))
        for N in (5, 7):
            m = np.array([float(r[f'mom{N}']) for r in rows]); out[N].append((m/m[0]).max())
    return {N: np.array(v) for N, v in out.items()}, n, fails

data = {}; info = {}
for eos in ('two_phase', 'maxwell'):
    for e in a.energies:
        d, n, f = maxima(Path(a.runs_root)/f'e{e}_table_{eos}', a.nev)
        data[(eos, e)] = d; info[(eos, e)] = (n, f)

def se(x): return float(np.std(x, ddof=1)/math.sqrt(len(x)))
res = {'energies': a.energies, 'nev_requested': a.nev, 'bootstrap': a.boot, 'seed': a.seed, 'table': {}, 'bootstrap_results': {}}
for eos in ('two_phase', 'maxwell'):
    for e in a.energies:
        n, f = info[(eos, e)]
        res['table'][f'{eos}_E{e}'] = {'n_events': n, 'recovery_failures': f,
            **{f'N{N}_mean': float(data[(eos, e)][N].mean()) for N in (5, 7)}, **{f'N{N}_se': se(data[(eos, e)][N]) for N in (5, 7)}}

def parab(x, y):   # vertex of a parabola through three points (x in ln E)
    A = np.polyfit(x, y, 2)
    return float(-A[1]/(2*A[0])) if A[0] < 0 else float('nan')

for N in (5, 7):
    B = a.boot
    # bootstrap means [eos][E] -> (B,)
    bm = {eos: np.array([[rng.choice(data[(eos, e)][N], size=len(data[(eos, e)][N]), replace=True).mean() for _ in range(B)] for e in a.energies]) for eos in ('two_phase', 'maxwell')}
    tp = bm['two_phase']; mx = bm['maxwell']
    arg = tp.argmax(0)
    prob = {int(e): float((arg == i).mean()) for i, e in enumerate(a.energies)}
    Eb = E[arg]
    # continuous estimate: vertex of parabola in ln E through argmax bin and neighbours (edge cases: discard)
    vert = []
    for b in range(B):
        i = arg[b]
        if 0 < i < len(E)-1:
            v = parab(np.log(E[i-1:i+2]), tp[i-1:i+2, b]); 
            if np.isfinite(v): vert.append(math.exp(v))
    vert = np.array(vert)
    i2, i3 = a.energies.index(2) if 2 in a.energies else None, a.energies.index(3) if 3 in a.energies else None
    r = {'prob_max_at_grid_energy': prob, 'argmax_grid_median': float(np.median(Eb)), 'argmax_grid_68_interval': [float(np.percentile(Eb, 16)), float(np.percentile(Eb, 84))],
         'argmax_grid_mean': float(Eb.mean()), 'argmax_grid_std': float(Eb.std()),
         'parabola_vertex_n_valid': int(len(vert)), 'parabola_vertex_median': float(np.median(vert)) if len(vert) else None,
         'parabola_vertex_68_interval': [float(np.percentile(vert, 16)), float(np.percentile(vert, 84))] if len(vert) else None}
    if i2 is not None and i3 is not None:
        d = tp[i2]-tp[i3]; r['diff_mean_2_minus_3'] = {'value': float(data[('two_phase', 2)][N].mean()-data[('two_phase', 3)][N].mean()), 'boot_se': float(d.std()), 'p_2_gt_3': float((d > 0).mean())}
    # Maxwell trend
    mm = np.array([data[('maxwell', e)][N].mean() for e in a.energies]); ms = np.array([se(data[('maxwell', e)][N]) for e in a.energies])
    slope_b = np.array([np.polyfit(E, mx[:, b], 1)[0] for b in range(B)])
    r['maxwell'] = {'slope_per_AGeV': float(np.polyfit(E, mm, 1, w=1/ms)[0]), 'slope_boot_se': float(slope_b.std()), 'p_slope_gt_0': float((slope_b > 0).mean()),
                    'diff_last_minus_first': float(mm[-1]-mm[0]), 'diff_last_minus_first_boot_se': float((mx[-1]-mx[0]).std()),
                    'p_last_gt_first': float((mx[-1] > mx[0]).mean()),
                    'p_monotonic_increasing': float(np.all(np.diff(mx, axis=0) > 0, axis=0).mean())}
    res['bootstrap_results'][f'N{N}'] = r
json.dump(res, open(a.out_prefix+'_numbers.json', 'w'), indent=1)

# table
L = ['| E_lab [A GeV] | n events | N=5 two-phase | N=5 Maxwell | N=7 two-phase | N=7 Maxwell |', '|---|---|---|---|---|---|']
for e in a.energies:
    t, m = res['table'][f'two_phase_E{e}'], res['table'][f'maxwell_E{e}']
    L.append(f"| {e} | {t['n_events']}/{m['n_events']} | {t['N5_mean']:.2f} ± {t['N5_se']:.2f} | {m['N5_mean']:.3f} ± {m['N5_se']:.3f} | {t['N7_mean']:.2f} ± {t['N7_se']:.2f} | {m['N7_mean']:.3f} ± {m['N7_se']:.3f} |")
open(a.out_prefix+'_table.md', 'w').write('\n'.join(L)+'\n')

# plot
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cb = {'two_phase': '#0072B2', 'maxwell': '#E69F00'}; lab = {'two_phase': 'two-phase EoS', 'maxwell': 'Maxwell EoS'}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, N in zip(axs, (5, 7)):
    for eos, mk in (('two_phase', 'o'), ('maxwell', 's')):
        ax.errorbar(E, [res['table'][f'{eos}_E{e}'][f'N{N}_mean'] for e in a.energies], [res['table'][f'{eos}_E{e}'][f'N{N}_se'] for e in a.energies],
                    color=cb[eos], marker=mk, ms=6, lw=2, capsize=3, label=f"this work, {lab[eos]} ({a.label or str(a.nev)+' events'})")
    if a.paper:
        for src, pre, ls, mk, off in (('arXiv:1209.2462 Fig. 4', 'p1209_fig4', ':', 'x', 0), ('arXiv:1302.2956 Fig. 6', 'p1302_fig6', '--', '+', 0)):
            for eos in ('two_phase', 'maxwell'):
                f = os.path.join(a.paper, f'{pre}_N{N}_{eos}.csv')
                if os.path.exists(f):
                    dd = np.genfromtxt(f, delimiter=',', comments='#', skip_header=1)
                    ax.plot(dd[:, 0], dd[:, 1], ls=ls, marker=mk, color=cb[eos], alpha=0.6, lw=1.2, ms=6, label=f'{src}, {lab[eos]}')
    ax.set_xlabel('beam energy (A GeV)'); ax.set_ylabel(f'mean maximum enhancement, N = {N}'); ax.set_yscale('log' if N == 7 else 'linear')
    ax.set_title(f'N = {N}', loc='left')
axs[0].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig(a.out_prefix+'_enhancement_vs_E.png', dpi=160)
print(open(a.out_prefix+'_table.md').read())
for N in (5, 7):
    r = res['bootstrap_results'][f'N{N}']
    print(f'N{N}', 'P(max at E)', {k: round(v, 3) for k, v in r['prob_max_at_grid_energy'].items()}, 'median %.2f 68%% %s' % (r['argmax_grid_median'], r['argmax_grid_68_interval']),
          'vertex', r['parabola_vertex_median'], r['parabola_vertex_68_interval'], r.get('diff_mean_2_minus_3'), r['maxwell'])
