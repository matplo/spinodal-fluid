#!/usr/bin/env python3
"""S1: paired comparison of dx=0.1 and dx=0.2 fm per-event maxima (same events)."""
import json, csv, sys, math, argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--fine-tp', required=True); ap.add_argument('--fine-mx', required=True)
ap.add_argument('--coarse-tp', required=True); ap.add_argument('--coarse-mx', required=True)
ap.add_argument('--n', type=int, default=20)
ap.add_argument('--out', default='s1_numbers.json')
ap.add_argument('--plot', default=None, help='PNG: mean enhancement vs time at both resolutions')
a = ap.parse_args()

def se(x):
    x = np.asarray(x, float); return float(x.std(ddof=1)/math.sqrt(len(x))) if len(x) > 1 else float('nan')

def load(d, ev):
    d = Path(d)/f'event_{ev:03d}'
    s = json.load(open(d/'summary.json'))
    rows = list(csv.DictReader(open(d/'observables.csv')))
    t = np.array([float(r['t_fm']) for r in rows])
    m = {N: np.array([float(r[f'mom{N}']) for r in rows]) for N in range(2, 8)}
    return s, t, m

def curves(path):
    out = []
    for ev in range(a.n):
        s, t, m = load(path, ev)
        out.append((s, t - t[0], {N: m[N]/m[N][0] for N in m}, m))
    return out

res = {}
for eos, fp, cp in (('two_phase', a.fine_tp, a.coarse_tp), ('maxwell', a.fine_mx, a.coarse_mx)):
    F, C = curves(fp), curves(cp)
    r = {'n_events': a.n}
    for tag, D in (('dx0.1', F), ('dx0.2', C)):
        e = {}
        for N in (2, 3, 4, 5, 6, 7):
            mx = np.array([c[2][N].max() for c in D])
            # mean curve on common grid t (0.5 fm hist times not needed: interpolate to 0.04 fm)
            tg = np.arange(0, 6.0001, 0.04)
            mc = np.mean([np.interp(tg, c[1], c[2][N]) for c in D], axis=0)
            i = int(np.argmax(mc))
            e[f'N{N}'] = dict(max_mean=float(mx.mean()), max_se=se(mx), per_event=mx.tolist(),
                              mean_curve_max=float(mc[i]), mean_curve_max_time_fm=float(tg[i]))
        e['wall_total_s_mean'] = float(np.mean([c[0]['wall_total_s'] for c in D]))
        e['wall_per_step_s_mean'] = float(np.mean([c[0]['wall_per_step_s'] for c in D]))
        e['steps'] = D[0][0]['steps']; e['dt_fm'] = D[0][0]['dt_fm']
        e['fluid_end_time_fm_mean'] = float(np.mean([c[0]['fluid_end_time_fm'] for c in D]))
        e['fluid_end_time_fm_min_max'] = [min(c[0]['fluid_end_time_fm'] for c in D), max(c[0]['fluid_end_time_fm'] for c in D)]
        e['recovery_failures'] = int(sum(c[0]['recovery_failures'] for c in D))
        e['peak_rss_MB_max'] = float(max(c[0]['peak_rss_MB'] for c in D))
        e['raw_moments_initial'] = {f'N{N}': float(np.mean([c[3][N][0] for c in D])) for N in range(2, 8)}
        r[tag] = e
    pr = {}
    for N in (2, 3, 4, 5, 6, 7):
        f = np.array(r['dx0.1'][f'N{N}']['per_event']); c = np.array(r['dx0.2'][f'N{N}']['per_event'])
        ratio = f/c
        # paired: mean of per-event ratios and its s.e.; also ratio of means with delta-method s.e.
        d = f - c
        mf, mc_ = f.mean(), c.mean()
        cov = np.cov(f, c, ddof=1)
        n = len(f)
        var_r = (cov[0, 0]/mc_**2 + mf**2*cov[1, 1]/mc_**4 - 2*mf*cov[0, 1]/mc_**3)/n
        pr[f'N{N}'] = dict(mean_of_ratios=float(ratio.mean()), mean_of_ratios_se=se(ratio),
                           ratio_of_means=float(mf/mc_), ratio_of_means_se=float(math.sqrt(max(var_r, 0.0))),
                           mean_diff=float(d.mean()), mean_diff_se=se(d),
                           median_ratio=float(np.median(ratio)), min_ratio=float(ratio.min()), max_ratio=float(ratio.max()),
                           pearson_r=float(np.corrcoef(f, c)[0, 1]))
    r['paired'] = pr
    res[eos] = r
json.dump(res, open(a.out, 'w'), indent=1)
for eos in res:
    for N in (5, 7):
        r = res[eos]
        print(eos, f'N{N}', 'dx0.1 %.3f±%.3f' % (r['dx0.1'][f'N{N}']['max_mean'], r['dx0.1'][f'N{N}']['max_se']),
              'dx0.2 %.3f±%.3f' % (r['dx0.2'][f'N{N}']['max_mean'], r['dx0.2'][f'N{N}']['max_se']),
              'ratio %.3f±%.3f' % (r['paired'][f'N{N}']['mean_of_ratios'], r['paired'][f'N{N}']['mean_of_ratios_se']))

if a.plot:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    col = {'two_phase': '#0072B2', 'maxwell': '#E69F00'}   # Okabe-Ito blue / orange
    lab = {'two_phase': 'two-phase EoS', 'maxwell': 'Maxwell EoS'}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, N in zip(axs, (5, 7)):
        for eos, fp, cp in (('two_phase', a.fine_tp, a.coarse_tp), ('maxwell', a.fine_mx, a.coarse_mx)):
            for D, ls, dxl, mk in ((curves(cp), '--', 'dx = 0.2 fm', 's'), (curves(fp), '-', 'dx = 0.1 fm', 'o')):
                tg = np.arange(0, 6.0001, 0.08)
                arr = np.array([np.interp(tg, c[1], c[2][N]) for c in D])
                m = arr.mean(0); e = arr.std(0, ddof=1)/np.sqrt(len(arr))
                ax.plot(tg, m, ls=ls, color=col[eos], lw=1.8, marker=mk, markevery=6, ms=4, mfc='white' if ls == '--' else col[eos],
                        label=f'{lab[eos]}, {dxl}')
                ax.fill_between(tg, m-e, m+e, color=col[eos], alpha=0.12, lw=0)
        ax.set_xlabel('time since start of fluid dynamics (fm/c)'); ax.set_ylabel(f'moment N = {N}, relative to its initial value')
        ax.set_title(f'N = {N}: mean over {a.n} events', loc='left')
        if N == 7: ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(a.plot, dpi=160)
