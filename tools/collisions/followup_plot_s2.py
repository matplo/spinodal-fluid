#!/usr/bin/env python3
"""Plot S2 numbers (followup_s2_initstate.py output): per-event maxima by initial-state variant, and initial raw moments vs the paper."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
num, out = sys.argv[1], sys.argv[2]
d = json.load(open(num)); labs = list(d)
names = {'baseline': 'sigma 1.0 fm,\nparticle gamma_z\n(baseline)', 'sigma0.7': 'sigma 0.7 fm', 'sigma0.5': 'sigma 0.5 fm', 'cmgamma': 'sigma 1.0 fm,\ncm gamma'}
col = {'two_phase': '#0072B2', 'maxwell': '#E69F00'}; mk = {'two_phase': 'o', 'maxwell': 's'}
lab = {'two_phase': 'two-phase EoS', 'maxwell': 'Maxwell EoS'}
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, N in zip(axs[:2], (5, 7)):
    for k, eos in enumerate(('two_phase', 'maxwell')):
        x = np.arange(len(labs)) + (k-0.5)*0.12
        ax.errorbar(x, [d[l][eos][f'N{N}']['max_mean'] for l in labs], [d[l][eos][f'N{N}']['max_se'] for l in labs],
                    fmt=mk[eos], color=col[eos], capsize=3, ms=6, label=lab[eos])
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([names[l] for l in labs], fontsize=7)
    ax.set_ylabel(f'per-event maximum of moment N = {N} (relative)'); ax.set_title(f'N = {N}, 3 A GeV, 50 events', loc='left'); ax.set_xlim(-0.5, len(labs)-0.5)
axs[0].legend(frameon=False, fontsize=8)
ax = axs[2]; Ns = np.arange(2, 8)
paper = [1.47, 2.61, 5.16, 11.0, 24.9, 59.1]
ax.plot(Ns, paper, 'k-', lw=2, marker='D', label='arXiv:1209.2462 Fig. 3, initial values')
for l, ls, m in zip(labs, ['-', '--', ':', '-.'], ['o', '^', 'v', 'x']):
    ax.plot(Ns, [d[l]['two_phase']['raw_initial_mean'][f'N{n}'] for n in Ns], ls=ls, marker=m, color='#0072B2', label=names[l].replace('\n', ' '))
ax.set_yscale('log'); ax.set_xlabel('moment order N'); ax.set_ylabel('initial <rho^N>/<rho>^N'); ax.set_title('initial moments', loc='left')
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(out, dpi=160)
