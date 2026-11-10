"""Paper figure: development instances, mean share of sure top-10 members labelled IN against N (shots per
subexperiment), BL1 / R-CP / ours, one panel per family. Data: derisk/results/a6/paper_numbers.json ('dev_in_vs_N').
python figs/fig_shots.py  -> figs/fig_shots.pdf
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, '..', 'derisk', 'results', 'a6', 'paper_numbers.json')))['dev_in_vs_N']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times'], 'font.size': 9,
                     'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
                     'pdf.fonttype': 42, 'axes.linewidth': 0.6})
NS = (1000, 10000, 100000, 1000000)
STY = {'BL1': dict(color='#bdbdbd', marker='s', ls=':', label='BL1', ms=8.0, mfc='none', mew=0.9, zorder=2),
       'R-CP': dict(color='#b8541a', marker='^', ls='--', label='R-CP', ms=6.0, mfc='none', mew=0.9, zorder=3),
       'R-PAIR-S6': dict(color='#1f4e8c', marker='o', ls='-', label='Ours', ms=3.2, mfc='white', mew=1.0, zorder=4)}
fig, axs = plt.subplots(1, 2, figsize=(88 / 25.4, 42 / 25.4), sharey=True)
for ax, (fam, title) in zip(axs, (('arith', 'Adder'), ('qaoa2', 'QAOA'))):
    for m, st in STY.items():
        xs = [N for N in NS if '%s|%s|%d' % (fam, m, N) in D]
        ax.plot(xs, [D['%s|%s|%d' % (fam, m, N)][0] for N in xs], lw=1.0, **st)
    ax.set_xscale('log')
    ax.set_xticks(NS)
    ax.set_xticklabels(['1k', '10k', '100k', '1M'])
    ax.set_title(title, fontsize=9, pad=2)
    ax.set_ylim(-0.08, 1.05)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel('IN share')
axs[0].legend(loc='upper left', frameon=False, handlelength=1.6, labelspacing=0.25, borderaxespad=0.1)
fig.supxlabel('Nominal shots per subexperiment, N = B/|S|', fontsize=9, y=0.02)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(os.path.join(HERE, 'fig_shots.pdf'), bbox_inches='tight', pad_inches=0.03)
fig.savefig(os.path.join(HERE, 'fig_shots.png'), dpi=300, bbox_inches='tight', pad_inches=0.03)
