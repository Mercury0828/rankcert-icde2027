"""Paper Fig. 1 from derisk/results/a6/fig1.json (python -m derisk.a6_fig1). Writes figs/fig1.pdf.

python figs/fig1.py
x: the 16 strings with the largest exact probabilities, by exact rank. Per string: our bounds (thick bar; dark = IN,
light = UND), R-CP bounds (thin bar, all UND), exact value (black dot), plug-in estimate (cross; filled box around the
tick label when top() returns the string). Dashed line: exact 10th value.
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, '..', 'derisk', 'results', 'a6', 'fig1.json')))
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times'], 'mathtext.fontset': 'stix',
                     'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 9, 'pdf.fonttype': 42, 'axes.linewidth': 0.6})
rows = sorted([r for r in D['rows'] if r['rank_true'] and r['rank_true'] <= 16], key=lambda r: r['rank_true'])
S = 100
fig, ax = plt.subplots(figsize=(88 / 25.4, 58 / 25.4))
for r in rows:
    i = r['rank_true']
    ax.plot([i - 0.34, i - 0.34], [r['rcp_L'] * S, r['rcp_U'] * S], color='#9a9a9a', lw=0.8, solid_capstyle='butt')
    c = '#1f4e8c' if r['ours_label'] == 1 else '#8fb3de'
    ax.errorbar(i, (r['ours_L'] + r['ours_U']) / 2 * S, yerr=(r['ours_U'] - r['ours_L']) / 2 * S, fmt='none', ecolor=c,
                elinewidth=2.2, capsize=1.8, capthick=0.8)
    ax.plot(i - 0.17, r['p'] * S, 'o', color='black', ms=2.4, zorder=5)
    ax.plot(i + 0.19, r['est'] * S, 'x', color='#c0392b', ms=3.4, mew=0.9, zorder=6)
ax.axhline(D['p10'] * S, color='black', lw=0.6, ls='--')
ax.set_xlim(0.3, 16.7)
ax.set_xticks([r['rank_true'] for r in rows])
plug = {r['rank_true'] for r in rows if r['rank_est'] is not None and r['rank_est'] <= D['k']}
labs = ax.set_xticklabels([str(r['rank_true']) for r in rows])
for t, r in zip(labs, rows):
    if r['rank_true'] in plug:
        t.set_bbox(dict(boxstyle='square,pad=0.15', fc='#e6e6e6', ec='black', lw=0.5))
ax.set_xlabel('Rank in target')
ax.set_ylabel('Probability (%)')
from matplotlib.lines import Line2D  # noqa: E402
h = [Line2D([], [], color='#1f4e8c', lw=3, label='Ours, in'), Line2D([], [], color='#8fb3de', lw=3, label='Ours, undecided'),
     Line2D([], [], color='#9a9a9a', lw=0.8, label='R-CP, undecided'),
     Line2D([], [], marker='o', color='black', ls='', ms=2.5, label='Target'),
     Line2D([], [], marker='x', color='#c0392b', ls='', ms=3.5, label='Estimate'),
     Line2D([], [], color='black', lw=0.6, ls='--', label='10th target value')]
ax.legend(handles=h, loc='upper right', ncol=2, frameon=False, handlelength=1.2, columnspacing=0.6, labelspacing=0.3, borderaxespad=0.1)
ax.set_ylim(0, 0.26)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, 'fig1.pdf'), bbox_inches='tight', pad_inches=0.03)
fig.savefig(os.path.join(HERE, 'fig1.png'), dpi=300, bbox_inches='tight', pad_inches=0.03)
print('rows', len(rows), 'plug-in top-10 ranks', sorted(plug))
