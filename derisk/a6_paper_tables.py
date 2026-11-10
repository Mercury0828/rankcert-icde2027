"""Write the LaTeX tables and number macros of paper section 6 from the result rows (no hand-typed numbers).

python -m derisk.a6_paper_tables
Inputs: derisk/results/a6/paper_numbers.json (python -m derisk.a6_paper_numbers), D:/runs/rankcert/h3/check_rows_tight.jsonl,
held-out rows D:/runs/rankcert/a6/heldout/. Outputs: paper/tables/{cov,a6,abl,search}.tex and paper/tables/numbers.tex.
"""
from __future__ import annotations

import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM = os.path.join(ROOT, 'derisk', 'results', 'a6', 'paper_numbers.json')
H3 = 'D:/runs/rankcert/h3/check_rows_tight.jsonl'
OUT = os.path.join(ROOT, 'paper', 'tables')
FAM = {'arith': 'Arithmetic', 'qaoa2': 'QAOA', 'bv': 'BV'}
SIZE = {'arith16': 'Adder, $m = 16$', 'arith20': 'Adder, $m = 20$', 'arith24': 'Adder, $m = 24$',
        'qaoa2_20': 'QAOA, $m = 20$', 'qaoa2_24': 'QAOA, $m = 24$'}
NAME = {'BL1': 'BL1', 'R-CP': 'R-CP', 'R-PAIR': 'Refined, uniform', 'R-PAIR-S': 'Refined, boundary',
        'R-PAIR6': 'Tighter, uniform', 'R-PAIR-S6': 'Tighter, boundary (ours)'}
ORDER = ['BL1', 'R-CP', 'R-PAIR', 'R-PAIR-S', 'R-PAIR6', 'R-PAIR-S6']


def macro(name, val):
    return '\\newcommand{\\%s}{%s}\n' % (name, val)


def pct(x):
    return '%.1f' % (100 * x)


def main():
    d = json.load(open(NUM))
    os.makedirs(OUT, exist_ok=True)
    mac = ''
    # Tab-Cov
    cov = d['coverage']
    tot = collections.defaultdict(lambda: [0, 0])
    lines = []
    for m in ORDER:
        cells = [NAME[m]]
        for sp in ('development', 'held-out'):
            v = cov.get('%s|%s' % (sp, m))
            if v is None:
                cells += ['', '', '']
                continue
            R, e, up, wi, wo = v
            cells += ['{:,}'.format(R).replace(',', '{,}'), '%d' % e, '%.4f' % up]
            tot[m][0] += R
            tot[m][1] += e
        lines.append(' & '.join(cells) + ' \\\\')
    with open(os.path.join(OUT, 'cov.tex'), 'w') as f:
        f.write('\\begin{table}[t]\\centering\\caption{Runs with at least one wrong \\IN{} or \\OUT{} label over the four '
                'queries of the run (Err.) and the one-sided $95\\%$ Clopper-Pearson upper limit of the error rate (Up.), '
                '$\\delta = 0.05$ per query.}\\label{tab:cov}\\small\\setlength{\\tabcolsep}{3pt}\n'
                '\\begin{tabular}{l|rrr|rrr}\\hline\n & \\multicolumn{3}{c|}{Development} & \\multicolumn{3}{c}{Held-out} \\\\\n'
                'Bounds & Runs & Err. & Up. & Runs & Err. & Up. \\\\ \\hline\n')
        f.write('\n'.join(lines) + '\n\\hline\\end{tabular}\\end{table}\n')
    ours = tot['R-PAIR-S6']
    allR = sum(v[0] for v in tot.values())
    allE = sum(v[1] for v in tot.values())
    mac += macro('CovRunsOurs', '{:,}'.format(ours[0]).replace(',', '{,}'))
    mac += macro('CovErrOurs', str(ours[1]))
    mac += macro('CovRunsAll', '{:,}'.format(allR).replace(',', '{,}'))
    mac += macro('CovErrAll', str(allE))
    # Tab-A6 (held-out): every registered held-out instance, complete N = 1e5 units only
    hi = d['heldout_instances']
    per = collections.defaultdict(list)
    for name, st in sorted(hi.items()):
        per[st['size']].append(st)
    rows = []
    fam_succ = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))   # succ, evaluated, registered
    bl1_out_any = False
    for size in sorted(per):
        sts = per[size]
        fam = sts[0]['family']
        ev = [s for s in sts if s['evaluated']]
        assert len(ev) == len(sts), 'held-out size %s incomplete' % size
        cells = [SIZE.get(size, size.replace('_', '\\_')), '%d' % len(ev)]
        cells.append('%.2f' % (sum(s['BL1'][0] for s in ev) / len(ev)))
        for m in ('R-CP', 'R-PAIR-S6'):
            fs = fam_succ[fam][m]
            fs[1] += len(ev)
            fs[2] += len(sts)
            s = sum(int(x[m + '_success']) for x in ev)
            fs[0] += s
            runs = [v for x in ev for v in x[m + '_in_reps']]
            mu = sum(runs) / len(runs)
            sd = (sum((v - mu) ** 2 for v in runs) / (len(runs) - 1)) ** 0.5
            cells.append('%.2f$\\,\\pm\\,$%.2f' % (sum(x[m][0] for x in ev) / len(ev), sd))
            cells.append('%.5f' % (sum(x[m][1] for x in ev) / len(ev)))
            cells.append('%d/%d' % (s, len(ev)))
        bl1_out_any = bl1_out_any or any(x['BL1_out_any'] for x in ev)
        rows.append(' & '.join(cells) + ' \\\\')
    with open(os.path.join(OUT, 'a6.tex'), 'w') as f:
        f.write('\\begin{table*}[t]\\centering\\caption{Held-out instances, top-$10$ query, $N = 10^5$ shots per '
                'subexperiment ($B \\approx 5 \\cdot 10^6$), $10$ repetitions each: mean share of sure members labelled '
                '\\IN{} ($\\pm$ standard deviation over all runs of the row), mean share of sure non-members labelled '
                '\\OUT{}, and instances that meet the success rule.'
                '}\\label{tab:a6}\\small\n\\begin{tabular}{lrr|rrr|rrr}\\hline\n'
                ' & & BL1 & \\multicolumn{3}{c|}{R-CP} & \\multicolumn{3}{c}{Ours} \\\\\n'
                'Circuit & Inst. & \\IN{} & \\IN{} & \\OUT{} & Succ. & \\IN{} & \\OUT{} & Succ. \\\\ \\hline\n')
        f.write('\n'.join(rows) + '\n\\hline\\end{tabular}\\end{table*}\n')
    for fam in ('arith', 'qaoa2'):
        F = 'Arith' if fam == 'arith' else 'Qaoa'
        for m, tag in (('R-PAIR-S6', 'Ours'), ('R-CP', 'Rcp')):
            s, e, n = fam_succ[fam][m]
            mac += macro('Succ%s%s' % (tag, F), '%d' % s)
            mac += macro('Eval%s%s' % (tag, F), '%d' % e)
            mac += macro('Reg%s%s' % (tag, F), '%d' % n)
    hm = d['heldout_means_1e5']
    for fam in ('arith', 'qaoa2'):
        for m, tag in (('R-PAIR-S6', 'Ours'), ('R-CP', 'Rcp'), ('BL1', 'Blone')):
            v = hm.get('%s|%s' % (fam, m))
            F = 'Arith' if fam == 'arith' else 'Qaoa'
            mac += macro('HoIn%s%s' % (tag, F), '%.2f' % v[0] if v else '--')
            mac += macro('HoIn%s%sPct' % (tag, F), '%.0f\\%%' % (100 * v[0]) if v else '--')
            mac += macro('HoOut%s%s' % (tag, F), '%.5f' % v[1] if v else '--')
    # Tab-Abl (development, N = 1e5)
    ab = d['ablation_dev_1e5']
    order = ['BL1', 'R-CP', 'R-PAIR', 'R-PAIR-S', 'R-PAIR6', 'R-PAIR-S6']
    rows = []
    for m in order:
        cells = [NAME[m]]
        for fam in ('arith', 'qaoa2'):
            v = ab.get('%s|%s' % (fam, m))
            cells += ['%.2f' % v[0], '%.5f' % v[1]] if v else ['--', '--']
        rows.append(' & '.join(cells) + ' \\\\')
    na = ab.get('arith|R-PAIR-S6', [0, 0, 0])[2]
    nq = ab.get('qaoa2|R-PAIR-S6', [0, 0, 0])[2]
    with open(os.path.join(OUT, 'abl.tex'), 'w') as f:
        f.write('\\begin{table}[t]\\centering\\caption{Ablation on the development instances ($%d$ arithmetic, $%d$ QAOA) at '
                '$N = 10^5$: mean top-$10$ \\IN{} and \\OUT{} shares over $10$ repetitions.}\\label{tab:abl}\\small\n'
                '\\begin{tabular}{lrr|rr}\\hline\n & \\multicolumn{2}{c|}{Arithmetic} & \\multicolumn{2}{c}{QAOA} \\\\\n'
                'Bounds & \\IN{} & \\OUT{} & \\IN{} & \\OUT{} \\\\ \\hline\n' % (na, nq))
        f.write('\n'.join(rows) + '\n\\hline\\end{tabular}\\end{table}\n')
    for m, tag in (('R-CP', 'Rcp'), ('R-PAIR', 'Rpair'), ('R-PAIR-S', 'RpairS'), ('R-PAIR6', 'Rsix'), ('R-PAIR-S6', 'Ours')):
        for fam in ('arith', 'qaoa2'):
            v = ab.get('%s|%s' % (fam, m))
            mac += macro('Abl%s%s' % (tag, 'Arith' if fam == 'arith' else 'Qaoa'), '%.2f' % v[0] if v else '--')
    # Tab-Search
    H = [json.loads(l) for l in open(H3)]
    by = collections.defaultdict(dict)
    for r in H:
        by[(r['m'], r['N'])][r['method']] = r
    rows = []
    for (m, N), v in sorted(by.items()):
        o, q, e = v.get('ours'), v.get('qcut'), v.get('enum')
        if not o or o.get('status') != 'ok':
            continue
        ens = e.get('extrapolated_seconds') if e else None
        assert o['wrong'] == 0, 'caption states that no label is wrong'
        rows.append('%d & $10^{%d}$ & %s & %s & %d & %s & %.1f & %.2f & %s \\\\' % (
            m, len(str(N)) - 1, '{:,}'.format(o['strings_evaluated']).replace(',', '{,}'),
            '{:.1e}'.format(o['strings_evaluated'] / 2 ** m).replace('e-0', 'e-'), o['n_in'], '{:,}'.format(o['n_und']).replace(',', '{,}'),
            o['seconds'], q['seconds'] if q else float('nan'), ('%.1f' % (ens / 86400)) if ens else '--'))
    with open(os.path.join(OUT, 'search.tex'), 'w') as f:
        f.write('\\begin{table}[t]\\centering\\caption{Search beyond enumeration (QAOA, top-$10$, one run per row): strings '
                'evaluated exactly and their share of $2^m$, \\IN{} and \\UND{} counts (all other strings are \\OUT{}), '
                'time of ours and of QCut\'s \\texttt{top()} in seconds, and extrapolated enumeration '
                'time in days.}\\label{tab:search}\\small\\setlength{\\tabcolsep}{2.5pt}\n'
                '\\begin{tabular}{rrrrrrrrr}\\hline\n$m$ & $N$ & Evaluated & Share & \\IN{} & \\UND{} & Ours & '
                '\\texttt{top()} & Enum. \\\\ \\hline\n')
        f.write('\n'.join(rows) + '\n\\hline\\end{tabular}\\end{table}\n')
    # Tab-Shots (development, IN share vs N)
    sh = d['dev_in_vs_N']
    rows = []
    for N in (1000, 10000, 100000, 1000000):
        cells = ['$10^{%d}$' % (len(str(N)) - 1)]
        for fam in ('arith', 'qaoa2'):
            for mm in ('BL1', 'R-CP', 'R-PAIR-S6'):
                v = sh.get('%s|%s|%d' % (fam, mm, N))
                cells.append('%.2f' % v[0] if v else '--')
        rows.append(' & '.join(cells) + ' \\\\')
    # Tab-Shots became Fig. shots (figs/fig_shots.py); only the N = 1e6 macros remain
    for fam, F in (('arith', 'Arith'), ('qaoa2', 'Qaoa')):
        for mm, tag in (('BL1', 'Blone'), ('R-CP', 'Rcp'), ('R-PAIR-S6', 'Ours')):
            v = sh.get('%s|%s|1000000' % (fam, mm))
            mac += macro('Six%s%s' % (tag, F), '%.2f' % v[0] if v else '--')
    # Fig. 1 running example (python -m derisk.a6_fig1): percentages with 3 decimals
    F1 = json.load(open(os.path.join(ROOT, 'derisk', 'results', 'a6', 'fig1.json')))
    byr = {r['rank_true']: r for r in F1['rows'] if r['rank_true']}
    pc = lambda v: '%.3f\\%%' % (100 * v)  # noqa: E731
    mac += macro('ExIn', str(F1['counts']['ours_in']))
    mac += macro('ExUnd', str(F1['counts']['ours_und']))
    mac += macro('ExRcpIn', str(F1['counts']['rcp_in']))
    mac += macro('ExRcpUnd', '{:,}'.format(F1['counts']['rcp_und']).replace(',', '{,}'))
    mac += macro('ExPilot', '{:,}'.format(F1['pilot_per_e']).replace(',', '{,}'))
    mac += macro('ExCollected', '{:,}'.format(F1['stats']['collected_all']).replace(',', '{,}'))
    for tag, r in (('One', byr[1]), ('Nine', byr[9]), ('Twelve', byr[12])):
        mac += macro('Ex%sP' % tag, pc(r['p']))
        mac += macro('Ex%sEst' % tag, pc(r['est']))
        mac += macro('Ex%sRcpL' % tag, pc(r['rcp_L']))
        mac += macro('Ex%sRcpU' % tag, pc(r['rcp_U']))
        mac += macro('Ex%sL' % tag, pc(r['ours_L']))
        mac += macro('Ex%sU' % tag, pc(r['ours_U']))
    mac += macro('ExPTen', pc(F1['p10']))
    mac += macro('ExPEleven', pc(F1['p11']))
    with open(os.path.join(OUT, 'numbers.tex'), 'w') as f:
        f.write('% generated by python -m derisk.a6_paper_tables; do not edit\n' + mac)
    print(open(os.path.join(OUT, 'numbers.tex')).read())


if __name__ == '__main__':
    main()
