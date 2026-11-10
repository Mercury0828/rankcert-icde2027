"""A6 development summary for the A6-FREEZE choice (derisk/BENCH_A6_PREREG.md §5 rule, applied to development seeds).

python -m derisk.a6_dev_summary [--reps-min 10]
Reads D:/runs/rankcert/a6/runs/ and runs6/ (development rows only; held-out rows are never read). For every candidate
method (R-PAIR, R-PAIR-S, R-PAIR6, R-PAIR-S6) and instance: instance success = at some N with S*N <= 1e7, mean top-10 IN
share >= 0.50, mean OUT share >= 0.999, and the Q2 condition against BL1 at the same N (IN >= 2x BL1's; if BL1's IN share is
0 at every admissible N, IN > 0 - reported as "BL1 = 0"; the "same IN with >= 4x fewer shots" clause is also checked).
Family success = >= 60 % of the family's instances. Units with fewer than --reps-min repeats are reported as incomplete.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

A6 = 'D:/runs/rankcert/a6'
CANDIDATES = ('R-PAIR', 'R-PAIR-S', 'R-PAIR6', 'R-PAIR-S6')
BMAX = 1e7


def load():
    rows = []
    for sub in ('runs', 'runs6'):
        for p in glob.glob(os.path.join(A6, sub, '*.jsonl')):
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get('split') != 'development':
                    continue
                rows.append(r)
    return rows


def shares(rs):
    """mean IN share, mean OUT share (None if undefined), repeats."""
    ins, outs = [], []
    for r in rs:
        e = r['top10']
        if e['sure_members'] == 0:
            return None, None, len(rs)
        ins.append(e['in_of_sure'] / e['sure_members'])
        outs.append(e['out_of_sure'] / e['sure_non'] if e['sure_non'] else None)
    if any(o is None for o in outs):
        return sum(ins) / len(ins), None, len(rs)
    return sum(ins) / len(ins), sum(outs) / len(outs), len(rs)


def main(reps_min):
    rows = load()
    by = collections.defaultdict(list)
    info = {}
    for r in rows:
        by[(r['instance'], r['method'], r['N'])].append(r)
        info[r['instance']] = (r['family'], r['S'])
    out = {}
    for inst, (fam, S) in sorted(info.items()):
        Ns = sorted({N for (i, m, N) in by if i == inst and S * N <= BMAX})
        bl1 = {N: shares(by[(inst, 'BL1', N)])[0] for N in Ns if by.get((inst, 'BL1', N))}
        bl1_zero = all((v or 0.0) == 0.0 for v in bl1.values())
        for meth in CANDIDATES:
            succ, best, incomplete = False, None, False
            for N in Ns:
                rs = by.get((inst, meth, N))
                if not rs:
                    incomplete = True
                    continue
                i, o, n = shares(rs)
                if n < reps_min:
                    incomplete = True
                if i is None or o is None:
                    continue
                b = bl1.get(N)
                if b is None:
                    q2 = False
                elif bl1_zero:
                    q2 = i > 0
                else:
                    q2 = i >= 2 * b or any(
                        (bl1.get(N2) or 0) <= i and S * N2 >= 4 * S * N for N2 in bl1 if bl1.get(N2) is not None)
                ok = i >= 0.5 and o >= 0.999 and q2
                succ = succ or ok
                if best is None or i >= best[1]:      # ties: the larger N (its OUT share is shown)
                    best = (N, round(i, 3), round(o, 5), n)
            out[(inst, meth)] = dict(family=fam, success=succ, best=best, incomplete=incomplete, bl1_zero=bl1_zero)
    fams = collections.defaultdict(lambda: collections.defaultdict(list))
    for (inst, meth), v in out.items():
        fams[v['family']][meth].append(v['success'])
    print('instance'.ljust(16) + ''.join(m.rjust(30) for m in CANDIDATES))
    for inst in sorted(info):
        cells = []
        for meth in CANDIDATES:
            v = out[(inst, meth)]
            tag = ('OK ' if v['success'] else '-- ') + str(v['best']) + ('*' if v['incomplete'] else '')
            cells.append(tag.rjust(30))
        print(inst.ljust(16) + ''.join(cells))
    print('\nfamily success (share of instances; bar 0.6); * = incomplete units')
    for fam, d in sorted(fams.items()):
        print(fam.ljust(8) + ''.join(('%s %.2f (%d)' % (m, sum(v) / len(v), len(v))).rjust(30) for m, v in d.items()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps-min', type=int, default=10)
    main(ap.parse_args().reps_min)
