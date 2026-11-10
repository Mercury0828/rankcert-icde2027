"""Numbers for paper section 6 (after A6-FREEZE, derisk/A6_FREEZE.json).

python -m derisk.a6_paper_numbers
(1) Tab-A6: held-out rows (./data/a6/heldout/), success rule of the benchmark protocol section 5 for the frozen
    R-PAIR-S6 and for R-CP, against BL1; per instance best (N, IN, OUT, reps) and the per-N means of every method.
(2) Tab-Abl: development rows at N = 1e5, mean top-10 IN / OUT share per family for BL1, R-CP, R-PAIR, R-PAIR-S, R-PAIR6,
    R-PAIR-S6 (instances where all six are present with >= 10 repeats).
(3) Tab-Cov: joint error rate (any wrong IN / OUT label over the four queries of a run) per method and split, all rows.
Writes derisk/results/a6/paper_numbers.json.
"""
from __future__ import annotations

import collections
import glob
import json
import os

from derisk.a6_dev_summary import BMAX, shares
from derisk.cov_summary import QUERIES, upper95

A6 = './data/a6'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'a6', 'paper_numbers.json')


def rows(subs):
    for sub in subs:
        for p in glob.glob(os.path.join(A6, sub, '*.jsonl')):
            for line in open(p):
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def success_table(rs, methods, reps_min=10):
    by = collections.defaultdict(list)
    info = {}
    for r in rs:
        by[(r['instance'], r['method'], r['N'])].append(r)
        info[r['instance']] = (r['family'], r['S'])
    res = {}
    for inst, (fam, S) in sorted(info.items()):
        Ns = sorted({N for (i, m, N) in by if i == inst and S * N <= BMAX})
        bl1 = {N: shares(by[(inst, 'BL1', N)])[0] for N in Ns if by.get((inst, 'BL1', N))}
        bl1_zero = all((v or 0.0) == 0.0 for v in bl1.values())
        per = {}
        for meth in methods + ['BL1']:
            per[meth] = {N: shares(by[(inst, meth, N)]) for N in Ns if by.get((inst, meth, N))}
        for meth in methods:
            succ, best = False, None
            for N, (i, o, n) in per[meth].items():
                if n < reps_min or i is None or o is None:
                    continue
                b = bl1.get(N)
                if b is None:
                    q2 = False
                elif bl1_zero:
                    q2 = i > 0
                else:
                    q2 = i >= 2 * b or any((bl1.get(N2) or 0) <= i and N2 >= 4 * N for N2 in bl1 if bl1.get(N2) is not None)
                succ = succ or (i >= 0.5 and o >= 0.999 and q2)
                if best is None or i >= best[1]:
                    best = (N, round(i, 3), round(o, 5), n)
            res[(inst, meth)] = dict(family=fam, success=succ, best=best, bl1_zero=bl1_zero,
                                     per_N={str(N): v for N, v in per[meth].items()},
                                     bl1_per_N={str(N): v for N, v in per['BL1'].items()})
    return res


def main():
    out = {}
    # (1) held-out
    ho = [r for r in rows(['heldout']) if r.get('split') == 'held-out']
    tab = success_table(ho, ['R-PAIR-S6', 'R-CP'])
    print('HELD-OUT (top-10; best (N, IN, OUT, reps); BL1 IN at 1e5)')
    fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for (inst, meth), v in sorted(tab.items()):
        fam[v['family']][meth].append(v['success'])
        b = v['bl1_per_N'].get('100000')
        print('%-16s %-10s %s %s  BL1@1e5 IN=%s' % (inst, meth, 'OK' if v['success'] else '--', v['best'],
                                                   None if b is None else round(b[0], 3)))
    print('held-out family success:')
    for f, d in sorted(fam.items()):
        print('  ', f, {m: '%d/%d' % (sum(x), len(x)) for m, x in d.items()})
    out['heldout'] = {'%s|%s' % k: v for k, v in tab.items()}
    out['heldout_family'] = {f: {m: [sum(x), len(x)] for m, x in d.items()} for f, d in fam.items()}
    # Registered held-out instances (bench.a6.table), status per instance at N = 1e5 with complete units only:
    # 'evaluated' iff R-PAIR-S6, R-CP and BL1 each have >= 10 repeats at N = 1e5 (reps 0-9). Means: per instance over
    # reps 0-9, then the unweighted mean over evaluated instances.
    from bench.a6 import table
    reg = {k: v for k, v in table().items() if v['split'] == 'held-out'}
    unit = collections.defaultdict(list)
    for r in ho:
        if r['N'] == 100000 and r['rep'] < 10:
            unit[(r['instance'], r['method'])].append(r)
    inst_stat = {}
    for name, v in sorted(reg.items()):
        ok = all(len(unit.get((name, m), [])) >= 10 for m in ('R-PAIR-S6', 'R-CP', 'BL1'))
        st = dict(family=v['family'], size=name.rsplit('_s', 1)[0], evaluated=ok)
        if ok:
            for m in ('R-PAIR-S6', 'R-CP', 'BL1'):
                st[m] = list(shares(unit[(name, m)][:10])[:2])
                rs = sorted(unit[(name, m)], key=lambda r: r['rep'])[:10]
                st[m + '_in_reps'] = [r['top10']['in_of_sure'] / r['top10']['sure_members'] for r in rs
                                      if r['top10']['sure_members']]
            for m in ('R-PAIR-S6', 'R-CP'):
                st[m + '_success'] = bool(tab.get((name, m), {}).get('success'))
            st['BL1_out_any'] = any(r['top10']['n_out'] > 0 for r in unit[(name, 'BL1')][:10])
        inst_stat[name] = st
    out['heldout_instances'] = inst_stat
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for name, st in inst_stat.items():
        if not st['evaluated']:
            continue
        for m in ('R-PAIR-S6', 'R-CP', 'BL1'):
            i, o = st[m]
            a = agg[(st['family'], m)]
            a[0] += i
            a[1] += o if o is not None else float('nan')
            a[2] += 1
    print('held-out means at N = 1e5 over evaluated instances (top-10 IN, OUT, instances)')
    out['heldout_means_1e5'] = {}
    for (f, m), (i, o, n) in sorted(agg.items()):
        print('  %-10s %-10s %.3f %.5f %d' % (f, m, i / n, o / n, n))
        out['heldout_means_1e5']['%s|%s' % (f, m)] = [i / n, o / n, n]
    print('held-out status:', {f: '%d evaluated of %d' % (sum(s['evaluated'] for s in inst_stat.values() if s['family'] == f),
                                                        sum(1 for s in inst_stat.values() if s['family'] == f))
                               for f in ('arith', 'qaoa2')})
    # (2) ablation on development rows at N = 1e5
    dev = [r for r in rows(['runs', 'runs6']) if r.get('split') == 'development' and r['N'] == 100000 and r['rep'] < 10]
    meths = ['BL1', 'R-CP', 'R-PAIR', 'R-PAIR-S', 'R-PAIR6', 'R-PAIR-S6']
    cell = collections.defaultdict(list)
    for r in dev:
        e = r['top10']
        if e['sure_members'] == 0 or not e['sure_non']:
            continue
        cell[(r['instance'], r['method'])].append((e['in_of_sure'] / e['sure_members'], e['out_of_sure'] / e['sure_non']))
    insts = sorted({i for (i, m) in cell if all(len(cell.get((i, mm), [])) >= 10 for mm in meths)})
    abl = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for i in insts:
        f = i.rsplit('_s', 1)[0].rstrip('0123456789').rstrip('_')
        for mm in meths:
            v = cell[(i, mm)][:10]
            a = abl[(f, mm)]
            a[0] += sum(x[0] for x in v) / len(v)
            a[1] += sum(x[1] for x in v) / len(v)
            a[2] += 1
    print('ABLATION (development, N = 1e5, reps 0-9; mean over instances of mean top-10 IN / OUT)')
    out['ablation_dev_1e5'] = {}
    for (f, mm), (i, o, n) in sorted(abl.items(), key=lambda kv: (kv[0][0], meths.index(kv[0][1]))):
        print('  %-8s %-10s IN %.3f OUT %.5f (%d instances)' % (f, mm, i / n, o / n, n))
        out['ablation_dev_1e5']['%s|%s' % (f, mm)] = [i / n, o / n, n]
    # (2b) development IN share vs N (top-10, reps 0-9, instances with all three methods at that N)
    devall = [r for r in rows(['runs', 'runs6']) if r.get('split') == 'development' and r['rep'] < 10]
    cellN = collections.defaultdict(list)
    for r in devall:
        e = r['top10']
        if e['sure_members'] == 0:
            continue
        cellN[(r['instance'], r['method'], r['N'])].append(e['in_of_sure'] / e['sure_members'])
    shots = {}
    for N in (1000, 10000, 100000, 1000000):
        for fam in ('arith', 'qaoa2'):
            insts = sorted({i for (i, m, n) in cellN if n == N and i.startswith(fam) and
                            all(len(cellN.get((i, mm, N), [])) >= 10 for mm in ('BL1', 'R-CP', 'R-PAIR-S6'))})
            for mm in ('BL1', 'R-CP', 'R-PAIR-S6'):
                if insts:
                    shots['%s|%s|%d' % (fam, mm, N)] = [sum(sum(cellN[(i, mm, N)][:10]) / 10 for i in insts) / len(insts),
                                                       len(insts)]
    out['dev_in_vs_N'] = shots
    print('DEV IN vs N', shots)
    # (3) coverage over all rows
    cov = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows(['runs', 'runs6', 'heldout', 'cov']):
        qs = [q for q in QUERIES if q in r]
        if not qs:
            continue
        a = cov[(r.get('split'), r['method'])]
        a[0] += 1
        a[1] += int(any(r[q]['wrong'] > 0 for q in qs))
        a[2] += sum(r[q]['wrong_in'] for q in qs)
        a[3] += sum(r[q]['wrong_out'] for q in qs)
    print('COVERAGE (runs, joint errors, up95, wrong IN, wrong OUT)')
    out['coverage'] = {}
    for (sp, mm), (R, e, wi, wo) in sorted(cov.items()):
        print('  %-12s %-10s R=%6d err=%d up95=%.5f wIN=%d wOUT=%d' % (sp, mm, R, e, upper95(e, R), wi, wo))
        out['coverage']['%s|%s' % (sp, mm)] = [R, e, upper95(e, R), wi, wo]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == '__main__':
    main()
