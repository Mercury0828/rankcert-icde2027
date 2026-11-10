"""Tab-Cov (H1): observed joint error rate of the IN / OUT labels against delta.

python -m derisk.cov_summary [--subs cov] [--split development]
A run is one (instance, method, N, rep) row of derisk.a6_run; it is a joint error if any of its queries (top-5, top-10,
tau = 0.01, tau = 0.05) has a wrong IN or OUT label (stats.labels.evaluate: an IN on a string that is not a sure member,
an OUT on a string that is not a sure non-member, so a label on a string tied at the boundary counts as wrong). The
guarantee is joint per query answer at level delta; counting any wrong label over the four queries of a run is stricter.
Reports, per (instance, method, N) and pooled per method: runs R, joint errors, rate and its one-sided 95 % Clopper-Pearson
upper limit, and the per-label wrong counts.

Tab-Cov runs: python -m derisk.a6_run <inst> --Ns <N> --reps 30-1029 --methods <frozen> --sub cov (reps >= 30 never
overlap the A6 evaluation repeats 0-29; same seeding scheme, so they are independent of them).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

from scipy.stats import beta

A6 = 'D:/runs/rankcert/a6'
QUERIES = ('top5', 'top10', 'tau0.01', 'tau0.05')


def upper95(x, n):
    return 1.0 if x == n else float(beta.ppf(0.95, x + 1, n - x))


def main(subs, split):
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])      # runs, joint errors, wrong IN, wrong OUT
    for sub in subs:
        for p in glob.glob(os.path.join(A6, sub, '*.jsonl')):
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if split and r.get('split') != split:
                    continue
                qs = [q for q in QUERIES if q in r]
                if not qs:
                    continue
                a = agg[(r['instance'], r['method'], r['N'])]
                a[0] += 1
                a[1] += int(any(r[q]['wrong'] > 0 for q in qs))
                a[2] += sum(r[q]['wrong_in'] for q in qs)
                a[3] += sum(r[q]['wrong_out'] for q in qs)
    pooled = collections.defaultdict(lambda: [0, 0, 0, 0])
    print('%-16s %-10s %8s %6s %6s %8s %8s %6s %6s' % ('instance', 'method', 'N', 'R', 'err', 'rate', 'up95', 'wIN', 'wOUT'))
    for (inst, meth, N), (R, e, wi, wo) in sorted(agg.items()):
        print('%-16s %-10s %8d %6d %6d %8.4f %8.4f %6d %6d' % (inst, meth, N, R, e, e / R, upper95(e, R), wi, wo))
        pm = pooled[meth]
        for i, v in enumerate((R, e, wi, wo)):
            pm[i] += v
    print('\npooled per method (delta = 0.05)')
    for meth, (R, e, wi, wo) in sorted(pooled.items()):
        print('%-10s R = %7d  joint errors = %5d  rate = %.5f  up95 = %.5f  wrong IN = %d  wrong OUT = %d'
              % (meth, R, e, e / R, upper95(e, R), wi, wo))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subs', default='cov')
    ap.add_argument('--split', default='', help="'development', 'held-out' or '' for all")
    a = ap.parse_args()
    main(a.subs.split(','), a.split)
