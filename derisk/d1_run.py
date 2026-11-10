"""D1 (decisiveness) and the D2 coverage loop share this runner.

python -m derisk.d1_run <circuit> --Ns 1000,10000,100000,1000000 --reps 30 --tag d1 [--methods R-H,R-CP,BL1-H,BL1-DKW]
Rows are appended to ./data/phase0/<tag>/<circuit>.jsonl; a restart skips (circuit, N, delta, method, rep)
units already present (checkpoint = the jsonl itself, flushed + fsynced after every unit; atomic per line).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import time

import numpy as np

from baselines import bl1
from rankcert.propagate.interval import dense_bounds, dense_point, factor_intervals, to_block_order
from rankcert.regions.regions import REGIONS
from stats.labels import evaluate, threshold_labels, threshold_truth, topk_labels, topk_truth

RUN = os.environ.get('RANKCERT_RUN', './data/phase0')
CIRCUIT_ID = {'qaoa20': 1, 'qaoa24': 2, 'adder16': 3, 'ghz20': 4, 'qaoa32': 5, 'qaoa40': 6}
KS = (5, 10)
TAUS = (0.01, 0.05)


def load(name):
    with open(os.path.join(RUN, 'ref', name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def draw_counts(qs, N, cid, rep):
    ss = np.random.SeedSequence([cid, int(N), int(rep), 20261107])
    rng = np.random.Generator(np.random.PCG64(ss))
    return [rng.multinomial(int(N), q / q.sum()).astype(float) for q in qs]


def bounds_for(method, model, n, delta, phat_block):
    if method in REGIONS:
        reg = REGIONS[method](model, n, delta)
        fi = factor_intervals(model, reg)
        L, U = dense_bounds(model, fi)
        return L, U, None
    if method.startswith('BL1-'):
        L, U, D = bl1.bounds(model, n, delta, method[4:], phat_block)
        return L, U, D
    raise ValueError(method)


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip()
    except Exception:
        return 'unknown'


def run(name, Ns, reps, tag, methods, delta=0.05, detail_reps=1, suffix='', rep_start=0):
    ref = load(name)
    model, qs, p_nat = ref['model'], ref['qs'], ref['p']
    p = to_block_order(model, p_nat)
    truths = {('k', k): topk_truth(p, k) for k in KS}
    truths.update({('t', t): threshold_truth(p, t) for t in TAUS})
    # boundary values for gap computation (top-k): p_(k) and p_(k+1)
    ps = np.sort(p)[::-1]
    out_dir = os.path.join(RUN, tag)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name + suffix + '.jsonl')
    done = set()
    if os.path.exists(path):
        # torn last line (interrupted write): truncate it so the recomputed row starts on a fresh line
        with open(path, 'rb+') as fb:
            data = fb.read()
            if data and not data.endswith(b'\n'):
                cut = data.rfind(b'\n') + 1
                fb.seek(cut)
                fb.truncate()
                with open(path + '.resume.log', 'a') as lg:
                    lg.write('%s truncated torn tail of %d bytes\n' % (time.strftime('%Y-%m-%dT%H:%M:%S'), len(data) - cut))
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r['N'], r['delta'], r['method'], r['rep']))
                except Exception:
                    pass  # a torn last line is recomputed (logged by count below)
    head = git_head()
    S = len(model.subexperiments)
    cid = CIRCUIT_ID[name]
    n_cuts = len(ref['manifest'].get('cut_locations', []))
    with open(path, 'a') as f:
        for N in Ns:
            for rep in range(rep_start, reps):
                todo = [m for m in methods if (N, delta, m, rep) not in done]
                if not todo:
                    continue
                n = draw_counts(qs, N, cid, rep)
                phat = dense_point(model, [v / v.sum() for v in n])
                for method in todo:
                    t0 = time.time()
                    L, U, D = bounds_for(method, model, n, delta, phat)
                    row = dict(circuit=name, cuts=n_cuts, N=N, total_shots=N * S, delta=delta,
                               method=method, rep=rep, commit=head, seconds=None, half_width=D)
                    for k in KS:
                        lab = topk_labels(L, U, k)
                        ev = evaluate(lab, truths[('k', k)])
                        row['top%d' % k] = ev
                        if rep < detail_reps:
                            und_sure = np.flatnonzero((lab == 0) & (truths[('k', k)] == 1))
                            row['top%d_und_sure' % k] = [
                                dict(x=int(x), p=float(p[x]), L=float(L[x]), U=float(U[x]), width=float(U[x] - L[x]),
                                     gap=float(p[x] - ps[k])) for x in und_sure[:20]]
                    for t in TAUS:
                        lab = threshold_labels(L, U, t)
                        row['tau%g' % t] = evaluate(lab, truths[('t', t)])
                    w = U - L
                    top_idx = np.argsort(-p)[:10]
                    row['width_top10_median'] = float(np.median(w[top_idx]))
                    row['width_all_median'] = float(np.median(w))
                    row['phat_max_abs_err'] = float(np.max(np.abs(phat - p)))
                    row['seconds'] = round(time.time() - t0, 2)
                    f.write(json.dumps(row) + '\n')
                    f.flush()
                    os.fsync(f.fileno())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('circuit')
    ap.add_argument('--Ns', default='1000,10000,100000,1000000')
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--tag', default='d1')
    ap.add_argument('--methods', default='R-H,R-CP,BL1-H,BL1-DKW')
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--rep-start', type=int, default=0)
    a = ap.parse_args()
    run(a.circuit, [int(x) for x in a.Ns.split(',')], a.reps, a.tag, a.methods.split(','), a.delta, suffix=a.suffix, rep_start=a.rep_start)
