"""A6 evaluation runner (derisk/BENCH_A6_PREREG.md v2).

python -m derisk.a6_run <instance> [<instance> ...] --Ns 1000,10000,100000,1000000 --reps 0-29
Rows -> D:/runs/rankcert/a6/runs/<instance>.jsonl (resumable; key (N, rep, method)).
Held-out instances (seeds 1006-1010) are refused unless the A6-FREEZE marker file derisk/A6_FREEZE.json exists (it is
written by the freeze commit and names the frozen methods by commit hash).

Methods at a common total budget B = S * N:
  R-CP     uniform N per subexperiment (stream 0), CP region delta = 0.05, interval propagation, Phase-0 labels
  BL6      = R-CP labels; its search cost is the Phase-0 QCut-style block search (rankcert/search/bnb.py), timed on rep 0
  BL1      stronger of BL1-H / BL1-DKW per unit (both recorded), stream 0 draws
  R-PAIR   pilot N//20 per e (stream 41), uniform inference (stream 42), rankcert.certify.rpair (mode 'auto': search,
         dense fallback when the collection exceeds AUTO_CAP; labels identical in all modes, tests/test_rpair_modes.py)
  R-PAIR-S same pilot, safeguarded allocation (beta = 1/2, derisk/repair5_run.py), inference stream 42, mode 'auto'
  R-PAIR6 / R-PAIR-S6  the same with certify variant 'r6' (repair 6, derisk/repair6_note.md: |A| = 128 from the same
         pilot; the allocation still uses the top-32 pilot strings, so R-PAIR-S6 draws the same inference counts as
         R-PAIR-S). Opt-in: --methods (default: the four registered methods).
Seeds: SeedSequence([family_id, m, instance_seed, N, rep, 20261107, stream]).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np

from baselines import bl1
from bench.a6 import A6, FAMILY_ID, HELD, table
from derisk.d1_run import git_head
from derisk.repair5_run import allocate_safeguarded
from rankcert.certify.rpair import NA6, certify, check_structure, expand, select_pilot
from rankcert.propagate.interval import dense_bounds, dense_point, factor_intervals, to_block_order
from rankcert.regions.regions import CPRegion
from rankcert.search.bitbnb import BitSearcher
from rankcert.search.bnb import Searcher
from stats.labels import evaluate, threshold_labels, threshold_truth, topk_labels, topk_truth

FREEZE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'A6_FREEZE.json')
KS = (5, 10)
TAUS = (0.01, 0.05)


def load_instance(name):
    row = table()[name]
    if row['seed'] in HELD and not os.path.exists(FREEZE):
        raise SystemExit('%s is held-out: refused before A6-FREEZE' % name)
    sub = 'embargo' if row['seed'] in HELD else 'ref'
    with open(os.path.join(A6, sub, name + '.pkl'), 'rb') as f:
        return row, pickle.load(f)


def draw(qs, sizes, key, stream):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(key) + [20261107, stream])))
    return [rng.multinomial(int(n), q / q.sum()).astype(float) for q, n in zip(qs, sizes)]


def evaluate_all(lab_by_q, truths):
    return {q: evaluate(lab_by_q[q], truths[q]) for q in truths}


BASE_METHODS = ('R-CP', 'BL1', 'R-PAIR', 'R-PAIR-S')
PAIR_METHODS = ('R-PAIR', 'R-PAIR-S', 'R-PAIR6', 'R-PAIR-S6')


def run(name, Ns, reps, want=BASE_METHODS, sub='runs'):
    row, ref = load_instance(name)
    model, qs = ref['model'], ref['qs']
    p = to_block_order(model, ref['p'])            # labels are in block order (review)
    S, m = len(qs), model.m
    M = 1 << m
    truths = {'top%d' % k: topk_truth(p, k) for k in KS}
    truths.update({'tau%g' % t: threshold_truth(p, t) for t in TAUS})
    path = os.path.join(A6, sub, name + '.jsonl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set()
    if os.path.exists(path):
        with open(path, 'rb+') as fb:
            data = fb.read()
            if data and not data.endswith(b'\n'):
                cut = data.rfind(b'\n') + 1
                fb.seek(cut)
                fb.truncate()
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r['N'], r['rep'], r['method']))
            except Exception:
                pass
    head = git_head()
    fam = FAMILY_ID[row['family']]
    with open(path, 'a') as f:
        def emit(rowd):
            f.write(json.dumps(rowd) + '\n')
            f.flush()
            os.fsync(f.fileno())

        for N in Ns:
            for rep in reps:
                key = (fam, m, row['seed'], N, rep)
                base = dict(instance=name, family=row['family'], split=row['split'], m=m, S=S, N=N, total_shots=S * N,
                            rep=rep, commit=head, delta=0.05)
                methods = tuple(mm for mm in want if check_structure(model) or mm not in PAIR_METHODS)
                todo = [mm for mm in methods if (N, rep, mm) not in done]   # R-PAIR family needs 2 subcircuits
                if not todo:
                    continue
                n0 = draw(qs, [N] * S, key, 0)
                if 'R-CP' in todo:
                    t0 = time.perf_counter()
                    fi = factor_intervals(model, CPRegion(model, n0, 0.05))
                    L, U = dense_bounds(model, fi)
                    labs = {'top%d' % k: topk_labels(L, U, k) for k in KS}
                    labs.update({'tau%g' % t: threshold_labels(L, U, t) for t in TAUS})
                    rr = dict(base, method='R-CP', seconds=round(time.perf_counter() - t0, 2), **evaluate_all(labs, truths))
                    if rep == 0:                                  # BL6 / ours search cost on the same region
                        for k in KS:
                            t1 = time.perf_counter()
                            Sb = Searcher(model, fi)
                            Sb.topk(k)
                            rr['BL6_search_top%d' % k] = dict(seconds=round(time.perf_counter() - t1, 3), leaves=Sb.leaves,
                                                             strings_evaluated=Sb.leaves * (1 << len(model.blocks[-1])))
                            t1 = time.perf_counter()
                            Bs = BitSearcher(model, fi)
                            res = Bs.topk(k)
                            rr['R3a_search_top%d' % k] = dict(seconds=round(time.perf_counter() - t1, 3),
                                                             strings_evaluated=Bs.stats['bottom_evaluated'],
                                                             candidates=int(len(res['cand'])))
                    emit(rr)
                if 'BL1' in todo:
                    t0 = time.perf_counter()
                    phat = dense_point(model, [v / v.sum() for v in n0])
                    best = None
                    for var in ('H', 'DKW'):
                        L, U, D = bl1.bounds(model, n0, 0.05, var, phat)
                        labs = {'top%d' % k: topk_labels(L, U, k) for k in KS}
                        labs.update({'tau%g' % t: threshold_labels(L, U, t) for t in TAUS})
                        ev = evaluate_all(labs, truths)
                        score = sum(ev[q]['in_of_sure'] + ev[q]['out_of_sure'] for q in ev)
                        if best is None or score > best[0]:
                            best = (score, var, ev)
                    emit(dict(base, method='BL1', variant=best[1], seconds=round(time.perf_counter() - t0, 2), **best[2]))
                if any(mm in todo for mm in PAIR_METHODS):
                    P = N // 20
                    pilot = draw(qs, [P] * S, key, 41)
                    sel = select_pilot(model, pilot, 'search')
                    sel6 = select_pilot(model, pilot, 'search', na=NA6) if any(mm.endswith('6') for mm in todo) else None
                    total = S * N - S * P
                    for meth in PAIR_METHODS:
                        if meth not in todo:
                            continue
                        if meth in ('R-PAIR', 'R-PAIR6'):
                            sizes = [N - P] * S
                        else:
                            sizes, _ = allocate_safeguarded(model, qs, pilot, sel, total)
                        inf = draw(qs, sizes, key, 42)
                        r6 = meth.endswith('6')
                        t0 = time.perf_counter()
                        ans = certify(model, pilot, inf, KS, mode='auto', selected=sel6 if r6 else sel, taus=TAUS,
                                      variant='r6' if r6 else 'r4')
                        dt = time.perf_counter() - t0
                        labs = {('top%d' % k): expand(M, ans[k]) for k in KS}
                        labs.update({('tau%g' % t): expand(M, ans['tau%g' % t]) for t in TAUS})
                        ev = evaluate_all(labs, truths)
                        emit(dict(base, method=meth, seconds=round(dt, 2), inference_per_e=[int(v) for v in sizes],
                                  pilot_per_e=P, empty=ans['empty'], path=ans['stats'].get('path'),
                                  answer_size={('top%d' % k): int(len(ans[k]['cand'])) for k in KS},
                                  certified={('top%d' % k): ans[k]['certified'] for k in KS}, **ev))
                print(name, N, rep, 'done', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('instances', nargs='+')
    ap.add_argument('--Ns', default='1000,10000,100000,1000000')
    ap.add_argument('--reps', default='0-29')
    ap.add_argument('--methods', default=','.join(BASE_METHODS))
    ap.add_argument('--sub', default='runs', help='row directory under the A6 root (separate writers use separate files)')
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.reps.split('-'))
    for nm in a.instances:
        run(nm, [int(x) for x in a.Ns.split(',')], list(range(lo, hi + 1)), tuple(a.methods.split(',')), a.sub)
