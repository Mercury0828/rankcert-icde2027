"""D1 repair attempt 5 (pre-registered in derisk/PREREG.md, appendix "repair attempt 5"): R-PAIR-S = R-PAIR-N with the
uniform-allocation safeguard of C1-region R4 (rounds/C1-region_R4_response.md, part (b), eq. (23)): reserve
b_e = ceil(BETA * U_e) for every subexperiment (U_e = balanced uniform split of the inference budget), distribute the rest
by the pilot Neyman weights w_e = sqrt(a_e) / sum_j sqrt(a_j) (floor + largest remainder, ties by index).
BETA = 1/2. Everything else (pilot, draws, inference, labels) is repair 4 unchanged (derisk/repair4_run.py).

python -m derisk.repair5_run <circuit> --Ns ... --reps a-b
Rows -> D:/runs/rankcert/phase0/repair5/<circuit>.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import math

import numpy as np

import derisk.repair4_run as r4

BETA = 0.5


def allocate_safeguarded(model, qs, pilot, selected, total, k=10):
    """R4 (b) safeguard: N_e >= BETA * U_e, sum N_e = total; pilot-only."""
    _, a = _ORIG_ALLOCATE(model, qs, pilot, selected, total, k)     # reuse the pilot variance weights a_e
    S = len(qs)
    U = np.full(S, total // S, dtype=int)
    U[: total - U.sum()] += 1                                         # balanced uniform split, ties by index
    b = np.array([math.ceil(BETA * u) for u in U], dtype=int)
    rest = int(total - b.sum())
    w = np.sqrt(np.maximum(a, 0.0))
    w = w / w.sum() if w.sum() > 0 else np.full(S, 1.0 / S)
    x = rest * w
    base = np.floor(x).astype(int)
    rem = rest - int(base.sum())
    order = np.lexsort((np.arange(S), -(x - base)))                   # largest remainder, ties by index
    base[order[:rem]] += 1
    n = b + base
    assert n.sum() == total and np.all(n >= BETA * U)
    return n, a


_ORIG_ALLOCATE = r4.allocate


def _run_unit(model, qs, p, truth_by_k, N, rep, cid):
    saved = r4.allocate
    r4.allocate = allocate_safeguarded
    try:
        return r4.run_unit(model, qs, p, truth_by_k, N, rep, cid)
    finally:
        r4.allocate = saved


def main(name, Ns, reps):
    import json
    import os
    import time
    from derisk.d1_run import CIRCUIT_ID, RUN, git_head, load
    from rankcert.propagate.interval import to_block_order
    from stats.labels import topk_truth
    ref = load(name)
    model, qs = ref['model'], ref['qs']
    if not r4.check_structure(model):
        print('%s: not applicable' % name)
        return
    p = to_block_order(model, ref['p'])
    truth_by_k = {k: topk_truth(p, k) for k in (5, 10)}
    path = os.path.join(RUN, 'repair5', name + '.jsonl')
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
                done.add((r['N'], r['rep']))
            except Exception:
                pass
    head = git_head()
    S = len(qs)
    with open(path, 'a') as f:
        for N in Ns:
            for rep in reps:
                if (N, rep) in done:
                    continue
                t0 = time.time()
                row = dict(circuit=name, N=N, total_shots=N * S, rep=rep, method='R-PAIR-S', beta=BETA, commit=head,
                           delta=0.05)
                row.update(_run_unit(model, qs, p, truth_by_k, N, rep, CIRCUIT_ID[name]))
                row['seconds'] = round(time.time() - t0, 1)
                f.write(json.dumps(row) + '\n')
                f.flush()
                os.fsync(f.fileno())
                print(name, N, rep, {q: (row[q]['in_of_sure'], row[q]['out_of_sure'], row[q]['wrong']) for q in
                                     ('top5', 'top10')}, row['seconds'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('circuit')
    ap.add_argument('--Ns', default='10000,100000,1000000')
    ap.add_argument('--reps', default='60-89')
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.reps.split('-'))
    main(a.circuit, [int(x) for x in a.Ns.split(',')], list(range(lo, hi + 1)))
