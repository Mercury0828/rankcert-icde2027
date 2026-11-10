"""R-PAIR-N = R-PAIR (variant r3,
derisk/repair3_run.py, unchanged inference) + a pilot-based variance-optimal (Neyman) allocation of the inference shots
across subexperiments (the derivation), fixed total budget S * N.

Draws (fresh, independent, all counted in the budget):
  pilot     : N // 20 shots per subexperiment, seed SeedSequence([cid, N, rep, 20261107, 41])
  inference : n_e shots for subexperiment e, seed SeedSequence([cid, N, rep, 20261107, 42]); sum_e n_e = S*N - S*(N//20)
Allocation (uses the pilot only): for the top-10 pairs (pilot ranks 5..12) compute the first-order influence functions
W_e of each pair difference at the pilot estimate (the derivation); a_e = sum over pairs of Var_{pilot q_e}(W_e);
n_e proportional to sqrt(a_e), with floor n_e >= FLOOR * (inference total / S); integer rounding by largest remainder.
Inference and labels: exactly R-PAIR (same deltas, A, pairs, rule), with per-subexperiment inference sizes n_e.

python -m derisk.repair4_run <circuit> --Ns 100000,1000000 --reps 30-59
Rows -> ./data/phase0/repair4/<circuit>.jsonl (resumable; key (N, rep); torn tail truncated).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from derisk.d1_run import CIRCUIT_ID, RUN, git_head, load
from derisk.repair3_run import D0, DP, DS, NA, PILOT_FRAC, check_structure
from rankcert.propagate.interval import dense_bounds, dense_point, factor_intervals, to_block_order
from rankcert.regions.regions import CPRegion
from stats.labels import evaluate, topk_truth

FLOOR = 0.2


def draw(qs, n_per, cid, N, rep, tag):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([cid, int(N), int(rep), 20261107, tag])))
    return [rng.multinomial(int(n), q / q.sum()).astype(float) for q, n in zip(qs, n_per)]


def make_feature(model, fi, qhat, cache):
    def feature(g, i, x):
        key = (g, i, x)
        if key in cache:
            return cache[key]
        use = model.uses[g][i]
        rows, signs, R = model.row_index(g, i)
        if x is None:
            a = signs
            lo, hi = fi['S'][g][model.block_of_sub[i]] if fi is not None else (np.nan, np.nan)
        else:
            b = model.block_of_sub[i]
            shift = sum(len(z) for z in model.blocks[:b])
            local = (x >> shift) & (R - 1)
            a = signs * (rows == local)
            if fi is not None:
                flo, fhi = fi['u'][g][b]
                lo, hi = flo[local], fhi[local]
            else:
                lo = hi = np.nan
        v = float(a @ qhat[use.e])
        r = max(v - lo, hi - v) if fi is not None else 0.0
        cache[key] = (use.e, a, v, r)
        return cache[key]
    return feature


def allocate(model, qs, pilot, selected, total, k=10):
    """Neyman allocation from the pilot only (returns integer n_e summing to total)."""
    S = len(qs)
    qp = [v / v.sum() for v in pilot]
    feature = make_feature(model, None, qp, {})
    lo_r, hi_r = max(1, k - 5), k + 2
    cand = [int(x) for x in selected[lo_r - 1:hi_r]]
    a = np.zeros(S)
    for j, x in enumerate(cand):
        for y in cand[j + 1:]:
            W = [np.zeros(len(q)) for q in qs]
            for sgn, s in ((1, x), (-1, y)):
                for g, c in enumerate(model.coefficients):
                    f0, f1 = feature(g, 0, s), feature(g, 1, s)
                    W[f0[0]] += sgn * (-c) * f0[1] * f1[2]
                    W[f1[0]] += sgn * (-c) * f1[1] * f0[2]
            a += np.array([float(qp[e] @ W[e] ** 2 - (qp[e] @ W[e]) ** 2) for e in range(S)])
    floor = FLOOR * total / S
    w = np.sqrt(np.maximum(a, 0.0))
    if w.sum() <= 0:
        n = np.full(S, total / S)
    else:
        n = np.full(S, floor)
        free = np.ones(S, bool)
        for _ in range(S):                       # water-filling with floors
            rest = total - n[~free].sum()
            share = rest * w[free] / w[free].sum() if w[free].sum() > 0 else np.full(free.sum(), rest / free.sum())
            low = share < floor
            if not low.any():
                n[free] = share
                break
            idx = np.flatnonzero(free)[low]
            n[idx] = floor
            free[idx] = False
    base = np.floor(n).astype(int)
    rem = int(total - base.sum())
    order = np.argsort(-(n - base), kind='stable')
    base[order[:rem]] += 1
    return base, a


def run_unit(model, qs, p, truth_by_k, N, rep, cid):
    M, S = len(p), len(qs)
    P = N // PILOT_FRAC
    pilot = draw(qs, [P] * S, cid, N, rep, 41)
    pilot_score = dense_point(model, [v / P for v in pilot])
    selected = np.argsort(-pilot_score, kind='stable')[:NA]
    n_alloc, a = allocate(model, qs, pilot, selected, S * N - S * P)
    counts = draw(qs, n_alloc, cid, N, rep, 42)
    ns = np.array([v.sum() for v in counts], dtype=float)
    qhat = [v / v.sum() for v in counts]
    reg = CPRegion(model, counts, D0)
    fi = factor_intervals(model, reg)
    L0, U0 = dense_bounds(model, fi)
    phat = dense_point(model, qhat)
    feature = make_feature(model, fi, qhat, {})

    def terms(x, mass):
        out = []
        for g, c in enumerate(model.coefficients):
            out.append((-c, feature(g, 0, x), feature(g, 1, x)))
            if mass:
                out.append((c / M, feature(g, 0, None), feature(g, 1, None)))
        return out

    def radius(terms_, t):
        W = [np.zeros(len(q)) for q in qs]
        D = [np.zeros(len(q)) for q in qs]
        rem = 0.
        for c, f0, f1 in terms_:
            rem += abs(c) * f0[3] * f1[3]
            for fa, fb in ((f0, f1), (f1, f0)):
                e, a_, v, r = fa
                W[e] += c * a_ * fb[2]
                D[e] += abs(c) * np.abs(a_) * fb[3]
        V, bmax = 0., 0.
        for e in range(S):
            mu = float(W[e] @ qhat[e])
            H = np.sqrt(reg.hi[e] @ ((W[e] - mu) ** 2))
            J = np.sqrt(reg.hi[e] @ (D[e] ** 2))
            V += (H + J) ** 2 / ns[e]
            osc = np.max(W[e] + D[e]) - np.min(W[e] - D[e])
            bmax = max(bmax, osc / ns[e])
        return np.sqrt(2 * V * t) + bmax * t / 3 + rem

    for x in selected[:4]:
        val = 1.0 / M + sum(c * f0[2] * f1[2] for c, f0, f1 in terms(int(x), True))
        if not abs(val - phat[int(x)]) <= 1e-9 * max(abs(phat[int(x)]), 1e-12):
            raise RuntimeError('term representation mismatch at x=%d' % int(x))
    ts = np.log(2 * NA / DS)
    L, U = L0.copy(), U0.copy()
    for x in selected:
        x = int(x)
        E = radius(terms(x, True), ts)
        L[x] = max(L[x], phat[x] - E)
        U[x] = min(U[x], phat[x] + E)
    empty = bool(np.any(L > U))
    out = {}
    for k, truth in truth_by_k.items():
        lo_r, hi_r = max(1, k - 5), k + 2
        cand = selected[lo_r - 1:hi_r]
        pairs = [(int(x), int(y)) for j, x in enumerate(cand) for y in cand[j + 1:]]
        tp = np.log(2 * len(pairs) / DP)
        unresolved = M - np.searchsorted(np.sort(U), L, side='left') - 1
        above = M - np.searchsorted(np.sort(L), U, side='right')
        ncert = 0
        for x, y in pairs:
            tt = terms(x, False) + [(-c, a_, b) for c, a_, b in terms(y, False)]
            E = radius(tt, tp)
            certified = bool(abs(phat[x] - phat[y]) > E)
            if phat[x] < phat[y]:
                x, y = y, x
            if certified:
                ncert += 1
                if U[y] >= L[x]:
                    unresolved[x] -= 1
                if L[x] <= U[y]:
                    above[y] += 1
        lab = np.zeros(M, dtype=np.int8)
        lab[unresolved <= k - 1] = 1
        lab[above >= k] = -1
        if empty:
            lab[:] = 0
        out['top%d' % k] = evaluate(lab, truth)
        out['top%d' % k].update(pairs=len(pairs), certified_pairs=ncert, tp=float(tp))
    out.update(empty=empty, pilot_per_e=int(P), inference_per_e=[int(v) for v in ns], ts=float(ts),
               alloc_a=[float(v) for v in a])
    return out


def main(name, Ns, reps):
    ref = load(name)
    model, qs = ref['model'], ref['qs']
    if not check_structure(model):
        print('%s: not applicable' % name)
        return
    p = to_block_order(model, ref['p'])
    truth_by_k = {k: topk_truth(p, k) for k in (5, 10)}
    path = os.path.join(RUN, 'repair4', name + '.jsonl')
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
                row = dict(circuit=name, N=N, total_shots=N * S, rep=rep, method='R-PAIR-N', commit=head, delta=0.05)
                row.update(run_unit(model, qs, p, truth_by_k, N, rep, CIRCUIT_ID[name]))
                row['seconds'] = round(time.time() - t0, 1)
                f.write(json.dumps(row) + '\n')
                f.flush()
                os.fsync(f.fileno())
                print(name, N, rep, {q: (row[q]['in_of_sure'], row[q]['out_of_sure'], row[q]['wrong'],
                                         row[q]['certified_pairs']) for q in ('top5', 'top10')},
                      row['inference_per_e'], row['seconds'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('circuit')
    ap.add_argument('--Ns', default='100000,1000000')
    ap.add_argument('--reps', default='30-59')
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.reps.split('-'))
    main(a.circuit, [int(x) for x in a.Ns.split(',')], list(range(lo, hi + 1)))
