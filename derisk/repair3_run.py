"""R-PAIR = the derivation
construction: pilot (5 % of each subexperiment's shots, exact hypergeometric split)
selects A = pilot top-32 strings and the pairs among pilot ranks max(1, k-5) .. k+2; inference shots give the all-string
CP region C0 (delta0 = 0.005, interval propagation L0, U0), Bernstein score bounds for A (delta_s = 0.005,
t_s = log(2|A|/delta_s)) and Bernstein difference bounds for the pairs (delta_p = 0.04, t_p = log(2|P|/delta_p));
three-valued top-k labels with certified orderings (R3 eq. 16). Faithful generalisation of an earlier prototype
(same arithmetic) to any 2-subcircuit instance and k in {5, 10}; one procedure per query k at delta = 0.05.

python -m derisk.repair3_run <circuit> --Ns 1000,10000,100000,1000000 --reps 0-29
Rows -> ./data/phase0/repair3/<circuit>.jsonl (resumable; key (N, rep, k)).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from derisk.d1_run import CIRCUIT_ID, RUN, draw_counts, git_head, load
from rankcert.propagate.interval import dense_bounds, dense_point, factor_intervals, to_block_order
from rankcert.regions.regions import CPRegion
from stats.labels import evaluate, topk_truth

D0, DS, DP = 0.005, 0.005, 0.04
NA = 32
PILOT_FRAC = 20          # pilot = N // 20


def check_structure(model):
    ok = all(len(row) == 2 for row in model.uses) and all(row[0].e != row[1].e for row in model.uses) \
        and all(len(u.held) > 0 for row in model.uses for u in row) and len(model.blocks) == 2
    return ok


def run_unit(model, qs, p, truth_by_k, N, rep, cid):
    M, S = len(p), len(qs)
    counts = draw_counts(qs, N, cid, rep)
    P = N // PILOT_FRAC
    rng = np.random.default_rng(17 + rep)
    pilot = [rng.multivariate_hypergeometric(v.astype(np.int64), P) for v in counts]
    counts = [v - a for v, a in zip(counts, pilot)]
    ns = np.array([v.sum() for v in counts], dtype=float)
    qhat = [v / v.sum() for v in counts]
    pilot_score = dense_point(model, [v / P for v in pilot])
    selected = np.argsort(-pilot_score, kind='stable')[:NA]
    reg = CPRegion(model, counts, D0)
    fi = factor_intervals(model, reg)
    L0, U0 = dense_bounds(model, fi)
    phat = dense_point(model, qhat)
    cache = {}

    def feature(g, i, x):
        key = (g, i, x)
        if key in cache:
            return cache[key]
        use = model.uses[g][i]
        rows, signs, R = model.row_index(g, i)
        if x is None:
            a = signs
            lo, hi = fi['S'][g][model.block_of_sub[i]]
        else:
            b = model.block_of_sub[i]
            shift = sum(len(z) for z in model.blocks[:b])
            local = (x >> shift) & (R - 1)
            a = signs * (rows == local)
            flo, fhi = fi['u'][g][b]
            lo, hi = flo[local], fhi[local]
        v = float(a @ qhat[use.e])
        r = max(v - lo, hi - v)
        cache[key] = (use.e, a, v, r)
        return cache[key]

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
                e, a, v, r = fa
                W[e] += c * a * fb[2]
                D[e] += abs(c) * np.abs(a) * fb[3]
        V, bmax = 0., 0.
        for e in range(S):
            mu = float(W[e] @ qhat[e])
            H = np.sqrt(reg.hi[e] @ ((W[e] - mu) ** 2))
            J = np.sqrt(reg.hi[e] @ (D[e] ** 2))
            V += (H + J) ** 2 / ns[e]
            osc = np.max(W[e] + D[e]) - np.min(W[e] - D[e])
            bmax = max(bmax, osc / ns[e])
        return np.sqrt(2 * V * t) + bmax * t / 3 + rem

    # algebra check: the term representation reproduces the plug-in score (guards the frozen-form assumption)
    for x in selected[:4]:
        val = 1.0 / M + sum(c * f0[2] * f1[2] for c, f0, f1 in terms(int(x), True))
        if not abs(val - phat[int(x)]) <= 1e-9 * max(abs(phat[int(x)]), 1e-12):
            raise RuntimeError('term representation mismatch at x=%d: %r vs %r' % (int(x), val, phat[int(x)]))
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
        lo_r, hi_r = max(1, k - 5), k + 2                      # pilot ranks (1-based, inclusive)
        cand = selected[lo_r - 1:hi_r]
        pairs = [(int(x), int(y)) for j, x in enumerate(cand) for y in cand[j + 1:]]
        tp = np.log(2 * len(pairs) / DP)
        unresolved = M - np.searchsorted(np.sort(U), L, side='left') - 1
        above = M - np.searchsorted(np.sort(L), U, side='right')
        ncert = 0
        for x, y in pairs:
            tt = terms(x, False) + [(-c, a, b) for c, a, b in terms(y, False)]
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
    out.update(empty=empty, pilot_per_e=int(P), inference_per_e=[int(v) for v in ns], ts=float(ts))
    return out


def main(name, Ns, reps):
    ref = load(name)
    model, qs = ref['model'], ref['qs']
    if not check_structure(model):
        print('%s: structure outside the 2-subcircuit / 2-held-factor form; not applicable' % name)
        return
    p = to_block_order(model, ref['p'])
    truth_by_k = {k: topk_truth(p, k) for k in (5, 10)}
    path = os.path.join(RUN, 'repair3', name + '.jsonl')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set()
    if os.path.exists(path):
        # torn last line (interrupted write): truncate it before appending
        with open(path, 'rb+') as fb:
            data = fb.read()
            if data and not data.endswith(b'\n'):
                cut = data.rfind(b'\n') + 1
                fb.seek(cut)
                fb.truncate()
                with open(path + '.resume.log', 'a') as lg:
                    lg.write('%s truncated torn tail of %d bytes\n' % (time.strftime('%Y-%m-%dT%H:%M:%S'), len(data) - cut))
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
                row = dict(circuit=name, N=N, total_shots=N * S, rep=rep, method='R-PAIR', commit=head, delta=0.05)
                row.update(run_unit(model, qs, p, truth_by_k, N, rep, CIRCUIT_ID[name]))
                row['seconds'] = round(time.time() - t0, 1)
                f.write(json.dumps(row) + '\n')
                f.flush()
                os.fsync(f.fileno())
                print(name, N, rep, {q: (row[q]['in_of_sure'], row[q]['out_of_sure'], row[q]['wrong'],
                                         row[q]['certified_pairs']) for q in ('top5', 'top10')}, row['seconds'],
                      flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('circuit')
    ap.add_argument('--Ns', default='1000,10000,100000,1000000')
    ap.add_argument('--reps', default='0-29')
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.reps.split('-'))
    main(a.circuit, [int(x) for x in a.Ns.split(',')], list(range(lo, hi + 1)))
