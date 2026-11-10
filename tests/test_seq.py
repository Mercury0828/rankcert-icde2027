"""Appended shots (rankcert/certify/seq.py, derisk/seq_note.md): allocation weights equal brute-force influence
variances; the total-level argument of certify reproduces the registered split; sequential runs issue no wrong or
conflicting labels and account the shots."""
import os
import pickle

import numpy as np
import pytest

from rankcert.certify.rpair import certify, expand
from rankcert.certify.seq import neyman_safeguarded, pair_weights, run, varmin_weights
from rankcert.propagate.interval import dense_point, to_block_order
from stats.labels import topk_truth

ARITH = 'D:/runs/rankcert/a6/ref/arith16_s1001.pkl'
BV = 'D:/runs/rankcert/a6/ref/bv16_s1001.pkl'


def _load(path):
    if not os.path.exists(path):
        pytest.skip('reference not built')
    ref = pickle.load(open(path, 'rb'))
    return ref


def _jacobian(model, q, e):
    """J[x, z] = d p(x) / d q_e(z) (block order), exact: p is affine in q_e."""
    Z = len(q[e])
    base = [v.copy() for v in q]
    base[e] = np.zeros(Z)
    p0 = dense_point(model, base)
    J = np.empty((1 << model.m, Z))
    for z in range(Z):
        qq = list(base)
        qq[e] = np.zeros(Z)
        qq[e][z] = 1.0
        J[:, z] = dense_point(model, qq) - p0
    return J


@pytest.mark.parametrize('path', [ARITH, BV])
def test_varmin_and_pair_weights_brute_force(path):
    ref = _load(path)
    model, qs = ref['model'], ref['qs']
    q = [v / v.sum() for v in qs]
    a = varmin_weights(model, q)
    rng = np.random.default_rng(0)
    p = dense_point(model, q)
    cand = [int(x) for x in np.argsort(-p, kind='stable')[:4]]
    ap = pair_weights(model, q, cand)
    es = sorted({int(e) for e in rng.choice(len(q), size=min(3, len(q)), replace=False)} | {0})
    for e in es:
        if len(q[e]) > 1024:
            continue
        J = _jacobian(model, q, e)
        brute = float((J ** 2 @ q[e] - (J @ q[e]) ** 2).sum())
        assert abs(a[e] - brute) <= 1e-9 + 1e-7 * abs(brute), (e, a[e], brute)
        bp = 0.0
        for j, x in enumerate(cand):
            for y in cand[j + 1:]:
                W = J[x] - J[y]
                bp += float(q[e] @ W ** 2 - (q[e] @ W) ** 2)
        assert abs(ap[e] - bp) <= 1e-12 + 1e-7 * abs(bp), (e, ap[e], bp)


def test_neyman_safeguarded():
    a = np.array([0.0, 1.0, 4.0, 9.0])
    n = neyman_safeguarded(a, 1001, 0.5)
    assert n.sum() == 1001 and np.all(n >= np.ceil(0.5 * 1001 // 4))
    assert np.array_equal(neyman_safeguarded(np.ones(4), 1001, 1.0), [251, 250, 250, 250])


def test_certify_delta_argument():
    ref = _load(ARITH)
    model, qs = ref['model'], ref['qs']
    rng = np.random.default_rng(3)
    pilot = [rng.multinomial(500, q / q.sum()).astype(float) for q in qs]
    counts = [rng.multinomial(9500, q / q.sum()).astype(float) for q in qs]
    for v in ('r4', 'r6'):
        a = certify(model, pilot, counts, variant=v)
        b = certify(model, pilot, counts, variant=v, delta=0.05)
        c = certify(model, pilot, counts, variant=v, delta=0.0005)
        M = 1 << model.m
        for k in (5, 10):
            la, lb, lc = expand(M, a[k]), expand(M, b[k]), expand(M, c[k])
            assert np.array_equal(la, lb)
            assert np.all((lc == 0) | (lc == la))       # a smaller level only removes labels


@pytest.mark.parametrize('rule', ['uniform', 'boundary', 'varmin'])
def test_sequential_run(rule):
    ref = _load(ARITH)
    model, qs = ref['model'], ref['qs']
    truth = topk_truth(to_block_order(model, ref['p']), 10)
    rng = np.random.default_rng([11, len(rule)])
    S = len(qs)
    recs = list(run(model, qs, rule, 1000, 4, rng, truth=truth))
    assert [r['round'] for r in recs] == [1, 2, 3, 4]
    assert recs[-1]['shots'] == S * 50 + S * 1000 * (1 + 2 + 4 + 8)
    assert all(r['conflicts'] == 0 and r['top10']['wrong'] == 0 for r in recs)
    ins = [r['top10']['in_of_sure'] for r in recs]
    assert ins == sorted(ins)                           # accumulated labels never disappear


def test_clopper_pearson_tiny_level():
    """C4-seq R1: the upper endpoint stays non-trivial and inside the Hoeffding interval at tiny levels, and is unchanged
    (ppf path) at the A6 levels."""
    from scipy.stats import beta
    from rankcert.regions.regions import clopper_pearson
    N = 536870912000.0
    for alpha in (1e-12, 1e-16, 9e-17):
        hi = clopper_pearson(np.array([0.0, 3.0]), N, alpha)[1]
        h = np.sqrt(np.log(2 / alpha) / (2 * N))
        assert np.all(hi < np.array([0.0, 3.0]) / N + h)
    k = np.array([0.0, 5.0, 100.0])
    hi = clopper_pearson(k, 1e5, 6.1e-9)[1]
    assert np.array_equal(hi, beta.ppf(1 - 6.1e-9 / 2, k + 1, 1e5 - k))
