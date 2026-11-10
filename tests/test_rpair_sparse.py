"""Sparse functional path of rankcert.certify.rpair (H3 widths): on A6 references, sparse=True and sparse=False give
identical labels (k = 5, 10; tau = 0.01, 0.05) for variants r4 and r6, and equal factor / score / pair radii to relative
1e-12 (radii on the first draw); the pooled LP equals linprog; auto activation is off on every A6 instance (dense path unchanged there)."""
import os
import pickle

import numpy as np
import pytest

from rankcert.certify.rpair import (D0, D0_6, DR_6, NA, NA6, SPARSE_AUTO, _lpmax, _Terms, certify, expand,
                                    select_pilot)
from rankcert.propagate.interval import factor_intervals
from rankcert.regions.regions import CPRegion

A6 = './data/a6/ref/'
REFS = [('arith16_s1001', 10 ** 4), ('bv16_s1001', 10 ** 3), ('qaoa2_20_s1001', 10 ** 5), ('qaoa2_24_s1001', 10 ** 5)]
_cache = {}


def _load(name):
    path = A6 + name + '.pkl'
    if not os.path.exists(path):
        pytest.skip('reference not built')
    if name not in _cache:
        ref = pickle.load(open(path, 'rb'))
        _cache[name] = (ref['model'], ref['qs'])
    return _cache[name]


def _draw(qs, N, rep):
    rng = np.random.default_rng([11, rep, N])
    pilot = [rng.multinomial(N // 20, q / q.sum()).astype(float) for q in qs]
    counts = [rng.multinomial(N, q / q.sum()).astype(float) for q in qs]
    return pilot, counts


def _close(a, b, rtol=1e-12):
    return abs(a - b) <= rtol * max(abs(a), abs(b)) + 1e-300


def test_pooled_lpmax_equals_linprog():
    """Merging cells of equal coefficient (pooled lo = sum lo, pooled hi = sum hi) leaves the LP optimum unchanged."""
    from scipy.optimize import linprog
    rng = np.random.default_rng(1)
    for _ in range(200):
        Z = int(rng.integers(4, 30))
        q = rng.dirichlet(np.ones(Z))
        lo = np.maximum(q - rng.random(Z) * .1, 0)
        hi = np.minimum(q + rng.random(Z) * .1, 1)
        lab = rng.integers(0, 3, Z)                     # three groups of equal coefficient
        gv = rng.normal(size=3)
        g = gv[lab]
        r = linprog(-g, A_eq=np.ones((1, Z)), b_eq=[1], bounds=list(zip(lo, hi)))
        plo = np.array([lo[lab == j].sum() for j in range(3)])
        phi = np.array([hi[lab == j].sum() for j in range(3)])
        keep = np.array([(lab == j).any() for j in range(3)])
        assert abs(-r.fun - _lpmax(plo[keep], phi[keep], gv[keep])) < 1e-9
        assert abs(_lpmax(lo, hi, g) - _lpmax(plo[keep], phi[keep], gv[keep])) < 1e-12


def test_auto_is_dense_on_a6():
    for name, _ in REFS:
        model, _ = _load(name)
        assert max(s.size for s in model.subexperiments) <= SPARSE_AUTO


@pytest.mark.parametrize('name,N', REFS)
@pytest.mark.parametrize('variant', ['r4', 'r6'])
def test_sparse_equals_dense(name, N, variant):
    model, qs = _load(name)
    r6 = variant == 'r6'
    for rep in range(2):
        pilot, counts = _draw(qs, N, rep)
        sel = select_pilot(model, pilot, 'search', na=NA6 if r6 else NA)
        # labels
        a = certify(model, pilot, counts, (5, 10), mode='auto', selected=sel, variant=variant, sparse=False)
        b = certify(model, pilot, counts, (5, 10), mode='auto', selected=sel, variant=variant, sparse=True)
        assert a['empty'] == b['empty']
        for k in (5, 10, 'tau0.01', 'tau0.05'):
            la, lb = expand(1 << model.m, a[k]), expand(1 << model.m, b[k])
            assert np.array_equal(la, lb), (name, variant, rep, k, int((la != lb).sum()))
            if k in (5, 10):
                assert a[k]['certified'] == b[k]['certified']
        if rep:                                         # radii on the first draw only (test time)
            continue
        # radii: factor, score (selected strings), pair
        ns = np.array([v.sum() for v in counts])
        qhat = [v / v.sum() for v in counts]
        reg = CPRegion(model, counts, D0_6 if r6 else D0)
        fi = factor_intervals(model, reg)
        Td = _Terms(model, fi, qhat, reg, ns, variant, sparse=False)
        Ts = _Terms(model, fi, qhat, reg, ns, variant, sparse=True)
        if r6:
            Td.tighten_factors(sel, DR_6)
            Ts.tighten_factors(sel, DR_6)
            dd = {(e, a_.tobytes()) for (e, a_, v, r) in Td.cache.values()}
            ds = {a_.ident(e) for (e, a_, v, r) in Ts.cache.values()}
            assert len(dd) == len(ds)
        assert Td.cache.keys() == Ts.cache.keys()
        for key in Td.cache:
            fd, fs_ = Td.cache[key], Ts.cache[key]
            assert fd[0] == fs_[0] and _close(fd[2], fs_[2]) and _close(fd[3], fs_[3]), (key, fd[2:], fs_[2:])
        t = 3.0
        for x in sel[:8]:
            assert _close(Td.radius(Td.terms(int(x), True), t), Ts.radius(Ts.terms(int(x), True), t))
        xs = [int(v) for v in sel[:5]]
        for j, x in enumerate(xs):
            for y in xs[j + 1:]:
                td = Td.terms(x, False) + [(-c, f) for c, f in Td.terms(y, False)]
                ts = Ts.terms(x, False) + [(-c, f) for c, f in Ts.terms(y, False)]
                assert _close(Td.radius(td, t), Ts.radius(ts, t))
                if r6:
                    assert _close(Td.bilinear(td, t), Ts.bilinear(ts, t))
