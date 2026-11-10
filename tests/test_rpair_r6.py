"""Repair 6: search = dense labels; the LP envelope is the exact LP optimum;
the bilinear-remainder and factor-radius bounds hold on Monte Carlo draws; the default variant is unchanged."""
import os
import pickle

import numpy as np
import pytest

from rankcert.certify.rpair import _lpmax, _Terms, certify, expand, select_pilot
from rankcert.propagate.interval import factor_intervals
from rankcert.regions.regions import CPRegion

REFS = [('./data/phase0/ref/adder16.pkl', 10 ** 4), ('./data/a6/ref/arith16_s1001.pkl', 10 ** 4),
        ('./data/a6/ref/bv16_s1001.pkl', 10 ** 3), ('./data/a6/ref/qaoa2_20_s1001.pkl', 10 ** 5)]


def _load(path):
    if not os.path.exists(path):
        pytest.skip('reference not built')
    ref = pickle.load(open(path, 'rb'))
    return ref['model'], ref['qs']


def test_lpmax_is_lp_optimum():
    from scipy.optimize import linprog
    rng = np.random.default_rng(0)
    for _ in range(300):
        Z = int(rng.integers(2, 12))
        q = rng.dirichlet(np.ones(Z))
        lo = np.maximum(q - rng.random(Z) * .1, 0)
        hi = np.minimum(q + rng.random(Z) * .1, 1)
        g = rng.random(Z)
        r = linprog(-g, A_eq=np.ones((1, Z)), b_eq=[1], bounds=list(zip(lo, hi)))
        assert abs(-r.fun - _lpmax(lo, hi, g)) < 1e-9


@pytest.mark.parametrize('path,N', REFS)
def test_r6_search_equals_dense(path, N):
    model, qs = _load(path)
    for rep in range(2):
        rng = np.random.default_rng([8, rep, N])
        pilot = [rng.multinomial(N // 20, q / q.sum()).astype(float) for q in qs]
        counts = [rng.multinomial(N - N // 20, q / q.sum()).astype(float) for q in qs]
        a = certify(model, pilot, counts, mode='dense', variant='r6')
        b = certify(model, pilot, counts, mode='auto', variant='r6')
        assert list(a['selected']) == list(b['selected'])
        for k in (5, 10, 'tau0.01', 'tau0.05'):
            la, lb = expand(1 << model.m, a[k]), expand(1 << model.m, b[k])
            assert np.array_equal(la, lb), (path, rep, k, int((la != lb).sum()))


@pytest.mark.parametrize('path,N', [REFS[1], REFS[3]])
def test_r6_bounds_hold_monte_carlo(path, N):
    """For the selected strings: |d| <= factor radius for every factor functional, |bilinear remainder| <= its bound, and
    |phat - p| <= score radius, on 40 independent draws (the nominal failure rates are far below 1/40)."""
    model, qs = _load(path)
    qt = [q / q.sum() for q in qs]
    T0 = None
    viol = dict(factor=0, bilinear=0, score=0)
    for rep in range(40):
        rng = np.random.default_rng([9, rep, N])
        pilot = [rng.multinomial(N // 20, q).astype(float) for q in qt]
        counts = [rng.multinomial(N, q).astype(float) for q in qt]
        sel = select_pilot(model, pilot, 'search', na=32)
        ns = np.array([v.sum() for v in counts])
        qhat = [v / v.sum() for v in counts]
        reg = CPRegion(model, counts, 0.005)
        T = _Terms(model, factor_intervals(model, reg), qhat, reg, ns, 'r6')
        T.tighten_factors(sel, 0.01)
        t = np.log(4 * len(sel) / 0.02)
        for x in sel[:8]:
            tt = T.terms(int(x), True)
            rem_true = 0.0
            for c, fs in tt:
                d = [float(a @ (qhat[e] - qt[e])) for (e, a, v, r) in fs]
                viol["factor"] += sum(abs(di) > fs[i][3] + 1e-12 for i, di in enumerate(d))
                if len(fs) == 2:
                    rem_true += c * d[0] * d[1]
            if all(len(fs) == 2 for _, fs in tt):
                viol['bilinear'] += abs(rem_true) > T.bilinear(tt, t)
            # true score from the true distributions through the same term representation
            p_true = 1.0 / T.M + sum(c * float(np.prod([float(a @ qt[e]) for (e, a, v, r) in fs])) for c, fs in tt)
            viol['score'] += abs(T.point(int(x)) - p_true) > T.radius(tt, t)
    assert viol == dict(factor=0, bilinear=0, score=0), viol


def test_default_variant_unchanged():
    """variant 'r4' (default) is the A6 R-PAIR / R-PAIR-S construction: compare with explicit variant='r4'."""
    model, qs = _load(REFS[1][0])
    rng = np.random.default_rng(5)
    pilot = [rng.multinomial(500, q / q.sum()).astype(float) for q in qs]
    counts = [rng.multinomial(9500, q / q.sum()).astype(float) for q in qs]
    a = certify(model, pilot, counts, mode='auto')
    b = certify(model, pilot, counts, mode='auto', variant='r4')
    assert len(a['selected']) == 32
    for k in (5, 10, 'tau0.01', 'tau0.05'):
        assert np.array_equal(expand(1 << model.m, a[k]), expand(1 << model.m, b[k]))
