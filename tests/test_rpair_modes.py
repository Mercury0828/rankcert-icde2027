"""R-PAIR certification: search mode (no 2^m vector) and auto mode (search with a dense fallback) give exactly the dense-mode labels."""
import os
import pickle

import numpy as np
import pytest

from rankcert.certify.rpair import certify, expand

REFS = [('D:/runs/rankcert/phase0/ref/qaoa20.pkl', 10 ** 6), ('D:/runs/rankcert/phase0/ref/adder16.pkl', 10 ** 4),
        ('D:/runs/rankcert/a6/ref/arith16_s1001.pkl', 10 ** 4), ('D:/runs/rankcert/a6/ref/arith16_s1002.pkl', 10 ** 3),
        ('D:/runs/rankcert/a6/ref/bv16_s1001.pkl', 10 ** 3), ('D:/runs/rankcert/a6/ref/qaoa2_20_s1001.pkl', 10 ** 5)]


@pytest.mark.parametrize('path,N', REFS)
def test_search_equals_dense(path, N):
    if not os.path.exists(path):
        pytest.skip('reference not built')
    ref = pickle.load(open(path, 'rb'))
    model, qs = ref['model'], ref['qs']
    for rep in range(2):
        rng = np.random.default_rng([7, rep, N])
        pilot = [rng.multinomial(N // 20, q / q.sum()).astype(float) for q in qs]
        counts = [rng.multinomial(N - N // 20, q / q.sum()).astype(float) for q in qs]
        a = certify(model, pilot, counts, mode='dense')
        b = certify(model, pilot, counts, mode='search')
        c = certify(model, pilot, counts, mode='auto', cap=10000)       # > beam frontier (8192); forces the dense fallback on wide regions
        d = certify(model, pilot, counts, mode='auto')
        assert list(a['selected']) == list(b['selected']) == list(c['selected']) == list(d['selected'])
        for k in (5, 10, 'tau0.01', 'tau0.05'):
            la = expand(1 << model.m, a[k])
            for o, tag in ((b, 'search'), (c, 'auto-cap1e4'), (d, 'auto')):
                lb = expand(1 << model.m, o[k])
                assert np.array_equal(la, lb), (path, rep, k, tag, int((la != lb).sum()))


@pytest.mark.parametrize('path,N', REFS)
def test_term_representation(path, N):
    """The product-term representation used by the bounds reproduces the plug-in score (any number of factors)."""
    if not os.path.exists(path):
        pytest.skip('reference not built')
    from rankcert.certify.rpair import _Terms
    from rankcert.propagate.interval import dense_point, factor_intervals
    from rankcert.regions.regions import CPRegion
    ref = pickle.load(open(path, 'rb'))
    model, qs = ref['model'], ref['qs']
    rng = np.random.default_rng(3)
    counts = [rng.multinomial(N, q / q.sum()).astype(float) for q in qs]
    qhat = [v / v.sum() for v in counts]
    reg = CPRegion(model, counts, 0.005)
    T = _Terms(model, factor_intervals(model, reg), qhat, reg, np.array([v.sum() for v in counts]))
    ph = dense_point(model, qhat)
    for x in list(np.argsort(-ph)[:5]) + [0, (1 << model.m) - 1]:
        assert abs(T.point(int(x)) - ph[int(x)]) <= 1e-9 * max(1e-12, abs(ph[int(x)])) + 1e-15
