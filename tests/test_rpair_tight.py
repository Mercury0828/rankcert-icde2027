"""certify(collect='tight') (H3 collection threshold, rpair.certify docstring): labels equal the enumeration labels
(mode='dense') on the A6 references for variants r4 and r6, 2 draws, several N (incl. N = 1e3 where bounds are wide and
N = 1e6 where pair certificates are frequent), k = 5, 10, tau = 0.01, 0.05; collect='r4' (default) output equals the
reviewed pre-change implementation (rpair.py at commit d22097e, before the sparse / tight work);
BitSearcher.beam_reached reaches the strings beam_lower ranks."""
import os
import subprocess
import types

import numpy as np
import pytest

from rankcert.certify.rpair import NA, NA6, certify, expand, select_pilot
from rankcert.propagate.interval import factor_intervals
from rankcert.regions.regions import CPRegion
from rankcert.search.bitbnb import BitSearcher

A6 = 'D:/runs/rankcert/a6/ref/'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = 'd22097e'      # reviewed pre-change rpair.py (before the sparse / tight work): default-behaviour oracle
GRID = [('arith16_s1001', 10 ** 3), ('arith16_s1001', 10 ** 4), ('arith16_s1001', 10 ** 5),
        ('bv16_s1001', 10 ** 3), ('bv16_s1001', 10 ** 4),
        ('qaoa2_20_s1001', 10 ** 3), ('qaoa2_20_s1001', 10 ** 5), ('qaoa2_20_s1001', 10 ** 6),
        # m = 24: ~70 s per case (2^24 enumeration reference); run with RANKCERT_SLOW_TESTS=1 (all passed 2026-11-08)
        *[pytest.param('qaoa2_24_s1001', N, marks=pytest.mark.skipif(not os.environ.get('RANKCERT_SLOW_TESTS'),
                                                                      reason='slow: set RANKCERT_SLOW_TESTS=1'))
          for N in (10 ** 5, 10 ** 6)]]
QUERIES = (5, 10, 'tau0.01', 'tau0.05')
_cache = {}
SUMMARY = []


def _load(name):
    path = A6 + name + '.pkl'
    if not os.path.exists(path):
        pytest.skip('reference not built')
    if name not in _cache:
        import pickle
        ref = pickle.load(open(path, 'rb'))
        _cache[name] = (ref['model'], ref['qs'])
    return _cache[name]


def _draw(qs, N, rep):
    rng = np.random.default_rng([12, rep, N])
    pilot = [rng.multinomial(N // 20, q / q.sum()).astype(float) for q in qs]
    counts = [rng.multinomial(N, q / q.sum()).astype(float) for q in qs]
    return pilot, counts


@pytest.mark.parametrize('name,N', GRID)
@pytest.mark.parametrize('variant', ['r4', 'r6'])
def test_tight_equals_enumeration(name, N, variant):
    model, qs = _load(name)
    M = 1 << model.m
    for rep in range(2):
        pilot, counts = _draw(qs, N, rep)
        sel = select_pilot(model, pilot, 'search', na=NA6 if variant == 'r6' else NA)
        kw = dict(ks=(5, 10), selected=sel, taus=(0.01, 0.05), variant=variant, sparse=True)
        a = certify(model, pilot, counts, mode='dense', **kw)
        b = certify(model, pilot, counts, mode='search', collect='tight', **kw)
        assert a['empty'] == b['empty']
        for q in QUERIES:
            la, lb = expand(M, a[q]), expand(M, b[q])
            assert np.array_equal(la, lb), (name, N, variant, rep, q, int((la != lb).sum()))
        st = b['stats']
        SUMMARY.append((name, N, variant, rep, a[5]['certified'] + a[10]['certified'], st.get('cert_winners'),
                        st.get('theta_cert', np.inf) < st.get('lambda_kmax', -np.inf), st.get('collected_all')))


def test_certificates_exercised():
    """The grid must contain pair certificates, including winners whose L pulls theta below lambda_kmax."""
    if not SUMMARY:
        pytest.skip('grid not run')
    assert sum(s[4] for s in SUMMARY) > 0
    assert any(s[6] for s in SUMMARY), SUMMARY


def _head_rpair():
    try:
        src = subprocess.check_output(['git', 'show', ORACLE + ':rankcert/certify/rpair.py'], cwd=REPO).decode('utf-8')
    except Exception:
        pytest.skip('git not available')
    mod = types.ModuleType('rpair_head')
    exec(compile(src, 'rpair_head', 'exec'), mod.__dict__)
    return mod


@pytest.mark.parametrize('variant', ['r4', 'r6'])
@pytest.mark.parametrize('mode', ['search', 'auto'])
def test_default_collect_equals_head(variant, mode):
    head = _head_rpair()
    model, qs = _load('qaoa2_20_s1001')
    pilot, counts = _draw(qs, 10 ** 5, 0)
    a = head.certify(model, pilot, counts, mode=mode, variant=variant)
    b = certify(model, pilot, counts, mode=mode, variant=variant)
    assert a.keys() == b.keys() and a['stats'] == b['stats'] and a['empty'] == b['empty']
    assert np.array_equal(a['selected'], b['selected'])
    for q in QUERIES:
        assert a[q].keys() == b[q].keys()
        for f in a[q]:
            va, vb = a[q][f], b[q][f]
            assert (np.array_equal(va, vb) if isinstance(va, np.ndarray) else va == vb), (q, f)


def test_beam_reached_matches_beam_lower():
    model, qs = _load('qaoa2_20_s1001')
    _, counts = _draw(qs, 10 ** 4, 0)
    fi = factor_intervals(model, CPRegion(model, counts, 0.005))
    idx, L = BitSearcher(model, fi).beam_reached()
    assert len(np.unique(idx)) == len(idx)
    BS = BitSearcher(model, fi)
    assert float(np.sort(L)[-10]) == BS.beam_lower(10)
    L2, _ = BS.values(idx)
    assert np.array_equal(L, L2)
