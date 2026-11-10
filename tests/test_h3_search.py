"""derisk/h3_search.py helpers on an A6 development reference (m = 20): the search-based exact top-k reference equals the
dense truth, the truth evaluation equals stats.labels.evaluate on the expanded labels, QCut top() values equal the dense
plug-in top values (outcome indices mapped to block order), and the enumeration chunk equals the head of dense_bounds."""
import os
import pickle

import numpy as np
import pytest

from derisk.a6_run import draw
from derisk.h3_search import enum_chunk, evaluate_vs_truth, exact_truth, natural_to_block, qcut_distribution
from rankcert.certify.rpair import NA6, certify, expand, select_pilot
from rankcert.propagate.interval import dense_bounds, dense_point, factor_intervals, to_block_order
from rankcert.regions.regions import CPRegion
from stats.labels import evaluate, topk_truth

REF = './data/a6/ref/qaoa2_20_s1001.pkl'


@pytest.fixture(scope='module')
def ref():
    if not os.path.exists(REF):
        pytest.skip('reference not built')
    return pickle.load(open(REF, 'rb'))


def test_truth_and_evaluation(ref):
    model, qs = ref['model'], ref['qs']
    p = to_block_order(model, ref['p'])
    tt = topk_truth(p, 10)
    tr = exact_truth(model, qs, 10)
    assert np.array_equal(np.flatnonzero(tt == 1), tr['sure_members'])
    assert np.array_equal(np.flatnonzero(tt != -1), tr['not_sure_non'])
    S, N = len(qs), 100000
    key = (2, model.m, 1001, N, 7)
    pilot, inf = draw(qs, [N // 20] * S, key, 41), draw(qs, [N] * S, key, 42)
    sel = select_pilot(model, pilot, 'search', na=NA6)
    ans = certify(model, pilot, inf, (10,), mode='search', selected=sel, taus=(), variant='r6')
    ev = evaluate_vs_truth(ans[10], tr, 1 << model.m)
    ref_ev = evaluate(expand(1 << model.m, ans[10]), tt)
    assert all(ev[k] == ref_ev[k] for k in ref_ev), (ev, ref_ev)


def test_qcut_top_and_chunk(ref):
    model, qs = ref['model'], ref['qs']
    S, N = len(qs), 10000
    inf = draw(qs, [N] * S, (2, model.m, 1001, N, 7), 42)
    qhat = [v / v.sum() for v in inf]
    top = qcut_distribution(model, qhat).top(10)
    ph = dense_point(model, qhat)
    for x, v in top:
        assert abs(ph[natural_to_block(model, x)] - v) < 1e-15
    assert abs(np.sort(ph)[::-1][9] - top[-1][1]) < 1e-15
    fi = factor_intervals(model, CPRegion(model, inf, 0.005))
    L, U = dense_bounds(model, fi)
    lead = model.m - len(model.blocks[-1])
    sl = dict(s=fi['s'], S=fi['S'], u=[[b for b in ug[:-1]] + [(ug[-1][0][:4], ug[-1][1][:4])] for ug in fi['u']])
    L2, U2 = dense_bounds(model, sl)
    assert np.array_equal(L2, L[:4 << lead]) and np.array_equal(U2, U[:4 << lead])
    ch = enum_chunk(model, fi, lead + 2)
    assert ch['chunk_strings'] == 4 << lead
