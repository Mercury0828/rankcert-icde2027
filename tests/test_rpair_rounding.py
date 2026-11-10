"""Sparse-path outward rounding: sparse decisions are a subset of dense
decisions (a sparse IN / OUT is always the dense label; the sparse path may only add UND), on a reviewer's exact-tie repro and
on random small cut models with thresholds placed exactly at dense interval endpoints. The A6 equality (no decision lost
by the margin) is tests/test_rpair_sparse.py::test_sparse_equals_dense."""
import numpy as np
import pytest

from rankcert.certify.rpair import D0, D0_6, DR_6, _Terms, certify, expand
from rankcert.propagate.interval import dense_bounds, factor_intervals
from rankcert.recon.model import CutModel, Subexperiment, Use
from rankcert.regions.regions import CPRegion


def _subset(dense, sparse):
    return bool(np.all((sparse == 0) | (sparse == dense)))


def _tie_model():
    return CutModel(6, [[0, 1, 2], [3, 4, 5]], np.array([-1.]), [Subexperiment((i,), 5, 0, 5) for i in range(2)],
                    [[Use(0, [0, 1, 2], [0, 2, 4]), Use(1, [3, 4, 5], [0, 2, 4])]], [0, 1])


def test_tie_repro():
    model = _tie_model()
    rng = np.random.default_rng(901)
    counts = [rng.multinomial(10 ** 6, rng.dirichlet(np.ones(32))).astype(float) for _ in range(2)]
    tau = 0.0035268944303920692
    kw = dict(ks=(5, 10), selected=np.arange(64), taus=(tau,), variant='r6')
    a = certify(model, counts, counts, sparse=False, **kw)
    b = certify(model, counts, counts, sparse=True, **kw)
    for q in (5, 10, 'tau%g' % tau):
        la, lb = expand(64, a[q]), expand(64, b[q])
        assert _subset(la, lb), q
    assert expand(64, b['tau%g' % tau])[8] in (0, expand(64, a['tau%g' % tau])[8])


def _random_model(rng):
    """G groups, 2 subcircuits of 3 held bits each, subexperiments with 0-2 QPD bits (pooled sign cells), random
    coefficients and dropped bits."""
    G = int(rng.integers(1, 4))
    subs, uses = [], []
    for g in range(G):
        row = []
        for i in range(2):
            w = int(rng.integers(0, 3))
            subs.append(Subexperiment((g, i), 3 + w, w, 3))
            row.append(Use(len(subs) - 1, [0, 1, 2] if i == 0 else [3, 4, 5], [0, 1, 2], int(rng.integers(0, 2))))
        uses.append(row)
    coef = rng.choice([-1.0, 1.0], size=G) * rng.uniform(0.2, 1.0, size=G)
    return CutModel(6, [[0, 1, 2], [3, 4, 5]], coef, subs, uses, [0, 1])


def _dense_endpoints(model, counts, variant, sel):
    """The dense path's refined endpoints LA, UA of the selected strings (certify arithmetic)."""
    from rankcert.certify.rpair import DS, DS_6
    r6 = variant == 'r6'
    reg = CPRegion(model, counts, D0_6 if r6 else D0)
    fi = factor_intervals(model, reg)
    ns = np.array([v.sum() for v in counts])
    T = _Terms(model, fi, [v / v.sum() for v in counts], reg, ns, variant, sparse=False)
    if r6:
        T.tighten_factors(sel, DR_6)
    L0, U0 = dense_bounds(model, fi)
    ts = np.log((4 if r6 else 2) * len(sel) / (DS_6 if r6 else DS))
    pA = np.array([T.point(int(x)) for x in sel])
    EA = np.array([T.radius(T.terms(int(x), True), ts) for x in sel])
    return np.maximum(L0[sel], pA - EA), np.minimum(U0[sel], pA + EA)


@pytest.mark.parametrize('variant', ['r4', 'r6'])
def test_random_models_sparse_subset_of_dense(variant):
    rng = np.random.default_rng(77)
    n_dec_dense = n_dec_sparse = 0
    for trial in range(60):
        model = _random_model(rng)
        N = int(rng.choice([10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]))
        counts = [rng.multinomial(N, rng.dirichlet(np.full(s.size, float(rng.choice([0.3, 1.0, 5.0]))))).astype(float)
                  for s in model.subexperiments]
        sel = np.arange(64)
        LA, UA = _dense_endpoints(model, counts, variant, sel)
        pick = rng.choice(64, 3, replace=False)
        taus = tuple(sorted({float(LA[pick[0]]), float(UA[pick[1]]), float(LA[pick[2]])}))   # exact ties
        kw = dict(ks=(1, 3, 5), selected=sel, taus=taus, variant=variant)
        a = certify(model, counts, counts, sparse=False, **kw)
        b = certify(model, counts, counts, sparse=True, **kw)
        for q in [1, 3, 5] + ['tau%g' % t for t in taus]:
            la, lb = expand(64, a[q]), expand(64, b[q])
            assert _subset(la, lb), (trial, q)
            n_dec_dense += int((la != 0).sum())
            n_dec_sparse += int((lb != 0).sum())
    assert n_dec_sparse > 0.9 * n_dec_dense, (n_dec_sparse, n_dec_dense)   # the margin drops only ties
