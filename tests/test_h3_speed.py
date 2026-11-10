"""H3 speed changes (m = 40 profile): regions.unique_inverse returns exactly np.unique(v, return_inverse=True) (so CPRegion
and every A6 number are unchanged), and the sparse support index of rpair._Terms equals its definition
{z : row_index(z) == local} (including subexperiments whose held bits are a strict subset of the output bits)."""
import os
import pickle

import numpy as np
import pytest

from rankcert.certify.rpair import _Terms
from rankcert.recon.model import CutModel, Subexperiment, Use
from rankcert.regions.regions import CPRegion, clopper_pearson, unique_inverse

A6 = './data/a6/ref/'


def _same_unique(v):
    u0, i0 = np.unique(v, return_inverse=True)
    u1, i1 = unique_inverse(v)
    assert u0.dtype == u1.dtype and i0.dtype == i1.dtype and i0.shape == i1.shape
    assert np.array_equal(u0, u1) and np.array_equal(i0, i1)


def test_unique_inverse_equals_numpy():
    rng = np.random.default_rng(3)
    for _ in range(50):
        Z = int(rng.integers(1, 5000))
        q = rng.dirichlet(np.full(Z, float(rng.choice([0.05, 1.0]))))
        _same_unique(rng.multinomial(int(rng.choice([10, 10 ** 3, 10 ** 6])), q).astype(float))
    _same_unique(np.array([0.0, 0.0, 3.0]))
    _same_unique(np.array([1.5, 0.0, 1.5]))                  # non-integral: np.unique path
    _same_unique(np.array([0.0, 1e12, 5.0]))                 # large values: np.unique path
    _same_unique(np.array([-1.0, 2.0]))                      # negative: np.unique path
    _same_unique(np.zeros(0))


def test_cpregion_bits_unchanged_on_a6():
    for name, N in (('qaoa2_20_s1001', 10 ** 5), ('arith16_s1001', 10 ** 3), ('qaoa2_24_s1001', 10 ** 6)):
        path = A6 + name + '.pkl'
        if not os.path.exists(path):
            pytest.skip('reference not built')
        ref = pickle.load(open(path, 'rb'))
        model, qs = ref['model'], ref['qs']
        rng = np.random.default_rng(4)
        counts = [rng.multinomial(N, q / q.sum()).astype(float) for q in qs]
        reg = CPRegion(model, counts, 0.005)
        for e, v in enumerate(counts):                       # reference: the former np.unique construction
            u, inv = np.unique(v, return_inverse=True)
            lo, hi = clopper_pearson(u, float(v.sum()), 0.005 / len(counts) / len(v))
            assert np.array_equal(reg.lo[e], lo[inv]) and np.array_equal(reg.hi[e], hi[inv])


def _support_of(model):
    T = _Terms.__new__(_Terms)
    T.model, T._csr = model, {}
    return T


def test_support_matches_definition():
    # held bits a strict subset of the output field (offsets 1, 3 of 4), 0-2 QPD bits, dropped bits
    rng = np.random.default_rng(5)
    for w in (0, 1, 2):
        subs = [Subexperiment((i,), 4 + w, w, 4) for i in range(2)]
        uses = [[Use(0, [0, 1], [1, 3], 1), Use(1, [2, 3], [0, 2], 0)]]
        model = CutModel(4, [[0, 1], [2, 3]], np.array([1.0]), subs, uses, [0, 1])
        T = _support_of(model)
        for i in range(2):
            rows, _, R = model.row_index(0, i)
            for local in range(R):
                assert np.array_equal(T._support(0, i, local), np.flatnonzero(rows == local))
    path = A6 + 'qaoa2_20_s1001.pkl'
    if os.path.exists(path):
        model = pickle.load(open(path, 'rb'))['model']
        T = _support_of(model)
        for g in range(model.groups):
            for i, use in enumerate(model.uses[g]):
                if not use.held:
                    continue
                rows, _, R = model.row_index(g, i)
                for local in rng.choice(R, 4, replace=False):
                    assert np.array_equal(T._support(g, i, int(local)), np.flatnonzero(rows == local))



def _interval_sparse_ref(reg, e, rows, signs, R):
    """CPRegion.interval_sparse as committed before the scalar cache (reference)."""
    lo, hi = reg.lo[e], reg.hi[e]
    cap = hi - lo
    Rm = 1.0 - lo.sum()
    Ctot = cap.sum()
    v0 = np.bincount(rows, weights=signs * lo, minlength=R)
    Cneg = np.bincount(rows, weights=(signs < 0) * cap, minlength=R)
    Cpos = np.bincount(rows, weights=(signs > 0) * cap, minlength=R)
    C0 = np.maximum(Ctot - Cneg - Cpos, 0.0)
    r1 = np.minimum(Rm, Cneg); rest = Rm - r1; r0 = np.minimum(rest, C0); r2 = np.maximum(rest - r0, 0.0)
    s1 = np.minimum(Rm, Cpos); rest = Rm - s1; s0 = np.minimum(rest, C0); s2 = np.maximum(rest - s0, 0.0)
    return v0 - r1 + r2, v0 + s1 - s2


@pytest.mark.parametrize('name,N', [('arith16_s1001', 10 ** 3), ('qaoa2_24_s1001', 10 ** 6),
                                    ('H3:h3qaoa2_28_s2001', 10 ** 5)])
def test_cp_scalar_cache_and_sign_identity(name, N):
    """CPRegion.interval_sparse with the cached (1 - sum lo, sum cap) equals the former code byte for byte (point and
    mass calls, repeated); every use's signs equal (-1)^dropped s_e over the full Z_e (the sparse path checks one
    period)."""
    from rankcert.certify.rpair import _SignPool
    path = ('./data/h3/ref/' + name[3:] if name.startswith('H3:') else A6 + name) + '.pkl'
    if not os.path.exists(path):
        pytest.skip('reference not built')
    ref = pickle.load(open(path, 'rb'))
    model, qs = ref['model'], ref['qs']
    rng = np.random.default_rng(6)
    counts = [rng.multinomial(N, q / q.sum()).astype(float) for q in qs]
    reg = CPRegion(model, counts, 0.005)
    for g, row in enumerate(model.uses):
        for i, use in enumerate(row):
            rows, signs, R = model.row_index(g, i)
            for rr, RR in ((rows, R), (np.zeros_like(rows), 1), (rows, R)):
                a = reg.interval_sparse(use.e, rr, signs, RR)
                b = _interval_sparse_ref(reg, use.e, rr, signs, RR)
                assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
            sub = model.subexperiments[use.e]
            P = _SignPool(sub, None, None, None)
            z = np.arange(sub.size)
            assert np.array_equal(signs, float((-1.0) ** use.dropped) * P.table[z & P.mask])
