"""Propagation validity: for points q' inside each R-CP region (random vertices of box ∩ simplex), the reconstructed
p(q') lies inside [L, U] for every string. Uses the built Phase-0 instances (skipped if not built)."""
import os
import numpy as np
import pytest

from derisk.d1_run import load, draw_counts, CIRCUIT_ID
from rankcert.regions.regions import CPRegion, HoeffdingRegion
from rankcert.propagate.interval import factor_intervals, dense_bounds, dense_point, to_block_order


def _feasible_point(lo, hi, rng):
    q = lo.copy()
    R = 1.0 - q.sum()
    for z in rng.permutation(len(q)):
        add = min(R, hi[z] - lo[z]) * (1.0 if rng.random() < 0.7 else rng.random())
        q[z] += add
        R -= add
        if R <= 0:
            break
    if R > 1e-15:  # finish filling
        for z in range(len(q)):
            add = min(R, hi[z] - q[z]); q[z] += add; R -= add
    return q


@pytest.mark.parametrize('name', ['ghz20', 'adder16', 'qaoa20'])
def test_points_in_region_inside_bounds(name):
    if not os.path.exists(os.path.join('D:/runs/rankcert/phase0/ref', name + '.pkl')):
        pytest.skip('instance not built')
    ref = load(name)
    model, qs = ref['model'], ref['qs']
    rng = np.random.default_rng(5)
    n = draw_counts(qs, 2000, CIRCUIT_ID[name], 999)
    reg = CPRegion(model, n, 0.05)
    L, U = dense_bounds(model, factor_intervals(model, reg))
    for _ in range(3):
        qp = [_feasible_point(reg.lo[e], reg.hi[e], rng) for e in range(len(qs))]
        pp = dense_point(model, qp)
        assert np.all(L <= pp + 1e-12) and np.all(pp <= U + 1e-12)
    # Hoeffding: the plug-in point is inside its own region
    regh = HoeffdingRegion(model, n, 0.05)
    Lh, Uh = dense_bounds(model, factor_intervals(model, regh))
    ph = dense_point(model, [v / v.sum() for v in n])
    assert np.all(Lh <= ph + 1e-12) and np.all(ph <= Uh + 1e-12)


@pytest.mark.parametrize('name', ['qaoa20', 'adder16', 'ghz20'])
def test_affine_bounds_contain_feasible_points(name):
    if not os.path.exists(os.path.join('D:/runs/rankcert/phase0/ref', name + '.pkl')):
        pytest.skip('instance not built')
    from rankcert.propagate.affine import affine_bounds
    ref = load(name)
    model, qs = ref['model'], ref['qs']
    rng = np.random.default_rng(11)
    n = draw_counts(qs, 5000, CIRCUIT_ID[name], 777)
    reg = CPRegion(model, n, 0.05)
    fi = factor_intervals(model, reg)
    ph = dense_point(model, [v / v.sum() for v in n])
    cand = np.argsort(-ph)[:40]
    L, U = affine_bounds(model, reg, n, cand, fi)
    assert np.all(L <= ph[cand] + 1e-12) and np.all(ph[cand] <= U + 1e-12)
    for _ in range(4):
        qp = [_feasible_point(reg.lo[e], reg.hi[e], rng) for e in range(len(qs))]
        pp = dense_point(model, qp)[cand]
        assert np.all(L <= pp + 1e-12) and np.all(pp <= U + 1e-12)
