"""The sparse functional path equals the dense path exactly (regions on the real Phase-0 instances)."""
import os
import numpy as np
import pytest
from derisk.d1_run import load, draw_counts, CIRCUIT_ID
from rankcert.regions.regions import REGIONS


@pytest.mark.parametrize('name', ['ghz20', 'adder16', 'qaoa20'])
def test_sparse_equals_dense(name):
    if not os.path.exists(os.path.join('D:/runs/rankcert/phase0/ref', name + '.pkl')):
        pytest.skip('instance not built')
    ref = load(name); model, qs = ref['model'], ref['qs']
    n = draw_counts(qs, 3000, CIRCUIT_ID[name], 4242)
    for meth, cls in REGIONS.items():
        reg = cls(model, n, 0.05)
        for g in range(model.groups):
            for i, use in enumerate(model.uses[g]):
                A = model.functional_matrix(g, i)
                rows, signs, R = model.row_index(g, i)
                a = reg.interval(use.e, A); b = reg.interval_sparse(use.e, rows, signs, R)
                assert np.allclose(a[0], b[0], atol=1e-12, rtol=0) and np.allclose(a[1], b[1], atol=1e-12, rtol=0)
                if use.held:
                    a = reg.interval(use.e, A.sum(axis=0, keepdims=True))
                    b = reg.interval_sparse(use.e, np.zeros_like(rows), signs, 1)
                    assert np.allclose(a[0], b[0], atol=1e-12, rtol=0) and np.allclose(a[1], b[1], atol=1e-12, rtol=0)


@pytest.mark.parametrize('name', ['ghz20', 'adder16', 'qaoa20'])
def test_row_index_matches_dense_matrix(name):
    if not os.path.exists(os.path.join('D:/runs/rankcert/phase0/ref', name + '.pkl')):
        pytest.skip('instance not built')
    model = load(name)['model']
    for g in range(model.groups):
        for i in range(len(model.uses[g])):
            A = model.functional_matrix(g, i)
            rows, signs, R = model.row_index(g, i)
            B = np.zeros_like(A)
            B[rows, np.arange(A.shape[1])] = signs
            assert R == A.shape[0] and np.array_equal(A, B)
