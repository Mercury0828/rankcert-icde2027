"""Property tests for the initial prototype (regions, propagation, labels)."""
import numpy as np
import pytest
from scipy.optimize import linprog

from rankcert.regions.regions import CPRegion, clopper_pearson
from rankcert.propagate.interval import imul
from stats.labels import topk_labels, threshold_labels, topk_truth


class _Model:  # minimal stand-in: CPRegion only needs len(n)
    pass


def _random_box(rng, Z):
    q = rng.dirichlet(np.ones(Z) * 0.3)
    n = rng.multinomial(200, q).astype(float)
    return n


@pytest.mark.parametrize('seed', range(25))
def test_cp_functional_is_exact_lp_optimum(seed):
    rng = np.random.default_rng(seed)
    Z = int(rng.integers(2, 9))
    n = [_random_box(rng, Z) for _ in range(2)]
    reg = CPRegion(_Model(), n, 0.05)
    A = rng.choice([-1.0, 0.0, 1.0], size=(4, Z))
    mn, mx = reg.interval(0, A)
    lo, hi = reg.lo[0], reg.hi[0]
    for r in range(A.shape[0]):
        res_min = linprog(A[r], A_eq=np.ones((1, Z)), b_eq=[1.0], bounds=list(zip(lo, hi)), method='highs')
        res_max = linprog(-A[r], A_eq=np.ones((1, Z)), b_eq=[1.0], bounds=list(zip(lo, hi)), method='highs')
        assert abs(res_min.fun - mn[r]) < 1e-9
        assert abs(-res_max.fun - mx[r]) < 1e-9


@pytest.mark.parametrize('seed', range(10))
def test_imul_contains_products(seed):
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(-1, 1, size=(2, 50)), axis=0)
    b = np.sort(rng.uniform(-1, 1, size=(2, 50)), axis=0)
    lo, hi = imul(a[0], a[1], b[0], b[1])
    for _ in range(200):
        x = rng.uniform(a[0], a[1])
        y = rng.uniform(b[0], b[1])
        assert np.all(lo <= x * y + 1e-15) and np.all(x * y <= hi + 1e-15)


def _brute_topk(L, U, k):
    n = len(L)
    lab = np.zeros(n, dtype=np.int8)
    for x in range(n):
        if sum(1 for y in range(n) if y != x and U[y] >= L[x]) <= k - 1:
            lab[x] = 1
        if sum(1 for y in range(n) if L[y] > U[x]) >= k:
            lab[x] = -1
    return lab


@pytest.mark.parametrize('seed', range(40))
def test_topk_labels_match_bruteforce_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 30))
    c = np.round(rng.uniform(0, 1, n), 1)           # many exact ties
    w = np.round(rng.uniform(0, 0.3, n), 1)
    L, U = c - w, c + w
    for k in (1, 2, 3, 5):
        assert np.array_equal(topk_labels(L, U, k), _brute_topk(L, U, k))


def test_topk_truth_ties():
    p = np.array([0.4, 0.2, 0.2, 0.1, 0.1, 0.0])
    t = topk_truth(p, 2)      # 0.4 sure member; the 0.2 pair straddles ranks 2/3 -> neither; rest sure non-members
    assert list(t) == [1, 0, 0, -1, -1, -1]
    t = topk_truth(p, 3)      # pair wholly inside top-3 -> members
    assert list(t) == [1, 1, 1, -1, -1, -1]


def test_clopper_pearson_zero_and_full():
    lo, hi = clopper_pearson(np.array([0.0, 10.0]), 10.0, 0.05)
    assert lo[0] == 0.0 and abs(hi[0] - (1 - 0.025 ** 0.1)) < 1e-12
    assert hi[1] == 1.0 and abs(lo[1] - 0.025 ** 0.1) < 1e-12


def test_search_tie_and_bound_regressions():
    """review counterexamples: exact ties must survive, and no branch is pruned by rounding."""
    from types import SimpleNamespace
    from rankcert.search.bnb import Searcher, expand_labels
    from rankcert.propagate.interval import dense_bounds
    for s_val, f0, f1, k in ((0.4, [0.4, 0.6], [0.4, 0.6], 2), (0.3, [0.3, 0.7], [0.4, 0.6], 2), (0.3, [0.3, 0.7], [0.4, 0.6], 1)):
        model = SimpleNamespace(groups=1, blocks=[[0], [1]], m=2)
        u0, u1 = np.array(f0), np.array(f1)
        fi = dict(s=(np.array([s_val]), np.array([s_val])), u=[[(u0, u0), (u1, u1)]], S=[[(1.0, 1.0), (1.0, 1.0)]])
        L, U = dense_bounds(model, fi)
        S = Searcher(model, fi)
        assert np.array_equal(expand_labels(4, S.topk(k)), topk_labels(L, U, k))
        for tau in np.unique(np.concatenate([L, U])):
            assert np.array_equal(expand_labels(4, S.threshold(float(tau))), threshold_labels(L, U, float(tau)))


def test_bl2_scalar_only_derivative():
    """review counterexample: a scalar-only subexperiment must contribute to the delta-method variance."""
    from rankcert.recon.model import CutModel, Subexperiment, Use
    from baselines.bl23 import bl2_bounds
    held = Subexperiment(key=None, num_clbits=2, written=1, out_width=1)
    scal = Subexperiment(key=None, num_clbits=1, written=1, out_width=0)
    model = CutModel(m=1, blocks=[[0]], coefficients=np.array([1.0]), subexperiments=[held, scal],
                     uses=[[Use(e=0, held=[0], offsets=[0]), Use(e=1, held=[], offsets=[])]], block_of_sub=[0, -1])
    n = [np.array([125.0, 375.0, 375.0, 125.0]), np.array([500.0, 500.0])]
    phat = np.array([0.5, 0.5])
    cand, L, U = bl2_bounds(model, n, 0.05, phat, n_cand=2)
    assert np.all(U - L > 0.03), (L, U)          # non-degenerate (expected ~ [0.482, 0.518])
