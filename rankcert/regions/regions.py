"""initial prototype confidence regions on raw categories and the functional intervals they induce.

Every region method returns, for the counts of all distinct subexperiments, an object with
    interval(e, A) -> (lo, hi)  for a dense functional matrix A (rows = functionals over Z_e),
valid simultaneously for every functional the model asks for, on one event of probability >= 1 - delta.

R-H  : Hoeffding per needed functional. A functional a.q with a in {-1,0,1}^Z is the mean of a bounded variable in
       [-1, 1] (range 2): Pr[|a.qhat - a.q| >= t] <= 2 exp(-N t^2 / 2). Union over the F distinct functionals the model
       uses (counted by `count_functionals`): t = sqrt(2 ln(2F/delta) / N). Interval clipped to the trivial range
       [-sum_{a<0}1, ...] = [-1, 1].
R-CP : exact two-sided Clopper-Pearson interval for every category z of Z_e (a FIXED set, all 2^{num_clbits} categories,
       observed or not) at level delta_e / |Z_e|, delta_e = delta / |S|; region C_e = box intersect simplex.
       min / max of a.q over C_e by the greedy fill (exact LP optimum for box intersect simplex).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import beta as _beta


def functional_set(model) -> dict:
    """Distinct functionals per subexperiment: {e: list of (g, i) uses}, plus the total count F of functional rows
    (point rows 2^{m_i} per distinct (e, held, offsets, dropped) and one mass row each)."""
    seen = {}
    for g, row in enumerate(model.uses):
        for i, use in enumerate(row):
            key = (use.e, tuple(use.held), tuple(use.offsets), use.dropped)
            seen.setdefault(key, (g, i))
    F = 0
    for (e, held, offs, dr) in seen:
        F += (1 << len(held)) + (1 if held else 0)
    return dict(keys=seen, F=F)


class HoeffdingRegion:
    name = 'R-H'

    def __init__(self, model, n: list, delta: float):
        self.n = n
        self.N = [float(v.sum()) for v in n]
        self.qhat = [v / v.sum() for v in n]
        self.F = functional_set(model)['F']
        self.delta = delta
        self.t = [math.sqrt(2.0 * math.log(2.0 * self.F / delta) / N) for N in self.N]

    def interval(self, e: int, A: np.ndarray):
        c = A @ self.qhat[e]
        lo = np.maximum(c - self.t[e], -1.0)
        hi = np.minimum(c + self.t[e], 1.0)
        return lo, hi

    def interval_sparse(self, e: int, rows: np.ndarray, signs: np.ndarray, R: int):
        """Same as interval() for the functional matrix with one nonzero (signs[z]) in row rows[z] of column z."""
        c = np.bincount(rows, weights=signs * self.qhat[e], minlength=R)
        return np.maximum(c - self.t[e], -1.0), np.minimum(c + self.t[e], 1.0)


def clopper_pearson(k: np.ndarray, N: float, alpha: float):
    """Exact two-sided (1 - alpha) Clopper-Pearson interval for k successes in N trials (vectorised)."""
    k = np.asarray(k, dtype=float)
    lo = np.where(k > 0, _beta.ppf(alpha / 2, k, N - k + 1), 0.0)
    if alpha < 1e-9:   # 1 - alpha/2 loses precision (rounds to 1 near 1e-16): use the upper-tail inverse. Never taken
        # at the A6 levels (smallest alpha there 6.1e-9), so those intervals are bit-for-bit unchanged (the derivation).
        hi = np.where(k < N, _beta.isf(alpha / 2, k + 1, N - k), 1.0)
    else:
        hi = np.where(k < N, _beta.ppf(1 - alpha / 2, k + 1, N - k), 1.0)
    return np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)


def unique_inverse(v: np.ndarray):
    """np.unique(v, return_inverse=True) for a count vector. When v holds nonnegative integers (shot counts) of moderate
    size, an O(|v|) bincount path returns the identical result (sorted distinct values in v's dtype, int64 inverse of
    v's shape); anything else goes to np.unique. Identical output, so every downstream number is unchanged (tested)."""
    v = np.asarray(v)
    if v.ndim == 1 and v.size and v.dtype.kind == 'f':
        vmin, vmax = float(v.min()), float(v.max())
        if vmin >= 0 and vmax <= max(2 * v.size, 1 << 24):
            iv = v.astype(np.int64)
            if np.array_equal(iv, v):
                present = np.bincount(iv) > 0
                lookup = np.cumsum(present) - 1
                return np.flatnonzero(present).astype(v.dtype), lookup[iv]
    return np.unique(v, return_inverse=True)


class CPRegion:
    name = 'R-CP'

    def __init__(self, model, n: list, delta: float):
        S = len(n)
        self.delta = delta
        self.lo, self.hi = [], []
        for v in n:
            N = float(v.sum())
            Z = len(v)
            alpha = delta / S / Z
            # unique count values -> one beta.ppf call each (counts repeat a lot, zeros especially)
            u, inv = unique_inverse(v)
            l, h = clopper_pearson(u, N, alpha)
            self.lo.append(l[inv])
            self.hi.append(h[inv])

    def interval(self, e: int, A: np.ndarray):
        lo, hi = self.lo[e], self.hi[e]
        cap = hi - lo
        R = 1.0 - lo.sum()                         # free mass to distribute (>= 0 since lo <= qhat)
        Ctot = cap.sum()
        neg = (A < 0)
        pos = (A > 0)
        v0 = A @ lo
        Cneg = neg.astype(float) @ cap
        Cpos = pos.astype(float) @ cap
        C0 = np.maximum(Ctot - Cneg - Cpos, 0.0)   # capacity; clip floating cancellation
        # minimum: fill a=-1 cells, then a=0, then a=+1
        r1 = np.minimum(R, Cneg)
        rest = R - r1
        r0 = np.minimum(rest, C0)
        r2 = np.maximum(rest - r0, 0.0)
        mn = v0 - r1 + r2
        # maximum: fill a=+1 cells, then a=0, then a=-1
        s1 = np.minimum(R, Cpos)
        rest = R - s1
        s0 = np.minimum(rest, C0)
        s2 = np.maximum(rest - s0, 0.0)
        mx = v0 + s1 - s2
        return mn, mx

    def interval_sparse(self, e: int, rows: np.ndarray, signs: np.ndarray, R: int):
        """Same as interval() for the functional matrix with one nonzero (signs[z]) in row rows[z] of column z."""
        lo, hi = self.lo[e], self.hi[e]
        cap = hi - lo
        c = self.__dict__.setdefault('_sums_cache', {})   # (1 - sum lo, sum cap) per subexperiment: the same
        if e not in c:                                     # expressions, evaluated once (point and mass calls share)
            c[e] = (1.0 - lo.sum(), cap.sum())
        Rm, Ctot = c[e]
        v0 = np.bincount(rows, weights=signs * lo, minlength=R)
        Cneg = np.bincount(rows, weights=(signs < 0) * cap, minlength=R)
        Cpos = np.bincount(rows, weights=(signs > 0) * cap, minlength=R)
        C0 = np.maximum(Ctot - Cneg - Cpos, 0.0)   # capacity; clip floating cancellation
        r1 = np.minimum(Rm, Cneg); rest = Rm - r1; r0 = np.minimum(rest, C0); r2 = np.maximum(rest - r0, 0.0)
        s1 = np.minimum(Rm, Cpos); rest = Rm - s1; s0 = np.minimum(rest, C0); s2 = np.maximum(rest - s0, 0.0)
        return v0 - r1 + r2, v0 + s1 - s2


REGIONS = {'R-H': HoeffdingRegion, 'R-CP': CPRegion}
