"""R-PAIR certification usable with or without the dense 2^m vector.

certify(model, pilot, counts, ks, mode) -> per query k a label answer {cand, cand_labels, rest}: strings in `cand` carry
the listed labels, every other string carries the label `rest` (OUT, or UND after the empty-interval fallback).
mode = 'dense' computes everything on the 2^m vector (reference); mode = 'search' never materialises it: the pilot
top-|A| comes from the bit-level search on point factors, L0_(k) from BitSearcher.topk, and only the strings with
U0 >= T_k = min(L0_(k), min_A L, min_A U) are collected (T3 / R4 eq. (21)–(22): every other string is OUT with the k
strings of largest L0 as witnesses). Both modes use the same arithmetic, so their labels are identical (tested).
Pilot ranking ties are broken by string index (block order), as in derisk/repair3_run.py.
mode = 'auto' runs the search with a frontier cap (AUTO_CAP) and falls back to the dense arithmetic when the collection
would exceed it (wide regions, where nearly every string is collected); labels are identical in all modes.

Sparse functional path (certify(sparse=None): auto, on iff max_e |Z_e| > SPARSE_AUTO; H3 widths m = 32-40 where |Z_e| is
up to 2^22). The dense path keeps one length-|Z_e| vector per factor functional (memory ~ |A| x groups x |Z_e|, and every
LP sorts |Z_e| cells). The sparse path stores a point functional by its support (<= 2^{QPD bits} categories) and a mass /
scalar functional as sigma * s_e (s_e = Subexperiment.signs(0), sigma = (-1)^dropped; asserted). Every vector the bounds
build (W_e, D_e, G_e and the factor-variance coefficients) is then "values on a small support U + a coefficient that
depends only on s_z off U", so each LP / second moment / oscillation is computed on U plus at most two pooled cells (one
per sign of s_z off U). Pooling is exact: cells with equal coefficient g enter the objective and the simplex constraint
only through their total mass Q, which ranges exactly over [sum lo, sum hi] (any such Q splits back into the cells), so
the LP optimum (and hi @ g, max, min) is unchanged. The dense path is unchanged bit for bit. The sparse route batches the
factor-radius LPs of one subexperiment (_tighten_points / _lpmax_rows: same greedy, different summation order) and, at
H3 widths, checks the full-functional sign identity on one period of the QPD bits (Subexperiment.signs is periodic in
z mod 2^written by construction; the full-length identity is asserted in tests/test_h3_speed.py).

Sparse vs dense rounding. The two paths compute the same mathematical bounds with different floating
summation orders (short support dots and pooled sums vs dots, sums and cumulative sums over all |Z_e| cells), so a decision
taken at an exact tie can differ. The sparse path therefore rounds outward: next to every score radius E, pair radius and
factor radius it carries a bound Delta >= |sparse value - dense value| and decides with E + Delta (+ the score bound
dp >= |p_sparse - p_dense|), and it declares the empty-interval fallback whenever the dense path could. Delta follows the
standard model (|fl(sum of n terms) - sum| <= gamma_n * sum |terms|, gamma_n = n u / (1 - n u), u = machine epsilon used
conservatively) with N = max_e |Z_e| + 64 for every long sum of either path:
  point factor value v (support dot vs |Z_e| dot): |dv| <= 2 gamma_N sum_z |a_z| qhat_z; mass / scalar values and CP / mass
  radii are computed by the same expression in both paths (dv = 0); factor radius r = max(v - lo, hi - v): dr <= dv + 2u(...);
  r6 factor radius min(r, rb): dr <= max(dr_CP, d rb), d rb from the variance LP below;
  products / sums of a term: first-order perturbation of the absolute-value polynomial (prod(|v| + dv) - prod |v|), plus
  rounding gamma_k of each path (k <= 4 n + 8, n factors);
  LP over box and simplex (_lpmax): 1-Lipschitz in max |dg| (total mass 1); its own rounding in either path is
  <= 16 gamma_N (LP + max g (1 + sum lo + sum hi)) (the dots lo @ g, take @ g are <= LP for g >= 0; the cumulative sum
  matters only at the greedy boundary, where the prefix is <= 1; pooled sums of lo / hi move a cell bound by
  <= gamma_N sum hi, which moves the LP by <= max g times that); hi @ g (variant r4) likewise;
  sqrt: |sqrt(x') - sqrt(x)| <= min(sqrt(eps), eps / sqrt(x)) for |x' - x| <= eps;
  max, min, max(., 0): 1-Lipschitz; the remainder term is a polynomial with nonnegative coefficients in (|v| + r, r).
Second-order terms are O(gamma_N^2) <= 1e-18 relative and are covered by the factor-2 / factor-16 slack. Because every
label rule is monotone (wider intervals and fewer certificates never add an IN or OUT, and a wider empty test only moves
to UND everywhere), the sparse labels are the dense labels except decisions within the outward-rounding margin, which are
dropped (UND): sparse decisions are a subset of dense decisions.
"""
from __future__ import annotations

import math

import numpy as np

from rankcert.propagate.interval import dense_bounds, factor_intervals
from rankcert.regions.regions import CPRegion
from rankcert.search.bitbnb import BitSearcher, FrontierCap

D0, DS, DP = 0.005, 0.005, 0.04
NA = 32
# Repair 6: pilot-selected set of NA6 strings; delta split region / scores / pairs /
# factor functionals; factor radii min(CP, functional Bernstein); variance envelope = LP over the region; bilinear remainder
# of two-factor products bounded by Bernstein conditional on the other side's samples (min with the deterministic bound).
NA6 = 128
D0_6, DS_6, DP_6, DR_6 = 0.005, 0.02, 0.015, 0.01
VARIANTS = ('r4', 'r6')
EPS = float(np.finfo(float).eps)          # used as the unit roundoff (conservative: u = eps / 2)


def _gamma(n):
    return n * EPS / (1.0 - n * EPS)


def _sqrt_mod(x, eps_):
    """sup |sqrt(x') - sqrt(x)| over x' >= 0 with |x' - x| <= eps_ (x >= 0)."""
    if eps_ <= 0:
        return 0.0
    s = float(np.sqrt(eps_))
    return min(s, float(eps_ / np.sqrt(x))) if x > 0 else s


def _rem_poly(A, r):
    """sum over subsets S, |S| >= 2, of prod_{i in S} r_i prod_{j not in S} A_j (= prod(A + r) - prod A - sum_i r_i
    prod_{j != i} A_j without cancellation; nonnegative coefficients, monotone in A, r >= 0)."""
    n = len(A)
    tot = 0.0
    for mask in range(1 << n):
        if bin(mask).count('1') < 2:
            continue
        p = 1.0
        for i in range(n):
            p *= r[i] if (mask >> i) & 1 else A[i]
        tot += p
    return tot


def _lpmax(lo, hi, g):
    """max_q sum_z q_z g_z over the box [lo, hi] intersected with the probability simplex (greedy, g descending)."""
    assert lo.sum() <= 1.0 + 1e-9 <= hi.sum() + 2e-9, 'empty region: sum lo > 1 or sum hi < 1'
    o = np.argsort(-g, kind='stable')
    cap = (hi - lo)[o]
    R = max(1.0 - float(lo.sum()), 0.0)
    take = np.minimum(cap, np.maximum(R - np.concatenate([[0.0], np.cumsum(cap)[:-1]]), 0.0))
    return float(lo @ g + take @ g[o])
def _lpmax_rows(lo, hi, g):
    """_lpmax for every row of (n, c) arrays: the same greedy, vectorised (sparse route; its rounding differs from the
    scalar call only in summation order, which the outward-rounding bound covers: module docstring)."""
    sl, sh = lo.sum(axis=1), hi.sum(axis=1)
    assert np.all(sl <= 1.0 + 1e-9) and np.all(1.0 + 1e-9 <= sh + 2e-9), 'empty region: sum lo > 1 or sum hi < 1'
    o = np.argsort(-g, axis=1, kind='stable')
    cap = np.take_along_axis(hi - lo, o, axis=1)
    R = np.maximum(1.0 - sl, 0.0)
    prev = np.concatenate([np.zeros((cap.shape[0], 1)), np.cumsum(cap, axis=1)[:, :-1]], axis=1)
    take = np.minimum(cap, np.maximum(R[:, None] - prev, 0.0))
    return (lo * g).sum(axis=1) + (take * np.take_along_axis(g, o, axis=1)).sum(axis=1)


AUTO_CAP = 1 << 18        # mode 'auto': search frontier cap before the dense fallback (labels identical either way)
SPARSE_AUTO = 1 << 14     # certify(sparse=None): sparse functional path iff max_e |Z_e| > SPARSE_AUTO (A6 max 2^14 stays dense)


class _SF:
    """Sparse factor functional over Z_e: point functional (values `vals` on sorted support `idx`, 0 elsewhere; sigma
    None) or full functional sigma * s_e (idx None)."""
    __slots__ = ('idx', 'vals', 'sigma', 'vabs', 'dv', 'dr')

    def __init__(self, idx=None, vals=None, sigma=None):
        self.idx, self.vals, self.sigma = idx, vals, sigma
        self.vabs = self.dv = self.dr = 0.0      # sum |a| qhat; bounds on |v_sparse - v_dense|, |r_sparse - r_dense|

    def ident(self, e):
        """Identity of the dense vector a (for the distinct-functional count of tighten_factors)."""
        if self.sigma is None:
            return (e, 0, self.idx.tobytes(), self.vals.tobytes())
        return (e, 1, float(self.sigma))


class _SignPool:
    """Per subexperiment: s_e(z) = (-1)^{#zero bits among the `written` QPD bits of z} (= Subexperiment.signs(0)), which
    depends only on z mod 2^written, and the pooled sums of lo / hi / qhat over the cells of each sign."""

    def __init__(self, sub, lo, hi, q):
        w = sub.written
        self.mask = (1 << w) - 1
        j = np.arange(1 << w)
        ones = np.zeros(1 << w, dtype=np.int64)
        for b in range(w):
            ones += (j >> b) & 1
        self.table = (-1.0) ** (w - ones)
        pos = self.table > 0
        rows = sub.size >> w

        def pool(x):
            if x is None:
                return None
            c = np.asarray(x, dtype=float).reshape(-1, 1 << w).sum(axis=0)
            return float(c[pos].sum()), float(c[~pos].sum())
        self.lo, self.hi, self.q = pool(lo), pool(hi), pool(q)
        self.n = (rows * int(pos.sum()), rows * int((~pos).sum()))

    def s(self, idx):
        return self.table[np.asarray(idx, dtype=np.int64) & self.mask]


class PointRegion:
    """Zero-width 'region' at given distributions: factor intervals collapse to the plug-in values."""
    def __init__(self, q):
        self.q = q

    def interval_sparse(self, e, rows, signs, R):
        c = np.bincount(rows, weights=signs * self.q[e], minlength=R)
        return c, c.copy()


def check_structure(model):
    """T2 needs each product to use a given subexperiment at most once (true by QCut's construction: one use per
    subcircuit per group, deduplicated by (subcircuit, fingerprint))."""
    return all(len({u.e for u in row}) == len(row) for row in model.uses)


def select_pilot(model, pilot, mode, na=NA):
    na = min(na, 1 << model.m)                     # tiny output spaces
    qp = [v / v.sum() for v in pilot]
    fi = factor_intervals(model, PointRegion(qp))
    if mode == 'dense':
        s, _ = dense_bounds(model, fi)
        return np.argsort(-s, kind='stable')[:na].astype(np.int64)
    res = BitSearcher(model, fi).topk(na)
    idx, val = res['cand'], res['cand_L']
    order = np.lexsort((idx, -val))
    return idx[order][:na].astype(np.int64)


class _Terms:
    def __init__(self, model, fi, qhat, reg, ns, variant='r4', sparse=False):
        """sparse: False (default here: external callers such as rankcert.certify.seq read the dense vector `a` of a
        feature), True, or None = auto (max_e |Z_e| > SPARSE_AUTO). certify() defaults to None."""
        self.model, self.fi, self.qhat, self.reg, self.ns = model, fi, qhat, reg, ns
        self.M = 1 << model.m
        self.cache = {}
        self.r6 = variant == 'r6'
        self.frozen = False
        if sparse is None:
            sparse = max(s.size for s in model.subexperiments) > SPARSE_AUTO
        self.sparse = bool(sparse)
        self._pools, self._csr, self._full_ok = {}, {}, set()
        self._gN = _gamma(max(s.size for s in model.subexperiments) + 64)   # longest sum of either path

    def point_bound(self, x):
        """(p(x), dp) with dp >= |p_sparse - p_dense| (0 on the dense path)."""
        p = self.point(x)
        if not self.sparse:
            return p, 0.0
        dp, mag, nt = 0.0, 1.0 / self.M, 0
        for c, fs in self.terms(x, True):
            av = np.array([abs(f[2]) for f in fs])
            up = av + np.array([f[1].dv for f in fs])
            dp += abs(c) * (float(np.prod(up)) - float(np.prod(av)))
            mag += abs(c) * float(np.prod(up))
            nt += 1
        return p, dp + 2 * _gamma(2 * nt + 16) * mag

    def radius_bound(self, terms_, t):
        """(E, dE) with dE >= |E_sparse - E_dense| (0 on the dense path)."""
        if not self.sparse:
            return self.radius(terms_, t), 0.0
        return self._radius_sparse_b(terms_, t)

    def tighten_factors(self, selected, delta_r):
        """r6: every factor functional used by the selected strings (a pilot-determined finite set, counted after all
        are built) gets radius min(CP radius, Bernstein radius at union level delta_r); no new functional may appear
        afterwards (pair constraints use a subset of them). The union counts distinct functionals (e, a), not cache keys
        (several (g, i, x) keys share one functional; review)."""
        if self.sparse:
            return self._tighten_sparse(selected, delta_r)
        for x in selected:
            self.terms(int(x), True)
        distinct = {(e, a.tobytes()) for (e, a, v, r) in self.cache.values()}
        tr = np.log(2 * len(distinct) / delta_r)
        for key, (e, a, v, r) in list(self.cache.items()):
            mu = float(a @ self.qhat[e])
            var = _lpmax(self.reg.lo[e], self.reg.hi[e], (a - mu) ** 2)
            rb = np.sqrt(2 * var * tr / self.ns[e]) + (float(a.max()) - float(a.min())) * tr / (3 * self.ns[e])
            self.cache[key] = (e, a, v, min(r, rb))
        self.frozen = True

    def feature(self, g, i, x):
        """Factor i of group g: (e, coefficient vector a over Z_e, plug-in value, radius r over the CP region).
        Held subcircuit: point functional u_{i,g}(x_i) (x given) or its mass S_{i,g} (x None);
        no held bits: the scalar E_{i,g} in both cases."""
        key = (g, i, x if self.model.uses[g][i].held else None)
        if key in self.cache:
            return self.cache[key]
        assert not self.frozen, 'factor functional outside the pilot-fixed union (r6)'
        if self.sparse:
            return self._feature_sparse(g, i, x, key)
        model, fi = self.model, self.fi
        use = model.uses[g][i]
        rows, signs, R = model.row_index(g, i)
        if not use.held:
            a = signs
            lo, hi = self.reg.interval_sparse(use.e, np.zeros_like(rows), signs, 1)
            lo, hi = float(lo[0]), float(hi[0])
        elif x is None:
            a = signs
            lo, hi = fi['S'][g][model.block_of_sub[i]]
        else:
            b = model.block_of_sub[i]
            shift = sum(len(z) for z in model.blocks[:b])
            local = (x >> shift) & (R - 1)
            a = signs * (rows == local)
            flo, fhi = fi['u'][g][b]
            lo, hi = flo[local], fhi[local]
        v = float(a @ self.qhat[use.e])
        r = max(v - lo, hi - v, 0.0)
        self.cache[key] = (use.e, a, v, r)
        return self.cache[key]

    def terms(self, x, mass):
        """p(x) = 1/M + sum_g [c_g/M prod_i (mass factors) - c_g prod_i (point factors)] (model docstring)."""
        out = []
        for g, c in enumerate(self.model.coefficients):
            I = range(len(self.model.uses[g]))
            out.append((-c, [self.feature(g, i, x) for i in I]))
            if mass:
                out.append((c / self.M, [self.feature(g, i, None) for i in I]))
        return out

    def point(self, x):
        return 1.0 / self.M + sum(c * float(np.prod([f[2] for f in fs])) for c, fs in self.terms(x, True))

    def radius(self, terms_, t):
        """R4 / T2 eq. (10)-(13) for products of any number of factors."""
        if self.sparse:
            return self._radius_sparse(terms_, t)
        S = len(self.qhat)
        W = [np.zeros(len(q)) for q in self.qhat]
        D = [np.zeros(len(q)) for q in self.qhat]
        rem = 0.
        for c, fs in terms_:
            vh = np.array([f[2] for f in fs])
            r = np.array([f[3] for f in fs])
            A = np.abs(vh) + r
            n = len(fs)
            rem += abs(c) * (np.prod(A + r) - np.prod(A) - sum(r[i] * np.prod(np.delete(A, i)) for i in range(n)))
            for i, (e, a, v, ri) in enumerate(fs):
                W[e] += c * a * np.prod(np.delete(vh, i))
                D[e] += abs(c) * np.abs(a) * (np.prod(np.delete(A, i)) - np.prod(np.abs(np.delete(vh, i))))
        V, bmax = 0., 0.
        for e in range(S):
            mu = float(W[e] @ self.qhat[e])
            if self.r6:                        # sup over the region (box and simplex) of the second moments
                H = np.sqrt(_lpmax(self.reg.lo[e], self.reg.hi[e], (W[e] - mu) ** 2))
                J = np.sqrt(_lpmax(self.reg.lo[e], self.reg.hi[e], D[e] ** 2))
            else:
                H = np.sqrt(self.reg.hi[e] @ ((W[e] - mu) ** 2))
                J = np.sqrt(self.reg.hi[e] @ (D[e] ** 2))
            V += (H + J) ** 2 / self.ns[e]
            osc = np.max(W[e] + D[e]) - np.min(W[e] - D[e])
            bmax = max(bmax, osc / self.ns[e])
        rem = max(rem, 0.0)
        if self.r6 and all(len(fs) == 2 for _, fs in terms_):
            rem = min(rem, self.bilinear(terms_, t))
        return np.sqrt(2 * V * t) + bmax * t / 3 + rem

    def bilinear(self, terms_, t, s=0):
        """r6: the remainder of two-factor products is sum c d_0 d_1 (d_i = a_i . (qhat_e - q_e) of factor i). Conditional
        on the samples of the side-(1-s) subexperiments it is a sum of independent centred shot terms of the side-s
        subexperiments with coefficient w_e(z) = sum c a_s(z) d_{1-s}, |w_e| <= G_e = sum |c| |a_s| r_{1-s}; Bernstein with
        E w^2 <= sup over the region of sum q G_e^2 and |X_j| <= 2 max G_e / N_e (holds with prob. >= 1 - 2 e^{-t})."""
        if self.sparse:
            return self._bilinear_sparse(terms_, t, s)
        side_s = {fs[s][0] for _, fs in terms_}
        side_o = {fs[1 - s][0] for _, fs in terms_}
        assert not (side_s & side_o), 'bilinear bound needs disjoint subexperiments on the two sides'
        G = {}
        for c, fs in terms_:
            e, a = fs[s][0], fs[s][1]
            G[e] = G.get(e, 0.0) + abs(c) * np.abs(a) * fs[1 - s][3]
        V2, b2 = 0.0, 0.0
        for e, g in G.items():
            V2 += _lpmax(self.reg.lo[e], self.reg.hi[e], g ** 2) / self.ns[e]
            b2 = max(b2, 2 * float(g.max()) / self.ns[e])
        return np.sqrt(2 * V2 * t) + b2 * t / 3

    # ---------------------------------------------------------------- sparse functional path (module docstring)
    def _pool(self, e):
        if e not in self._pools:
            lo = getattr(self.reg, 'lo', None)
            hi = getattr(self.reg, 'hi', None)
            self._pools[e] = _SignPool(self.model.subexperiments[e], None if lo is None else lo[e],
                                       None if hi is None else hi[e], self.qhat[e])
        return self._pools[e]

    def _support(self, g, i, local):
        """Categories z of Z_e whose held output bits equal `local` (sorted). The row of z depends only on its output
        bits z >> written, so the support is a union of aligned runs of 2^written categories."""
        use = self.model.uses[g][i]
        sub = self.model.subexperiments[use.e]
        k = (use.e, tuple(use.held), tuple(use.offsets))
        if k not in self._csr:
            # model.row_index: row(z) has bit t = bit (W - 1 - offsets[t]) of the output integer z >> written. So the
            # outputs with row = local are base(local) + every combination of the remaining output bits (ascending), and
            # no sort or index over |Z_e| is needed (H3 profile: the former argsort-based index cost ~3.5 s and 0.8 GB at
            # m = 40).
            W, w = sub.out_width, sub.written
            pos = [W - 1 - off for off in use.offsets]
            free = np.zeros(1, dtype=np.int64)
            for b in range(W):                     # ascending bits -> ascending combinations
                if b not in pos:
                    free = np.concatenate([free, free + (1 << b)])
            self._csr[k] = (pos, free, w, sub.size >> len(pos))
        pos, free, w, size = self._csr[k]
        base = 0
        for t, b in enumerate(pos):
            base |= ((local >> t) & 1) << b
        idx = (((base + free)[:, None] << w) | np.arange(1 << w, dtype=np.int64)[None, :]).ravel()
        rows = self.model.row_index(g, i)[0]
        assert len(idx) == size and np.all(rows[idx] == local)   # O(support) check of the definition
        return idx

    def _feature_sparse(self, g, i, x, key):
        model, fi = self.model, self.fi
        use = model.uses[g][i]
        rows, signs, R = model.row_index(g, i)
        e = use.e
        if not use.held or x is None:
            sigma = float((-1.0) ** use.dropped)
            if (e, use.dropped) not in self._full_ok:      # all full functionals of e are +-s_e (one period: the
                P = self._pool(e)                          # signs depend only on z mod 2^written by construction)
                k = len(P.table)
                assert np.array_equal(signs[:k], sigma * P.table), 'signs != (-1)^dropped s_e'
                self._full_ok.add((e, use.dropped))
            a = _SF(sigma=sigma)
            v = float(signs @ self.qhat[e])                # same expression as the dense path: dv = dr = 0
            a.vabs = 1.0
            if not use.held:
                lo, hi = self.reg.interval_sparse(e, np.zeros_like(rows), signs, 1)
                lo, hi = float(lo[0]), float(hi[0])
            else:
                lo, hi = fi['S'][g][model.block_of_sub[i]]
        else:
            b = model.block_of_sub[i]
            shift = sum(len(z) for z in model.blocks[:b])
            local = (x >> shift) & (R - 1)
            idx = self._support(g, i, int(local))
            a = _SF(idx=idx, vals=signs[idx])
            v = float(a.vals @ self.qhat[e][idx])
            flo, fhi = fi['u'][g][b]
            lo, hi = flo[local], fhi[local]
            a.vabs = float(self.qhat[e][idx].sum())
            a.dv = 2 * self._gN * a.vabs + 2 * EPS * abs(v)
            a.dr = a.dv + 2 * EPS * (abs(v) + abs(float(lo)) + abs(float(hi)))
        r = max(v - lo, hi - v, 0.0)
        self.cache[key] = (e, a, v, r)
        return self.cache[key]

    def _off(self, e, idx):
        """Pooled off-support cells of e by sign of s_z: list of (lo, hi, sign) with at least one cell."""
        P = self._pool(e)
        lo_e, hi_e = self.reg.lo[e], self.reg.hi[e]
        s = P.s(idx)
        out = []
        for k, sg in ((0, 1.0), (1, -1.0)):
            sel = (s > 0) if sg > 0 else (s < 0)
            if P.n[k] - int(sel.sum()) <= 0:
                continue
            ids = idx[sel]
            lo = max(P.lo[k] - float(lo_e[ids].sum()), 0.0)
            hi = max(P.hi[k] - float(hi_e[ids].sum()), lo)
            out.append((lo, hi, sg))
        return out

    def _cells(self, e, idx, g_sup, g_off_fn):
        """(lo, hi, g) over the support cells and the pooled off-support cells (g_off_fn(sign) -> coefficient)."""
        off = self._off(e, idx)
        lo = np.concatenate([self.reg.lo[e][idx], [c[0] for c in off]])
        hi = np.concatenate([self.reg.hi[e][idx], [c[1] for c in off]])
        g = np.concatenate([g_sup, [g_off_fn(c[2]) for c in off]])
        return lo, hi, g, off

    def _tighten_sparse(self, selected, delta_r):
        for x in selected:
            self.terms(int(x), True)
        distinct = {a.ident(e) for (e, a, v, r) in self.cache.values()}
        tr = np.log(2 * len(distinct) / delta_r)
        empty = np.zeros(0, dtype=np.int64)
        groups = {}
        for key, (e, a, v, r) in self.cache.items():
            if a.sigma is None:
                groups.setdefault((e, len(a.idx)), []).append(key)
        for (e, k), keys in groups.items():                 # point functionals: one batched LP per (e, |support|)
            self._tighten_points(e, keys, tr)
        for key, (e, a, v, r) in list(self.cache.items()):
            if a.sigma is None:
                continue
            mu = v                                          # = a @ qhat[e] (feature value)
            if a.sigma is None:
                lo, hi, g, off = self._cells(e, a.idx, (a.vals - mu) ** 2, lambda sg: mu ** 2)
                amax = max(float(a.vals.max()), 0.0) if off else float(a.vals.max())
                amin = min(float(a.vals.min()), 0.0) if off else float(a.vals.min())
            else:
                sig = a.sigma
                lo, hi, g, off = self._cells(e, empty, np.zeros(0), lambda sg: (sig * sg - mu) ** 2)
                amax = max(sig * c[2] for c in off)
                amin = min(sig * c[2] for c in off)
            var = _lpmax(lo, hi, g)
            rb = np.sqrt(2 * var * tr / self.ns[e]) + (amax - amin) * tr / (3 * self.ns[e])
            # dense mu = a @ qhat = its v: |dmu| <= dv; |a| <= 1 -> |dg| <= 2 (1 + |mu|) dv + dv^2
            dg = 2 * (1.0 + abs(mu)) * a.dv + a.dv ** 2 + 4 * EPS * float(g.max())
            eps_ = self._lp_eps(var, g, dg, lo, hi)
            drb = float(np.sqrt(2 * tr / self.ns[e])) * (_sqrt_mod(var, eps_) + 2 * EPS * float(np.sqrt(var))) \
                + 8 * EPS * float(rb)
            a.dr = max(a.dr, drb)
            self.cache[key] = (e, a, v, min(r, rb))
        self.frozen = True

    def _tighten_points(self, e, keys, tr):
        """Batched factor radii of the point functionals `keys` of subexperiment e (same support size): cells = the
        support plus one pooled cell per sign of s_z off the support (a pooled cell with no category is a zero-width,
        zero-coefficient pad, which changes neither the LP nor its error bound). Same mathematics as the scalar path."""
        ents = [self.cache[k] for k in keys]
        IDX = np.stack([a.idx for (_, a, _, _) in ents])
        VALS = np.stack([a.vals for (_, a, _, _) in ents])
        mu = np.array([v for (_, _, v, _) in ents])
        dv = np.array([a.dv for (_, a, _, _) in ents])
        r_cp = np.array([r for (_, _, _, r) in ents])
        P = self._pool(e)
        lo_e, hi_e = self.reg.lo[e], self.reg.hi[e]
        LO, HI, SU = lo_e[IDX], hi_e[IDX], P.s(IDX)
        lo_c, hi_c, g_c, has_off = [LO], [HI], [(VALS - mu[:, None]) ** 2], np.zeros(len(keys), bool)
        for kk, sg in ((0, 1.0), (1, -1.0)):
            sel = SU == sg
            cnt = P.n[kk] - sel.sum(axis=1)
            present = cnt > 0
            lo_o = np.maximum(P.lo[kk] - (LO * sel).sum(axis=1), 0.0)
            hi_o = np.maximum(P.hi[kk] - (HI * sel).sum(axis=1), lo_o)
            lo_c.append(np.where(present, lo_o, 0.0)[:, None])
            hi_c.append(np.where(present, hi_o, 0.0)[:, None])
            g_c.append(np.where(present, mu ** 2, 0.0)[:, None])
            has_off |= present
        lo, hi, g = np.concatenate(lo_c, axis=1), np.concatenate(hi_c, axis=1), np.concatenate(g_c, axis=1)
        var = _lpmax_rows(lo, hi, g)
        amax = np.where(has_off, np.maximum(VALS.max(axis=1), 0.0), VALS.max(axis=1))
        amin = np.where(has_off, np.minimum(VALS.min(axis=1), 0.0), VALS.min(axis=1))
        ns = self.ns[e]
        rb = np.sqrt(2 * var * tr / ns) + (amax - amin) * tr / (3 * ns)
        gm = g.max(axis=1)
        dg = 2 * (1.0 + np.abs(mu)) * dv + dv ** 2 + 4 * EPS * gm
        eps_ = dg + 16 * self._gN * (np.abs(var) + gm * (1.0 + lo.sum(axis=1) + hi.sum(axis=1)))
        sq_e = np.sqrt(np.maximum(eps_, 0.0))
        sqv = np.sqrt(np.maximum(var, 0.0))
        smod = np.where(var > 0, np.minimum(sq_e, eps_ / np.where(var > 0, sqv, 1.0)), sq_e)
        smod = np.where(eps_ > 0, smod, 0.0)
        drb = float(np.sqrt(2 * tr / ns)) * (smod + 2 * EPS * sqv) + 8 * EPS * rb
        for j, key in enumerate(keys):
            e_, a, v, r = ents[j]
            a.dr = max(a.dr, float(drb[j]))
            self.cache[key] = (e_, a, v, min(r, float(rb[j])))

    def _lp_eps(self, LP, g, dg, lo, hi):
        """Bound on |LP_sparse - LP_dense| for coefficients within dg of each other (module docstring)."""
        dgm = float(np.max(dg)) if np.size(dg) else 0.0
        gm = float(np.max(np.abs(g))) if np.size(g) else 0.0
        return dgm + 16 * self._gN * (abs(float(LP)) + gm * (1.0 + float(np.sum(lo)) + float(np.sum(hi))))

    def _accumulate(self, lst, e):
        """Sum of contributions (a, cw, cd) in order: W = sum cw * a, D = sum cd * |a| (dense-path arithmetic order),
        as (U, W on U, D on U, lam, lamD) with W = lam * s_z and D = lamD off U."""
        pts = [a.idx for a, _, _ in lst if a.sigma is None]
        U = np.unique(np.concatenate(pts)) if pts else np.zeros(0, dtype=np.int64)
        sU = self._pool(e).s(U)
        WU, DU = np.zeros(len(U)), np.zeros(len(U))
        lam = lamD = 0.0
        for a, cw, cd in lst:
            if a.sigma is None:
                pos = np.searchsorted(U, a.idx)
                WU[pos] += cw[0] * a.vals * cw[1]
                DU[pos] += cd
            else:
                WU += cw[0] * (a.sigma * sU) * cw[1]
                lam += cw[0] * a.sigma * cw[1]
                DU += cd
                lamD += cd
        return U, sU, WU, DU, lam, lamD

    def _accumulate_b(self, lst, U):
        """Nonnegative bound arrays accumulated like _accumulate: items (a, cw, cd, (dW, Wabs, dD, Dabs))."""
        n = len(U)
        out = [np.zeros(n) for _ in range(4)]
        offc = [0.0] * 4
        for a, _, _, b in lst:
            if a.sigma is None:
                pos = np.searchsorted(U, a.idx)
                for j in range(4):
                    out[j][pos] += b[j]
            else:
                for j in range(4):
                    out[j] += b[j]
                    offc[j] += b[j]
        return out, offc

    def _radius_sparse(self, terms_, t):
        return self._radius_sparse_b(terms_, t)[0]

    def _radius_sparse_b(self, terms_, t):
        """(E, dE): E exactly as the sparse radius always computed it; dE >= |E_sparse - E_dense| (module docstring)."""
        gN = self._gN
        acc = {}
        rem = 0.
        drem, remabs = 0.0, 0.0
        prod = math.prod                     # scalar products of <= 3 factors: same IEEE operations as np.prod, without the
        for c, fs in terms_:                 # per-call array overhead (H3 m = 40 profile)
            vh = [f[2] for f in fs]
            r = [f[3] for f in fs]
            av = [abs(v) for v in vh]
            A = [x + y for x, y in zip(av, r)]
            n = len(fs)
            rem += abs(c) * (prod([x + y for x, y in zip(A, r)]) - prod(A)
                             - sum(r[i] * prod(A[:i] + A[i + 1:]) for i in range(n)))
            dv = [f[1].dv for f in fs]
            drr = [f[1].dr for f in fs]
            A2 = [x + y + z for x, y, z in zip(A, dv, drr)]
            r2 = [x + y for x, y in zip(r, drr)]
            av2 = [x + y for x, y in zip(av, dv)]
            big = abs(c) * prod([x + y for x, y in zip(A2, r2)])
            drem += abs(c) * (_rem_poly(A2, r2) - _rem_poly(A, r)) + 2 * _gamma(4 * n + 8) * big
            remabs += big
            for i, (e, a, v, ri) in enumerate(fs):
                Pab, Pup = prod(av[:i] + av[i + 1:]), prod(av2[:i] + av2[i + 1:])
                Ai, Aup = prod(A[:i] + A[i + 1:]), prod(A2[:i] + A2[i + 1:])
                bnd = (abs(c) * (Pup - Pab + 4 * n * EPS * Pup), abs(c) * Pup,
                       abs(c) * ((Aup - Ai) + (Pup - Pab) + 8 * n * EPS * (Aup + Pup)), abs(c) * (Aup + Pup))
                acc.setdefault(e, []).append((a, (c, prod(vh[:i] + vh[i + 1:])), abs(c) * (Ai - Pab), bnd))
        drem += 2 * _gamma(len(terms_) + 2) * remabs
        V, bmax = 0., 0.
        dV, Vabs, dbmax = 0.0, 0.0, 0.0
        for e in sorted(acc):
            U, sU, WU, DU, lam, lamD = self._accumulate([x[:3] for x in acc[e]], e)
            (dWU, WaU, dDU, DaU), (dlam, lama, dlamD, lamDa) = self._accumulate_b(acc[e], U)
            gK = 2 * _gamma(len(acc[e]) + 2)
            dWU, dlam = dWU + gK * WaU, dlam + gK * lama
            dDU, dlamD = dDU + gK * DaU, dlamD + gK * lamDa
            P = self._pool(e)
            off = self._off(e, U)
            qU = self.qhat[e][U]
            qoff = {sg: P.q[0 if sg > 0 else 1] - float(qU[sU == sg].sum()) for _, _, sg in off}
            mu = float(WU @ qU) + sum(lam * sg * qoff[sg] for _, _, sg in off)
            dmu = float(dWU @ qU) + dlam + 3 * gN * (float((np.abs(WU) + dWU) @ qU) + abs(lam) + dlam)
            loc = np.concatenate([self.reg.lo[e][U], [o[0] for o in off]])
            hic = np.concatenate([self.reg.hi[e][U], [o[1] for o in off]])
            gW = np.concatenate([(WU - mu) ** 2, [(lam * o[2] - mu) ** 2 for o in off]])
            gD = np.concatenate([DU ** 2, [lamD ** 2 for _ in off]])
            ddW = np.concatenate([dWU, [dlam for _ in off]]) + dmu
            bW = np.abs(np.concatenate([WU - mu, [lam * o[2] - mu for o in off]]))
            dgW = 2 * bW * ddW + ddW ** 2 + 4 * EPS * gW
            ddD = np.concatenate([dDU, [dlamD for _ in off]])
            Dc = np.abs(np.concatenate([DU, [lamD for _ in off]]))
            dgD = 2 * Dc * ddD + ddD ** 2 + 4 * EPS * gD
            if self.r6:
                LH, LJ = _lpmax(loc, hic, gW), _lpmax(loc, hic, gD)
                H, J = np.sqrt(LH), np.sqrt(LJ)
                eH, eJ = self._lp_eps(LH, gW, dgW, loc, hic), self._lp_eps(LJ, gD, dgD, loc, hic)
            else:
                LH, LJ = hic @ gW, hic @ gD
                H, J = np.sqrt(LH), np.sqrt(LJ)
                sh = 1.0 + float(hic.sum())
                eH = float(hic @ dgW) + 16 * gN * (float(LH) + float(gW.max()) * sh)
                eJ = float(hic @ dgD) + 16 * gN * (float(LJ) + float(gD.max()) * sh)
            V += (H + J) ** 2 / self.ns[e]
            hiW = np.concatenate([WU + DU, [lam * o[2] + lamD for o in off]])
            loW = np.concatenate([WU - DU, [lam * o[2] - lamD for o in off]])
            osc = np.max(hiW) - np.min(loW)
            bmax = max(bmax, osc / self.ns[e])
            s_ = float(H + J)
            ds = _sqrt_mod(float(LH), eH) + _sqrt_mod(float(LJ), eJ) + 2 * EPS * s_
            dV += (2 * s_ * ds + ds ** 2 + 4 * EPS * (s_ + ds) ** 2) / self.ns[e]
            Vabs += (s_ + ds) ** 2 / self.ns[e]
            dosc = 2 * float(np.max(ddW - dmu + ddD)) + 4 * EPS * (float(np.max(np.abs(hiW))) + float(np.max(np.abs(loW))))
            dbmax = max(dbmax, (dosc + EPS * abs(float(osc))) / self.ns[e])
        dV += 2 * _gamma(len(acc) + 2) * Vabs
        rem = max(rem, 0.0)
        if self.r6 and all(len(fs) == 2 for _, fs in terms_):
            bil, dbil = self._bilinear_sparse_b(terms_, t)
            rem = min(rem, bil)
            drem = max(drem, dbil)
        E = np.sqrt(2 * V * t) + bmax * t / 3 + rem
        dE = float(np.sqrt(2 * t)) * _sqrt_mod(float(V), dV) + dbmax * t / 3 + drem + 16 * EPS * float(E)
        return E, 2 * dE                          # factor-2 slack for second-order terms

    def _bilinear_sparse(self, terms_, t, s=0):
        return self._bilinear_sparse_b(terms_, t, s)[0]

    def _bilinear_sparse_b(self, terms_, t, s=0):
        side_s = {fs[s][0] for _, fs in terms_}
        side_o = {fs[1 - s][0] for _, fs in terms_}
        assert not (side_s & side_o), 'bilinear bound needs disjoint subexperiments on the two sides'
        acc = {}
        for c, fs in terms_:
            e, a = fs[s][0], fs[s][1]
            ro, dro = fs[1 - s][3], fs[1 - s][1].dr
            acc.setdefault(e, []).append((a, (0.0, 0.0), abs(c) * ro,
                                          (0.0, 0.0, abs(c) * dro + 2 * EPS * abs(c) * ro, abs(c) * (ro + dro))))
        V2, b2 = 0.0, 0.0
        dV2, V2abs, db2 = 0.0, 0.0, 0.0
        for e, lst in acc.items():
            U, sU, _, GU, _, lamG = self._accumulate([x[:3] for x in lst], e)
            (_, _, dGU, GaU), (_, _, dlamG, lamGa) = self._accumulate_b(lst, U)
            gK = 2 * _gamma(len(lst) + 2)
            dGU, dlamG = dGU + gK * GaU, dlamG + gK * lamGa
            off = self._off(e, U)
            loc = np.concatenate([self.reg.lo[e][U], [o[0] for o in off]])
            hic = np.concatenate([self.reg.hi[e][U], [o[1] for o in off]])
            gG = np.concatenate([GU, [lamG for _ in off]])
            L2 = _lpmax(loc, hic, gG ** 2)
            V2 += L2 / self.ns[e]
            b2 = max(b2, 2 * float(gG.max()) / self.ns[e])
            dG = np.concatenate([dGU, [dlamG for _ in off]])
            dg2 = 2 * np.abs(gG) * dG + dG ** 2 + 4 * EPS * gG ** 2
            e2 = self._lp_eps(L2, gG ** 2, dg2, loc, hic)
            dV2 += (e2 + 2 * EPS * L2) / self.ns[e]
            V2abs += (L2 + e2) / self.ns[e]
            db2 = max(db2, (2 * float(dG.max()) + 4 * EPS * abs(float(gG.max()))) / self.ns[e])
        dV2 += 2 * _gamma(len(acc) + 2) * V2abs
        val = np.sqrt(2 * V2 * t) + b2 * t / 3
        dval = float(np.sqrt(2 * t)) * _sqrt_mod(float(V2), dV2) + db2 * t / 3 + 16 * EPS * float(val)
        return val, 2 * dval


COLLECTS = ('r4', 'tight')


def certify(model, pilot, counts, ks=(5, 10), mode='search', selected=None, taus=(0.01, 0.05), cap=AUTO_CAP,
            variant='r4', delta=None, sparse=None, collect='r4'):
    """variant 'r4': the derivation construction (A6 methods R-PAIR / R-PAIR-S); 'r6': repair 6.
    delta: total level (default None = the registered 0.05 split); otherwise every share is scaled by delta / 0.05.
    sparse: None = auto (sparse functional path iff max_e |Z_e| > SPARSE_AUTO), True / False forced (module docstring).

    collect (search / auto modes; labels are the enumeration labels either way, only the collected set differs):
      'r4'    theta = min(L', min_A L, min_A U, query thresholds) (default; unchanged).
      'tight' theta = min(lambda_kmax, min over certificate winners x of L(x), taus), where L, U are the final bounds
              (refined on A), lambda_k = k-th largest of {L(x): x in A} u {L0(x): x reached by the beam, x not in A}
              (distinct strings), and a certificate winner is an x in A certified above some y for some queried k.
              C = {x : U0(x) >= theta}, W = C u A (refined bounds on A); counts over W; no IN filter.
      Proof that the 'tight' labels equal enumeration. Let L_(k) be the k-th
      largest L over all 2^m strings and Z = {z : L(z) >= L_(k)}, |Z| >= k. (i) lambda_k <= L_(k) (k-th largest of a
      subset of the values of distinct strings) and lambda_kmax <= lambda_k, so theta <= L_(k) for every queried k.
      (ii) Z is inside W: z in A, or z not in A with U0(z) >= L0(z) = L(z) >= L_(k) >= theta; so L_(k) is the k-th
      largest L in W. (iii) Every y outside W is not in A and has L(y) <= U(y) = U0(y) < theta. (iv) Outside y: every
      z in Z has L(z) >= L_(k) >= theta > U(y), so above(y) >= k: OUT in enumeration (OUT overrides IN), = rest.
      (v) x in W, unresolved(x) = #{y != x: U(y) >= L(x)} minus the certificates x wins against such y: if
      L(x) >= theta, no outside y has U(y) >= L(x), so the count over W is exact; if L(x) < theta, x is not a
      certificate winner (theta <= L of every winner), x is not in Z (L(x) < theta <= L_(k)), and every z in Z (in W)
      has U(z) >= L(z) >= L_(k) > L(x) with no certificate removing it, so both counts are >= k: not IN either way.
      (vi) above(x) = #{z: L(z) > U(x)} plus certificates against x: if U(x) >= theta, no outside z has
      L(z) > U(x) (L(z) < theta), exact; if U(x) < theta (refined A strings only), Z gives >= k in both: OUT.
      (vii) Certificates involve only A, inside W. (viii) Threshold tau: theta <= tau, so {U0 >= tau} u A is inside W
      and every other string has U = U0 < tau: OUT. The reported top-k candidate set is W filtered to
      {U >= L_(k)} u A (dropped strings are non-A with U < L_(k): OUT by Z, asserted).
    """
    assert check_structure(model) and variant in VARIANTS and collect in COLLECTS
    r6 = variant == 'r6'
    d0, ds, dp = (D0_6, DS_6, DP_6) if r6 else (D0, DS, DP)
    dr = DR_6
    if delta is not None:
        f = delta / 0.05
        d0, ds, dp, dr = d0 * f, ds * f, dp * f, dr * f
    nev = 4 if r6 else 2                       # r6: each score / pair constraint also carries the bilinear event
    M = 1 << model.m
    ns = np.array([v.sum() for v in counts], dtype=float)
    qhat = [v / v.sum() for v in counts]
    if selected is None:
        selected = select_pilot(model, pilot, mode, na=NA6 if r6 else NA)
    reg = CPRegion(model, counts, d0)
    fi = factor_intervals(model, reg)
    T = _Terms(model, fi, qhat, reg, ns, variant, sparse=sparse)
    if r6:
        T.tighten_factors(selected, dr)
    if mode == 'dense':
        L0, U0 = dense_bounds(model, fi)
        L0A, U0A = L0[selected], U0[selected]
        BS = None
    else:
        BS = BitSearcher(model, fi, cap=cap if mode == 'auto' else None)
        L0A, U0A = BS.values(selected)
    ts = np.log(nev * len(selected) / ds)
    pA = np.array([T.point(int(x)) for x in selected])
    dpA = None
    if T.sparse:                               # outward rounding (module docstring): decisions only beyond the margin
        dpA = np.array([T.point_bound(int(x))[1] for x in selected])
        Eb = [T.radius_bound(T.terms(int(x), True), ts) for x in selected]
        EA = np.array([b[0] for b in Eb])
        dEA = np.array([b[1] for b in Eb])
        Eup = (EA + dEA + dpA) * (1 + 8 * EPS)
        Edn = np.maximum(EA - dEA - dpA, 0.0) * (1 - 8 * EPS)
        LA = np.maximum(L0A, pA - Eup)
        UA = np.minimum(U0A, pA + Eup)
        empty = bool(np.any(np.maximum(L0A, pA - Edn) > np.minimum(U0A, pA + Edn)))   # dense could be empty
        # review: when the inner radius is not positive the sparse centre need not lie in every dense interval,
        # so the dense test cannot be excluded locally: fall back (all UND) conservatively.
        empty = empty or bool(np.any(EA - dEA - dpA <= 0.0))
    else:
        EA = np.array([T.radius(T.terms(int(x), True), ts) for x in selected])
        LA = np.maximum(L0A, pA - EA)
        UA = np.minimum(U0A, pA + EA)
        empty = bool(np.any(LA > UA))
    out = dict(empty=empty, selected=selected, stats={})

    def certs_for(k):
        lo_r, hi_r = max(1, k - 5), k + 2
        cand_pairs = [int(x) for x in selected[lo_r - 1:hi_r]]
        pairs = [(x, y) for j, x in enumerate(cand_pairs) for y in cand_pairs[j + 1:]]
        tp = np.log(nev * len(pairs) / dp)
        pos = {int(x): j for j, x in enumerate(selected)}
        certs = []
        for x, y in pairs:
            tt = T.terms(x, False) + [(-c, fs) for c, fs in T.terms(y, False)]
            if T.sparse:
                E, dE = T.radius_bound(tt, tp)
                d = pA[pos[x]] - pA[pos[y]]
                E = (E + dE + dpA[pos[x]] + dpA[pos[y]]) * (1 + 8 * EPS) + 4 * EPS * abs(d)
            else:
                E = T.radius(tt, tp)
                d = pA[pos[x]] - pA[pos[y]]
            if abs(d) > E:
                certs.append((x, y) if d > 0 else (y, x))
        return certs

    pre = None
    tight = collect == 'tight' and mode in ('search', 'auto') and not empty
    if mode in ('search', 'auto') and not empty:
        if tight:                              # docstring: collect='tight'
            pre = {k: certs_for(k) for k in ks}
            posS = {int(x): j for j, x in enumerate(selected)}
            winners = sorted({x for k in ks for x, _ in pre[k]})
            th_cert = min([float(LA[posS[x]]) for x in winners], default=np.inf)
            bidx, bL = BS.beam_reached()
            pool = np.concatenate([LA, bL[~np.isin(bidx, selected)]])
            kmax = max(ks)
            lam = float(-np.partition(-pool, kmax - 1)[kmax - 1]) if len(pool) >= kmax else -np.inf
            theta = min([lam, th_cert] + [float(t) for t in taus])
            out['stats'].update(collect='tight', theta=theta, lambda_kmax=lam, theta_cert=th_cert,
                                cert_winners=len(winners))
        else:
            # One collection serves every query: theta <= L' <= L_(kmax) <= L_(k) and theta <= every query threshold, so
            # each query's set {U0 >= threshold} is a filter of it, and L_(k) is the k-th largest L0 in it (every string
            # with L0 >= L_(k) has U0 >= L0 >= theta). Identical to separate BS.topk(k) / BS.collect(.) calls (tested).
            Lp = BS.beam_lower(max(ks))
            theta = min([Lp, float(LA.min()), float(UA.min())] + [min(t, float(UA.min())) for t in taus])
        try:
            cidx, cL, cU = BS.collect(theta)
            cLs = -np.sort(-cL)[:max(ks)]
            out['stats']['collected_all'] = int(len(cidx))
            mode = 'search'
            if tight:                          # W = C u A with the raw (L0, U0); A refined per query below
                inC = np.isin(selected, cidx)
                widx = np.concatenate([cidx, np.asarray(selected, np.int64)[~inC]])
                wL0 = np.concatenate([cL, L0A[~inC]])
                wU0 = np.concatenate([cU, U0A[~inC]])
                wA = np.isin(widx, selected)
                out['stats']['working'] = int(len(widx))
        except FrontierCap:                    # mode 'auto' only: the collection would be huge -> dense arithmetic
            if mode != 'auto' or model.m > 26:
                raise
            L0, U0 = dense_bounds(model, fi)   # bit-identical to BS values (rankcert.search.bitbnb docstring)
            mode = 'dense'
        out['stats']['path'] = mode
    tight = tight and mode == 'search'        # after a dense fallback the dense branch below applies
    for k in ks:
        certs = pre[k] if pre is not None else certs_for(k)
        if empty:
            out[k] = dict(cand=np.zeros(0, np.int64), cand_labels=np.zeros(0, np.int8), rest=0, certified=len(certs))
            continue
        if mode == 'dense':
            idx = np.arange(M, dtype=np.int64)
            L, U = L0.copy(), U0.copy()
        elif tight:
            idx, L, U = widx, wL0.copy(), wU0.copy()
        else:
            Lk = float(cLs[k - 1])
            Tk = min(Lk, float(LA.min()), float(UA.min()))
            sub = cU >= Tk                     # = BS.collect(Tk), same walk order
            idx, L, U = cidx[sub], cL[sub].copy(), cU[sub].copy()
        posA = _positions(idx, selected, mode)
        where = {int(x): int(j) for x, j in zip(selected, posA)}
        L[posA] = LA
        U[posA] = UA
        n = len(idx)
        unresolved = n - np.searchsorted(np.sort(U), L, side='left') - 1
        above = n - np.searchsorted(np.sort(L), U, side='right')
        for x, y in certs:                     # x certified above y
            jx, jy = where[x], where[y]
            if U[jy] >= L[jx]:
                unresolved[jx] -= 1
            if L[jx] <= U[jy]:
                above[jy] += 1
        lab = np.zeros(n, dtype=np.int8)
        isin = unresolved <= k - 1
        if mode == 'search' and not tight:
            isin &= L >= Tk                    # strings below T_k have >= k unresolved competitors (T3 argument)
        lab[isin] = 1
        lab[above >= k] = -1
        if tight:                              # report W filtered to {U >= L_(k)} u A (docstring)
            Lk = float(-np.partition(-L, k - 1)[k - 1])
            keep = (U >= Lk) | wA
            assert np.all(lab[~keep] == -1)
            out[k] = dict(cand=idx[keep], cand_labels=lab[keep], rest=-1, certified=len(certs),
                          collected=int(keep.sum()), working=int(n), Lk=Lk)
            continue
        out[k] = dict(cand=idx, cand_labels=lab, rest=-1, certified=len(certs), collected=int(n))
    for tau in taus:                           # threshold corollary of T3 on the refined endpoints
        key = 'tau%g' % tau
        if empty:
            out[key] = dict(cand=np.zeros(0, np.int64), cand_labels=np.zeros(0, np.int8), rest=0)
            continue
        if mode == 'dense':
            idx = np.arange(M, dtype=np.int64)
            L, U = L0.copy(), U0.copy()
        elif tight:
            sub = (wU0 >= tau) | wA            # every other string has U = U0 < tau: OUT (docstring (viii))
            idx, L, U = widx[sub], wL0[sub].copy(), wU0[sub].copy()
        else:
            sub = cU >= min(tau, float(UA.min()))   # = BS.collect(.); every other string has U0 < tau: OUT
            idx, L, U = cidx[sub], cL[sub].copy(), cU[sub].copy()
        posA = _positions(idx, selected, mode)
        L[posA] = LA
        U[posA] = UA
        lab = np.zeros(len(idx), dtype=np.int8)
        lab[L > tau] = 1
        lab[U < tau] = -1
        out[key] = dict(cand=idx, cand_labels=lab, rest=-1, collected=int(len(idx)))
    return out


def _positions(idx, xs, mode):
    """Positions of the strings xs (all present) in the index array idx."""
    xs = np.asarray(xs, dtype=np.int64)
    if mode == 'dense':
        return xs
    order = np.argsort(idx, kind='stable')
    return order[np.searchsorted(idx[order], xs)]


def expand(M, ans):
    lab = np.full(M, ans['rest'], dtype=np.int8)
    lab[ans['cand']] = ans['cand_labels']
    return lab
