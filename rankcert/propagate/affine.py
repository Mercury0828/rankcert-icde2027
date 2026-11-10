"""Repair route R-2 (guide §7b): first-order (affine) propagation with an exact linear part and a bounded remainder.

For one string x write q_e = qhat_e + d_e with d_e in D_e = C_e - qhat_e (C_e = the region of subexperiment e, a box
intersected with the simplex for R-CP). Every factor is u = uhat + A d, so for group g with held blocks b = 0..s-1
    prod_b u_gb = prod_b uhat_gb + sum_b (prod_{b' != b} uhat_gb') (A_gb d_{e(g,b)}) + R_g,
    |R_g| <= prod_b (|uhat_gb| + r_gb) - prod_b |uhat_gb| - sum_b r_gb prod_{b' != b} |uhat_gb'|
where r_gb = max |A_gb d| over the region (from the factor interval). Collecting the linear terms by subexperiment gives
one coefficient vector w_e(x) per e, and max / min of sum_e w_e . d_e over prod_e D_e is the sum of independent exact LP
optima (box intersect simplex: greedy fill in coefficient order). Scalar-only factors enter s_g as intervals (as in the
naive propagation). The bound for p(x) = c0 - T(x) is
    L = c0_lo - That(x) - max_lin - sum_g |s_g|max R_g ,   U = c0_hi - That(x) - min_lin + sum_g |s_g|max R_g.
It holds on the same event as the naive interval bound, so the intersection of both is valid; it is applied to a
candidate set of strings (the ones whose labels can change), all other strings keep the naive bound.
Requires a region with per-category boxes (R-CP: lo, hi). Phase-0 prototype; validity of this construction is a
C2-propagate target (not proved here; containment is tested on random feasible points).
"""
from __future__ import annotations

import numpy as np


def lp_box_simplex(w: np.ndarray, lo: np.ndarray, hi: np.ndarray):
    """(min, max) of w . q over {lo <= q <= hi, sum q = 1} (exact, greedy)."""
    R = 1.0 - lo.sum()
    cap = hi - lo
    base = w @ lo

    def fill(order):
        c = np.cumsum(cap[order])
        take = np.clip(np.minimum(cap[order], R - (c - cap[order])), 0.0, None)
        return float(w[order] @ take)
    o = np.argsort(w)
    return base + fill(o), base + fill(o[::-1])


def affine_bounds(model, region, n, cand_block_idx, fi):
    """Affine L, U for the candidate strings (block-order indices). `region` must expose lo / hi per subexperiment."""
    qhat = [v / v.sum() for v in n]
    scal_hat, tabs_hat = model.weights(qhat)
    nb = len(model.blocks)
    # local values of the candidates
    loc = []
    rest = np.asarray(cand_block_idx, dtype=np.int64).copy()
    for bits in model.blocks:
        loc.append(rest & ((1 << len(bits)) - 1))
        rest >>= len(bits)
    s_lo, s_hi = fi['s']
    smax = np.maximum(np.abs(s_lo), np.abs(s_hi))
    s_mid = np.asarray(scal_hat, float)          # s_g at qhat (point; scalar-only factors' spread goes to remainder)
    C = len(cand_block_idx)
    That = np.zeros(C)
    rem = np.zeros(C)
    W = [np.zeros((C, len(q))) for q in qhat]
    for g in range(model.groups):
        uh = [tabs_hat[g][b][loc[b]] for b in range(nb)]
        r = []
        for b in range(nb):
            lo, hi = fi['u'][g][b]
            r.append(np.maximum(uh[b] - lo[loc[b]], hi[loc[b]] - uh[b]))
        prod_hat = np.prod(uh, axis=0)
        That += s_mid[g] * prod_hat
        # remainder of the product beyond first order
        absu = [np.abs(x) for x in uh]
        full = np.prod([absu[b] + r[b] for b in range(nb)], axis=0)
        first = np.zeros(C)
        for b in range(nb):
            first += r[b] * np.prod([absu[bb] for bb in range(nb) if bb != b], axis=0)
        R_g = full - np.prod(absu, axis=0) - first
        # scalar-only factors: s_g in [s_lo, s_hi]; the deviation of s_g from s_mid times the full product bound
        ds = max(abs(s_hi[g] - s_mid[g]), abs(s_mid[g] - s_lo[g]))
        rem += smax[g] * R_g + ds * full
        # linear part, collected per subexperiment
        for i, use in enumerate(model.uses[g]):
            if not use.held:
                continue
            b = model.block_of_sub[i]
            rows, signs, _ = model.row_index(g, i)
            other = np.prod([uh[bb] for bb in range(nb) if bb != b], axis=0) if nb > 1 else np.ones(C)
            coef = s_mid[g] * other                       # d T / d u_gb at qhat
            W[use.e] += coef[:, None] * (signs[None, :] * (rows[None, :] == loc[b][:, None]))
    lin_min = np.zeros(C)
    lin_max = np.zeros(C)
    for e, We in enumerate(W):
        if not np.any(We):
            continue
        lo, hi = region.lo[e], region.hi[e]
        base = We @ qhat[e]
        for c in range(C):
            mn, mx = lp_box_simplex(We[c], lo, hi)
            lin_min[c] += mn - base[c]
            lin_max[c] += mx - base[c]
    # constant term c0 as in the naive propagation
    c_lo = c_hi = 0.0
    from rankcert.propagate.interval import imul
    for g in range(model.groups):
        mlo, mhi = s_lo[g], s_hi[g]
        for b in range(nb):
            a, c = fi['S'][g][b]
            mlo, mhi = imul(mlo, mhi, a, c)
        c_lo += mlo
        c_hi += mhi
    sc = 1.0 / (1 << model.m)
    c0_lo, c0_hi = (1.0 + c_lo) * sc, (1.0 + c_hi) * sc
    L = c0_lo - That - lin_max - rem
    U = c0_hi - That - lin_min + rem
    return L, U
