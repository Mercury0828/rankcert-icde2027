"""Naive interval propagation through the separable affine form.

    p(x) = c0 - sum_g s_g prod_b u_{g,b}(x_b),   c0 = (1 + sum_g s_g prod_b S_{g,b}) / 2^m,
    s_g = parity * c_g * prod_{subcircuits without held bits} E_{g,i},   S_{g,b} = sum_y u_{g,b}(y).

Every factor is replaced by its interval over the region; products use all four corner products; shared subexperiments
are treated as independent intervals (valid on the region event). Strings are indexed in BLOCK ORDER:
index(x) = sum_b x_b * prod_{b' < b} 2^{m_b'} (use `to_block_order` to permute a natural-order array).
"""
from __future__ import annotations

import numpy as np


def imul(alo, ahi, blo, bhi):
    c = (alo * blo, alo * bhi, ahi * blo, ahi * bhi)
    lo = np.minimum(np.minimum(c[0], c[1]), np.minimum(c[2], c[3]))
    hi = np.maximum(np.maximum(c[0], c[1]), np.maximum(c[2], c[3]))
    return lo, hi


def iouter(alo, ahi, blo, bhi):
    """Interval outer product, result index = i_b * len(a) + i_a (a = lower block)."""
    return imul(alo[None, :], ahi[None, :], blo[:, None], bhi[:, None])


def factor_intervals(model, region) -> dict:
    """Interval of every factor: s_g (scalar), u_{g,b} (arrays over block values), S_{g,b} (scalar mass)."""
    G = model.groups
    nb = len(model.blocks)
    s_lo = np.array(model.coefficients, dtype=float)
    s_hi = s_lo.copy()
    u = [[None] * nb for _ in range(G)]
    S = [[None] * nb for _ in range(G)]
    cache = {}
    for g in range(G):
        for i, use in enumerate(model.uses[g]):
            key = (use.e, tuple(use.held), tuple(use.offsets), use.dropped)
            if key not in cache:
                rows, signs, R = model.row_index(g, i)
                lo, hi = region.interval_sparse(use.e, rows, signs, R)
                if use.held:
                    # mass functional = sum of all rows = signs as a single row
                    mlo, mhi = region.interval_sparse(use.e, np.zeros_like(rows), signs, 1)
                    cache[key] = (lo, hi, float(mlo[0]), float(mhi[0]))
                else:
                    cache[key] = (lo, hi, None, None)
            lo, hi, mlo, mhi = cache[key]
            if use.held:
                b = model.block_of_sub[i]
                u[g][b] = (lo, hi)
                S[g][b] = (mlo, mhi)
            else:
                a, c = imul(s_lo[g], s_hi[g], float(lo[0]), float(hi[0]))
                s_lo[g], s_hi[g] = a, c
    return dict(s=(s_lo, s_hi), u=u, S=S)


def dense_bounds(model, fi: dict):
    """Dense L, U (block order) from factor intervals."""
    G = model.groups
    nb = len(model.blocks)
    m = model.m
    s_lo, s_hi = fi['s']
    sum_lo = None
    sum_hi = None
    c_lo = c_hi = 0.0
    for g in range(G):
        # point term
        lo, hi = fi['u'][g][0]
        for b in range(1, nb):
            blo, bhi = fi['u'][g][b]
            lo, hi = iouter(lo, hi, blo, bhi)
            lo, hi = lo.ravel(), hi.ravel()
        lo, hi = imul(lo, hi, s_lo[g], s_hi[g])
        if sum_lo is None:
            sum_lo, sum_hi = lo.copy(), hi.copy()
        else:
            sum_lo += lo
            sum_hi += hi
        # mass term
        mlo, mhi = s_lo[g], s_hi[g]
        for b in range(nb):
            a, c = fi['S'][g][b]
            mlo, mhi = imul(mlo, mhi, a, c)
        c_lo += mlo
        c_hi += mhi
    scale = 1.0 / (1 << m)
    c0_lo = (1.0 + c_lo) * scale
    c0_hi = (1.0 + c_hi) * scale
    L = c0_lo - sum_hi
    U = c0_hi - sum_lo
    return L, U


def dense_point(model, qs: list) -> np.ndarray:
    """Plug-in (or exact, if qs exact) p in block order."""
    scal, tabs = model.weights(qs)
    nb = len(model.blocks)
    m = model.m
    T = None
    tot = 0.0
    for g in range(model.groups):
        t = tabs[g][0]
        for b in range(1, nb):
            t = np.multiply.outer(tabs[g][b], t).ravel()
        t = scal[g] * t
        T = t if T is None else T + t
        mass = scal[g]
        for b in range(nb):
            mass *= float(tabs[g][b].sum())
        tot += mass
    return (1.0 + tot) / (1 << m) - T


def to_block_order(model, arr_natural: np.ndarray) -> np.ndarray:
    from rankcert.recon.model import block_index_map
    maps = block_index_map(model.m, model.blocks)
    idx = np.zeros(1 << model.m, dtype=np.int64)
    mult = 1
    for b, mp in enumerate(maps):
        idx += mp * mult
        mult <<= len(model.blocks[b])
    out = np.empty_like(arr_natural)
    out[idx] = arr_natural
    return out
