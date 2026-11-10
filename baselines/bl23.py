"""BL2 (delta-method normal intervals, ShotQC-style variance propagation, Bonferroni over the candidate set) and BL3
(percentile max-type bootstrap over the candidate set, B resamples). Neither has a finite-sample guarantee (N1, N11);
they are evaluated on coverage and decisiveness only. Candidate set C = the `n_cand` strings with the
largest plug-in estimates; the §P4.3 rules are applied with C as the universe; strings outside C are UND.

Block-order indexing as in rankcert.propagate.interval. The constant c0 = (1 + sum_g s_g prod_b S_gb) / 2^m is held at its
plug-in value in the variance (its contribution is O(2^-m)); this is stated, not hidden.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _block_values(model, x_block_idx):
    """Per block, the local value of each candidate (block-order index)."""
    vals = []
    rest = x_block_idx.copy()
    for bits in model.blocks:
        w = len(bits)
        vals.append(rest & ((1 << w) - 1))
        rest = rest >> w
    return vals


def candidate_estimates(model, qhat, cand, with_grad=True):
    """p_hat on the candidates and, for each distinct subexperiment e, the gradient d p_hat(x) / d q_e (|C| x |Z_e|)."""
    loc = _block_values(model, cand)
    nb = len(model.blocks)
    scal, tabs = model.weights(qhat)
    T = np.zeros(len(cand))
    mass = 0.0
    grads = [np.zeros((len(cand), len(q))) for q in qhat]
    for g in range(model.groups):
        fac = [tabs[g][b][loc[b]] for b in range(nb)]
        prod = scal[g] * np.prod(fac, axis=0)
        T += prod
        mm = scal[g]
        for b in range(nb):
            mm *= float(tabs[g][b].sum())
        mass += mm
        # gradient through every functional of the group: held block rows and scalar-only rows
        for i, use in enumerate(model.uses[g] if with_grad else []):
            if not use.held:
                rows, signs, R = model.row_index(g, i)
                coef = float(model.coefficients[g])
                for i2, u2 in enumerate(model.uses[g]):
                    if i2 != i and not u2.held:
                        r2, s2, _ = model.row_index(g, i2)
                        coef *= float(np.bincount(r2, weights=s2 * qhat[u2.e], minlength=1)[0])
                blockprod = np.prod(fac, axis=0) if nb else np.ones(len(cand))
                grads[use.e] += -(coef * blockprod)[:, None] * signs[None, :]
                continue
            b = model.block_of_sub[i]
            rows, signs, R = model.row_index(g, i)
            other = scal[g] * np.prod([fac[bb] for bb in range(nb) if bb != b], axis=0) if nb > 1 else np.full(len(cand), scal[g])
            # dp/dq_e(z) = -other(x) * sign(z) * [row(z) == x_b]
            grads[use.e] += -(other[:, None] * (signs[None, :] * (rows[None, :] == loc[b][:, None])))
    phat = (1.0 + mass) / (1 << model.m) - T
    return phat, grads


def bl2_bounds(model, n, delta, phat_block, n_cand=256):
    cand = np.argsort(-phat_block)[:n_cand]
    qhat = [v / v.sum() for v in n]
    ph, grads = candidate_estimates(model, qhat, cand)
    var = np.zeros(len(cand))
    for e, G in enumerate(grads):
        q = qhat[e]
        N = float(n[e].sum())
        var += ((G ** 2) @ q - (G @ q) ** 2) / N
    z = norm.ppf(1 - delta / (2 * len(cand)))
    sd = np.sqrt(np.maximum(var, 0.0))
    return cand, ph - z * sd, ph + z * sd


def bl3_bounds(model, n, delta, phat_block, rng, B=2000, n_cand=256):
    cand = np.argsort(-phat_block)[:n_cand]
    qhat = [v / v.sum() for v in n]
    ph, _ = candidate_estimates(model, qhat, cand)
    mx = np.empty(B)
    for b in range(B):
        nb_ = [rng.multinomial(int(v.sum()), q) for v, q in zip(n, qhat)]
        pb, _ = candidate_estimates(model, [v / v.sum() for v in nb_], cand, with_grad=False)
        mx[b] = np.max(np.abs(pb - ph))
    c = float(np.quantile(mx, 1 - delta))
    return cand, ph - c, ph + c
