"""Appended shots: rounds with fresh batches, adaptive allocation, labels valid at every
round on one event of probability >= 1 - delta.

Round 0 is the pilot. Round t >= 1 draws a fresh batch of b_t = S * N1 * 2^(t-1) shots split over the subexperiments by
an allocation rule that sees only the pooled past counts and the labels issued so far; the round certificate is
rankcert.certify.rpair.certify on the batch alone at level delta_t = delta * 2^(-t), with the pooled past counts as
side information (selection of the refined set). Issued labels are accumulated (union over rounds; on the event of T5
they never conflict, and a conflict is counted if it ever happens).

Allocation rules (all F_{t-1}-measurable): 'uniform'; 'boundary' (ours: Neyman weights from the pair-difference
influence variances of the still-undecided top-k boundary strings at the pooled estimate); 'varmin' (ShotQC-style:
Neyman weights that minimise the summed variance of the reconstruction over all 2^m strings, computed exactly through
the separable form). Every rule keeps n_e >= BETA * (uniform share), as the R-PAIR-S safeguard.
"""
from __future__ import annotations

import math

import numpy as np

from rankcert.certify.rpair import PointRegion, _Terms, certify, expand, select_pilot
from rankcert.propagate.interval import factor_intervals

BETA = 0.5
RULES = ('uniform', 'boundary', 'varmin')


def neyman_safeguarded(a, total, beta=BETA):
    """Integer n_e summing to total, n_e >= ceil(beta * U_e) with U the balanced integer uniform split (U_e = total / S
    when S divides total, as in `run`), the rest proportional to sqrt(a_e) (largest remainder, ties by index)."""
    S = len(a)
    U = np.full(S, total // S, dtype=np.int64)
    U[: total - U.sum()] += 1
    b = np.array([math.ceil(beta * u) for u in U], dtype=np.int64)
    rest = int(total - b.sum())
    w = np.sqrt(np.maximum(np.asarray(a, dtype=float), 0.0))
    w = w / w.sum() if w.sum() > 0 else np.full(S, 1.0 / S)
    x = rest * w
    base = np.floor(x).astype(np.int64)
    rem = rest - int(base.sum())
    order = np.lexsort((np.arange(S), -(x - base)))
    base[order[:rem]] += 1
    n = b + base
    assert n.sum() == total
    return n


def _point_terms(model, q):
    """_Terms at the plug-in q (zero-width region): feature(g, i, x) -> (e, a, value, 0)."""
    reg = PointRegion(q)
    return _Terms(model, factor_intervals(model, reg), q, reg, None)


def pair_weights(model, q, cand):
    """a_e = sum over pairs (x, y) of cand of Var_{q_e}(W_e), W_e = first-order influence of subexperiment e on
    p(x) - p(y) (mass terms cancel in the difference). General number of subcircuits."""
    T = _point_terms(model, q)
    S = len(q)
    a = np.zeros(S)
    for j, x in enumerate(cand):
        for y in cand[j + 1:]:
            W = [np.zeros(len(v)) for v in q]
            for sgn, s in ((1.0, int(x)), (-1.0, int(y))):
                for g, c in enumerate(model.coefficients):
                    fs = [T.feature(g, i, s) for i in range(len(model.uses[g]))]
                    vals = [f[2] for f in fs]
                    for i, (e, av, v, _) in enumerate(fs):
                        other = float(np.prod(vals[:i] + vals[i + 1:]))
                        W[e] += sgn * (-c) * other * av
            a += np.array([float(q[e] @ W[e] ** 2 - (q[e] @ W[e]) ** 2) for e in range(S)])
    return a


def varmin_weights(model, q):
    """a_e = sum_x Var_{z ~ q_e}(psi_{e,x}(z)), psi_{e,x}(z) = d p(x) / d q_e(z) (first-order contribution of one shot
    of e to p_hat(x), incl. the mass term). With D_x = d T'(x) / d q_e and T'(x) = sum_g c_g prod_i f_{g,i}(x_i):
      sum_x Var(psi) = sum_x Var(D_x) - (1/M) Var(sum_x D_x)
      sum_x Var(D_x) = sum_{g,g' in G_e} c_g c_g' K_gg' (M2_gg' - M1_gg'),  K_gg' = prod_{j != i} k_j(g, g'),
    k_j = sum_{x_j} u_{g,j}(x_j) u_{g',j}(x_j) (held j) or E_{g,j} E_{g',j} (no held bits); M2 = sum_z q s_g s_g' [r_g = r_g']
    (held i; without the row condition otherwise), M1 = sum_{x_i} u_g u_g' (held i) or E_g E_g'."""
    S = len(q)
    M = float(1 << model.m)
    vec = {}

    def fac(g, i):
        if (g, i) not in vec:
            use = model.uses[g][i]
            rows, signs, R = model.row_index(g, i)
            if use.held:
                vec[(g, i)] = (rows, signs, np.bincount(rows, weights=signs * q[use.e], minlength=R))
            else:
                vec[(g, i)] = (None, signs, np.array([float(signs @ q[use.e])]))
        return vec[(g, i)]

    uses_of = [[] for _ in range(S)]
    for g, row in enumerate(model.uses):
        for i, use in enumerate(row):
            uses_of[use.e].append((g, i))
    a = np.zeros(S)
    for e in range(S):
        U = uses_of[e]
        if not U:
            continue
        i0 = U[0][1]
        assert all(i == i0 for _, i in U), 'a subexperiment belongs to one subcircuit'
        held = model.uses[U[0][0]][i0].held
        n = len(U)
        G = [g for g, _ in U]
        c = np.array([model.coefficients[g] for g in G], dtype=float)
        K = np.ones((n, n))
        tot = np.ones(n)
        for j in range(len(model.uses[G[0]])):
            if j == i0:
                continue
            vs = [fac(g, j)[2] for g in G]
            for r in range(n):
                tot[r] *= float(vs[r].sum())
                for s_ in range(n):
                    K[r, s_] *= float(vs[r] @ vs[s_])
        qe = q[e]
        M2 = np.zeros((n, n))
        M1 = np.zeros((n, n))
        for r in range(n):
            rr, sr, ur = fac(*U[r])
            for s_ in range(n):
                rs, ss, us = fac(*U[s_])
                same = (rr == rs) if held else 1.0
                M2[r, s_] = float(qe @ (sr * ss * same))
                M1[r, s_] = float(ur @ us)
        cc = np.outer(c, c)
        sumvar = float((cc * K * (M2 - M1)).sum())
        # sum_x D_x(z) = sum_g c_g s_g(z) prod_{j != i} (sum_{x_j} f_{g,j})
        Dsum = sum(c[r] * tot[r] * fac(*U[r])[1] for r in range(n))
        varsum = float(qe @ Dsum ** 2 - (qe @ Dsum) ** 2)
        a[e] = max(sumvar - varsum / M, 0.0)
    return a


def run(model, qs, rule, N1, rounds, rng, k=10, delta=0.05, variant='r6', mode='auto', pilot_per_e=None,
        truth=None, beta=BETA):
    """One sequential run on the true distributions qs (simulation). Yields one record per round with the cumulative
    shots and, if truth (topk_truth array, block order) is given, the evaluation of the accumulated top-k labels."""
    from stats.labels import evaluate
    assert rule in RULES
    S = len(qs)
    qt = [v / v.sum() for v in qs]
    M = 1 << model.m
    n0 = pilot_per_e if pilot_per_e is not None else max(N1 // 20, 1)
    pooled = [rng.multinomial(n0, v).astype(float) for v in qt]
    shots = S * n0
    lab = np.zeros(M, dtype=np.int8)
    conflicts = 0
    for t in range(1, rounds + 1):
        b = S * N1 * (1 << (t - 1))
        qp = [v / v.sum() for v in pooled]
        if rule == 'uniform':
            sizes = neyman_safeguarded(np.ones(S), b, beta=1.0)
        elif rule == 'varmin':
            sizes = neyman_safeguarded(varmin_weights(model, qp), b, beta)
        else:
            ranked = select_pilot(model, pooled, mode, na=k + 12)
            und = [int(x) for x in ranked if lab[int(x)] == 0][:8]
            cand = und if len(und) >= 2 else [int(x) for x in ranked[max(1, k - 5) - 1:k + 2]]
            sizes = neyman_safeguarded(pair_weights(model, qp, cand), b, beta)
        assert sizes.min() >= 1, 'every subexperiment gets a positive batch (hypothesis of the round certificate)'
        batch = [rng.multinomial(int(n), v).astype(float) for n, v in zip(sizes, qt)]
        dt = delta * 2.0 ** (-t)
        out = certify(model, pooled, batch, ks=(k,), mode=mode, taus=(), variant=variant, delta=dt)
        new = expand(M, out[k])
        conflicts += int(((new == 1) & (lab == -1)).sum() + ((new == -1) & (lab == 1)).sum())
        lab = np.where(lab == 0, new, lab)
        shots += b
        pooled = [p + c for p, c in zip(pooled, batch)]
        rec = dict(round=t, batch=int(b), shots=int(shots), delta_t=dt, conflicts=conflicts,
                   min_n=int(sizes.min()), max_n=int(sizes.max()), empty=bool(out['empty']))
        if truth is not None:
            rec['top%d' % k] = evaluate(lab, truth)
        yield rec
