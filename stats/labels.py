"""Three-valued labels (guide §P4.3), ground truth from the exact target, decided shares, joint errors, Clopper-Pearson.

Label codes: +1 IN, -1 OUT, 0 UND.
Top-k rules: IN(x) iff |{y != x : U(y) >= L(x)}| <= k-1 ; OUT(x) iff |{y : L(y) > U(x)}| >= k.
Threshold:   IN(x) iff L(x) > tau ; OUT(x) iff U(x) < tau.
Ground truth (exact p): sure member iff |{y != x : p(y) >= p(x)}| <= k-1 ; sure non-member iff |{y : p(y) > p(x)}| >= k;
top-k ties grouped at tolerance tol = rtol * max|p|, rtol = 1e-12 (amended from 1e-9 before any D1/D2 run, PREREG
appendix), because structural ties are computed with floating error ~1e-16 relative; threshold truth is exact.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as _beta


def topk_labels(L: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    n = len(L)
    Us = np.sort(U)
    Ls = np.sort(L)
    cnt_U_ge_Lx = n - np.searchsorted(Us, L, side='left')          # |{y : U(y) >= L(x)}|, includes x (U(x) >= L(x))
    cnt_L_gt_Ux = n - np.searchsorted(Ls, U, side='right')         # |{y : L(y) > U(x)}|
    lab = np.zeros(n, dtype=np.int8)
    lab[(cnt_U_ge_Lx - 1) <= k - 1] = 1
    lab[cnt_L_gt_Ux >= k] = -1
    return lab


def threshold_labels(L, U, tau):
    lab = np.zeros(len(L), dtype=np.int8)
    lab[L > tau] = 1
    lab[U < tau] = -1
    return lab


def topk_truth(p: np.ndarray, k: int, rtol: float = 1e-12) -> np.ndarray:
    """+1 sure member, -1 sure non-member, 0 neither."""
    tol = rtol * float(np.max(np.abs(p)))
    ps = np.sort(p)
    n = len(p)
    ge = n - np.searchsorted(ps, p - tol, side='left')
    gt = n - np.searchsorted(ps, p + tol, side='right')
    t = np.zeros(n, dtype=np.int8)
    t[(ge - 1) <= k - 1] = 1
    t[gt >= k] = -1
    return t


def threshold_truth(p, tau):
    """Exact comparison with the fixed constant tau (no tolerance band: a band would count correct labels of strings
    within the band as wrong — review)."""
    t = np.zeros(len(p), dtype=np.int8)
    t[p > tau] = 1
    t[p < tau] = -1
    return t


def evaluate(lab: np.ndarray, truth: np.ndarray) -> dict:
    """Decided shares against the truth and the number of wrong labels."""
    sm = truth == 1
    sn = truth == -1
    wrong_in = int(((lab == 1) & (truth != 1)).sum())
    wrong_out = int(((lab == -1) & (truth != -1)).sum())
    return dict(sure_members=int(sm.sum()), in_of_sure=int(((lab == 1) & sm).sum()),
                sure_non=int(sn.sum()), out_of_sure=int(((lab == -1) & sn).sum()),
                n_in=int((lab == 1).sum()), n_out=int((lab == -1).sum()), n_und=int((lab == 0).sum()),
                wrong_in=wrong_in, wrong_out=wrong_out, wrong=wrong_in + wrong_out)


def clopper_pearson_ci(x: int, n: int, level: float = 0.95):
    a = 1 - level
    lo = 0.0 if x == 0 else float(_beta.ppf(a / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(_beta.ppf(1 - a / 2, x + 1, n - x))
    return lo, hi
