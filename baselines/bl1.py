"""BL1 — uniform bound + the same three-valued rules (guide §P6.3; G7's approach, stated as a corrected, explicitly
simultaneous bound for the cut estimator's product structure). Never attributed to Neumann et al. as written.

Both variants give ONE half-width Delta for every string x around the plug-in estimate p_hat(x):
  every factor value (point rows u_{g,b}(y) and mass rows S_{g,b}, scalar rows E) deviates by at most eps from its
  truth on the good event; with |u|, |uhat| <= 1, a product of r factors deviates by at most r * eps (telescoping); so
      Delta = sum_g |c_g| r_g eps  +  2^{-m} sum_g |c_g| r_g eps_S
  where r_g = number of factors (held blocks + scalar rows) of group g.
BL1-H  : eps = Hoeffding with union over all F factor values: sqrt(2 ln(2F/delta) / N) (same F as R-H).
BL1-DKW: cells: Massart-DKW per subexperiment at level delta/2 split over |S| subexperiments,
         sup_z |qhat(z) - q(z)| <= 2 eps_D, eps_D = sqrt(ln(2|S| / (delta/2)) / (2N)); a point row with 2^{r_e} supported
         cells deviates by at most 2^{r_e} * 2 eps_D; mass / scalar rows: Hoeffding at level delta/2 over their count.
"""
from __future__ import annotations

import math

import numpy as np

from rankcert.regions.regions import functional_set


def _factor_count(model, g):
    nb = len(model.blocks)
    scalars = sum(1 for use in model.uses[g] if not use.held)
    return nb + scalars


def half_width(model, n: list, delta: float, variant: str) -> float:
    N = min(float(v.sum()) for v in n)
    G = model.groups
    if variant == 'H':
        F = functional_set(model)['F']
        eps = math.sqrt(2.0 * math.log(2.0 * F / delta) / N)
        eps_S = eps
        eps_point = [eps] * G
    elif variant == 'DKW':
        S = len(model.subexperiments)
        eps_D = math.sqrt(math.log(2.0 * S / (delta / 2)) / (2.0 * N))
        fs = functional_set(model)
        n_mass = sum(1 for k in fs['keys'])          # one mass / scalar row per distinct use
        eps_S = math.sqrt(2.0 * math.log(2.0 * n_mass / (delta / 2)) / N)
        eps_point = []
        for g in range(G):
            worst = 0.0
            for use in model.uses[g]:
                if use.held:
                    sub = model.subexperiments[use.e]
                    r = sub.written                  # supported cells of one point row: 2^{written} (dropped are constant)
                    worst = max(worst, (1 << r) * 2.0 * eps_D)
                else:
                    worst = max(worst, eps_S)
            eps_point.append(min(worst, 2.0))
    else:
        raise ValueError(variant)
    c = np.abs(np.asarray(model.coefficients, dtype=float))
    D = 0.0
    for g in range(G):
        r = _factor_count(model, g)
        # telescoping: |prod uhat - prod u| <= r * eps because |uhat|, |u| <= 1 (review; the earlier
        # (1 + eps)^r - 1 was needlessly loose and would have weakened the baseline)
        D += c[g] * r * eps_point[g]
        D += c[g] * r * eps_S / (1 << model.m)
    return D


def bounds(model, n: list, delta: float, variant: str, p_hat_block: np.ndarray):
    D = half_width(model, n, delta, variant)
    return p_hat_block - D, p_hat_block + D, D
