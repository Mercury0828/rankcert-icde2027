"""D0 on the initial instances: QCut reproduction to 1e-12 from Aer raw counts (all x), exact reference vs uncut,
structure statistics, and validation of every exact q_e against 10^6 Aer shots (total variation + chi-square).
python -m derisk.d0_run ghz20 qaoa20 adder16 qaoa24
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import QCut as ck
from qiskit_aer import AerSimulator
from scipy.stats import chisquare

from bench import build as B
from bench.exact import exact_category_distribution
from derisk.d0_check import check
from derisk.d1_run import git_head

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def validate_refs(model, qs, shots=10 ** 6, seed=11):
    sim = AerSimulator(seed_simulator=seed)
    rows = []
    for e, (sub, q) in enumerate(zip(model.subexperiments, qs)):
        res = sim.run(sub.circuit, shots=shots).result().get_counts()
        n = np.zeros(sub.size)
        for k, v in res.items():
            n[sub.category_index(k)] += v
        tv = 0.5 * float(np.abs(n / shots - q).sum())
        mask = q * shots >= 5
        obs = n[mask]
        exp = q[mask] * shots
        # pool the low-expectation cells into one bin
        lo_obs, lo_exp = shots - obs.sum(), shots - exp.sum()
        if lo_exp >= 5:
            obs = np.append(obs, lo_obs)
            exp = np.append(exp, lo_exp)
        exp = exp * obs.sum() / exp.sum()
        chi = chisquare(obs, exp)
        outside = float(n[q <= 0].sum())          # shots observed in categories of exact probability 0
        rows.append(dict(e=e, Z=sub.size, tv=tv, chi2_p=float(chi.pvalue), dof=int(len(obs) - 1),
                         shots_in_zero_prob_cells=outside))
    return rows


def main(names):
    out = dict(commit=git_head(), records=[])
    for nm in names:
        man, qc_cut, qc_plain, qubits = B.make(nm)
        rec, model, qs, p_exact = check(nm, qc_cut, qc_plain, qubits, shots=10 ** 4, seed=7, out=lambda r: None)
        val = validate_refs(model, qs)
        rec['ref_validation'] = dict(max_tv=max(r['tv'] for r in val), min_chi2_p=min(r['chi2_p'] for r in val),
                                     n_chi2_p_below_0_01=sum(1 for r in val if r['chi2_p'] < 0.01),
                                     shots_in_zero_prob_cells=sum(r['shots_in_zero_prob_cells'] for r in val),
                                     per_subexperiment=val)
        rec['manifest_sha256'] = json.load(open(os.path.join(RES, 'manifest_%s.json' % nm)))['manifest_sha256']
        out['records'].append(rec)
        print(nm, {k: rec[k] for k in ('groups', 'subexperiments', 'max_abs_exact_vs_uncut', 'max_abs_ours_vs_qcut',
                                       'qcut_top10_values_match', 'kappa')}, rec['ref_validation']['max_tv'],
              rec['ref_validation']['min_chi2_p'], rec['ref_validation']['shots_in_zero_prob_cells'], flush=True)
    with open(os.path.join(RES, 'd0_reproduction.json'), 'w') as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv[1:])
