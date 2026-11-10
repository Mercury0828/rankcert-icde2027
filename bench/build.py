"""Build the initial benchmark instances, their cut models, exact references and manifests.

Usage: python -m bench.build <name> [<name> ...]   names: qaoa20 qaoa24 adder16 ghz20 qaoa32 qaoa40
Writes ./data/phase0/ref/<name>.npz (exact q_e per distinct subexperiment, exact p if m <= 24) and
derisk/results/manifest_<name>.json (circuit + cut manifest and its SHA-256).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import QCut as ck

from bench import circuits as C
from bench.exact import exact_category_distribution, uncut_distribution
from rankcert.recon.qcut_adapter import build_model, exact_options

RUN = os.environ.get('RANKCERT_RUN', './data/phase0')
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'derisk', 'results')

SPECS = {
    'qaoa20': dict(kind='qaoa', n=20, crossing=1, seed=101),
    'qaoa24': dict(kind='qaoa', n=24, crossing=2, seed=202),
    'qaoa32': dict(kind='qaoa', n=32, crossing=1, seed=303),
    'qaoa40': dict(kind='qaoa', n=40, crossing=1, seed=404),
    'adder16': dict(kind='adder', nbits=7, hadamard=[2, 4, 1], fixed_a=53, fixed_b=75, cut_at=4),
    'ghz20': dict(kind='ghz', n=20),
}


def make(name):
    sp = SPECS[name]
    man = dict(name=name, spec=sp)
    if sp['kind'] == 'qaoa':
        edges, cross = C.two_cluster_graph(sp['n'], sp['crossing'], sp['seed'])
        g, b, val = C.optimize_qaoa_p1(sp['n'], edges, cross)
        man.update(edges=edges, cross=cross, gamma=g, beta=b, expected_cut=val,
                   max_cut=None if sp['n'] > 24 else int(C.maxcut_values(sp['n'], edges + cross).max()))
        qc_cut = C.qaoa_circuit(sp['n'], edges, cross, [g], [b], cut=True)
        qc_plain = C.qaoa_circuit(sp['n'], edges, cross, [g], [b], cut=False)
    elif sp['kind'] == 'adder':
        qc_cut = C.cuccaro_adder(sp['nbits'], sp['hadamard'], sp['fixed_a'], sp['fixed_b'], sp['cut_at'], cut=True)
        qc_plain = C.cuccaro_adder(sp['nbits'], sp['hadamard'], sp['fixed_a'], sp['fixed_b'], sp['cut_at'], cut=False)
    else:
        qc_cut, qc_plain = C.ghz(sp['n'], cut=True), C.ghz(sp['n'], cut=False)
    n = qc_plain.num_qubits
    qubits = list(range(n))
    return man, qc_cut, qc_plain, qubits


def build(name):
    t0 = time.time()
    man, qc_cut, qc_plain, qubits = make(name)
    cc = ck.get_locations_and_subcircuits(qc_cut, options=exact_options())
    ex = ck.get_experiment_circuits(cc, qubits=qubits)
    model = build_model(ex)
    qs = [exact_category_distribution(s.circuit) for s in model.subexperiments]
    m = len(qubits)
    man.update(m=m, groups=model.groups, subexperiments=len(model.subexperiments),
               subcircuit_widths=[s.num_qubits for s in cc.subcircuits], cut_locations=[str(c) for c in cc.cut_locations],
               blocks=model.blocks, Z_sizes=[s.size for s in model.subexperiments],
               coefficients=[float(c) for c in model.coefficients], kappa=float(2 * np.abs(model.coefficients).sum()),
               support_exact=[int((q > 1e-15).sum()) for q in qs])
    out = dict(qs=np.array(qs, dtype=object))
    if m <= 24:
        p = model.dense_p(qs)
        man['p_min'] = float(p.min())
        man['p_sum'] = float(p.sum())
        if m <= 28:
            pu = uncut_distribution(qc_plain, qubits)
            man['max_abs_exact_vs_uncut'] = float(np.max(np.abs(p - pu)))
        order = np.argsort(-p)
        man['top20'] = [(int(x), float(p[x])) for x in order[:20]]
        man['ceilings'] = {str(k): sure_counts(p, k) for k in (1, 5, 10, 20)}
        man['threshold_counts'] = {str(t): int((p > t).sum()) for t in (0.01, 0.05)}
        out['p'] = p
    manifest_core = {k: man[k] for k in ('name', 'spec', 'm', 'groups', 'subexperiments', 'cut_locations', 'blocks',
                                         'coefficients') if k in man}
    for k in ('edges', 'cross', 'gamma', 'beta'):
        if k in man:
            manifest_core[k] = man[k]
    man['manifest_sha256'] = C.manifest_hash(manifest_core)
    man['build_seconds'] = round(time.time() - t0, 1)
    os.makedirs(os.path.join(RUN, 'ref'), exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RUN, 'ref', name + '.pkl'), 'wb') as f:
        pickle.dump(dict(model=strip(model), qs=qs, p=out.get('p'), manifest=man), f)
    with open(os.path.join(RES, 'manifest_%s.json' % name), 'w') as f:
        json.dump(man, f, indent=1)
    return man


def strip(model):
    """Model without circuit objects (picklable, small)."""
    import copy
    m2 = copy.copy(model)
    m2.subexperiments = [copy.copy(s) for s in model.subexperiments]
    for s in m2.subexperiments:
        s.circuit = None
        s.key = None
    return m2


def sure_counts(p, k, rtol=1e-9):
    """Number of sure members / sure non-members of the top-k under p, ties grouped at tolerance rtol * max p."""
    tol = rtol * float(np.max(np.abs(p)))
    ps = np.sort(p)[::-1]
    # sure member: |{y != x : p(y) >= p(x) - tol}| <= k - 1 ; sure non-member: |{y : p(y) > p(x) + tol}| >= k
    asc = ps[::-1]
    ge = len(p) - np.searchsorted(asc, p - tol, side='left')      # count of y with p(y) >= p(x) - tol (incl. x)
    gt = len(p) - np.searchsorted(asc, p + tol, side='right')     # count of y with p(y) > p(x) + tol
    sure_in = int(((ge - 1) <= k - 1).sum())
    sure_out = int((gt >= k).sum())
    return dict(sure_members=sure_in, sure_non_members=sure_out, neither=int(len(p) - sure_in - sure_out))


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        man = build(nm)
        print(json.dumps({k: man[k] for k in man if k not in ('edges', 'top20', 'coefficients', 'Z_sizes',
                                                               'support_exact', 'blocks')}, default=str))
