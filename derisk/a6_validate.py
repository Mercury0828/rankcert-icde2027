"""A6 integrity check 2: Aer-sample validation of the exact q_e.

python -m derisk.a6_validate            -> development seeds 1001-1002 of every (family, size) at 1e6 shots / subexperiment
                                           and held-out seed 1006 of every (family, size) at 1e5 shots / subexperiment
Rule: chi-square with pooled low-expectation cells per subexperiment (derisk/d0_run.validate_refs); Bonferroni-adjusted
minimum p (min p * number of subexperiments, capped at 1) >= 0.01; no shot in a zero-probability cell.
The circuits are rebuilt with the frozen builder and must reproduce the manifest circuit hash; the stored q_e are used.
For held-out instances only the pass / fail flags and the test statistics are written (no distribution descriptors).
Aer runs single-threaded (shared machine). Output: derisk/results/a6/validation.json (resumable per instance).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import QCut as ck
from qiskit_aer import AerSimulator

from bench import circuits as C
from bench.a6 import A6, HELD, RES, make, table
from derisk.d0_run import validate_refs
from derisk.d1_run import git_head
from rankcert.recon.qcut_adapter import build_model, exact_options

OUT = os.path.join(RES, 'validation.json')


def targets():
    out = []
    for name, r in table().items():
        if r['seed'] in (1001, 1002):
            out.append((name, 10 ** 6))
        elif r['seed'] == 1006:
            out.append((name, 10 ** 5))
    return out


def validate(name, shots, seed=11):
    t0 = time.time()
    r, pr, qc_cut, qc_plain, qubits = make(name)
    man = json.load(open(os.path.join(RES, 'manifest_%s.json' % name)))
    h = C.manifest_hash(dict(qasm=str(qc_plain.data), cut=str(qc_cut.data)))
    held = r['seed'] in HELD
    ref = pickle.load(open(os.path.join(A6, 'embargo' if held else 'ref', name + '.pkl'), 'rb'))
    cc = ck.get_locations_and_subcircuits(qc_cut, options=exact_options())
    model = build_model(ck.get_experiment_circuits(cc, qubits=qubits))
    same = (h == man['circuit_sha256'] and len(model.subexperiments) == len(ref['qs'])
            and all(s.size == len(q) for s, q in zip(model.subexperiments, ref['qs'])))
    rec = dict(name=name, split=r['split'], shots=shots, circuit_hash_ok=bool(h == man['circuit_sha256']),
               structure_ok=bool(same))
    if same:
        import derisk.d0_run as d0
        orig = d0.AerSimulator
        d0.AerSimulator = lambda seed_simulator: AerSimulator(seed_simulator=seed_simulator, max_parallel_threads=1)
        try:
            rows = validate_refs(model, ref['qs'], shots=shots, seed=seed)
        finally:
            d0.AerSimulator = orig
        S = len(rows)
        # A test with 0 degrees of freedom (one cell after pooling) has p = NaN and no power: recorded as vacuous; its
        # subexperiment is still covered by the zero-probability-cell rule. A NaN p with dof > 0 fails the check
        # (review: NaN must never pass through min()).
        tested = [x for x in rows if x['dof'] > 0]
        nan = [x['e'] for x in tested if not (x['chi2_p'] == x['chi2_p'])]
        min_p = min((x['chi2_p'] for x in tested if x['chi2_p'] == x['chi2_p']), default=1.0)
        rec.update(subexperiments=S, chi2_tested=len(tested), chi2_vacuous=S - len(tested), chi2_nan=nan,
                   min_chi2_p=min_p, bonferroni_min_p=min(1.0, min_p * len(tested)),
                   max_tv=max(x['tv'] for x in rows),
                   shots_in_zero_prob_cells=float(sum(x['shots_in_zero_prob_cells'] for x in rows)))
        rec['pass'] = bool(not nan and rec['bonferroni_min_p'] >= 0.01 and rec['shots_in_zero_prob_cells'] == 0)
        if not held:
            rec['per_subexperiment'] = rows
    else:
        rec['pass'] = False
    rec['seconds'] = round(time.time() - t0, 1)
    return rec


def main(names=None):
    out = json.load(open(OUT)) if os.path.exists(OUT) else dict(records={})
    out['commit'] = git_head()
    for name, shots in targets():
        if names and name not in names:
            continue
        if name in out['records']:
            continue
        rec = validate(name, shots)
        out['records'][name] = rec
        json.dump(out, open(OUT, 'w'), indent=1)
        print(name, {k: rec.get(k) for k in ('shots', 'structure_ok', 'bonferroni_min_p', 'max_tv',
                                              'shots_in_zero_prob_cells', 'pass', 'seconds')}, flush=True)
    bad = [n for n, r in out['records'].items() if not r['pass']]
    print('a6 validation done; failed: %s' % (bad or 'none'), flush=True)
    if bad:
        raise SystemExit(1)               # run_a6_dev.sh stops: a failed check is diagnosed and repaired first (§2)


if __name__ == '__main__':
    main(sys.argv[1:] or None)
