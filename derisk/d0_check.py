"""D0 check for one circuit: (1) exact reconstruction limit (our formula at exact q_e) vs uncut statevector;
(2) our plug-in estimate from QCut's raw counts vs QCut 2.4.0's estimate_probabilities (dense array and top());
(3) structure statistics (|G|, |S|, |Z_e|, observed support sizes)."""
from __future__ import annotations

import time

import numpy as np
import QCut as ck
from qiskit_aer import AerSimulator

from bench.exact import exact_category_distribution, uncut_distribution
from rankcert.recon.qcut_adapter import build_model, counts_vectors, exact_options


def check(name, qc_cut, qc_plain, qubits, shots, seed, finder=False, out=print):
    t0 = time.time()
    opts = exact_options()
    cc = ck.get_locations_and_subcircuits(qc_cut, options=opts)
    ex = ck.get_experiment_circuits(cc, qubits=list(qubits))
    model = build_model(ex)
    S = len(model.subexperiments)
    qs = [exact_category_distribution(s.circuit) for s in model.subexperiments]
    p_exact = model.dense_p(qs)
    p_uncut = uncut_distribution(qc_plain, qubits)
    err_ref = float(np.max(np.abs(p_exact - p_uncut)))
    # run through QCut on Aer and compare estimators
    job = ck.submit_experiments(ex, shots=shots, backend=AerSimulator(seed_simulator=seed))
    raw = job.result()
    n = counts_vectors(model, raw)
    qhat = [v / v.sum() for v in n]
    p_ours = model.dense_p(qhat)
    probs = ck.estimate_probabilities(raw)
    p_qcut = np.asarray(probs.probabilities, dtype=float) if not callable(probs.probabilities) else np.asarray(probs.probabilities(), dtype=float)
    err_est = float(np.max(np.abs(p_ours - p_qcut)))
    top_q = probs.top(10)
    m = len(qubits)
    top_ours = sorted(range(1 << m), key=lambda x: -p_ours[x])[:10]
    key = lambda x: format(x, '0%db' % m)
    top_match = all(abs(p_ours[int(k, 2)] - v) < 1e-12 for k, v in top_q.items())
    rec = dict(name=name, m=m, groups=model.groups, subexperiments=S,
               Z_sizes=[s.size for s in model.subexperiments],
               observed_support=[int((v > 0).sum()) for v in n],
               shots_per_subexperiment=shots, total_shots=int(sum(v.sum() for v in n)),
               max_abs_exact_vs_uncut=err_ref, max_abs_ours_vs_qcut=err_est, qcut_top10_values_match=top_match,
               qcut_top10=[(k, v) for k, v in top_q.items()], ours_top10=[(key(x), float(p_ours[x])) for x in top_ours],
               kappa=float(np.abs(model.coefficients).sum() * 2), seconds=round(time.time() - t0, 1),
               negative_estimates=int((p_ours < 0).sum()), sum_p_ours=float(p_ours.sum()))
    out(rec)
    return rec, model, qs, p_exact


if __name__ == '__main__':
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RZZGate
    import sys
    toy_cut = QuantumCircuit(4); toy = QuantumCircuit(4)
    for c, isc in ((toy_cut, True), (toy, False)):
        c.h(range(4)); c.rzz(0.7, 0, 1)
        if isc:
            c.append(**ck.cutGate(RZZGate(0.9), 1, 2))
        else:
            c.rzz(0.9, 1, 2)
        c.rzz(0.5, 2, 3); c.rx(0.4, range(4))
    check('toy4', toy_cut, toy, [0, 1, 2, 3], 2000, 7)
    check('toy4-subset', toy_cut, toy, [2, 0, 3], 2000, 8)
