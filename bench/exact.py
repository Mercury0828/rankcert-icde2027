"""Exact references: the exact joint distribution q_e over ALL classical bits of a subexperiment circuit
(output bits and every mid-circuit QPD measurement record), and the exact uncut distribution.

Method (noiseless): deferred measurement. A measurement that is not the last operation on its qubit is replaced by a CX
onto a fresh ancilla that is read at the end; the joint distribution of all recorded bits is unchanged (principle of
deferred measurement; valid because the circuits contain no reset and no classically controlled operation, which is
asserted). A final measurement reads its qubit directly. The joint distribution is read from the exact statevector.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

_IGNORED = {'barrier', 'delay'}


def exact_category_distribution(qc: QuantumCircuit) -> np.ndarray:
    """Exact q_e as a dense vector over categories z (bit j of z = classical bit j of the circuit)."""
    last_on_qubit = {}
    for idx, inst in enumerate(qc.data):
        if inst.operation.name in _IGNORED:
            continue
        for q in inst.qubits:
            last_on_qubit[qc.find_bit(q).index] = idx
    n_mid = sum(1 for idx, inst in enumerate(qc.data)
                if inst.operation.name == 'measure' and last_on_qubit[qc.find_bit(inst.qubits[0]).index] != idx)
    anc = QuantumRegister(n_mid, 'anc') if n_mid else None
    work = QuantumCircuit(qc.num_qubits + n_mid)
    reader = {}                       # clbit index -> work-qubit index holding its final value
    a = qc.num_qubits
    for idx, inst in enumerate(qc.data):
        op = inst.operation
        name = op.name
        if name in _IGNORED:
            continue
        if name == 'reset':
            raise ValueError('reset not supported by the deferred-measurement reference')
        if getattr(op, 'condition', None) is not None:
            raise ValueError('classically controlled operation not supported')
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if name == 'measure':
            q = qs[0]
            c = qc.find_bit(inst.clbits[0]).index
            if last_on_qubit[q] == idx:
                reader[c] = q
            else:
                work.cx(q, a)
                reader[c] = a
                a += 1
            continue
        if inst.clbits:
            raise ValueError('operation %s writes classical bits' % name)
        work.append(op, qs)
    sv = Statevector.from_instruction(work)
    C = qc.num_clbits
    written = [c for c in range(C) if c in reader]
    # A classical bit that no instruction writes keeps its initial value 0 on Aer and on IBM backends. QCut keeps such
    # bits when a QPD term measures on one cut of a register but not on another (compact_qpd_register handles only the
    # all-or-nothing case); its estimator then reads the bit as 0, i.e. a constant sign -1. Model it as constant 0.
    probs_w = sv.probabilities(qargs=[reader[c] for c in written])
    probs_w = np.clip(probs_w, 0.0, None)
    probs_w = probs_w / probs_w.sum()
    if len(written) == C:
        return probs_w
    full = np.zeros(1 << C)
    idx = np.zeros(len(probs_w), dtype=np.int64)
    w = np.arange(len(probs_w), dtype=np.int64)
    for t, c in enumerate(written):
        idx |= ((w >> t) & 1) << c
    full[idx] = probs_w
    return full


def uncut_distribution(qc: QuantumCircuit, qubits: list) -> np.ndarray:
    """Exact distribution of the uncut circuit (measurements removed) over `qubits`: bit b of x = qubits[b]."""
    clean = qc.remove_final_measurements(inplace=False)
    sv = Statevector.from_instruction(clean)
    return sv.probabilities(qargs=list(qubits))
