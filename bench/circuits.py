"""Benchmark circuits for Phase 0 (guide §7a D1). Every builder returns the circuit twice: with QCut cut markers on the
declared cut gates (`cut=True`) and with the plain gates (`cut=False`, for the uncut exact reference).

Graphs for QAOA are built so that the declared cuts separate them (guide §7a D1):
  (i)  two random 3-regular graphs on n/2 vertices, one edge removed in each, one bridge edge between the two freed
       vertex pairs' endpoints -> exactly one crossing edge -> one gate cut;
  (ii) two random 3-regular graphs on n/2 vertices joined by a double-edge swap (remove (a,b) in cluster A and (c,d) in
       cluster B, add (a,c) and (b,d)) -> 3-regular, exactly two crossing edges -> two gate cuts.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, RZZGate

import QCut as ck


# ------------------------------------------------------------------ graphs
def random_regular(n: int, d: int, rng: np.random.Generator, max_tries: int = 10000) -> list:
    """Uniform-ish random simple d-regular graph by the pairing model with rejection; connected."""
    for _ in range(max_tries):
        stubs = np.repeat(np.arange(n), d)
        rng.shuffle(stubs)
        pairs = stubs.reshape(-1, 2)
        edges = set()
        ok = True
        for a, b in pairs:
            a, b = int(a), int(b)
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if ok and _connected(n, edges):
            return sorted(edges)
    raise RuntimeError('no regular graph found')


def _connected(n, edges):
    adj = {i: set() for i in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen, stack = {0}, [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == n


def two_cluster_graph(n: int, crossing: int, seed: int) -> tuple:
    """Return (edges, crossing_edges) on vertices 0..n-1; cluster A = 0..n/2-1, cluster B = n/2..n-1."""
    rng = np.random.default_rng(seed)
    h = n // 2
    A = random_regular(h, 3, rng)
    B = [(a + h, b + h) for a, b in random_regular(h, 3, rng)]
    if crossing == 1:
        ea = A.pop(int(rng.integers(len(A))))
        eb = B.pop(int(rng.integers(len(B))))
        cross = [(ea[0], eb[0])]
    elif crossing == 2:
        ea = A.pop(int(rng.integers(len(A))))
        eb = B.pop(int(rng.integers(len(B))))
        cross = [(ea[0], eb[0]), (ea[1], eb[1])]
    else:
        raise ValueError(crossing)
    return sorted(A + B), cross


def maxcut_values(n: int, edges: list) -> np.ndarray:
    """Cut size of every x in {0,1}^n (bit v of x = side of vertex v)."""
    x = np.arange(1 << n, dtype=np.int64)
    cut = np.zeros(1 << n, dtype=np.int64)
    for a, b in edges:
        cut += ((x >> a) ^ (x >> b)) & 1
    return cut


def qaoa_circuit(n: int, edges: list, cross: list, gammas, betas, cut: bool) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for a, b in edges:
            qc.rzz(-gamma, a, b)
        for a, b in cross:
            if cut:
                qc.append(**ck.cutGate(RZZGate(-gamma), a, b))
            else:
                qc.rzz(-gamma, a, b)
        qc.rx(2 * beta, range(n))
    return qc


def edge_zz_p1(edges_all: list, a: int, b: int, gamma: float, beta: float) -> float:
    """Exact <Z_a Z_b> for p = 1 QAOA by the light cone: only cost gates incident to a or b matter."""
    from qiskit.quantum_info import Statevector, SparsePauliOp
    inc = [e for e in edges_all if a in e or b in e]
    verts = sorted({v for e in inc for v in e})
    loc = {v: i for i, v in enumerate(verts)}
    qc = QuantumCircuit(len(verts))
    qc.h(range(len(verts)))
    for u, v in inc:
        qc.rzz(-gamma, loc[u], loc[v])
    qc.rx(2 * beta, [loc[a], loc[b]])
    lab = ['I'] * len(verts)
    lab[len(verts) - 1 - loc[a]] = 'Z'
    lab[len(verts) - 1 - loc[b]] = 'Z'
    return float(np.real(Statevector.from_instruction(qc).expectation_value(SparsePauliOp(''.join(lab)))))


def qaoa_expected_cut(n, edges, cross, gammas, betas) -> float:
    """Exact expected cut for p = 1 (light cone); sum over edges of (1 - <ZZ>) / 2."""
    assert len(gammas) == 1
    E = list(edges) + list(cross)
    return float(sum((1 - edge_zz_p1(E, a, b, gammas[0], betas[0])) / 2 for a, b in E))


def optimize_qaoa_p1(n, edges, cross, grid: int = 12) -> tuple:
    """Classical optimizer for p = 1 angles: grid search + Nelder-Mead on the exact (light-cone) expected cut."""
    from scipy.optimize import minimize
    best = (-1.0, 0.0, 0.0)
    for g in np.linspace(0.05, np.pi - 0.05, grid):
        for b in np.linspace(0.05, np.pi / 2 - 0.05, grid // 2):
            v = qaoa_expected_cut(n, edges, cross, [g], [b])
            if v > best[0]:
                best = (v, g, b)
    res = minimize(lambda t: -qaoa_expected_cut(n, edges, cross, [t[0]], [t[1]]), [best[1], best[2]],
                   method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-8, maxiter=300))
    return float(res.x[0]), float(res.x[1]), float(-res.fun)


# ------------------------------------------------------------------ adder (Cuccaro ripple-carry, CutQC family)
def cuccaro_adder(nbits: int, hadamard_inputs: list, fixed_a: int, fixed_b: int, cut_at: int, cut: bool,
                  ry_inputs: dict | None = None):
    """Ripple-carry adder (Cuccaro et al. MAJ / UMA): |c0>|b>|a>|z> -> |c0>|a+b>|a>|z xor carry>.

    Qubits: 0 = c0; for bit i: b_i = 1 + 2i, a_i = 2 + 2i; z = 2 nbits + 1. `hadamard_inputs` are put in uniform
    superposition (several equal output peaks); other inputs take the bits of fixed_a / fixed_b.
    Cut (cut=True): the carry wire a_{cut_at-1} is the only wire shared by bits < cut_at and bits >= cut_at; it is cut
    twice with QCut wire cuts, before MAJ(cut_at) and after UMA(cut_at), so the segment between the cuts belongs to the
    high part -> two wire cuts, two subcircuits.
    """
    from QCut import cut as wire_cut
    n = 2 * nbits + 2
    qc = QuantumCircuit(n)
    c0, z = 0, n - 1
    bq = [1 + 2 * i for i in range(nbits)]
    aq = [2 + 2 * i for i in range(nbits)]
    for i in range(nbits):
        if (fixed_a >> i) & 1 and aq[i] not in hadamard_inputs and aq[i] not in (ry_inputs or {}):
            qc.x(aq[i])
        if (fixed_b >> i) & 1 and bq[i] not in hadamard_inputs and bq[i] not in (ry_inputs or {}):
            qc.x(bq[i])
    for q in hadamard_inputs:
        qc.h(q)
    for q, theta in sorted((ry_inputs or {}).items()):     # A6 F-ARITH: graded uncertain inputs (default: none)
        qc.ry(theta, q)

    def maj(x, y, w):
        qc.cx(w, y); qc.cx(w, x); qc.ccx(x, y, w)

    def uma(x, y, w):
        qc.ccx(x, y, w); qc.cx(w, x); qc.cx(x, y)

    carry = aq[cut_at - 1]
    for i in range(nbits):
        if cut and i == cut_at:
            qc.append(wire_cut(), [carry])
        maj(c0 if i == 0 else aq[i - 1], bq[i], aq[i])
    qc.cx(aq[nbits - 1], z)
    for i in reversed(range(nbits)):
        uma(c0 if i == 0 else aq[i - 1], bq[i], aq[i])
        if cut and i == cut_at:
            qc.append(wire_cut(), [carry])
    return qc


# ------------------------------------------------------------------ GHZ sanity
def ghz(n: int, cut: bool) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        if cut and i == n // 2 - 1:
            qc.append(**ck.cutGate(CXGate(), i, i + 1))
        else:
            qc.cx(i, i + 1)
    return qc


def manifest_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
