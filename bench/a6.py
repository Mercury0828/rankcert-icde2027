"""A6 peaked benchmark family: deterministic builders, parameter table, manifests (derisk/BENCH_A6_PREREG.md v2).

python -m bench.a6 list                      -> the instance table (no building)
python -m bench.a6 build <name> [<name> ...] -> exact references + manifests
Development seeds 1001-1005, held-out 1006-1010. Held-out exact distributions and descriptors are written only to the
embargo directory D:/runs/rankcert/a6/embargo/ (never to the repository) until the A6-FREEZE commit exists.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
import time

import numpy as np
import QCut as ck
from qiskit import QuantumCircuit

from bench import circuits as C
from bench.build import strip, sure_counts
from bench.exact import exact_category_distribution, uncut_distribution
from rankcert.recon.qcut_adapter import build_model, exact_options

A6 = os.environ.get('RANKCERT_A6', 'D:/runs/rankcert/a6')
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'derisk', 'results', 'a6')
FAMILY_ID = {'arith': 1, 'qaoa2': 2, 'bv': 3}
DEV = tuple(range(1001, 1006))
HELD = tuple(range(1006, 1011))


# ------------------------------------------------------------------ parameter table
def table():
    rows = []
    for nbits in (7, 9, 11):
        for s in DEV + HELD:
            rows.append(dict(name='arith%d_s%d' % (2 * nbits + 2, s), family='arith', nbits=nbits, seed=s))
    for n in (20, 24):
        for s in DEV + HELD:
            rows.append(dict(name='qaoa2_%d_s%d' % (n, s), family='qaoa2', n=n, seed=s))
    for n in (16, 20):
        for s in (1001, 1002):
            rows.append(dict(name='bv%d_s%d' % (n, s), family='bv', n=n, seed=s))
    for r in rows:
        r['split'] = 'held-out' if r['seed'] in HELD else 'development'
    return {r['name']: r for r in rows}


# ------------------------------------------------------------------ F-ARITH
def arith_params(nbits, seed):
    rng = np.random.default_rng([FAMILY_ID['arith'], nbits, seed])
    cut_at = nbits // 2 + 1
    bq = [1 + 2 * i for i in range(nbits)]
    aq = [2 + 2 * i for i in range(nbits)]
    low = sorted(bq[:cut_at] + aq[:cut_at])
    high = sorted(bq[cut_at:] + aq[cut_at:])
    chosen = sorted([int(q) for q in rng.choice(low, 2, replace=False)] + [int(q) for q in rng.choice(high, 2, replace=False)])
    thetas = rng.uniform(math.pi / 8, 3 * math.pi / 8, size=4)
    fixed_a = int(rng.integers(0, 1 << nbits))
    fixed_b = int(rng.integers(0, 1 << nbits))
    return dict(nbits=nbits, cut_at=cut_at, ry={q: float(t) for q, t in zip(chosen, thetas)}, fixed_a=fixed_a, fixed_b=fixed_b)


def make_arith(nbits, seed):
    pr = arith_params(nbits, seed)
    mk = lambda cut: C.cuccaro_adder(nbits, [], pr['fixed_a'], pr['fixed_b'], pr['cut_at'], cut=cut, ry_inputs=pr['ry'])
    return pr, mk(True), mk(False)


# ------------------------------------------------------------------ F-QAOA2
def edge_zz_p2(E, a, b, g, bt):
    """Exact <Z_a Z_b> for p = 2 QAOA by the backward light cone: layer-2 cost gates incident to {a, b}, layer-2 mixer on
    {a, b}; layer-1 mixer on S1 = {a, b} + neighbours; layer-1 cost gates incident to S1 (support up to distance 2)."""
    from qiskit.quantum_info import SparsePauliOp, Statevector
    nb = {}
    for u, v in E:
        nb.setdefault(u, set()).add(v)
        nb.setdefault(v, set()).add(u)
    S1 = {a, b} | nb.get(a, set()) | nb.get(b, set())
    c1 = [e for e in E if e[0] in S1 or e[1] in S1]
    c2 = [e for e in E if a in e or b in e]
    verts = sorted({v for e in c1 for v in e} | S1)
    loc = {v: i for i, v in enumerate(verts)}
    qc = QuantumCircuit(len(verts))
    qc.h(range(len(verts)))
    for u, v in c1:
        qc.rzz(-g[0], loc[u], loc[v])
    qc.rx(2 * bt[0], [loc[v] for v in sorted(S1)])
    for u, v in c2:
        qc.rzz(-g[1], loc[u], loc[v])
    qc.rx(2 * bt[1], [loc[a], loc[b]])
    lab = ['I'] * len(verts)
    lab[len(verts) - 1 - loc[a]] = 'Z'
    lab[len(verts) - 1 - loc[b]] = 'Z'
    return float(np.real(Statevector.from_instruction(qc).expectation_value(SparsePauliOp(''.join(lab)))))


def expected_cut_p2(edges, cross, g, bt):
    E = list(edges) + list(cross)
    return float(sum((1 - edge_zz_p2(E, a, b, g, bt)) / 2 for a, b in E))


def optimize_p2(n, edges, cross):
    from scipy.optimize import minimize
    g1, b1, _ = C.optimize_qaoa_p1(n, edges, cross)
    best = None
    trace = []
    for gm in (0.75, 1.0, 1.25):
        for bm in (0.75, 1.0, 1.25):
            x0 = [gm * g1, gm * g1, bm * b1, bm * b1]
            res = minimize(lambda t: -expected_cut_p2(edges, cross, t[:2], t[2:]), x0, method='Nelder-Mead',
                           options=dict(xatol=1e-4, fatol=1e-6, maxiter=400))
            val = float(-res.fun)
            trace.append(dict(start=[gm, bm], value=val, x=[float(v) for v in res.x], nit=int(res.nit)))
            if best is None or val > best[0]:
                best = (val, [float(v) for v in res.x])
    return dict(p1=(g1, b1), gammas=best[1][:2], betas=best[1][2:], expected_cut=best[0], trace=trace)


def make_qaoa2(n, seed):
    edges, cross = C.two_cluster_graph(n, 1, seed)
    opt = optimize_p2(n, edges, cross)
    pr = dict(n=n, edges=edges, cross=cross, **opt)
    qc_cut = C.qaoa_circuit(n, edges, cross, opt['gammas'], opt['betas'], cut=True)
    qc_plain = C.qaoa_circuit(n, edges, cross, opt['gammas'], opt['betas'], cut=False)
    return pr, qc_cut, qc_plain


# ------------------------------------------------------------------ F-SANITY (Bernstein-Vazirani)
def make_bv(n, seed):
    from QCut import cut as wire_cut
    rng = np.random.default_rng([FAMILY_ID['bv'], n, seed])
    w = n // 2
    s = sorted(int(i) for i in rng.choice(n, w, replace=False))

    def mk(cut):
        qc = QuantumCircuit(n + 1)
        anc = n
        qc.x(anc)
        qc.h(range(n + 1))
        for j, i in enumerate(s):
            if cut and j == math.ceil(w / 2):
                qc.append(wire_cut(), [anc])
            qc.cx(i, anc)
        qc.h(range(n))
        return qc
    return dict(n=n, hidden=s), mk(True), mk(False)


# ------------------------------------------------------------------ build
def make(name):
    r = table()[name]
    if r['family'] == 'arith':
        pr, qc_cut, qc_plain = make_arith(r['nbits'], r['seed'])
        qubits = list(range(qc_plain.num_qubits))
    elif r['family'] == 'qaoa2':
        pr, qc_cut, qc_plain = make_qaoa2(r['n'], r['seed'])
        qubits = list(range(r['n']))
    else:
        pr, qc_cut, qc_plain = make_bv(r['n'], r['seed'])
        qubits = list(range(r['n']))
    return r, pr, qc_cut, qc_plain, qubits


def build(name):
    t0 = time.time()
    r, pr, qc_cut, qc_plain, qubits = make(name)
    cc = ck.get_locations_and_subcircuits(qc_cut, options=exact_options())
    ex = ck.get_experiment_circuits(cc, qubits=qubits)
    model = build_model(ex)
    m = len(qubits)
    widths = [s.num_qubits for s in cc.subcircuits]
    ncut = len(cc.cut_locations)
    eligible = (1 <= ncut <= 3) and max(widths) <= 25 and m <= 24
    man = dict(name=name, row=r, params={k: v for k, v in pr.items() if k != 'trace'}, m=m, cuts=ncut,
               cut_locations=[str(c) for c in cc.cut_locations], subcircuit_widths=widths, groups=model.groups,
               subexperiments=len(model.subexperiments), blocks=model.blocks,
               coefficients=[float(c) for c in model.coefficients], eligible=bool(eligible),
               circuit_sha256=C.manifest_hash(dict(qasm=str(qc_plain.data), cut=str(qc_cut.data))))
    if 'trace' in pr:
        man['optimizer_trace_sha256'] = C.manifest_hash(pr['trace'])
    os.makedirs(RES, exist_ok=True)
    if not eligible:
        man['reason'] = 'structural: cuts=%d, max width=%d, m=%d' % (ncut, max(widths), m)
        json.dump(man, open(os.path.join(RES, 'manifest_%s.json' % name), 'w'), indent=1, default=str)
        return man
    qs = [exact_category_distribution(s.circuit) for s in model.subexperiments]
    p = model.dense_p(qs)
    pu = uncut_distribution(qc_plain, qubits)
    integ = dict(max_abs_exact_vs_uncut=float(np.max(np.abs(p - pu))), p_sum=float(p.sum()))
    man['integrity'] = integ
    man['integrity_ok'] = bool(integ['max_abs_exact_vs_uncut'] <= 1e-10)
    desc = dict(top20=[(int(x), float(p[x])) for x in np.argsort(-p)[:20]],
                ceilings={str(k): sure_counts(p, k) for k in (5, 10)}, max_p=float(p.max()),
                top10_mass=float(np.sort(p)[::-1][:10].sum()))
    held = r['split'] == 'held-out'
    refdir = os.path.join(A6, 'embargo' if held else 'ref')
    os.makedirs(refdir, exist_ok=True)
    with open(os.path.join(refdir, name + '.pkl'), 'wb') as f:
        pickle.dump(dict(model=strip(model), qs=qs, p=p, manifest=man, descriptors=desc), f)
    if held:
        man['descriptors'] = 'EMBARGOED (see %s until A6-FREEZE)' % refdir
    else:
        man['descriptors'] = desc
    man['build_seconds'] = round(time.time() - t0, 1)
    json.dump(man, open(os.path.join(RES, 'manifest_%s.json' % name), 'w'), indent=1, default=str)
    return man


if __name__ == '__main__':
    if sys.argv[1] == 'list':
        for k, v in table().items():
            print(k, v)
    else:
        for nm in sys.argv[2:]:
            mm = build(nm)
            print(nm, dict(eligible=mm['eligible'], cuts=mm['cuts'], widths=mm['subcircuit_widths'],
                           integrity_ok=mm.get('integrity_ok'), seconds=mm.get('build_seconds')), flush=True)
