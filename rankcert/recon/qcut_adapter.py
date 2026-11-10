"""Build a CutModel from a QCut 2.4.0 CutExperiment, and read raw counts into category vectors.

The layout rule (which character of a subcircuit's output field carries which queried qubit) follows the composition
QCut documents in `postprocess._bit_layout`: the circuit-wide outcome is the subcircuits' output fields concatenated
(in subcircuit order), reversed, permuted by `map_qubit`, and reversed again. We re-implement it here; correctness is
validated independently by comparing the exact reconstruction with the uncut statevector distribution (bench.exact).
"""
from __future__ import annotations

import numpy as np

from QCut.cutlocation import SingleQubitCutLocation
from QCut.options import CutOptions
from QCut.utils.circuit_utils import circuit_fingerprint

from .model import CutModel, Subexperiment, Use


def circuit_key(qc) -> tuple:
    """"""
    items = []
    for inst in qc.data:
        op = inst.operation
        params = []
        for p in op.params:
            try:
                params.append(round(float(p), 12))
            except (TypeError, ValueError):
                params.append(str(p))
        qs = tuple(qc.find_bit(q).index for q in inst.qubits)
        cs = tuple(qc.find_bit(c).index for c in inst.clbits)
        items.append((op.name, tuple(params), qs, cs))
    cregs = tuple((r.name, r.size) for r in qc.cregs)
    return (qc.num_qubits, cregs, tuple(items))


def output_layout(widths: list, positions: list, map_qubit) -> list:
    """Per subcircuit: (held outcome positions b, char offsets inside its output field)."""
    starts = np.cumsum([0] + list(widths[:-1]))
    ids = np.concatenate([np.arange(s, s + w) for s, w in zip(starts, widths)][::-1])
    if map_qubit is not None:
        ids = np.array([ids[map_qubit[k]] for k in sorted(map_qubit.keys(), reverse=True)])
    ids = ids[::-1]
    owner = np.concatenate([[s] * w for s, w in enumerate(widths)])
    out = []
    for idx, start in enumerate(starts):
        held = [b for b, p in enumerate(positions) if owner[ids[p]] == idx]
        out.append((held, [int(ids[positions[b]] - start) for b in held]))
    return out


def build_model(experiment) -> CutModel:
    """CutModel from a CutExperiment built with `qubits=` (probability reconstruction)."""
    assert experiment.can_reconstruct_probabilities
    assert not experiment.sampled, "expansion must be exact (A1)"
    assert not experiment.communicates, "communicating wire cuts change shot counts adaptively (A3); not supported"
    qubits = list(experiment.qubits)
    m = len(qubits)
    wire_cuts = len([c for c in experiment.cut_locations if isinstance(c, SingleQubitCutLocation)])
    parity = float((-1) ** (wire_cuts + 1))

    subexps, index = [], {}
    fp_cache: dict = {}
    uses, coefficients = [], []
    layout = None
    for g, (exp_g, coef) in enumerate(zip(experiment.experiments, experiment.coefficients)):
        circuits = exp_g[0]                                    # one measurement setting for Z strings
        row = []
        widths = []
        for i in sorted(circuits):
            qc = circuits[i]
            written, dropped = experiment.qpd_bits[(g, 0, i)]
            meas = [r for r in qc.cregs if r.name == 'meas']
            width = meas[0].size if meas else 0
            widths.append(width)
            # Subexperiment identity = exactly QCut's deduplication key (execution/circuit_knitting.py _deduplicate):
            # (subcircuit index, circuit_fingerprint). A distinct key is a separately submitted circuit, i.e. an
            # independent shot batch (A2).
            key = (i, circuit_fingerprint(qc, fp_cache))
            if key not in index:
                index[key] = len(subexps)
                subexps.append(Subexperiment(key=key, num_clbits=qc.num_clbits, written=written,
                                             out_width=qc.num_clbits - written, circuit=qc))
            e = index[key]
            assert subexps[e].written == written and subexps[e].out_width == width, (g, i)
            row.append(Use(e=e, held=[], offsets=[], dropped=dropped))
        lay = output_layout(widths, qubits, experiment.map_qubit)
        if layout is None:
            layout = [h for h, _ in lay]
        assert [h for h, _ in lay] == layout, "subcircuits hold different bits in different groups"
        for i, (held, offs) in enumerate(lay):
            row[i].held, row[i].offsets = held, offs
        uses.append(row)
        coefficients.append(parity * float(coef))

    blocks, block_of_sub = [], []
    for held in layout:
        if held:
            block_of_sub.append(len(blocks))
            blocks.append(held)
        else:
            block_of_sub.append(-1)
    assert sorted(b for bl in blocks for b in bl) == list(range(m))
    return CutModel(m=m, blocks=blocks, coefficients=np.array(coefficients), subexperiments=subexps,
                    uses=uses, block_of_sub=block_of_sub)


def counts_vectors(model: CutModel, raw_result) -> tuple:
    """Category count vectors n_e and shot counts N_e read from a QCut RawResult (groups x subcircuits).

    Uses of the same subexperiment must carry identical counts (QCut deduplication shares one result).
    """
    n = [None] * len(model.subexperiments)
    for g, row in enumerate(model.uses):
        for i, use in enumerate(row):
            cr = raw_result.results[g][0][i]
            counts = cr.counts()
            sub = model.subexperiments[use.e]
            vec = np.zeros(sub.size)
            for k, v in counts.items():
                vec[sub.category_index(k)] += v
            if n[use.e] is None:
                n[use.e] = vec
            else:
                assert np.array_equal(n[use.e], vec), "two uses of one subexperiment carry different counts"
    return n


def exact_options(**kw) -> CutOptions:
    """The options every RANKCERT experiment uses: exact expansion (A1), no communicating wire cuts (A3)."""
    base = dict(expansion='exact', wire_cut_communication='never')
    base.update(kw)
    return CutOptions(**base)
