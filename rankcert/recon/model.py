"""Cut model: the separable affine form of a QCut (2.4.0) reconstruction, built from the experiment metadata.

Notation follows guide §P4. For every group g and subcircuit i the factor weight is a signed linear functional of the raw
category distribution q_e of the subexperiment e = e(i, g) that QCut actually ran (after deduplication):

    u_{i,g}(y) = sum_{z in Z_e} [out_i(z) = y] * sign(z) * q_e(z)

where z is a raw category (all classical bits of the subexperiment circuit: its output bits and its QPD measurement bits),
out_i(z) are the bits of z that carry the queried qubits held by subcircuit i (indexed by *measured* bit values, i.e. the
complement of QCut's internal mode index), and sign(z) = (-1)^dropped * prod_{written qpd bits b} (+1 if b == '1' else -1).
A subcircuit that holds no queried qubit contributes the scalar functional E_{i,g} = sum_z sign(z) q_e(z).

The reconstruction (QCut theory page; distribution.py / probabilities.py at v2.4.0) is, with parity = (-1)^(w+1),
w = number of wire cuts, and s_g = parity * c_g * prod_{i without held bits} E_{i,g}:

    T'(x) := sum_g s_g prod_{i held} u_{i,g}(x_i)          (= QCut's T[complement x])
    p(x)  = (1 + sum_y T'(y)) / 2^m - T'(x),   sum_y T'(y) = sum_g s_g prod_{i held} sum_{y_i} u_{i,g}(y_i).

Everything here is computed from our own reading of the raw counts; QCut's estimator is used only as the object we compare
against (D0).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Subexperiment:
    """One distinct circuit that is run (after QCut's deduplication)."""
    key: tuple                      # identity: (id(result object), index, scale, label_filter) at build time
    num_clbits: int                 # total classical bits of the circuit = log2 |Z_e|
    written: int                    # QPD bits written by the circuit (rightmost `written` chars of a counts key)
    out_width: int                  # number of non-QPD (output) bits = num_clbits - written
    circuit: object = None          # the qiskit circuit (for exact references)

    @property
    def size(self) -> int:
        return 1 << self.num_clbits

    def category_index(self, key: str) -> int:
        """Counts key ('01 1' style, spaces removed, leftmost char = highest bit) -> integer category index."""
        bits = key.replace(' ', '')
        assert len(bits) == self.num_clbits, (key, self.num_clbits)
        return int(bits, 2)

    def signs(self, dropped: int = 0) -> np.ndarray:
        """sign(z) for every category z (dense, length |Z_e|); `dropped` constant -1 bits belong to the use."""
        z = np.arange(self.size)
        qpd = z & ((1 << self.written) - 1)            # the rightmost `written` characters are the low bits
        ones = np.zeros(self.size, dtype=np.int64)
        for b in range(self.written):
            ones += (qpd >> b) & 1
        zeros = self.written - ones
        return ((-1.0) ** dropped) * ((-1.0) ** zeros)

    def out_bits(self) -> np.ndarray:
        """Output-bit string (as an integer whose bit t = char t from the right of the output field) per category."""
        z = np.arange(self.size)
        return z >> self.written


@dataclass
class Use:
    """Use of subexperiment e by group g on subcircuit i."""
    e: int
    held: list                      # positions b of the queried outcome held by this subcircuit (ascending)
    offsets: list                   # for each held position, the char index (from the left) inside the output field
    dropped: int = 0                # QPD bits of this use that are constant -1 (QCut's qpd_bits[(g, obs, i)][1])


@dataclass
class CutModel:
    m: int                                  # width of the queried outcome
    blocks: list                            # per held subcircuit (in model order): list of outcome positions
    coefficients: np.ndarray                # parity * c_g per group
    subexperiments: list                    # list[Subexperiment]
    uses: list                              # uses[g] = list over ALL subcircuits of Use (held may be [])
    block_of_sub: list = field(default_factory=list)   # per subcircuit index: block index or -1 (no held bits)

    @property
    def groups(self) -> int:
        return len(self.coefficients)

    # ---- functionals -------------------------------------------------------------------------------------------
    def functional_matrix(self, g: int, i: int) -> np.ndarray:
        """A dense (2^{m_i}, |Z_e|) matrix A with u_{i,g}(y) = A[y] @ q_e, entries in {-1, 0, +1}.

        Row index y uses the outcome-position order of block_of_sub[i]: bit t of y = outcome position blocks[b][t].
        For a subcircuit without held bits the matrix has one row (the scalar functional E_{i,g}).
        """
        use = self.uses[g][i]
        sub = self.subexperiments[use.e]
        sg = sub.signs(use.dropped)
        if not use.held:
            return sg[None, :]
        outs = sub.out_bits()
        W = sub.out_width
        width = len(use.held)
        y = np.zeros(sub.size, dtype=np.int64)
        for t, off in enumerate(use.offsets):
            # char `off` from the left of the output field = bit (W-1-off) of the output integer
            y |= ((outs >> (W - 1 - off)) & 1) << t
        A = np.zeros((1 << width, sub.size))
        A[y, np.arange(sub.size)] = sg
        return A

    def row_index(self, g: int, i: int) -> tuple:
        """Cached sparse form of functional_matrix(g, i): (row index of every category, its sign, number of rows).
        Every column of the functional matrix has exactly one nonzero, so u = bincount(row, sign * q)."""
        cache = self.__dict__.setdefault('_row_cache', {})
        use = self.uses[g][i]
        key = (use.e, tuple(use.held), tuple(use.offsets), use.dropped)
        if key not in cache:
            # built directly (no dense matrix): row of category z = its local output value y(z), entry = sign(z)
            sub = self.subexperiments[use.e]
            signs = sub.signs(use.dropped)
            if not use.held:
                rows = np.zeros(sub.size, dtype=np.int64)
                R = 1
            else:
                outs = sub.out_bits()
                W = sub.out_width
                rows = np.zeros(sub.size, dtype=np.int64)
                for t, off in enumerate(use.offsets):
                    rows |= ((outs >> (W - 1 - off)) & 1) << t
                R = 1 << len(use.held)
            cache[key] = (rows, signs, R)
        return cache[key]

    def weights(self, qs: list) -> tuple:
        """Factor weights at given category distributions qs[e] (dense vectors).

        Returns (scal, tabs): scal[g] = s_g (coefficient times scalar functionals), tabs[g][b] = vector u_{i,g} over the
        2^{m_b} values of block b.
        """
        scal = np.array(self.coefficients, dtype=float).copy()
        tabs = []
        for g in range(self.groups):
            row = [None] * len(self.blocks)
            for i, use in enumerate(self.uses[g]):
                rows, signs, R = self.row_index(g, i)
                v = np.bincount(rows, weights=signs * qs[use.e], minlength=R)
                if not use.held:
                    scal[g] *= float(v[0])
                else:
                    row[self.block_of_sub[i]] = v
            tabs.append(row)
        return scal, tabs

    def dense_p(self, qs: list) -> np.ndarray:
        """p(x) for every x in {0,1}^m (dense; m <= ~26). Index x: bit b of x = outcome position b."""
        scal, tabs = self.weights(qs)
        return dense_from_weights(self.m, self.blocks, scal, tabs)


def block_index_map(m: int, blocks: list) -> list:
    """For each block b, an index array giving, for every outcome x (0..2^m-1), the local value x_b."""
    x = np.arange(1 << m, dtype=np.int64)
    maps = []
    for bits in blocks:
        loc = np.zeros(1 << m, dtype=np.int64)
        for t, pos in enumerate(bits):
            loc |= ((x >> pos) & 1) << t
        maps.append(loc)
    return maps


def dense_from_weights(m: int, blocks: list, scal: np.ndarray, tabs: list) -> np.ndarray:
    maps = block_index_map(m, blocks)
    T = np.zeros(1 << m)
    total = 0.0
    for g in range(len(scal)):
        term = np.full(1 << m, scal[g])
        mass = scal[g]
        for b, mp in enumerate(maps):
            term *= tabs[g][b][mp]
            mass *= float(tabs[g][b].sum())
        T += term
        total += mass
    return (1.0 + total) / (1 << m) - T
