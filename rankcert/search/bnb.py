"""Phase-0 branch-and-bound search on interval endpoints (guide §P5.2 T4 ROUTE; QCut's _Search idea on interval
endpoints, credited). Block-level branching: the first s-1 blocks are fixed one at a time (children of a node are all
values of the next block, ordered by bound); the last block is evaluated as a vector ("leaf").

Bounds for a branch B fixing blocks 0..j-1: each group term interval is s_g * prod_{b<j} [u_gb(x_b)] * prod_{b>=j} H_gb
with H_gb = [min_y lo_gb(y), max_y hi_gb(y)] (inclusion-monotone interval arithmetic), so with Hlo_g(B) its lower end,
    U_bar(B) = c0_hi - sum_g Hlo_g(B) >= max_{y in B} U(y),    L_bar(B) = c0_lo - sum_g Hlo_g(B) >= max_{y in B} L(y).

Top-k: phase A finds L_(k), the k-th largest lower bound (best-first on L_bar, stop when no branch can beat it);
phase B collects the candidate set C = {x : U(x) >= L_(k)} (prune branches with U_bar < L_(k): all their strings are
OUT, because OUT(x) <=> |{y : L(y) > U(x)}| >= k <=> U(x) < L_(k)). IN / UND are decided inside C (every y that can
block an IN candidate x has U(y) >= L(x) >= L_(k), hence lies in C).
Threshold tau: C = {x : U(x) >= tau}; IN iff L > tau inside C; everything else OUT.
Strings are in block order (index = sum_b x_b prod_{b'<b} 2^{m_b'}), as in rankcert.propagate.interval.
"""
from __future__ import annotations

import heapq

import numpy as np

from rankcert.propagate.interval import imul


class Searcher:
    def __init__(self, model, fi):
        self.model = model
        self.nb = len(model.blocks)
        self.widths = [len(b) for b in model.blocks]
        self.G = model.groups
        s_lo, s_hi = fi['s']
        self.s = (np.asarray(s_lo, float), np.asarray(s_hi, float))
        self.u = fi['u']
        c_lo = c_hi = 0.0
        for g in range(self.G):
            mlo, mhi = s_lo[g], s_hi[g]
            for b in range(self.nb):
                a, c = fi['S'][g][b]
                mlo, mhi = imul(mlo, mhi, a, c)
            c_lo += mlo
            c_hi += mhi
        sc = 1.0 / (1 << model.m)
        self.c0 = ((1.0 + c_lo) * sc, (1.0 + c_hi) * sc)
        self.H = [[(float(self.u[g][b][0].min()), float(self.u[g][b][1].max())) for b in range(self.nb)]
                  for g in range(self.G)]
        self.leaves = 0
        self.nodes = 0

    # ---- bounds for the children of a node (prefix fixes blocks 0..j-1; children = all values of block j)
    # Arithmetic order mirrors rankcert.propagate.interval.dense_bounds exactly (block products left to right, the
    # coefficient s_g last, groups summed in order), so leaf values are bit-identical to enumeration (review
    # FATAL-1). Branch bounds carry an outward rounding margin so that a branch is never pruned because of rounding.
    def child_bounds(self, prefix_iv: list, j: int):
        """prefix_iv[g] = (lo, hi) interval of prod_{b<j} u_gb(x_b) (None at the root). Returns (U_bar, L_bar, child)."""
        tot_lo = None
        mag = None
        child = []
        for g in range(self.G):
            if prefix_iv[g] is None:
                lo, hi = self.u[g][j][0], self.u[g][j][1]
            else:
                plo, phi = prefix_iv[g]
                lo, hi = imul(plo, phi, self.u[g][j][0], self.u[g][j][1])     # vector over block-j values
            child.append((lo, hi))
            hlo, hhi = lo, hi
            for b in range(j + 1, self.nb):
                hlo, hhi = imul(hlo, hhi, self.H[g][b][0], self.H[g][b][1])
            hlo, hhi = imul(hlo, hhi, self.s[0][g], self.s[1][g])
            tot_lo = hlo if tot_lo is None else tot_lo + hlo
            a = np.maximum(np.abs(hlo), np.abs(hhi))
            mag = a if mag is None else mag + a
        self.nodes += len(tot_lo)
        slack = 1e-12 * (mag + abs(self.c0[0]) + abs(self.c0[1])) + 1e-300
        return self.c0[1] - tot_lo + slack, self.c0[0] - tot_lo + slack, child

    def leaf(self, prefix_iv: list):
        """Exact L, U over the last block given the prefix intervals (j = nb - 1)."""
        j = self.nb - 1
        slo = shi = None
        for g in range(self.G):
            if prefix_iv[g] is None:
                lo, hi = self.u[g][j][0], self.u[g][j][1]
            else:
                plo, phi = prefix_iv[g]
                # same corner order as iouter(prefix, u_j): a = prefix (lower blocks), b = block j
                lo, hi = imul(plo, phi, self.u[g][j][0], self.u[g][j][1])
            lo, hi = imul(lo, hi, self.s[0][g], self.s[1][g])
            slo = lo if slo is None else slo + lo
            shi = hi if shi is None else shi + hi
        self.leaves += 1
        return self.c0[0] - shi, self.c0[1] - slo

    def _root(self):
        return [None] * self.G

    def _walk(self, visit, bound_key, threshold_fn):
        """Depth-first over blocks 0..nb-2 in order of decreasing bound; `threshold_fn()` gives the current pruning
        value (a child with bound < it is pruned, all its strings decided by the caller's rule); `visit(index_offset,
        stride_of_last_block, L, U)` is called on leaves (string index = offset + x_last * stride)."""
        stride = [1]
        for w in self.widths[:-1]:
            stride.append(stride[-1] << w)
        last_stride = stride[self.nb - 1]

        def rec(prefix_iv, j, offset):
            if j == self.nb - 1:
                L, U = self.leaf(prefix_iv)
                visit(offset, last_stride, L, U)
                return
            Ub, Lb, child = self.child_bounds(prefix_iv, j)
            key = Ub if bound_key == 'U' else Lb
            for v in np.argsort(-key, kind='stable'):
                if key[v] < threshold_fn():
                    break
                civ = [(child[g][0][v], child[g][1][v]) for g in range(self.G)]
                rec(civ, j + 1, offset + int(v) * stride[j])
        rec(self._root(), 0, 0)
        return last_stride

    # ---- queries
    def topk(self, k: int):
        best = []          # min-heap of the k largest L values
        kth = lambda: best[0] if len(best) >= k else -np.inf

        def visit_a(off, st, L, U):
            top = L if len(L) <= k else L[np.argpartition(-L, k - 1)[:k]]
            for v in top:
                if len(best) < k:
                    heapq.heappush(best, float(v))
                elif v > best[0]:
                    heapq.heapreplace(best, float(v))
        # phase A: prune a branch only when its L_bar is strictly below the current k-th value
        self._walk(visit_a, 'L', lambda: np.nextafter(kth(), np.inf) if np.isfinite(kth()) else -np.inf)
        Lk = kth()
        cand_idx, cand_L, cand_U = [], [], []

        def visit_b(off, st, L, U):
            sel = np.flatnonzero(U >= Lk)
            if sel.size:
                cand_idx.append(off + sel * st)
                cand_L.append(L[sel])
                cand_U.append(U[sel])
        self._walk(visit_b, 'U', lambda: Lk)
        idx = np.concatenate(cand_idx) if cand_idx else np.zeros(0, np.int64)
        CL = np.concatenate(cand_L) if cand_L else np.zeros(0)
        CU = np.concatenate(cand_U) if cand_U else np.zeros(0)
        Us = np.sort(CU)
        cnt = len(CU) - np.searchsorted(Us, CL, side='left')        # |{y in C : U(y) >= L(x)}| incl. x
        lab = np.where((cnt - 1) <= k - 1, 1, 0).astype(np.int8)
        return dict(Lk=Lk, cand=idx, cand_labels=lab, cand_L=CL, cand_U=CU)

    def threshold(self, tau: float):
        cand_idx, cand_L = [], []

        def visit(off, st, L, U):
            sel = np.flatnonzero(U >= tau)
            if sel.size:
                cand_idx.append(off + sel * st)
                cand_L.append(L[sel])
        self._walk(visit, 'U', lambda: tau)
        idx = np.concatenate(cand_idx) if cand_idx else np.zeros(0, np.int64)
        CL = np.concatenate(cand_L) if cand_L else np.zeros(0)
        return dict(cand=idx, cand_labels=np.where(CL > tau, 1, 0).astype(np.int8))


def expand_labels(n_strings: int, res: dict) -> np.ndarray:
    """Full label vector implied by a search answer (everything outside the candidate set is OUT)."""
    lab = np.full(n_strings, -1, dtype=np.int8)
    lab[res['cand']] = res['cand_labels']
    return lab
