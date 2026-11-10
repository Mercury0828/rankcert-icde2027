"""R-3a: bit-level branch and bound
over all m output bits, processed level by level and vectorised over the frontier.

A branch at a level fixes blocks 0..j-1 completely (values v_0..v_{j-1}) and the high bits of block j, i.e.
x_j in [a 2^r, (a+1) 2^r). Its bound per group g is the interval product
    s_g * prod_{b<j} u_gb(v_b) * hull_r(u_gj)[a] * prod_{b>j} H_gb
with hull_r the min-of-lo / max-of-hi over the range (inclusion-monotone interval arithmetic), so
    U_bar = c0_hi - sum_g lo_g + slack >= max_{y in branch} U(y)
(outward slack 1e-12 * magnitude as in rankcert.search.bnb). At the bottom (all bits fixed) L, U are computed with the
arithmetic order of rankcert.propagate.interval.dense_bounds (block products left to right, s_g last, groups summed in
order), so values are bit-identical to enumeration.

Top-k: a beam (width `beam`, by U_bar) reaches k real strings whose exact L values give L' <= L_(k); the full walk keeps
branches with U_bar >= L'; L_(k) is read exactly from the survivors (every string with L >= L_(k) survives since
U >= L >= L_(k) >= L'); candidates C = {U >= L_(k)}; IN / UND inside C as in rankcert.search.bnb.
Threshold tau: keep U_bar >= tau; IN iff L > tau.
String index in block order: sum_b x_b * prod_{b'<b} 2^{w_b'}.
"""
from __future__ import annotations

import os

import numpy as np

from rankcert.propagate.interval import imul


class FrontierCap(RuntimeError):
    pass


class BitSearcher:
    def __init__(self, model, fi, cap=None, chunk=2 * 10 ** 6):
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
        # range hulls: hull[g][b][r] = (lo_min, hi_max) over consecutive ranges of 2^r values (r = 0 .. w_b)
        self.hull = []
        for g in range(self.G):
            hg = []
            for b in range(self.nb):
                lo, hi = self.u[g][b]
                lv, hv = [np.asarray(lo, float)], [np.asarray(hi, float)]
                for r in range(1, self.widths[b] + 1):
                    lv.append(np.minimum(lv[-1][0::2], lv[-1][1::2]))
                    hv.append(np.maximum(hv[-1][0::2], hv[-1][1::2]))
                hg.append((lv, hv))
            self.hull.append(hg)
        self.cap = int(float(os.environ.get("RANKCERT_FRONTIER_CAP", 2e7))) if cap is None else cap
        self.chunk = chunk
        self.stats = dict(levels=[], bottom_evaluated=0, branch_bounds=0)

    # ---- bounds for a frontier: vals = list of int arrays for blocks < j, a = partial prefix of block j at level r
    def _ubar(self, vals, j, a, r):
        tot_lo = None
        mag = None
        for g in range(self.G):
            lo = hi = None
            for b in range(j):
                ulo, uhi = self.u[g][b]
                if lo is None:
                    lo, hi = ulo[vals[b]], uhi[vals[b]]
                else:
                    lo, hi = imul(lo, hi, ulo[vals[b]], uhi[vals[b]])
            hl, hh = self.hull[g][j][0][r][a], self.hull[g][j][1][r][a]
            if lo is None:
                lo, hi = hl, hh
            else:
                lo, hi = imul(lo, hi, hl, hh)
            for b in range(j + 1, self.nb):
                Hl, Hh = self.hull[g][b][0][-1][0], self.hull[g][b][1][-1][0]
                lo, hi = imul(lo, hi, Hl, Hh)
            lo, hi = imul(lo, hi, self.s[0][g], self.s[1][g])
            tot_lo = lo.copy() if tot_lo is None else tot_lo + lo
            t = np.maximum(np.abs(lo), np.abs(hi))
            mag = t if mag is None else mag + t
        self.stats['branch_bounds'] += len(a)
        slack = 1e-12 * (mag + abs(self.c0[0]) + abs(self.c0[1])) + 1e-300
        return self.c0[1] - tot_lo + slack

    def _exact(self, vals):
        """Exact L, U of fully fixed strings, dense_bounds arithmetic order."""
        slo = shi = None
        for g in range(self.G):
            lo, hi = self.u[g][0][0][vals[0]], self.u[g][0][1][vals[0]]
            for b in range(1, self.nb):
                lo, hi = imul(lo, hi, self.u[g][b][0][vals[b]], self.u[g][b][1][vals[b]])
            lo, hi = imul(lo, hi, self.s[0][g], self.s[1][g])
            if slo is None:
                slo, shi = lo.copy(), hi.copy()
            else:
                slo += lo
                shi += hi
        self.stats['bottom_evaluated'] += len(vals[0])
        return self.c0[0] - shi, self.c0[1] - slo

    def _chunked_ubar(self, vals, j, a, r):
        n = len(a)
        if n <= self.chunk:
            return self._ubar(vals, j, a, r)
        out = np.empty(n)
        for s0 in range(0, n, self.chunk):
            sl = slice(s0, s0 + self.chunk)
            out[sl] = self._ubar([v[sl] for v in vals], j, a[sl], r)
        return out

    def _walk(self, keep):
        """Level walk. keep(Ubar) -> boolean mask (or index array) of children to keep. Returns the bottom frontier
        (list of per-block value arrays)."""
        vals = []
        a = np.zeros(1, np.int64)
        for j in range(self.nb):
            w = self.widths[j]
            for r in range(w - 1, -1, -1):
                # children: append one bit (high bits first)
                a = np.concatenate([2 * a, 2 * a + 1])
                vals = [np.concatenate([v, v]) for v in vals]
                if len(a) > self.cap:
                    raise FrontierCap('frontier %d > cap %d' % (len(a), self.cap))
                if j == self.nb - 1 and r == 0:
                    self.stats['levels'].append(len(a))
                    return vals + [a]
                ub = self._chunked_ubar(vals, j, a, r)
                sel = keep(ub)
                a = a[sel]
                vals = [v[sel] for v in vals]
                self.stats['levels'].append(len(a))
            vals = vals + [a]
            a = np.zeros(len(vals[0]), np.int64)
        return vals

    def beam_lower(self, k: int, beam: int = 4096):
        """Valid lower bound L' <= L_(k): the k-th largest exact L among the strings a beam (width `beam`, by U_bar)
        reaches (-inf if fewer than k)."""
        def keep_beam(ub):
            if len(ub) <= beam:
                return np.arange(len(ub))
            return np.argpartition(-ub, beam - 1)[:beam]
        bv = self._walk(keep_beam)
        Lb, _ = self._exact(bv)
        return float(np.sort(Lb)[-k]) if len(Lb) >= k else -np.inf

    def beam_reached(self, beam: int = 4096):
        """The strings the beam of beam_lower reaches (same walk) with their exact L: (block-order idx, L). Distinct
        strings (each bottom node is one string)."""
        def keep_beam(ub):
            if len(ub) <= beam:
                return np.arange(len(ub))
            return np.argpartition(-ub, beam - 1)[:beam]
        bv = self._walk(keep_beam)
        Lb, _ = self._exact(bv)
        stride = [1]
        for w in self.widths[:-1]:
            stride.append(stride[-1] << w)
        idx = np.zeros(len(Lb), np.int64)
        for b in range(self.nb):
            idx += bv[b].astype(np.int64) * stride[b]
        return idx, Lb

    def topk(self, k: int, beam: int = 4096):
        # 1) beam for a valid lower bound L' <= L_(k)
        Lp = self.beam_lower(k, beam)
        beam_stats = dict(levels=self.stats['levels'])
        self.stats['levels'] = []
        # 2) full walk
        vals = self._walk(lambda ub: np.flatnonzero(ub >= Lp))
        L, U = self._exact(vals)
        Lk = float(-np.partition(-L, k - 1)[k - 1])
        sel = np.flatnonzero(U >= Lk)
        stride = [1]
        for w in self.widths[:-1]:
            stride.append(stride[-1] << w)
        idx = np.zeros(len(sel), np.int64)
        for b in range(self.nb):
            idx += vals[b][sel].astype(np.int64) * stride[b]
        CL, CU = L[sel], U[sel]
        Us = np.sort(CU)
        cnt = len(CU) - np.searchsorted(Us, CL, side='left')
        lab = np.where((cnt - 1) <= k - 1, 1, 0).astype(np.int8)
        self.stats['beam_levels'] = beam_stats['levels']
        self.stats['Lprime'] = Lp
        return dict(Lk=Lk, cand=idx, cand_labels=lab, cand_L=CL, cand_U=CU)

    def _split(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        vals, sh = [], 0
        for w in self.widths:
            vals.append((idx >> sh) & ((1 << w) - 1))
            sh += w
        return vals

    def values(self, idx):
        """Exact (L, U) of given strings (block-order indices), same arithmetic as the bottom level / dense_bounds."""
        return self._exact(self._split(idx))

    def collect(self, tau: float):
        """All strings with U >= tau: (idx, L, U), never materialising the 2^m vector."""
        vals = self._walk(lambda ub: np.flatnonzero(ub >= tau))
        L, U = self._exact(vals)
        sel = np.flatnonzero(U >= tau)
        stride = [1]
        for w in self.widths[:-1]:
            stride.append(stride[-1] << w)
        idx = np.zeros(len(sel), np.int64)
        for b in range(self.nb):
            idx += vals[b][sel].astype(np.int64) * stride[b]
        return idx, L[sel], U[sel]

    def threshold(self, tau: float):
        vals = self._walk(lambda ub: np.flatnonzero(ub >= tau))
        L, U = self._exact(vals)
        sel = np.flatnonzero(U >= tau)
        stride = [1]
        for w in self.widths[:-1]:
            stride.append(stride[-1] << w)
        idx = np.zeros(len(sel), np.int64)
        for b in range(self.nb):
            idx += vals[b][sel].astype(np.int64) * stride[b]
        return dict(cand=idx, cand_labels=np.where(L[sel] > tau, 1, 0).astype(np.int8))
