"""H3 / F-Search (the method description "Q3 Search cost (H3)"): cost of the R-PAIR6 label computation at output widths where 2^m
enumeration is infeasible, against (a) full enumeration and (b) QCut's uncertified exact `top()` on the plug-in estimate.

python -m derisk.h3_search --ms 24,28,32,36,40 --seeds 2001-2003 --N 100000 [--reps 0] [--build-only]

Instances. F-QAOA2 exactly as in the A6 family (bench.a6.make_qaoa2: two random 3-regular clusters of n/2 vertices joined
by one crossing edge, p = 2, one gate cut per layer -> 2 cuts, 2 subcircuits of n/2 qubits, all n qubits queried, m = n),
development-style seeds 2001-2003 (new instances, not in the A6 table), one instance per (m, seed), name
h3qaoa2_<m>_s<seed>. Angles: the same optimiser as A6 (bench.a6.optimize_p2: p = 1 grid + Nelder-Mead, then nine
Nelder-Mead starts on the exact light-cone expected cut), unchanged. Its cost is linear in n, not exponential: each
objective evaluation is one <= 14-qubit statevector per edge (3n/2 edges); measured on this machine 0.044 / 0.069 / 0.12 /
0.18 s per evaluation at n = 16 / 24 / 32 / 40, i.e. about 10-15 min per instance at n = 40 (A6 n = 24 builds: ~320 s).
`--smoke-angles` (smoke checks only, never for H3 rows) replaces it by the p = 1 optimum repeated in both layers; such
instances are named *_smoke and live in a separate directory.

Stage 1 (untimed, resumable; skipped when the pickle exists): circuit, QCut cut experiment, CutModel, the exact category
distribution q_e of every distinct subexperiment (bench.exact: one statevector of n/2 + (QPD mid-circuit bits) qubits per
subexperiment, never the 2^m vector), and an exact top-k reference computed by the bit-level search on the exact point
factors (L = U = p; no dense p). Written to ./data/h3/ref/<name>.pkl (+ manifest json) -- never to the repo.
At m <= 24 the reference is also checked against the uncut statevector (A6 integrity check).

Stage 2 (timed, rows resumable by key (instance, N, rep, method)): pilot N//20 per subexperiment (stream 41) and uniform
inference counts N per subexperiment (stream 42), SeedSequence([family_id, m, seed, N, rep, 20261107, stream])
(derisk.a6_run.draw). Methods (one row each):
  ours  pilot selection (select_pilot, search, |A| = 128) + rankcert.certify.rpair.certify(mode='search', variant='r6',
        ks=(10,), taus=()); wall (perf_counter), process CPU, peak RSS (psutil sampler thread, 5 ms), BitSearcher counters
        (branch bounds = branches expanded, frontier nodes kept, strings evaluated exactly, search seconds) per searcher,
        collected count, labels n_in / n_out / n_und (n_out as a count), errors against the exact top-k reference.
        certify uses its default sparse=None (sparse functional path iff max_e |Z_e| > rpair.SPARSE_AUTO = 2^14, i.e.
        m >= 26 here, where max |Z_e| = 2^(m/2 + 2)). Guard for the dense path: one length-|Z_e| vector per factor functional (~|A| x groups x held
        subcircuits of them, 193 MB measured at m = 20); the script estimates that footprint and refuses the run
        (status 'refused_memory_estimate', no crash on the shared machine) when it exceeds --mem-budget-gb.
        collect='tight' (default here; rpair.certify docstring): theta = min(lambda_kmax, min L of certificate winners,
        taus); rows record theta, lambda_kmax, theta_cert, cert_winners and whether the winners pulled theta down.
  qcut  QCut 2.4.0 SeparableDistribution.top(10) (QCut/execution/distribution.py) on the plug-in estimate of the same
        inference counts, distribution built from our factor weights as in derisk/d3_run.qcut_top_seconds (construction
        timed separately); node counts from a second, instrumented pass (_Search subclass) when the timed pass took
        <= --qcut-count-max-seconds.
  enum  m <= --enum-max: certify(mode='dense') on the same inputs and selection (measured; labels compared with 'ours').
        Always: per-string cost of the enumeration arithmetic (dense_bounds + the top-k label pass) on a contiguous chunk
        of 2^--enum-chunk-log2 strings of the same instance, times 2^m -> 'extrapolated_seconds' (EXTRAPOLATED, linear: the
        sort in the label pass is n log n, so this underestimates enumeration); memory = chunk peak bytes per string x 2^m;
        'extrapolated_total_seconds' adds our certify's non-search time (regions, radii, pairs), which enumeration pays too.
Every row carries the git commit (derisk.d1_run.git_head). Rows -> ./data/h3/rows.jsonl.
Single-threaded BLAS is set unless the caller set it (timings comparable across methods).
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import gc
import json
import pickle
import platform
import threading
import time

import numpy as np

H3 = os.environ.get('RANKCERT_H3', './data/h3')
K = 10
FAMILY = 'qaoa2'

try:
    import psutil
except ImportError:          # pragma: no cover - psutil is installed in the pinned env
    psutil = None


# ------------------------------------------------------------------ measurement helpers
class Meter:
    """Wall (perf_counter), process CPU (process_time), peak RSS during the block (sampler thread, 5 ms)."""

    def __init__(self, interval=0.005):
        self.interval = interval

    def _rss(self):
        return self.proc.memory_info().rss if psutil else 0

    def __enter__(self):
        self.proc = psutil.Process() if psutil else None
        self.base = self._rss()
        self.peak = self.base
        self._stop = threading.Event()

        def loop():
            while not self._stop.wait(self.interval):
                r = self._rss()
                if r > self.peak:
                    self.peak = r
        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()
        self.t0, self.c0 = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self.cpu = time.process_time() - self.c0
        self._stop.set()
        self._th.join()
        self.peak = max(self.peak, self._rss())
        return False

    def record(self):
        mb = 1.0 / 2 ** 20
        return dict(seconds=round(self.seconds, 4), cpu_seconds=round(self.cpu, 4),
                    peak_rss_MB=round(self.peak * mb, 1), base_rss_MB=round(self.base * mb, 1),
                    peak_delta_MB=round((self.peak - self.base) * mb, 1))


PHASE = ['start']                    # current stage-2 phase (watchdog message)


def rss_watchdog(limit_gb, interval=0.05):
    """Shared-machine guard: hard exit (code 3, message on stderr) when the process RSS exceeds limit_gb."""
    if not psutil or not limit_gb:
        return
    proc = psutil.Process()
    lim = limit_gb * 2 ** 30

    def loop():
        while True:
            r = proc.memory_info().rss
            if r > lim:
                import sys
                sys.stderr.write('RSS %.2f GB > limit %.2f GB in phase %s: exiting\n' % (r / 2 ** 30, limit_gb,
                                                                                       PHASE[0]))
                sys.stderr.flush()
                os._exit(3)
            time.sleep(interval)
    threading.Thread(target=loop, daemon=True).start()


def _recording_searcher():
    """BitSearcher subclass that records every instance (pilot selection and certify each build one) and times its
    walks / exact evaluations; patched into rankcert.certify.rpair for the measured call only."""
    from rankcert.search.bitbnb import BitSearcher

    class Rec(BitSearcher):
        made = []

        def __init__(self, *a, **kw):
            t0 = time.perf_counter()
            super().__init__(*a, **kw)
            self.timing = dict(init=time.perf_counter() - t0, walk=0.0, exact=0.0)
            Rec.made.append(self)

        def _walk(self, keep):
            t0 = time.perf_counter()
            try:
                return super()._walk(keep)
            finally:
                self.timing['walk'] += time.perf_counter() - t0

        def _exact(self, vals):
            t0 = time.perf_counter()
            try:
                return super()._exact(vals)
            finally:
                self.timing['exact'] += time.perf_counter() - t0

        def beam_lower(self, k, beam=4096):
            self.Lprime = super().beam_lower(k, beam)
            return self.Lprime

        def collect(self, tau):
            self.theta = float(tau)               # certify: theta = min(L', min_A L, min_A U)
            return super().collect(tau)
    return Rec


def searcher_stats(S):
    lv = S.stats['levels'] + S.stats.get('beam_levels', [])
    return dict(theta=getattr(S, 'theta', None), Lprime=getattr(S, 'Lprime', None),
                branch_bounds=int(S.stats['branch_bounds']), frontier_nodes=int(sum(lv)),
                max_frontier=int(max(lv) if lv else 0), strings_evaluated=int(S.stats['bottom_evaluated']),
                search_seconds=round(sum(S.timing.values()), 4), timing={k: round(v, 4) for k, v in S.timing.items()})


# ------------------------------------------------------------------ stage 1
def name_of(m, seed, smoke=False):
    return 'h3qaoa2_%d_s%d%s' % (m, seed, '_smoke' if smoke else '')


def ref_dir(smoke=False):
    return os.path.join(H3, 'ref_smoke' if smoke else 'ref')


def make_instance(n, seed, smoke=False):
    from bench import circuits as C
    from bench.a6 import make_qaoa2
    if not smoke:
        return make_qaoa2(n, seed)
    edges, cross = C.two_cluster_graph(n, 1, seed)
    g1, b1, val = C.optimize_qaoa_p1(n, edges, cross)
    pr = dict(n=n, edges=edges, cross=cross, gammas=[g1, g1], betas=[b1, b1], angles='SMOKE: p=1 optimum repeated')
    return (pr, C.qaoa_circuit(n, edges, cross, pr['gammas'], pr['betas'], cut=True),
            C.qaoa_circuit(n, edges, cross, pr['gammas'], pr['betas'], cut=False))


def exact_truth(model, qs, k=K, rtol=1e-12):
    """Exact top-k reference without the dense vector: bit-level search on the exact point factors (L = U = p).
    Same semantics as stats.labels.topk_truth (ties at tol = rtol * max p): returns the sure members and the strings that
    are NOT sure non-members (block-order indices), both subsets of C = {p >= p_(k) - 2 tol}, which is exact for both sets
    (a non-sure-non-member has p >= p_(k) - tol; every y counted for such x has p(y) >= p(x) - tol)."""
    from rankcert.certify.rpair import PointRegion
    from rankcert.propagate.interval import factor_intervals
    from rankcert.search.bitbnb import BitSearcher
    fi = factor_intervals(model, PointRegion([q / q.sum() for q in qs]))
    BS = BitSearcher(model, fi)
    res = BS.topk(k)
    pk = float(res['Lk'])
    pmax = float(np.max(res['cand_L']))
    tol = rtol * pmax
    idx, L, U = BS.collect(pk - 2 * tol)
    p = L
    ps = np.sort(p)
    n = len(p)
    ge = n - np.searchsorted(ps, p - tol, side='left')
    gt = n - np.searchsorted(ps, p + tol, side='right')
    order = np.argsort(-p, kind='stable')[:20]
    return dict(k=k, pk=pk, pmax=pmax, tol=tol, collected=int(n), max_LU_gap=float(np.max(U - L)) if n else 0.0,
                sure_members=np.sort(idx[(ge - 1) <= k - 1]), not_sure_non=np.sort(idx[gt < k]),
                top20=[(int(idx[j]), float(p[j])) for j in order])


def build(m, seed, smoke=False, force=False):
    import QCut as ck
    from bench.build import strip
    from bench.circuits import manifest_hash
    from bench.exact import exact_category_distribution, uncut_distribution
    from derisk.d1_run import git_head
    from rankcert.certify.rpair import check_structure
    from rankcert.recon.qcut_adapter import build_model, exact_options
    name = name_of(m, seed, smoke)
    rd = ref_dir(smoke)
    path = os.path.join(rd, name + '.pkl')
    if os.path.exists(path) and not force:
        return None
    os.makedirs(rd, exist_ok=True)
    t0 = time.perf_counter()
    with Meter() as mo:
        pr, qc_cut, qc_plain = make_instance(m, seed, smoke)
    t_opt = mo.seconds
    qubits = list(range(m))
    cc = ck.get_locations_and_subcircuits(qc_cut, options=exact_options())
    ex = ck.get_experiment_circuits(cc, qubits=qubits)
    model = build_model(ex)
    assert check_structure(model), 'R-PAIR needs one use per subexperiment per product'
    sims, qs = [], []
    for s in model.subexperiments:
        with Meter() as ms:
            q = exact_category_distribution(s.circuit)
        qs.append(q)
        sims.append(dict(num_clbits=s.num_clbits, qubits=s.circuit.num_qubits, **ms.record()))
    with Meter() as mt:
        truth = exact_truth(model, qs)
    man = dict(name=name, family=FAMILY, m=m, seed=seed, smoke=bool(smoke), commit=git_head(),
               params={k: v for k, v in pr.items() if k != 'trace'}, cuts=len(cc.cut_locations),
               cut_locations=[str(c) for c in cc.cut_locations], subcircuit_widths=[s.num_qubits for s in cc.subcircuits],
               groups=model.groups, subexperiments=len(model.subexperiments), blocks=model.blocks,
               Z_sizes=[s.size for s in model.subexperiments], qs_MB=round(sum(q.nbytes for q in qs) / 2 ** 20, 1),
               circuit_sha256=manifest_hash(dict(qasm=str(qc_plain.data), cut=str(qc_cut.data))),
               optimizer_seconds=round(t_opt, 1), simulation=sims,
               simulation_seconds_total=round(sum(r['seconds'] for r in sims), 1),
               simulation_peak_delta_MB_max=max(r['peak_delta_MB'] for r in sims),
               truth=dict(pk=truth['pk'], pmax=truth['pmax'], collected=truth['collected'],
                          sure_members=int(len(truth['sure_members'])), not_sure_non=int(len(truth['not_sure_non'])),
                          seconds=round(mt.seconds, 2), top20=truth['top20']))
    if 'trace' in pr:
        man['optimizer_trace_sha256'] = manifest_hash(pr['trace'])
    if m <= 24:                                   # A6 integrity check: exact reconstruction vs uncut statevector
        from rankcert.propagate.interval import to_block_order
        from stats.labels import topk_truth
        p = model.dense_p(qs)
        man['max_abs_exact_vs_uncut'] = float(np.max(np.abs(p - uncut_distribution(qc_plain, qubits))))
        tt = topk_truth(to_block_order(model, p), K)
        man['truth_matches_dense'] = bool(np.array_equal(np.flatnonzero(tt == 1), truth['sure_members']) and
                                          np.array_equal(np.flatnonzero(tt != -1), truth['not_sure_non']))
        del p
    man['build_seconds'] = round(time.perf_counter() - t0, 1)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        sm = strip(model)
        sm.__dict__.pop('_row_cache', None)      # rebuilt lazily; would add ~16 B per category to the pickle
        pickle.dump(dict(model=sm, qs=qs, truth=truth, manifest=man), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    with open(os.path.join(rd, name + '.json'), 'w') as f:
        json.dump(man, f, indent=1, default=str)
    return man


# ------------------------------------------------------------------ stage 2 pieces
def evaluate_vs_truth(ans, truth, M):
    """Wrong labels and decided shares of a certify answer {cand, cand_labels, rest} against exact_truth."""
    cand, lab, rest = ans['cand'], ans['cand_labels'], ans['rest']
    n_in = int((lab == 1).sum())
    n_und = int((lab == 0).sum()) + (M - len(cand) if rest == 0 else 0)
    n_out = M - n_in - n_und
    order = np.argsort(cand, kind='stable')
    sc = cand[order]

    def label_of(xs):
        xs = np.asarray(xs, dtype=np.int64)
        pos = np.searchsorted(sc, xs)
        hit = (pos < len(sc)) & (sc[np.minimum(pos, len(sc) - 1)] == xs) if len(sc) else np.zeros(len(xs), bool)
        out = np.full(len(xs), rest, dtype=np.int8)
        out[hit] = lab[order[pos[hit]]]
        return out
    ins = cand[lab == 1]
    wrong_in = int((~np.isin(ins, truth['sure_members'])).sum())
    wrong_out = int((label_of(truth['not_sure_non']) == -1).sum())
    in_of_sure = int((label_of(truth['sure_members']) == 1).sum())
    sure_non = M - len(truth['not_sure_non'])
    return dict(n_in=n_in, n_out=n_out, n_und=n_und, sure_members=int(len(truth['sure_members'])),
                in_of_sure=in_of_sure, sure_non=sure_non, out_of_sure=n_out - wrong_out,
                wrong_in=wrong_in, wrong_out=wrong_out, wrong=wrong_in + wrong_out)


def r6_cache_bytes(model, n_selected):
    """Footprint of the r6 factor-functional cache (rpair._Terms.cache): one dense float64 vector of length |Z_e| per key
    (g, i, x) for held subcircuits (x in the selected set) and per (g, i, None); tighten_factors also builds a bytes copy
    of every vector for its distinct-functional count, hence the factor 2."""
    tot = 0
    for g, row in enumerate(model.uses):
        for use in row:
            z = model.subexperiments[use.e].size * 8
            tot += z * (n_selected + 1) if use.held else z
    return 2 * tot


def qcut_distribution(model, qhat):
    """QCut 2.4.0 SeparableDistribution of the plug-in estimate, built from our factor weights (derisk/d3_run.py)."""
    from QCut.execution.distribution import SeparableDistribution
    scal, tabs = model.weights(qhat)
    modes = []
    for b, bits in enumerate(model.blocks):
        w = len(bits)
        comp = np.arange(1 << w) ^ ((1 << w) - 1)       # QCut indexes modes by the complement of the measured bits
        modes.append(np.stack([tabs[g][b][comp] for g in range(model.groups)], axis=1))
    del tabs
    return SeparableDistribution(model.m, [list(b) for b in model.blocks], modes, np.asarray(scal))


def qcut_count(dist, count=K):
    """Instrumented replay of SeparableDistribution.top's branch and bound (same _Search code, counting calls)."""
    from QCut.execution.distribution import _Search

    class Counting(_Search):
        def __init__(self, *a):
            super().__init__(*a)
            self.nodes = 0
            self.leaves = 0

        def descend(self, level, weights, blocks):
            self.nodes += 1
            return super().descend(level, weights, blocks)

        def leaf(self, weights, blocks):
            self.leaves += 1
            return super().leaf(weights, blocks)
    s = Counting(dist, min(count, 1 << dist.width))
    s.descend(0, dist.coefficients, [])
    rows_last = int(dist.modes[-1].idx.size)
    return dict(nodes=s.nodes, leaves=s.leaves, last_mode_rows=rows_last, strings_evaluated=s.leaves * rows_last,
                mode_rows=[int(md.idx.size) for md in dist.modes])


def natural_to_block(model, x):
    """QCut outcome (bit b = outcome position b) -> our block-order index."""
    idx, sh = 0, 0
    for bits in model.blocks:
        loc = 0
        for t, pos in enumerate(bits):
            loc |= ((x >> pos) & 1) << t
        idx |= loc << sh
        sh += len(bits)
    return idx


def enum_chunk(model, fi, log2):
    """Per-string cost of the enumeration arithmetic: dense_bounds on a contiguous slice of the last block (first
    2^log2 strings in block order, same arithmetic as rankcert.propagate.interval.dense_bounds) + the top-k label pass."""
    from rankcert.propagate.interval import dense_bounds
    from stats.labels import topk_labels
    lead = model.m - len(model.blocks[-1])
    rows = max(1, 1 << max(0, log2 - lead))
    rows = min(rows, 1 << len(model.blocks[-1]))
    u = [[(lo, hi) for lo, hi in ug] for ug in fi['u']]
    for g in range(len(u)):
        lo, hi = u[g][-1]
        u[g][-1] = (lo[:rows], hi[:rows])
    fi2 = dict(s=fi['s'], S=fi['S'], u=u)
    with Meter() as mm:
        L, U = dense_bounds(model, fi2)
        topk_labels(L, U, K)
    n = len(L)
    return dict(chunk_strings=int(n), per_string_seconds=mm.seconds / n, per_string_cpu=mm.cpu / n,
                per_string_peak_bytes=max(mm.peak - mm.base, 0) / n, **{'chunk_' + k: v for k, v in mm.record().items()})


# ------------------------------------------------------------------ stage 2
def load_ref(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def done_keys(path):
    done = set()
    if os.path.exists(path):
        with open(path, 'rb+') as fb:                       # drop a torn last line (crash mid-write)
            data = fb.read()
            if data and not data.endswith(b'\n'):
                fb.seek(data.rfind(b'\n') + 1)
                fb.truncate()
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r['instance'], r['N'], r['rep'], r['method']))
            except Exception:
                pass
    return done


def run_instance(name, ref, N, reps, rows_path, methods=('ours', 'qcut', 'enum'), enum_max=24, chunk_log2=22,
                 qcut_count_max=120.0, mem_budget_gb=8.0, seed=None, sparse=None, collect='tight'):
    from bench.a6 import FAMILY_ID
    from derisk.a6_run import draw
    from derisk.d1_run import git_head
    from rankcert.certify import rpair
    from rankcert.certify.rpair import NA6, certify, expand
    from rankcert.propagate.interval import factor_intervals
    from rankcert.regions.regions import CPRegion
    model, qs, truth = ref['model'], ref['qs'], ref['truth']
    man = ref.get('manifest', {})
    seed = man.get('seed', seed)
    S, m = len(qs), model.m
    M = 1 << m
    done = done_keys(rows_path)
    os.makedirs(os.path.dirname(rows_path), exist_ok=True)
    head = git_head()
    env = dict(python=platform.python_version(), machine=platform.node(),
               threads={v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    with open(rows_path, 'a') as f:
        def emit(r):
            f.write(json.dumps(r, default=float) + '\n')
            f.flush()
            os.fsync(f.fileno())
            print({k: r.get(k) for k in ('instance', 'N', 'rep', 'method', 'status', 'seconds', 'qcut_top_seconds',
                                          'measured_seconds', 'extrapolated_seconds', 'collected', 'n_in', 'n_und',
                                          'wrong')}, flush=True)

        for rep in reps:
            todo = [mt for mt in methods if (name, N, rep, mt) not in done]
            if not todo:
                continue
            key = (FAMILY_ID[FAMILY], m, seed, N, rep)
            P = N // 20
            PHASE[0] = 'draw'
            pilot = draw(qs, [P] * S, key, 41)
            inf = draw(qs, [N] * S, key, 42)
            if rep == reps[-1]:                 # memory (m = 40: each category vector set is ~0.9 GB): q_e no longer needed
                ref['qs'] = qs = None
                gc.collect()
            base = dict(instance=name, family=FAMILY, m=m, seed=seed, S=S, groups=model.groups, N=N, rep=rep, k=K,
                        pilot_per_e=P, inference_per_e=N, total_shots=S * (N + P), commit=head, variant='r6',
                        Z_sizes_max=max(s.size for s in model.subexperiments), env=env)
            ours_ans = sel = None
            if 'ours' in todo or ('enum' in todo and m <= enum_max):
                sparse_on = (max(s.size for s in model.subexperiments) > rpair.SPARSE_AUTO if sparse is None
                             else bool(sparse))
                est = 0 if sparse_on else r6_cache_bytes(model, NA6)
                if est > mem_budget_gb * 2 ** 30:
                    if 'ours' in todo:
                        emit(dict(base, method='ours', status='refused_memory_estimate',
                                  r6_cache_estimate_GB=round(est / 2 ** 30, 2), mem_budget_gb=mem_budget_gb))
                else:
                    Rec = _recording_searcher()
                    orig = rpair.BitSearcher
                    rpair.BitSearcher = Rec
                    status = 'ok'
                    try:
                        PHASE[0] = 'select'
                        with Meter() as m_sel:
                            sel = rpair.select_pilot(model, pilot, 'search', na=NA6)
                        sel_stats = [searcher_stats(s_) for s_ in Rec.made]   # release the searchers (memory)
                        del Rec.made[:]
                        if m > enum_max:                # pilot only feeds the selection (certify gets `selected`)
                            pilot = None
                        gc.collect()
                        PHASE[0] = 'certify'
                        with Meter() as m_cert:
                            try:
                                ours_ans = certify(model, pilot, inf, (K,), mode='search', selected=sel, taus=(),
                                                   variant='r6', sparse=sparse, collect=collect)
                            except rpair.FrontierCap as exc:
                                status = 'frontier_cap: %s' % exc
                    finally:
                        rpair.BitSearcher = orig
                    cert_stats = [searcher_stats(s_) for s_ in Rec.made]
                    del Rec.made[:]
                    gc.collect()
                    if 'ours' in todo:
                        row = dict(base, method='ours', status=status, sparse=bool(sparse_on), collect=collect,
                                   r6_cache_estimate_GB=round(est / 2 ** 30, 2),
                                   seconds=round(m_sel.seconds + m_cert.seconds, 4),
                                   cpu_seconds=round(m_sel.cpu + m_cert.cpu, 4),
                                   peak_rss_MB=max(m_sel.record()['peak_rss_MB'], m_cert.record()['peak_rss_MB']),
                                   select=m_sel.record(), certify=m_cert.record(),
                                   searchers=dict(select=sel_stats, certify=cert_stats))
                        cs = row['searchers']['certify']
                        row['search_seconds_in_certify'] = round(sum(s['search_seconds'] for s in cs), 4)
                        row['nonsearch_seconds_in_certify'] = round(m_cert.seconds - row['search_seconds_in_certify'], 4)
                        row['theta'] = cs[-1]['theta'] if cs else None
                        row['branch_bounds'] = int(sum(s['branch_bounds'] for s in cs + row['searchers']['select']))
                        row['strings_evaluated'] = int(sum(s['strings_evaluated']
                                                           for s in cs + row['searchers']['select']))
                        row['frac_strings_evaluated'] = row['strings_evaluated'] / float(M)
                        if ours_ans is not None:
                            a = ours_ans[K]
                            st = ours_ans['stats']
                            row.update({k: st.get(k) for k in ('theta', 'lambda_kmax', 'theta_cert', 'cert_winners',
                                                               'working')})
                            if collect == 'tight' and 'theta_cert' in st:   # (absent when empty: no collection)
                                row['theta_pulled_by_certs'] = bool(st['theta_cert'] < st['lambda_kmax'])
                                row['theta_minus_lambda'] = st['theta'] - st['lambda_kmax']
                            row.update(empty=ours_ans['empty'], collected_all=ours_ans['stats'].get('collected_all'),
                                       collected=a.get('collected', 0), certified_pairs=a['certified'],
                                       **evaluate_vs_truth(a, truth, M))
                        emit(row)
            if 'qcut' in todo:
                qhat = [v / v.sum() for v in inf]
                PHASE[0] = 'qcut'
                with Meter() as m_build:
                    dist = qcut_distribution(model, qhat)
                with Meter() as m_top:
                    top = dist.top(K)
                row = dict(base, method='qcut', status='ok', qcut_top_seconds=round(m_top.seconds, 4),
                           seconds=round(m_top.seconds, 4), cpu_seconds=round(m_top.cpu, 4),
                           peak_rss_MB=m_top.record()['peak_rss_MB'], top=m_top.record(), construct=m_build.record(),
                           top10=[(natural_to_block(model, x), float(v)) for x, v in top])
                if m_top.seconds <= qcut_count_max:
                    row['counts'] = qcut_count(dist)
                if m <= enum_max:                         # untimed check: top values = dense plug-in top values
                    from rankcert.propagate.interval import dense_point
                    ph = dense_point(model, qhat)
                    ref_top = np.sort(ph)[::-1][:K]
                    row['check_top_values_maxabs'] = float(np.max(np.abs(np.array([v for _, v in top]) - ref_top)))
                    del ph
                del dist
                emit(row)
            if 'enum' in todo:
                fi = factor_intervals(model, CPRegion(model, inf, rpair.D0_6))
                PHASE[0] = 'enum'
                ch = enum_chunk(model, fi, min(chunk_log2, m))
                row = dict(base, method='enum', status='ok', chunk=ch,
                           extrapolated=True, extrapolated_seconds=ch['per_string_seconds'] * M,
                           extrapolated_cpu_seconds=ch['per_string_cpu'] * M,
                           extrapolated_memory_GB=max(ch['per_string_peak_bytes'], 48.0) * M / 2 ** 30,
                           extrapolation_note='EXTRAPOLATED: per-string cost of dense_bounds + top-k label pass on a '
                                              'contiguous chunk x 2^m (linear; the label sort is n log n -> underestimate); '
                                              'memory = chunk peak bytes per string (>= 6 float64) x 2^m; '
                                              'extrapolated_total_seconds adds the non-search part of our certify '
                                              '(regions, radii, pair constraints), which enumeration also pays')
                if ours_ans is not None and 'ours' in todo:
                    nonsearch = m_cert.seconds - sum(s_['search_seconds'] for s_ in cert_stats)
                    row.update(nonsearch_seconds_ours=round(nonsearch, 4),
                               extrapolated_total_seconds=row['extrapolated_seconds'] + nonsearch)
                del fi
                if m <= enum_max and sel is not None:
                    with Meter() as m_den:
                        dans = certify(model, pilot, inf, (K,), mode='dense', selected=sel, taus=(), variant='r6',
                                       sparse=sparse)
                    row.update(measured=True, measured_seconds=round(m_den.seconds, 4), dense=m_den.record(),
                               extrapolation_ratio_measured_over_chunk=m_den.seconds / max(row['extrapolated_seconds'],
                                                                                            1e-12))
                    if ours_ans is not None:
                        row['labels_equal_search'] = bool(np.array_equal(expand(M, dans[K]), expand(M, ours_ans[K])))
                    del dans
                else:
                    row['measured'] = False
                emit(row)


def parse_range(s):
    out = []
    for part in s.split(','):
        if '-' in part:
            lo, hi = (int(v) for v in part.split('-'))
            out.extend(range(lo, hi + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ms', default='24,28,32,36,40')
    ap.add_argument('--seeds', default='2001-2003')
    ap.add_argument('--N', type=int, default=100000)
    ap.add_argument('--reps', default='0')
    ap.add_argument('--build-only', action='store_true')
    ap.add_argument('--methods', default='ours,qcut,enum')
    ap.add_argument('--enum-max', type=int, default=24, help='measure dense enumeration up to this m (memory ~2^m x 50 B)')
    ap.add_argument('--enum-chunk-log2', type=int, default=22)
    ap.add_argument('--qcut-count-max-seconds', type=float, default=120.0)
    ap.add_argument('--mem-budget-gb', type=float, default=8.0, help='refuse r6 when its cache estimate exceeds this')
    ap.add_argument('--collect', default='tight', choices=('tight', 'r4'), help="certify collection rule (H3: 'tight')")
    ap.add_argument('--sparse', default='auto', choices=('auto', 'on', 'off'), help='certify sparse functional path')
    ap.add_argument('--rss-limit-gb', type=float, default=8.0, help='hard exit above this RSS (0 = off)')
    ap.add_argument('--rows', default=os.path.join(H3, 'rows.jsonl'))
    ap.add_argument('--ref', default=None, help='stage 2 only, on this existing reference pickle (e.g. an A6 dev '
                    'instance; needs a "truth" entry or a dense "p"); rows keyed by the pickle name')
    ap.add_argument('--smoke-angles', action='store_true', help='smoke checks only: p=1 angles, *_smoke instances')
    a = ap.parse_args(argv)
    rss_watchdog(a.rss_limit_gb)
    methods = tuple(a.methods.split(','))
    reps = parse_range(a.reps)
    kw = dict(methods=methods, enum_max=a.enum_max, chunk_log2=a.enum_chunk_log2,
              qcut_count_max=a.qcut_count_max_seconds, mem_budget_gb=a.mem_budget_gb,
              sparse={'auto': None, 'on': True, 'off': False}[a.sparse], collect=a.collect)
    if a.ref:
        ref = load_ref(a.ref)
        if 'truth' not in ref:
            from rankcert.propagate.interval import to_block_order
            from stats.labels import topk_truth
            tt = topk_truth(to_block_order(ref['model'], ref['p']), K)
            ref['truth'] = dict(sure_members=np.flatnonzero(tt == 1), not_sure_non=np.flatnonzero(tt != -1))
        nm = os.path.splitext(os.path.basename(a.ref))[0]
        seed = int(nm.rsplit('_s', 1)[1]) if '_s' in nm else 0
        run_instance(nm, ref, a.N, reps, a.rows, seed=seed, **kw)
        return
    for m in parse_range(a.ms):
        for seed in parse_range(a.seeds):
            name = name_of(m, seed, a.smoke_angles)
            man = build(m, seed, a.smoke_angles)
            if man is not None:
                print('built', name, {k: man.get(k) for k in ('cuts', 'subcircuit_widths', 'groups', 'subexperiments',
                                                              'optimizer_seconds', 'simulation_seconds_total',
                                                              'simulation_peak_delta_MB_max', 'qs_MB', 'build_seconds',
                                                              'max_abs_exact_vs_uncut', 'truth_matches_dense')},
                      flush=True)
            if a.build_only:
                continue
            ref = load_ref(os.path.join(ref_dir(a.smoke_angles), name + '.pkl'))
            run_instance(name, ref, a.N, reps, a.rows, **kw)
            del ref


if __name__ == '__main__':
    main()
