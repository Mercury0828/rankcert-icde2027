"""Data of paper Fig. 1 (running example): one development instance at N shots per subexperiment, rep 0.

python -m derisk.a6_fig1 [instance] [N]          (default qaoa2_24_s1001, N = 1e5)
Same draws as derisk.a6_run (rep 0): R-CP and the plug-in estimate use the uniform stream-0 counts; ours (R-PAIR-S6)
uses the pilot (stream 41) and the allocated inference counts (stream 42). For the strings with the 16 largest exact
probabilities, and for every string in the plug-in top-10, writes the exact probability, the plug-in estimate, the R-CP
interval and label, and our refined interval and label (top-10 query).
The frozen certify() (rankcert/certify/rpair.py) is not modified: its source is executed with one extra line that also
returns the refined endpoints of the reported candidates; the labels are asserted equal to the unmodified call.
Writes derisk/results/a6/fig1.json.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import textwrap

import numpy as np

import rankcert.certify.rpair as RP
from bench.a6 import FAMILY_ID
from derisk.a6_run import draw, load_instance
from derisk.repair5_run import allocate_safeguarded
from rankcert.propagate.interval import dense_bounds, dense_point, factor_intervals, to_block_order
from rankcert.regions.regions import CPRegion
from stats.labels import topk_labels

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'a6', 'fig1.json')
K = 10


def certify_with_bounds():
    src = textwrap.dedent(inspect.getsource(RP.certify))
    a = "out[k] = dict(cand=idx[keep], cand_labels=lab[keep], rest=-1, certified=len(certs),"
    b = "out[k] = dict(cand_L=L[keep], cand_U=U[keep], cand=idx[keep], cand_labels=lab[keep], rest=-1, certified=len(certs),"
    c = "out[k] = dict(cand=idx, cand_labels=lab, rest=-1, certified=len(certs), collected=int(n))"
    d = "out[k] = dict(cand_L=L, cand_U=U, cand=idx, cand_labels=lab, rest=-1, certified=len(certs), collected=int(n))"
    assert src.count(a) == 1 and src.count(c) == 1
    ns = dict(RP.__dict__)
    exec(compile(src.replace(a, b).replace(c, d), RP.__file__, 'exec'), ns)
    return ns['certify']


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'qaoa2_24_s1001'
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    row, ref = load_instance(name)
    assert row['split'] == 'development'
    model, qs = ref['model'], ref['qs']
    p = to_block_order(model, ref['p'])
    S, M = len(qs), 1 << model.m
    key = (FAMILY_ID[row['family']], model.m, row['seed'], N, 0)
    n0 = draw(qs, [N] * S, key, 0)
    ph = dense_point(model, [v / v.sum() for v in n0])
    Lc, Uc = dense_bounds(model, factor_intervals(model, CPRegion(model, n0, 0.05)))
    labc = topk_labels(Lc, Uc, K)
    P = N // 20
    pilot = draw(qs, [P] * S, key, 41)
    sel = RP.select_pilot(model, pilot, 'search')
    sel6 = RP.select_pilot(model, pilot, 'search', na=RP.NA6)
    sizes, _ = allocate_safeguarded(model, qs, pilot, sel, S * N - S * P)
    inf = draw(qs, sizes, key, 42)
    cert = certify_with_bounds()
    ans = cert(model, pilot, inf, (K,), mode='auto', selected=sel6, taus=(), variant='r6')
    ref_ans = RP.certify(model, pilot, inf, (K,), mode='auto', selected=sel6, taus=(), variant='r6')
    assert np.array_equal(ans[K]['cand'], ref_ans[K]['cand'])
    assert np.array_equal(ans[K]['cand_labels'], ref_ans[K]['cand_labels'])
    lab = RP.expand(M, ans[K])
    pos = {int(x): j for j, x in enumerate(ans[K]['cand'])}
    order_true = np.argsort(-p, kind='stable')
    top_est = np.argsort(-ph, kind='stable')[:K]
    show = list(dict.fromkeys([int(x) for x in order_true[:16]] + [int(x) for x in top_est]))
    rank_true = {int(x): r + 1 for r, x in enumerate(order_true[:64])}
    rank_est = {int(x): r + 1 for r, x in enumerate(np.argsort(-ph, kind='stable')[:64])}
    rows = []
    for x in show:
        j = pos.get(x)
        rows.append(dict(x=x, rank_true=rank_true.get(x), rank_est=rank_est.get(x), p=float(p[x]), est=float(ph[x]),
                         rcp_L=float(Lc[x]), rcp_U=float(Uc[x]), rcp_label=int(labc[x]),
                         ours_L=None if j is None else float(ans[K]['cand_L'][j]),
                         ours_U=None if j is None else float(ans[K]['cand_U'][j]), ours_label=int(lab[x])))
    st = {k_: (v if isinstance(v, (int, float, str)) else str(v)) for k_, v in ans['stats'].items()}
    out = dict(instance=name, m=model.m, S=S, N=N, B=S * N, k=K, pilot_per_e=P, stats=st, reported=int(len(ans[K]['cand'])), p10=float(p[order_true[K - 1]]),
               p11=float(p[order_true[K]]), plugin_top10_outside_true=[int(x) for x in top_est if rank_true.get(int(x), 99) > K],
               counts=dict(ours_in=int((lab == 1).sum()), ours_und=int((lab == 0).sum()), rcp_in=int((labc == 1).sum()),
                           rcp_und=int((labc == 0).sum())), rows=rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
