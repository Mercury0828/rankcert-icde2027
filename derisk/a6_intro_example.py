"""Intro example (paper section 1): plug-in top-10 of the estimate vs the top-10 of the exact reconstruction.

python -m derisk.a6_intro_example <instance> [N]
Uniform draw at N shots per subexperiment with the A6 seeding (stream 0, rep 0), the plug-in estimate (what QCut's top()
ranks), and the number of its top-10 strings outside the exact top-10. Development instances only.
"""
import sys

import numpy as np

from bench.a6 import FAMILY_ID
from derisk.a6_run import draw, load_instance
from rankcert.propagate.interval import dense_point, to_block_order

name = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
row, ref = load_instance(name)
assert row['split'] == 'development'
model, qs = ref['model'], ref['qs']
p = to_block_order(model, ref['p'])
key = (FAMILY_ID[row['family']], model.m, row['seed'], N, 0)
n0 = draw(qs, [N] * len(qs), key, 0)
ph = dense_point(model, [v / v.sum() for v in n0])
top_true = set(np.argsort(-p, kind='stable')[:10].tolist())
top_est = np.argsort(-ph, kind='stable')[:10].tolist()
miss = [x for x in top_est if x not in top_true]
print(name, 'm=%d S=%d N=%d B=%d' % (model.m, len(qs), N, len(qs) * N), 'missed %d of 10' % len(miss))
p10 = np.sort(p)[::-1][9]
for x in miss:
    print('  missed string rank_est=%d est=%.3e true=%.3e (true 10th = %.3e)' % (top_est.index(x) + 1, ph[x], p[x], p10))
