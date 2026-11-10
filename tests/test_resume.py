"""Kill -> restart -> resume test for the D1/D2 runner (guide §P6.5): a torn last line is truncated, finished units are
skipped, the unfinished unit is recomputed with the same seeds, and the resumed file equals an uninterrupted run."""
import json, os, shutil, tempfile
import pytest
from derisk import d1_run as D


def _rows(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def test_kill_restart_resume(monkeypatch):
    if not os.path.exists('D:/runs/rankcert/phase0/ref/ghz20.pkl'):
        pytest.skip('instance not built')
    tmp = tempfile.mkdtemp(dir='D:/runs/rankcert')
    # build an uninterrupted reference
    monkeypatch.setattr(D, 'RUN', tmp)
    os.makedirs(os.path.join(tmp, 'ref'))
    shutil.copy('D:/runs/rankcert/phase0/ref/ghz20.pkl', os.path.join(tmp, 'ref', 'ghz20.pkl'))
    D.run('ghz20', [1000], 3, 'full', ['R-CP', 'BL1-H'])
    full = _rows(os.path.join(tmp, 'full', 'ghz20.jsonl'))
    # simulate a kill: keep 3 complete rows and a torn fourth line
    os.makedirs(os.path.join(tmp, 'part'))
    p = os.path.join(tmp, 'part', 'ghz20.jsonl')
    with open(os.path.join(tmp, 'full', 'ghz20.jsonl')) as f:
        lines = f.readlines()
    with open(p, 'w') as f:
        f.writelines(lines[:3])
        f.write(lines[3][:25])                         # torn
    D.run('ghz20', [1000], 3, 'part', ['R-CP', 'BL1-H'])
    part = _rows(p)
    key = lambda r: (r['N'], r['method'], r['rep'])
    strip = lambda r: {k: v for k, v in r.items() if k not in ('seconds',)}
    assert sorted(map(key, part)) == sorted(map(key, full))
    fa = {key(r): strip(r) for r in full}
    for r in part:
        assert strip(r) == fa[key(r)]
    assert os.path.exists(p + '.resume.log')
    shutil.rmtree(tmp)
