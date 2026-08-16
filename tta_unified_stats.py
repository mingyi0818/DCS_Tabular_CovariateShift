"""Paired t-tests for the unified-protocol TTA comparison (Table 16).

Reads results/tta_unified_5seed_results.json and computes, per split,
paired t-tests of TabPFN-DCS against Random / Tent / AdapTable / SelfTrain
on accuracy, plus per-seed values. Appends the results to revision_stats.json.
"""
import json
import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR

SRC = os.path.join(RESULT_DIR, 'tta_unified_5seed_results.json')
DST = os.path.join(RESULT_DIR, 'revision_stats.json')

with open(SRC) as f:
    data = json.load(f)

# errors check
n_errors = len(data.get('errors', []))

per_seed = {}
for blk in data['results']:
    split = blk['split']
    seed = blk['seed']
    for method, m in blk['results'].items():
        if 'accuracy' in m:
            per_seed.setdefault((split, method), {})[seed] = m['accuracy']

out = {'n_error_entries': n_errors, 'n_blocks': len(data['results']),
       'config': data['config'], 'tests': {}}

for split in ['iid', 'temporal']:
    dcs = per_seed[(split, 'dcs')]
    seeds = sorted(dcs.keys())
    out['tests'][split] = {'seeds': seeds, 'dcs_values': [dcs[s] for s in seeds]}
    for other in ['random', 'knn', 'tent', 'adaptable', 'self_training']:
        ov = per_seed[(split, other)]
        o_seeds = sorted(ov.keys())
        assert o_seeds == seeds
        a = np.array([dcs[s] for s in seeds])
        b = np.array([ov[s] for s in seeds])
        t, p = stats.ttest_rel(a, b)
        diff = a - b
        sd = diff.std(ddof=1)
        d = diff.mean() / sd if sd > 0 else float('inf')
        out['tests'][split][f'dcs_vs_{other}'] = {
            'other_values': b.tolist(),
            'mean_diff_pp': float(diff.mean() * 100),
            't': float(t), 'p': float(p), 'df': len(seeds) - 1,
            'cohens_d': float(d),
        }

with open(DST) as f:
    rev = json.load(f)
rev['M4_tta_unified'] = out
with open(DST, 'w') as f:
    json.dump(rev, f, indent=2, ensure_ascii=False)

print(f"blocks={out['n_blocks']} errors={out['n_error_entries']}")
for split in ['iid', 'temporal']:
    print(f"\n[{split}] DCS vs X (paired t, df=4)")
    for k, v in out['tests'][split].items():
        if k.startswith('dcs_vs_'):
            print(f"  {k}: diff={v['mean_diff_pp']:+.2f}pp t={v['t']:.3f} "
                  f"p={v['p']:.4f} d={v['cohens_d']:.2f}")
