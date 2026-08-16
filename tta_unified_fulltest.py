"""Unified-protocol 5-seed TTA comparison: FULL test set (n_test=7,327), local GPU.

Replaces the mixed-protocol Table 16 data (tta_combined_5seed_results.json),
which merged a 2-seed full-test run (n_test=7327) with a 3-seed subsampled
run (n_test_max=2000). This script re-runs all 5 seeds x 2 splits x 6 methods
with the full test set so every entry shares one protocol, matching the
cloud protocol of Table 3 (n_test=7,327).

Resume support: completed (split, seed) combos are skipped on restart.
Output: results/tta_unified_5seed_results.json
"""
import os
import sys
import json
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tta_5seed_local as base
from config import RESULT_DIR
from splits import prepare_split
from context_shield_methods import set_seed, json_safe

SEEDS = [42, 123, 456, 789, 2024]
DATASET = 'adult'
SPLITS = ['iid', 'temporal']
CONTEXT_SIZE = 10000
OUTPUT_NAME = 'tta_unified_5seed_results.json'


def main():
    output_path = os.path.join(RESULT_DIR, OUTPUT_NAME)

    all_results = {
        'experiment': 'dcs_vs_tta_5seed_unified_fulltest',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'dataset': DATASET,
            'splits': SPLITS,
            'seeds': SEEDS,
            'context_size': CONTEXT_SIZE,
            'methods': {m: base.METHOD_LABELS[m] for m in base.METHODS},
            'tabpfn_mode': 'local_gpu',
            'device': base.DEVICE,
            'n_test': 'full (no subsampling)',
            'protocol': 'unified: identical to Table 3 test protocol (n_test=7,327)',
        },
        'results': [],
        'errors': [],
    }

    # Resume: keep already-completed (split, seed) blocks
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            prev = json.load(f)
        if prev.get('experiment') == all_results['experiment']:
            for blk in prev.get('results', []):
                if 'results' in blk and len(blk['results']) == len(base.METHODS):
                    all_results['results'].append(blk)
                    done.add((blk['split'], blk['seed']))
            all_results['errors'] = prev.get('errors', [])
            print(f"Resuming: {len(done)} (split, seed) blocks already complete")

    def save():
        with open(output_path, 'w') as f:
            json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    print(f"Running {DATASET}, splits={SPLITS}, seeds={SEEDS}, full test set")
    t_start = time.time()

    for seed in SEEDS:
        for split_type in SPLITS:
            if (split_type, seed) in done:
                continue
            print(f"\n[{DATASET}/{split_type}/seed={seed}] elapsed={time.time()-t_start:.0f}s")
            set_seed(seed)
            try:
                split_data = prepare_split(DATASET, split_type, seed=seed)
            except Exception as e:
                print(f"  ERROR preparing split: {e}")
                all_results['errors'].append({'split': split_type, 'seed': seed,
                                              'phase': 'prepare_split', 'error': str(e)})
                continue

            X_train, y_train = split_data['X_train'], split_data['y_train']
            X_test, y_test = split_data['X_test'], split_data['y_test']  # FULL test set
            print(f"  train={X_train.shape}, test={X_test.shape}")

            results_for_combo = {}
            for method in base.METHODS:
                label = base.METHOD_LABELS[method]
                print(f"    {label}...", end=' ', flush=True)
                t0 = time.time()
                try:
                    ctx = min(CONTEXT_SIZE, len(X_train))
                    metrics = base.run_tta_method(method, X_train, y_train, X_test, y_test,
                                                  n_select=ctx, seed=seed)
                    metrics['elapsed'] = float(time.time() - t0)
                    print(f"acc={metrics['accuracy']:.4f} ({metrics['elapsed']:.1f}s)")
                    results_for_combo[method] = metrics
                except Exception as e:
                    msg = f"{type(e).__name__}: {str(e)[:200]}"
                    print(f"FAILED: {msg}")
                    results_for_combo[method] = {'error': msg}
                    all_results['errors'].append({'split': split_type, 'seed': seed,
                                                  'method': method,
                                                  'traceback': traceback.format_exc()[:500]})
                # save after each combo
                all_results['results'] = [b for b in all_results['results']
                                          if not (b['split'] == split_type and b['seed'] == seed)]
                all_results['results'].append({
                    'dataset': DATASET, 'split': split_type, 'seed': seed,
                    'split_info': {'n_train': int(len(X_train)), 'n_test': int(len(X_test)),
                                   'n_features': int(X_train.shape[1])},
                    'context_size': int(min(CONTEXT_SIZE, len(X_train))),
                    'results': dict(results_for_combo),
                })
                save()

    # ---- Summary over all completed blocks ----
    summary = {}
    for split_type in SPLITS:
        for method in base.METHODS:
            accs, f1s, sels = [], [], []
            for blk in all_results['results']:
                if blk['split'] != split_type:
                    continue
                m = blk['results'].get(method)
                if m and 'accuracy' in m:
                    accs.append(m['accuracy'])
                    f1s.append(m['f1_macro'])
                    if 'selection_time' in m:
                        sels.append(m['selection_time'])
            key = f"{DATASET}_{split_type}_{base.METHOD_LABELS[method]}"
            summary[key] = {
                'accuracy_mean': float(np.mean(accs)) if accs else None,
                'accuracy_std': float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                'f1_macro_mean': float(np.mean(f1s)) if f1s else None,
                'f1_macro_std': float(np.std(f1s, ddof=1)) if len(f1s) > 1 else None,
                'selection_time_mean': float(np.mean(sels)) if sels else None,
                'selection_time_std': float(np.std(sels, ddof=1)) if len(sels) > 1 else None,
                'n_seeds': len(accs),
            }
    all_results['summary'] = summary
    save()

    print("\nSUMMARY (mean±std, n seeds)")
    for k, v in summary.items():
        if v['n_seeds'] > 0:
            print(f"  {k}: acc={v['accuracy_mean']:.4f}±{v['accuracy_std']:.4f} "
                  f"f1={v['f1_macro_mean']:.4f} sel={v['selection_time_mean']} n={v['n_seeds']}")
    print(f"\nTotal elapsed: {(time.time()-t_start)/60:.1f} min -> {output_path}")


if __name__ == '__main__':
    main()
