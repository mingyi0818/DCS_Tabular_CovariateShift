"""Combine TTA comparison results into a single 5-seed file.

Sources:
  - results/tta_comparison_results.json   (2 seeds: [42, 123], 2 splits)
  - results/tta_5seed_results.json        (3 seeds: [456, 789, 2024], 2 splits,
    BUT contains many duplicate entries that must be deduplicated)

Target:
  - results/tta_combined_5seed_results.json
    Expected: 5 seeds x 2 splits = 10 unique entries, each with 6 methods
    {random, knn, dcs, tent, adaptable, self_training}

This script does NOT run any model. It only combines/deduplicates existing
JSON results and recomputes the 5-seed summary statistics.
"""
import os
import sys
import json
import time
import numpy as np
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR

OLD_2SEED_PATH = os.path.join(RESULT_DIR, 'tta_comparison_results.json')
NEW_3SEED_PATH = os.path.join(RESULT_DIR, 'tta_5seed_results.json')
COMBINED_PATH = os.path.join(RESULT_DIR, 'tta_combined_5seed_results.json')

ALL_SEEDS = [42, 123, 456, 789, 2024]
ALL_SPLITS = ['iid', 'temporal']
ALL_METHODS = ['random', 'knn', 'dcs', 'tent', 'adaptable', 'self_training']


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def entry_key(e):
    """Unique key for a TTA result entry: (dataset, split, seed)."""
    return (e.get('dataset'), e.get('split'), e.get('seed'))


def load_and_dedup(path, label):
    """Load a TTA results file and deduplicate by (dataset, split, seed).

    The 5-seed file in particular has the same (dataset, split, seed) entry
    repeated 6 times (a known bug in the original generator). We keep only
    the first occurrence of each key.
    """
    if not os.path.exists(path):
        print(f"  [{label}] FILE NOT FOUND: {path}")
        return [], {}
    with open(path, 'r') as f:
        data = json.load(f)
    raw_results = data.get('results', [])
    print(f"  [{label}] Loaded {len(raw_results)} raw entries from {os.path.basename(path)}")

    deduped = OrderedDict()
    for e in raw_results:
        k = entry_key(e)
        if k in deduped:
            continue
        deduped[k] = e
    deduped_list = list(deduped.values())
    print(f"  [{label}] After deduplication: {len(deduped_list)} unique entries")

    # Report the unique (split, seed) pairs seen
    pairs = sorted([(e['split'], e['seed']) for e in deduped_list])
    print(f"  [{label}] Unique (split, seed) pairs: {pairs}")
    return deduped_list, deduped


def main():
    print("=" * 80)
    print("Combine TTA results into a single 5-seed file")
    print("=" * 80)

    print("\n[1/3] Loading source files...")
    old_results, old_map = load_and_dedup(OLD_2SEED_PATH, 'old-2seed')
    new_results, new_map = load_and_dedup(NEW_3SEED_PATH, 'new-3seed')

    print("\n[2/3] Merging into 5-seed combined file...")
    # Merge: prefer new 3-seed entries when there is key overlap (there
    # shouldn't be, since seeds are disjoint: [42,123] vs [456,789,2024]).
    combined_map = OrderedDict()
    for k, e in old_map.items():
        combined_map[k] = e
    for k, e in new_map.items():
        if k in combined_map:
            print(f"  Overlap detected for {k}; preferring new 3-seed entry.")
        combined_map[k] = e

    # Sort by (split_order, seed) for readability
    split_order = {'iid': 0, 'temporal': 1}
    combined_list = sorted(
        combined_map.values(),
        key=lambda e: (split_order.get(e['split'], 99), e['seed']),
    )

    # Verify we have all expected 5 seeds x 2 splits = 10 entries
    expected_keys = {(ds, sp, sd) for ds in ['adult'] for sp in ALL_SPLITS for sd in ALL_SEEDS}
    got_keys = set(entry_key(e) for e in combined_list)
    missing = expected_keys - got_keys
    extra = got_keys - expected_keys
    print(f"  Combined entries: {len(combined_list)} (expected {len(expected_keys)})")
    if missing:
        print(f"  MISSING keys: {sorted(missing)}")
    if extra:
        print(f"  EXTRA keys:   {sorted(extra)}")

    # Per-entry verification: each entry should have all 6 methods
    print("\n  Per-entry method check:")
    for e in combined_list:
        methods_present = sorted(e.get('results', {}).keys())
        n_methods = len(methods_present)
        missing_methods = set(ALL_METHODS) - set(methods_present)
        status = "OK" if n_methods == 6 and not missing_methods else "WARN"
        print(f"    [{status}] split={e['split']:<9} seed={e['seed']:<5} "
              f"methods={n_methods}/6 {methods_present}")

    # ---- Compute 5-seed summary (mean +/- std) per (split, method) ----
    print("\n[3/3] Computing 5-seed summary statistics...")
    summary = {}
    for split in ALL_SPLITS:
        for method in ALL_METHODS:
            accs, f1s, aucs = [], [], []
            seeds_used = []
            for e in combined_list:
                if e['split'] != split:
                    continue
                m_res = e.get('results', {}).get(method)
                if not m_res:
                    continue
                accs.append(m_res['accuracy'])
                f1s.append(m_res['f1_macro'])
                aucs.append(m_res['auc'])
                seeds_used.append(e['seed'])
            if accs:
                key = f"adult_{split}_{method}"
                summary[key] = {
                    'accuracy_mean': float(np.mean(accs)),
                    'accuracy_std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                    'f1_macro_mean': float(np.mean(f1s)),
                    'f1_macro_std': float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                    'auc_mean': float(np.mean(aucs)),
                    'auc_std': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                    'n_seeds': len(accs),
                    'seeds': sorted(seeds_used),
                }

    # ---- Build combined output object ----
    combined = {
        'experiment': 'dcs_vs_tta_baselines_5seed_combined',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'description': ('Combined 5-seed TTA comparison: DCS vs TTA baselines. '
                        'Merges the original 2-seed file (seeds 42,123) with the '
                        '3-seed extension file (seeds 456,789,2024). Duplicates '
                        'in the 3-seed file were removed.'),
        'config': {
            'dataset': 'adult',
            'splits': ALL_SPLITS,
            'seeds': ALL_SEEDS,
            'context_size': 10000,
            'methods': {
                'random': 'TabPFN-Random',
                'knn': 'TabPFN-KNN',
                'dcs': 'TabPFN-DCS',
                'tent': 'TabPFN-Tent',
                'adaptable': 'TabPFN-AdapTable',
                'self_training': 'TabPFN-SelfTrain',
            },
            'source_files': {
                'old_2seed': 'tta_comparison_results.json',
                'new_3seed': 'tta_5seed_results.json',
            },
            'note': ('The 2-seed file used full test set (n_test=7327); the '
                     '3-seed extension used n_test_max=2000. Both files share '
                     'identical methodology and 6 methods per entry.'),
        },
        'results': combined_list,
        'summary': summary,
    }

    with open(COMBINED_PATH, 'w') as f:
        json.dump(json_safe(combined), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved combined file to: {COMBINED_PATH}")

    # ---- Print 5-seed mean +/- std per split/method ----
    print("\n" + "=" * 80)
    print("5-SEED SUMMARY (mean +/- std)")
    print("=" * 80)
    for split in ALL_SPLITS:
        print(f"\n  --- Split: {split} ---")
        print(f"  {'Method':<16} {'Accuracy':<22} {'F1-Macro':<22} {'AUC':<22} {'N':<4}")
        print("  " + "-" * 86)
        for method in ALL_METHODS:
            key = f"adult_{split}_{method}"
            s = summary.get(key)
            if s:
                print(f"  {method:<16} "
                      f"{s['accuracy_mean']:.4f}+/-{s['accuracy_std']:.4f}  "
                      f"{s['f1_macro_mean']:.4f}+/-{s['f1_macro_std']:.4f}  "
                      f"{s['auc_mean']:.4f}+/-{s['auc_std']:.4f}  "
                      f"{s['n_seeds']:<4d}")

    # ---- Final verification ----
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    n_entries = len(combined_list)
    print(f"  Total entries:     {n_entries} (expected 10 = 5 seeds x 2 splits)")
    seeds_in_combined = sorted(set(e['seed'] for e in combined_list))
    splits_in_combined = sorted(set(e['split'] for e in combined_list))
    print(f"  Seeds present:     {seeds_in_combined}")
    print(f"  Splits present:    {splits_in_combined}")
    for split in ALL_SPLITS:
        seeds_for_split = sorted(set(e['seed'] for e in combined_list if e['split'] == split))
        print(f"    {split}: seeds={seeds_for_split} ({len(seeds_for_split)} seeds)")
    all_have_6 = all(len(e.get('results', {})) == 6 for e in combined_list)
    print(f"  All entries have 6 methods: {all_have_6}")
    if n_entries == 10 and all_have_6 and set(seeds_in_combined) == set(ALL_SEEDS):
        print("  PASS: 5 seeds x 2 splits = 10 entries, each with 6 methods.")
    else:
        print("  WARNING: verification failed (see above).")

    print("\n" + "=" * 80)
    print("TTA 5-seed combination complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
