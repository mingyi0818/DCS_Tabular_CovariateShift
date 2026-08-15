"""M7 Multi Train Fraction Experiment with FIXED 15% test window.

Tests DCS-Logistic vs TabPFN-Random at different train fractions while
keeping the test window fixed at the last 15% of data.

Protocol:
  - Sort data by temporal column (age for Adult)
  - test = last 15% of data (85% to 100%) — SAME for all fractions
  - train = first `fraction` of data (0% to `fraction`%)
  - Gap between train end and test start is unused

  - 60%: train=0-60%, test=85-100% (gap=60-85%, 25% unused)
  - 70%: train=0-70%, test=85-100% (gap=70-85%, matches main experiment)
  - 80%: train=0-80%, test=85-100% (gap=80-85%, 5% unused)

This ensures n_test is constant across fractions and the 70% fraction
exactly matches the main experiment's temporal split (n_test=7327).

5 seeds: 42, 123, 456, 789, 2024
Methods: DCS-Logistic, TabPFN-Random
Save to: results/multi_fraction_results.json

Run: cd d:\\ResearchPaperPrepare\\67_DCS_Tabular_CovariateShift
     python code/m7_multi_fraction.py
"""
import os
import sys
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR, DATASETS
from splits import SPLIT_CONFIG, load_raw_dataframe, encode_features
from context_shield_methods import (
    dcs_selection, random_context_selection, run_tabpfn,
    set_seed, json_safe,
)

SEEDS = [42, 123, 456, 789, 2024]
TRAIN_FRACTIONS = [0.6, 0.7, 0.8]
TEST_FRACTION = 0.15  # Fixed: last 15%
CONTEXT_SIZE = 10000
K_CLUSTERS = 50
DATASET = 'adult'


def sort_temporal(df, dataset_name, seed):
    """Sort dataframe by temporal column (same logic as splits.split_temporal)."""
    cfg = SPLIT_CONFIG[dataset_name]
    temporal_col = cfg['temporal_col']

    if temporal_col is None:
        raise ValueError(f"Dataset {dataset_name} has no temporal column")

    df_sorted = df.copy()

    if 'temporal_order' in cfg and cfg['temporal_order']:
        order_map = {m: i for i, m in enumerate(cfg['temporal_order'])}
        df_sorted['_temporal_order'] = df_sorted[temporal_col].map(
            lambda x: order_map.get(x, 0)
        )
        rng = np.random.RandomState(seed)
        df_sorted['_temporal_jitter'] = rng.uniform(0, 0.5, size=len(df_sorted))
        df_sorted = df_sorted.sort_values(['_temporal_order', '_temporal_jitter'])
        df_sorted = df_sorted.drop(['_temporal_order', '_temporal_jitter'], axis=1)
    else:
        rng = np.random.RandomState(seed)
        std_val = df_sorted[temporal_col].std()
        jitter_scale = 0.01 * std_val if std_val > 0 else 0.01
        df_sorted['_temporal_jitter'] = rng.uniform(0, jitter_scale, size=len(df_sorted))
        df_sorted['_sort_key'] = df_sorted[temporal_col] + df_sorted['_temporal_jitter']
        df_sorted = df_sorted.sort_values('_sort_key')
        df_sorted = df_sorted.drop(['_temporal_jitter', '_sort_key'], axis=1)

    return df_sorted


def prepare_fixed_test_split(dataset_name, train_fraction, seed):
    """Prepare split with fixed 15% test window (last 15% of data).

    Args:
        dataset_name: Key in DATASETS
        train_fraction: Fraction for training (0.6, 0.7, 0.8)
        seed: Random seed (for temporal jitter)

    Returns:
        dict with X_train, X_test, y_train, y_test, n_train, n_test, etc.
    """
    df, target_col = load_raw_dataframe(dataset_name)

    # Sort by temporal column
    df_sorted = sort_temporal(df, dataset_name, seed)

    n = len(df_sorted)
    n_train = int(n * train_fraction)
    # Fixed test window: last 15%
    n_test_start = int(n * (1.0 - TEST_FRACTION))

    train_df = df_sorted.iloc[:n_train].copy()
    test_df = df_sorted.iloc[n_test_start:].copy()

    # Encode features (fit on train, apply to test)
    X_train_df, y_train, encoders = encode_features(train_df, target_col)
    X_test_df, y_test, _ = encode_features(test_df, target_col, fit_df=train_df)

    # Ensure test has same columns as train
    for col in X_train_df.columns:
        if col not in X_test_df.columns:
            X_test_df[col] = 0
    X_test_df = X_test_df[X_train_df.columns]

    feature_names = list(X_train_df.columns)

    # Scale (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_test = scaler.transform(X_test_df.values)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'n_total': int(n),
        'train_fraction': float(train_fraction),
        'test_fraction': float(TEST_FRACTION),
        'n_test_start': int(n_test_start),
        'n_train_end': int(n_train),
        'gap_size': int(n_test_start - n_train),
    }


def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = 0.0
    if y_proba is not None:
        try:
            if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


def main():
    print("=" * 80)
    print("M7 Multi Train Fraction (Fixed 15% Test Window)")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Dataset: {DATASET}")
    print(f"Seeds: {SEEDS}")
    print(f"Train fractions: {TRAIN_FRACTIONS}")
    print(f"Test fraction: {TEST_FRACTION} (fixed)")
    print(f"Context size: {CONTEXT_SIZE}")
    print(f"K clusters: {K_CLUSTERS}")
    os.makedirs(RESULT_DIR, exist_ok=True)

    all_results = {
        'experiment': 'multi_train_fraction_fixed_test',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'reviewer_issue': 'M7/M10: statistical inference over context-selection seeds on a single fixed split',
        'config': {
            'dataset': DATASET,
            'split': 'temporal',
            'seeds': SEEDS,
            'train_fractions': TRAIN_FRACTIONS,
            'test_fraction': TEST_FRACTION,
            'test_window': 'last 15% (fixed across all fractions)',
            'methods': ['DCS-Logistic', 'TabPFN-Random'],
            'context_size': CONTEXT_SIZE,
            'K': K_CLUSTERS,
            'dcs_fix': 'largest_remainder_method',
        },
        'results': {},
    }

    for fraction in TRAIN_FRACTIONS:
        print(f"\n{'=' * 60}")
        print(f"Train fraction: {fraction}")
        print(f"{'=' * 60}")

        fraction_key = str(fraction)
        all_results['results'][fraction_key] = {
            'train_fraction': float(fraction),
            'seeds': [],
        }

        for seed in SEEDS:
            print(f"\n  [seed={seed}]")
            set_seed(seed)

            # Prepare split
            try:
                split_data = prepare_fixed_test_split(
                    DATASET, fraction, seed=seed
                )
            except Exception as e:
                print(f"    ERROR split failed: {e}")
                all_results['results'][fraction_key]['seeds'].append({
                    'seed': int(seed),
                    'error': f'Split failed: {e}',
                })
                continue

            X_train = split_data['X_train']
            X_test = split_data['X_test']
            y_train = split_data['y_train']
            y_test = split_data['y_test']
            n_train = split_data['n_train']
            n_test = split_data['n_test']

            print(f"    train={n_train}, test={n_test}, "
                  f"gap={split_data['gap_size']}")

            seed_result = {
                'seed': int(seed),
                'n_train': int(n_train),
                'n_test': int(n_test),
                'n_total': int(split_data['n_total']),
                'gap_size': int(split_data['gap_size']),
            }

            # --- DCS-Logistic ---
            print(f"    Running DCS-Logistic...", end=' ', flush=True)
            try:
                t0 = time.time()
                dcs_idx = dcs_selection(
                    X_train, X_test, CONTEXT_SIZE,
                    n_clusters=K_CLUSTERS, method='logistic', seed=seed
                )
                selection_time = time.time() - t0

                # Run TabPFN with selected context
                import tabpfn_client
                if not getattr(main, '_tabpfn_init', False):
                    tabpfn_client.init()
                    main._tabpfn_init = True
                from tabpfn_client import TabPFNClassifier

                X_ctx = X_train[dcs_idx]
                y_ctx = y_train[dcs_idx]

                clf = TabPFNClassifier()
                t0 = time.time()
                clf.fit(X_ctx, y_ctx)
                fit_time = time.time() - t0

                t0 = time.time()
                y_pred = clf.predict(X_test)
                predict_time = time.time() - t0

                try:
                    y_proba = clf.predict_proba(X_test)
                except Exception:
                    y_proba = None

                metrics = compute_metrics(y_test, y_pred, y_proba)
                metrics['selection_time'] = float(selection_time)
                metrics['fit_time'] = float(fit_time)
                metrics['predict_time'] = float(predict_time)
                metrics['n_context'] = int(len(y_ctx))

                seed_result['DCS-Logistic'] = metrics
                print(f"acc={metrics['accuracy']:.4f}, "
                      f"f1m={metrics['f1_macro']:.4f}, "
                      f"n_ctx={len(y_ctx)}, "
                      f"sel={selection_time:.2f}s")
            except Exception as e:
                seed_result['DCS-Logistic'] = {'error': str(e)}
                print(f"FAILED: {e}")

            # --- TabPFN-Random ---
            print(f"    Running TabPFN-Random...", end=' ', flush=True)
            try:
                t0 = time.time()
                random_idx = random_context_selection(
                    X_train, CONTEXT_SIZE, seed=seed
                )
                selection_time = time.time() - t0

                from tabpfn_client import TabPFNClassifier

                X_ctx = X_train[random_idx]
                y_ctx = y_train[random_idx]

                clf = TabPFNClassifier()
                t0 = time.time()
                clf.fit(X_ctx, y_ctx)
                fit_time = time.time() - t0

                t0 = time.time()
                y_pred = clf.predict(X_test)
                predict_time = time.time() - t0

                try:
                    y_proba = clf.predict_proba(X_test)
                except Exception:
                    y_proba = None

                metrics = compute_metrics(y_test, y_pred, y_proba)
                metrics['selection_time'] = float(selection_time)
                metrics['fit_time'] = float(fit_time)
                metrics['predict_time'] = float(predict_time)
                metrics['n_context'] = int(len(y_ctx))

                seed_result['TabPFN-Random'] = metrics
                print(f"acc={metrics['accuracy']:.4f}, "
                      f"f1m={metrics['f1_macro']:.4f}, "
                      f"n_ctx={len(y_ctx)}")
            except Exception as e:
                seed_result['TabPFN-Random'] = {'error': str(e)}
                print(f"FAILED: {e}")

            # Compute delta
            dcs_res = seed_result.get('DCS-Logistic', {})
            rand_res = seed_result.get('TabPFN-Random', {})
            if 'accuracy' in dcs_res and 'accuracy' in rand_res:
                delta = dcs_res['accuracy'] - rand_res['accuracy']
                seed_result['delta_accuracy'] = float(delta)
                seed_result['delta_pp'] = float(delta * 100)
                print(f"    Delta (DCS - Random): {delta*100:+.4f}pp")

            all_results['results'][fraction_key]['seeds'].append(seed_result)

            # Save incrementally
            with open(os.path.join(RESULT_DIR, 'multi_fraction_results.json'), 'w') as f:
                json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY: Mean +/- Std over seeds")
    print("=" * 80)
    print(f"{'Fraction':<10} {'Method':<16} {'Accuracy':<18} {'F1-Macro':<18} {'Delta':<12}")
    print("-" * 80)

    summary = {}
    for fraction in TRAIN_FRACTIONS:
        fk = str(fraction)
        fraction_data = all_results['results'][fk]

        # Check n_test consistency
        n_tests = [s.get('n_test') for s in fraction_data['seeds'] if 'n_test' in s]
        if n_tests:
            print(f"\n  Fraction {fraction}: n_test values = {n_tests} "
                  f"(all same: {len(set(n_tests)) == 1})")

        for method in ['DCS-Logistic', 'TabPFN-Random']:
            accs = [s[method]['accuracy'] for s in fraction_data['seeds']
                    if method in s and 'accuracy' in s.get(method, {})]
            f1s = [s[method]['f1_macro'] for s in fraction_data['seeds']
                   if method in s and 'f1_macro' in s.get(method, {})]
            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                mean_f1 = np.mean(f1s)
                std_f1 = np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0
                print(f"  {fraction:<10} {method:<16} "
                      f"{mean_acc:.4f}+/-{std_acc:.4f}  "
                      f"{mean_f1:.4f}+/-{std_f1:.4f}")

                summary[f"{fk}_{method}"] = {
                    'accuracy_mean': float(mean_acc),
                    'accuracy_std': float(std_acc),
                    'f1_macro_mean': float(mean_f1),
                    'f1_macro_std': float(std_f1),
                    'n_seeds': len(accs),
                    'values': accs,
                }

        # Delta
        dcs_accs = [s['DCS-Logistic']['accuracy'] for s in fraction_data['seeds']
                    if 'DCS-Logistic' in s and 'accuracy' in s.get('DCS-Logistic', {})]
        rand_accs = [s['TabPFN-Random']['accuracy'] for s in fraction_data['seeds']
                     if 'TabPFN-Random' in s and 'accuracy' in s.get('TabPFN-Random', {})]
        if dcs_accs and rand_accs and len(dcs_accs) == len(rand_accs):
            deltas = [d - r for d, r in zip(dcs_accs, rand_accs)]
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0
            print(f"  {'':10} {'Delta':<16} {mean_delta*100:+.4f}+/-{std_delta*100:.4f}pp")

            # Paired t-test
            from scipy import stats as sp_stats
            t_stat, t_p = sp_stats.ttest_rel(dcs_accs, rand_accs)
            print(f"  {'':10} {'t-test':<16} t={t_stat:.4f}, p={t_p:.6f}")

            summary[f"{fk}_delta"] = {
                'mean': float(mean_delta),
                'std': float(std_delta),
                'delta_pp': float(mean_delta * 100),
                'paired_t_test': {
                    't_statistic': float(t_stat),
                    'p_value': float(t_p),
                    'df': int(len(deltas) - 1),
                    'significant_at_0.05': bool(t_p < 0.05),
                },
                'values': deltas,
            }

    all_results['summary'] = summary

    # Verify n_test=7327 for 70% fraction
    fraction_70 = all_results['results'].get('0.7', {})
    n_tests_70 = [s.get('n_test') for s in fraction_70.get('seeds', []) if 'n_test' in s]
    if n_tests_70:
        print(f"\n  Verification: 70% fraction n_test = {n_tests_70[0]} "
              f"(expected 7327: {'PASS' if n_tests_70[0] == 7327 else 'FAIL'})")

    output_path = os.path.join(RESULT_DIR, 'multi_fraction_results.json')
    with open(output_path, 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    print("=" * 80)
    print("M7 Multi Train Fraction Experiment Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
