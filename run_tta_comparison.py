"""DCS vs TTA Baselines Comparison Experiment.

Runs DCS (our method) against three Test-Time Adaptation baselines on
the Adult dataset under IID and temporal covariate shift:

Methods compared:
  1. TabPFN-Random      — random context (lower bound)
  2. TabPFN-KNN         — KNN context selection
  3. TabPFN-DCS         — our Diversity-Constrained Density-Ratio Selection
  4. TabPFN-Tent        — entropy-minimizing context selection (Wang 2021)
  5. TabPFN-AdapTable   — shift-aware uncertainty calibration (Kim 2024)
  6. TabPFN-SelfTrain   — self-training with pseudo-labels

Datasets:
  - Adult (IID split):   random 70/15/15, no covariate shift
  - Adult (Temporal split): sorted by age, shift between train/test

If TabReD datasets are available (cooking-time, weather), they are also
tested as additional real-world temporal shift datasets.

Results saved to: results/tta_comparison_results.json

Run:
    cd d:\\ResearchPaperPrepare\\67_DCS_Tabular_CovariateShift
    python code/run_tta_comparison.py
"""
import os
import sys
import json
import time
import traceback
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR, CONFIG
from splits import prepare_split, get_supported_splits
from context_shield_methods import json_safe, compute_metrics
from tta_baselines import run_tta_method

# ============================================================================
# Experiment Configuration
# ============================================================================

# Datasets and splits to test
# Adult is the primary dataset; TabReD datasets are tested if available
PRIMARY_DATASETS = ['adult']
PRIMARY_SPLITS = ['iid', 'temporal']

# Seeds for multi-run statistics
SEEDS = [42, 123]  # Reduced to 2 seeds for TTA experiments (TabPFN API calls are slow)

# Context size (TabPFN limit)
CONTEXT_SIZE = 10000

# Methods to compare
METHODS = [
    'random',        # TabPFN-Random: random context (baseline)
    'knn',           # TabPFN-KNN: KNN context selection
    'dcs',           # TabPFN-DCS: our method
    'tent',          # TabPFN-Tent: entropy minimization
    'adaptable',     # TabPFN-AdapTable: uncertainty calibration
    'self_training', # TabPFN-SelfTrain: pseudo-label augmentation
]

# Human-readable method names for reporting
METHOD_LABELS = {
    'random': 'TabPFN-Random',
    'knn': 'TabPFN-KNN',
    'dcs': 'TabPFN-DCS',
    'tent': 'TabPFN-Tent',
    'adaptable': 'TabPFN-AdapTable',
    'self_training': 'TabPFN-SelfTrain',
}


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_single_experiment(dataset_name, split_type, seed, context_size=None):
    """Run all TTA methods on one dataset/split/seed combination.

    Args:
        dataset_name: dataset key (e.g., 'adult', 'cooking-time')
        split_type: 'iid', 'temporal', or 'tabred_default'
        seed: random seed
        context_size: context size override

    Returns:
        dict mapping method_name -> metrics dict
    """
    if context_size is None:
        context_size = CONTEXT_SIZE

    # Load data
    if dataset_name in PRIMARY_DATASETS:
        # Use standard splits.py framework
        split_data = prepare_split(dataset_name, split_type, seed=seed)
        if split_data is None:
            return None
        X_train = split_data['X_train']
        y_train = split_data['y_train']
        X_test = split_data['X_test']
        y_test = split_data['y_test']
        split_info = split_data['split_info']
    else:
        # Try TabReD loader
        try:
            from tabred_loader import load_tabred_dataset
            data = load_tabred_dataset(dataset_name, max_train=context_size,
                                       max_test=2000, seed=seed)
            if data is None:
                return None
            X_train = data['X_train']
            y_train = data['y_train']
            X_test = data['X_test']
            y_test = data['y_test']
            split_info = data['split_info']
        except ImportError:
            print(f"  [SKIP] tabred_loader not available for {dataset_name}")
            return None

    print(f"  train={X_train.shape}, test={X_test.shape}, "
          f"features={X_train.shape[1]}, classes={len(np.unique(y_train))}")

    results = {}

    for method in METHODS:
        label = METHOD_LABELS[method]
        print(f"    Running {label}...", end=' ', flush=True)
        t0 = time.time()

        try:
            # Adjust context size if train is smaller
            ctx = min(context_size, len(X_train))

            metrics = run_tta_method(method, X_train, y_train, X_test, y_test,
                                     n_select=ctx, seed=seed)

            elapsed = time.time() - t0
            if 'error' in metrics:
                print(f"FAILED ({elapsed:.1f}s): {metrics['error'][:80]}")
                results[method] = {
                    'error': metrics['error'],
                    'elapsed': float(elapsed),
                }
            else:
                print(f"acc={metrics['accuracy']:.4f} f1m={metrics['f1_macro']:.4f} "
                      f"({elapsed:.1f}s)")
                metrics['elapsed'] = float(elapsed)
                results[method] = metrics

        except Exception as e:
            elapsed = time.time() - t0
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"FAILED ({elapsed:.1f}s): {error_msg}")
            results[method] = {
                'error': error_msg,
                'elapsed': float(elapsed),
                'traceback': traceback.format_exc()[:500],
            }

    return {
        'dataset': dataset_name,
        'split': split_type,
        'seed': seed,
        'split_info': {
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'n_features': int(X_train.shape[1]),
            'n_classes': int(len(np.unique(y_train))),
        },
        'context_size': int(min(context_size, len(X_train))),
        'results': results,
    }


def run_tabred_experiments(seeds, context_size=5000):
    """Run TTA comparison on TabReD datasets if available.

    TabReD datasets use a smaller context size because:
    1. They have more features (192 for cooking-time) which increases
       TabPFN memory usage.
    2. We subsample test data for speed.

    Returns list of experiment results, or empty list if not available.
    """
    try:
        from tabred_loader import check_tabred_available
        availability = check_tabred_available()
    except ImportError:
        return []

    tabred_results = []
    for ds_name, available in availability.items():
        if not available:
            print(f"\n  [SKIP] TabReD dataset '{ds_name}' not available")
            continue

        print(f"\n{'='*60}")
        print(f"TabReD Dataset: {ds_name}")
        print(f"{'='*60}")

        for seed in seeds:
            print(f"\n  [{ds_name}/tabred_default/seed={seed}]")
            set_seed(seed)

            result = run_single_experiment(
                ds_name, 'tabred_default', seed, context_size=context_size
            )

            if result is not None:
                tabred_results.append(result)

    return tabred_results


def compute_summary(all_results):
    """Compute mean ± std summary across seeds for each method.

    Args:
        all_results: list of per-seed experiment result dicts

    Returns:
        summary dict keyed by 'dataset_split_method' with mean/std metrics
    """
    summary = {}

    # Group results by dataset + split
    groups = {}
    for exp in all_results:
        key = f"{exp['dataset']}_{exp['split']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)

    for group_key, experiments in groups.items():
        for method in METHODS:
            metrics_list = []
            for exp in experiments:
                m = exp['results'].get(method)
                if m and 'error' not in m:
                    metrics_list.append(m)

            if not metrics_list:
                continue

            accs = [m['accuracy'] for m in metrics_list]
            f1s = [m['f1_macro'] for m in metrics_list]
            aucs = [m.get('auc', 0) for m in metrics_list]

            summary_key = f"{group_key}_{method}"
            summary[summary_key] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                'f1_macro_mean': float(np.mean(f1s)),
                'f1_macro_std': float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                'auc_mean': float(np.mean(aucs)),
                'auc_std': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                'n_seeds': len(metrics_list),
            }

    return summary


def print_summary_table(summary, all_results):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print("SUMMARY: DCS vs TTA Baselines (Mean ± Std over seeds)")
    print("=" * 100)

    # Get unique dataset/split combinations
    combos = sorted(set(
        f"{exp['dataset']}_{exp['split']}" for exp in all_results
    ))

    for combo in combos:
        print(f"\n  [{combo}]")
        print(f"  {'Method':<25} {'Accuracy':<18} {'F1-Macro':<18} {'AUC':<12} {'N'}")
        print(f"  {'-'*80}")

        for method in METHODS:
            label = METHOD_LABELS[method]
            key = f"{combo}_{method}"
            s = summary.get(key)
            if s:
                acc_str = f"{s['accuracy_mean']:.4f}±{s['accuracy_std']:.4f}"
                f1_str = f"{s['f1_macro_mean']:.4f}±{s['f1_macro_std']:.4f}"
                auc_str = f"{s['auc_mean']:.4f}"
                print(f"  {label:<25} {acc_str:<18} {f1_str:<18} {auc_str:<12} {s['n_seeds']}")
            else:
                print(f"  {label:<25} {'N/A':<18} {'N/A':<18} {'N/A':<12} 0")

    # Improvement analysis vs Random baseline
    print("\n" + "=" * 100)
    print("IMPROVEMENT ANALYSIS (vs TabPFN-Random)")
    print("=" * 100)

    for combo in combos:
        baseline_key = f"{combo}_random"
        baseline = summary.get(baseline_key, {})
        if not baseline:
            continue

        baseline_acc = baseline['accuracy_mean']
        print(f"\n  [{combo}] Baseline (Random) = {baseline_acc:.4f}")
        print(f"  {'Method':<25} {'Accuracy':<12} {'Δ (pp)':<10} {'F1-Macro':<12}")
        print(f"  {'-'*60}")

        for method in METHODS:
            if method == 'random':
                continue
            label = METHOD_LABELS[method]
            key = f"{combo}_{method}"
            s = summary.get(key)
            if s:
                delta = (s['accuracy_mean'] - baseline_acc) * 100
                print(f"  {label:<25} {s['accuracy_mean']:.4f}      "
                      f"{delta:+.2f}      {s['f1_macro_mean']:.4f}")


def ensure_adult_dataset():
    """Ensure the Adult dataset is available; download if missing."""
    from config import DATASETS
    adult_path = DATASETS['adult']['path']
    if os.path.exists(adult_path):
        # Quick sanity check
        import pandas as pd
        try:
            df = pd.read_csv(adult_path, nrows=5)
            if 'income' in df.columns:
                print(f"  [OK] Adult dataset found: {adult_path}")
                return True
        except Exception:
            pass

    print(f"  [INFO] Adult dataset not found at {adult_path}")
    print(f"  Attempting to download...")
    try:
        from download_adult import download_adult
        return download_adult()
    except ImportError:
        # Try running download_adult.py as subprocess
        import subprocess
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'download_adult.py')
        result = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            return os.path.exists(adult_path)
        else:
            print(f"  [FAIL] download_adult.py failed: {result.stderr[:300]}")
            return False
    except Exception as e:
        print(f"  [FAIL] Could not download Adult dataset: {e}")
        return False


def main():
    print("=" * 100)
    print("DCS vs TTA Baselines Comparison Experiment")
    print("=" * 100)
    print(f"Primary datasets: {PRIMARY_DATASETS}")
    print(f"Splits: {PRIMARY_SPLITS}")
    print(f"Seeds: {SEEDS}")
    print(f"Context size: {CONTEXT_SIZE}")
    print(f"Methods: {[METHOD_LABELS[m] for m in METHODS]}")
    print(f"Results file: {os.path.join(RESULT_DIR, 'tta_comparison_results.json')}")
    print("=" * 100)

    # Ensure Adult dataset is available
    print("\n--- Checking Adult dataset ---")
    if not ensure_adult_dataset():
        print("  [ERROR] Adult dataset is required but could not be loaded.")
        print("          Please run: python code/download_adult.py")
        print("          Or download manually from: https://archive.ics.uci.edu/dataset/2/adult")
        return

    all_results = {
        'experiment': 'dcs_vs_tta_baselines',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'primary_datasets': PRIMARY_DATASETS,
            'primary_splits': PRIMARY_SPLITS,
            'seeds': SEEDS,
            'context_size': CONTEXT_SIZE,
            'methods': {m: METHOD_LABELS[m] for m in METHODS},
        },
        'results': [],
        'tabred_results': [],
        'summary': {},
        'errors': [],
    }

    # ---- Part 1: Primary datasets (Adult) ----
    print("\n" + "=" * 80)
    print("PART 1: Primary Datasets (Adult)")
    print("=" * 80)

    for ds_name in PRIMARY_DATASETS:
        supported = get_supported_splits(ds_name)
        for split_type in PRIMARY_SPLITS:
            if split_type not in supported:
                print(f"\n  [SKIP] {ds_name} does not support {split_type} split")
                continue

            for seed in SEEDS:
                print(f"\n[{ds_name}/{split_type}/seed={seed}]")
                set_seed(seed)

                exp_result = run_single_experiment(ds_name, split_type, seed)
                if exp_result is not None:
                    all_results['results'].append(exp_result)

                    # Save incrementally
                    output_path = os.path.join(RESULT_DIR, 'tta_comparison_results.json')
                    with open(output_path, 'w') as f:
                        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    # ---- Part 2: TabReD datasets (if available) ----
    print("\n" + "=" * 80)
    print("PART 2: TabReD Datasets (real temporal shift)")
    print("=" * 80)

    try:
        tabred_results = run_tabred_experiments(SEEDS, context_size=5000)
        all_results['tabred_results'] = tabred_results
    except Exception as e:
        error_msg = f"TabReD experiments failed: {type(e).__name__}: {e}"
        print(f"\n  [ERROR] {error_msg}")
        all_results['errors'].append({
            'phase': 'tabred_experiments',
            'error': error_msg,
            'traceback': traceback.format_exc()[:500],
        })

    # ---- Summary ----
    print("\n" + "=" * 100)
    print("Computing summary statistics...")
    print("=" * 100)

    # Combine primary and tabred results for summary
    combined_results = all_results['results'] + all_results['tabred_results']
    summary = compute_summary(combined_results)
    all_results['summary'] = summary

    print_summary_table(summary, combined_results)

    # ---- Save final results ----
    all_results['timestamp_end'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    output_path = os.path.join(RESULT_DIR, 'tta_comparison_results.json')
    with open(output_path, 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    print(f"\n{'='*100}")
    print(f"Results saved to: {output_path}")
    print(f"Total experiments: {len(all_results['results'])} primary + "
          f"{len(all_results['tabred_results'])} TabReD")
    print(f"{'='*100}")
    print("DCS vs TTA Baselines Experiment Complete!")
    print("=" * 100)


if __name__ == '__main__':
    main()
