"""5-seed Sensitivity analysis using LOCAL GPU TabPFN.

Re-runs the sensitivity experiment with all 5 seeds [42, 123, 456, 789, 2024]
using LOCAL GPU TabPFN (device='cuda'), addressing the reviewer concern that
the original sensitivity_results.json only had 3 seeds.

Tests:
  1. n_clusters (K) sensitivity: [5, 10, 20, 30, 50, 100, 200] on Adult/temporal
  2. context_size sensitivity: [1000, 2000, 5000, 8000, 10000] on Adult/temporal

Method: DCS-Logistic (logistic regression domain classifier + k-means clustering)

Results saved to: results/sensitivity_5seed_results.json
"""
import os
import sys
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans

# Local TabPFN setup (must be before importing tabpfn)
os.environ['TABPFN_MODEL_CACHE_DIR'] = r'E:\datasets\tabpfn_models'
try:
    from tabpfn_client.config import get_access_token
    token = get_access_token()
    if token:
        os.environ['TABPFN_TOKEN'] = token
        os.environ['HF_TOKEN'] = token
        auth_path = os.path.expanduser('~/.cache/tabpfn/auth_token')
        os.makedirs(os.path.dirname(auth_path), exist_ok=True)
        with open(auth_path, 'w') as f:
            f.write(token)
except Exception:
    pass

from tabpfn import TabPFNClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR, DATASETS
from splits import SPLIT_CONFIG, load_raw_dataframe, encode_features
from context_shield_methods import dcs_selection, set_seed, json_safe

SEEDS = [42, 123, 456, 789, 2024]
DATASET = 'adult'
SPLIT = 'temporal'

# Sensitivity parameters (same as sensitivity_exp.py)
N_CLUSTERS_VALUES = [5, 10, 20, 30, 50, 100, 200]
CONTEXT_SIZE_VALUES = [1000, 2000, 5000, 8000, 10000]
DEFAULT_N_CLUSTERS = 50
DEFAULT_CONTEXT_SIZE = 10000

DEVICE = 'cuda'
N_TEST_MAX = 2000  # Subsample test for speed (matches comprehensive_local_gpu.py)


def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if y_proba.ndim > 1:
        proba_pos = y_proba[:, 1]
    else:
        proba_pos = y_proba
    try:
        auc = roc_auc_score(y_true, proba_pos)
    except Exception:
        auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


def run_local_tabpfn(X_ctx, y_ctx, X_test, y_test):
    t0 = time.time()
    clf = TabPFNClassifier(device=DEVICE)
    clf.fit(X_ctx, y_ctx)
    fit_time = time.time() - t0
    t0 = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - t0
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        y_proba = np.column_stack([1 - y_pred, y_pred])
    m = compute_metrics(y_test, y_pred, y_proba)
    m['fit_time'] = float(fit_time)
    m['predict_time'] = float(predict_time)
    m['n_context'] = int(len(y_ctx))
    return m


def sort_temporal(df, dataset_name, seed):
    """Sort by temporal column (matches comprehensive_local_gpu.py)."""
    cfg = SPLIT_CONFIG.get(dataset_name, {})
    tc = cfg.get('temporal_col')
    if tc is None:
        return df
    d = df.copy()
    if 'temporal_order' in cfg and cfg['temporal_order']:
        om = {m: i for i, m in enumerate(cfg['temporal_order'])}
        d['_o'] = d[tc].map(lambda x: om.get(x, 0))
        rng = np.random.RandomState(seed)
        d['_j'] = rng.uniform(0, 0.5, size=len(d))
        d = d.sort_values(['_o', '_j']).drop(['_o', '_j'], axis=1)
    else:
        rng = np.random.RandomState(seed)
        sv = d[tc].std()
        js = 0.01 * sv if sv > 0 else 0.01
        d['_j'] = rng.uniform(0, js, size=len(d))
        d['_k'] = d[tc] + d['_j']
        d = d.sort_values('_k').drop(['_j', '_k'], axis=1)
    return d


def prepare_split(dataset_name, seed):
    """Prepare data split matching comprehensive_local_gpu.py (subsamples test to N_TEST_MAX)."""
    df, tc = load_raw_dataframe(dataset_name)
    d = sort_temporal(df, dataset_name, seed)
    n = len(d)
    nt = int(n * 0.7); nts = int(n * 0.85)
    tr = d.iloc[:nt].copy(); te = d.iloc[nts:].copy()
    Xtr_df, ytr, _ = encode_features(tr, tc)
    Xte_df, yte, _ = encode_features(te, tc, fit_df=tr)
    for c in Xtr_df.columns:
        if c not in Xte_df.columns:
            Xte_df[c] = 0
    Xte_df = Xte_df[Xtr_df.columns]
    sc = StandardScaler()
    X_train = sc.fit_transform(Xtr_df.values)
    X_test = sc.transform(Xte_df.values)
    y_train, y_test = ytr, yte
    # Subsample test for speed (matches comprehensive_local_gpu.py)
    if len(X_test) > N_TEST_MAX:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_test), N_TEST_MAX, replace=False)
        X_test = X_test[idx]; y_test = y_test[idx]
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'n_train': len(X_train), 'n_test': len(X_test)}


def run_single(X_train, y_train, X_test, y_test, n_select, n_clusters, seed):
    """Run DCS-Logistic with specific parameters."""
    t0 = time.time()
    idx = dcs_selection(X_train, X_test, n_select,
                        n_clusters=n_clusters, method='logistic', seed=seed)
    selection_time = time.time() - t0

    metrics = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
    metrics['selection_time'] = float(selection_time)
    metrics['n_clusters'] = int(n_clusters)
    metrics['n_select'] = int(n_select)
    metrics['n_context_actual'] = int(len(idx))
    return metrics


def main():
    print("=" * 80)
    print("5-Seed Sensitivity Analysis using LOCAL GPU TabPFN")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Device: {DEVICE}, Max test: {N_TEST_MAX}")
    print(f"Seeds: {SEEDS}")
    print(f"Dataset: {DATASET}/{SPLIT}")
    print(f"Method: DCS-Logistic")
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Check GPU
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    all_results = {
        'experiment': 'sensitivity_analysis_5seed_local_gpu',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'dataset': DATASET,
            'split': SPLIT,
            'seeds': SEEDS,
            'n_clusters_values': N_CLUSTERS_VALUES,
            'context_size_values': CONTEXT_SIZE_VALUES,
            'default_n_clusters': DEFAULT_N_CLUSTERS,
            'default_context_size': DEFAULT_CONTEXT_SIZE,
            'tabpfn_mode': 'local_gpu',
            'device': DEVICE,
            'n_test_max': N_TEST_MAX,
            'method': 'DCS-Logistic',
        },
        'n_clusters_sensitivity': [],
        'context_size_sensitivity': [],
    }

    output_path = os.path.join(RESULT_DIR, 'sensitivity_5seed_results.json')

    # === 1. n_clusters sensitivity ===
    print(f"\n[1/2] n_clusters sensitivity (context_size={DEFAULT_CONTEXT_SIZE})...")
    for n_clusters in N_CLUSTERS_VALUES:
        for seed in SEEDS:
            print(f"  n_clusters={n_clusters}, seed={seed}...", end=' ', flush=True)
            try:
                set_seed(seed)
                split_data = prepare_split(DATASET, seed)
                X_tr = split_data['X_train']
                y_tr = split_data['y_train']
                X_te = split_data['X_test']
                y_te = split_data['y_test']

                metrics = run_single(X_tr, y_tr, X_te, y_te,
                                     DEFAULT_CONTEXT_SIZE, n_clusters, seed)
                print(f"acc={metrics['accuracy']:.4f}, sel={metrics['selection_time']:.2f}s")

                all_results['n_clusters_sensitivity'].append({
                    'n_clusters': int(n_clusters),
                    'seed': int(seed),
                    'metrics': metrics,
                })
            except Exception as e:
                print(f"FAILED: {e}")
                all_results['n_clusters_sensitivity'].append({
                    'n_clusters': int(n_clusters),
                    'seed': int(seed),
                    'error': str(e),
                })

            # Save incrementally
            with open(output_path, 'w') as f:
                json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    # === 2. context_size sensitivity ===
    print(f"\n[2/2] context_size sensitivity (n_clusters={DEFAULT_N_CLUSTERS})...")
    for ctx_size in CONTEXT_SIZE_VALUES:
        for seed in SEEDS:
            print(f"  context_size={ctx_size}, seed={seed}...", end=' ', flush=True)
            try:
                set_seed(seed)
                split_data = prepare_split(DATASET, seed)
                X_tr = split_data['X_train']
                y_tr = split_data['y_train']
                X_te = split_data['X_test']
                y_te = split_data['y_test']

                metrics = run_single(X_tr, y_tr, X_te, y_te,
                                     ctx_size, DEFAULT_N_CLUSTERS, seed)
                print(f"acc={metrics['accuracy']:.4f}, sel={metrics['selection_time']:.2f}s")

                all_results['context_size_sensitivity'].append({
                    'context_size': int(ctx_size),
                    'seed': int(seed),
                    'metrics': metrics,
                })
            except Exception as e:
                print(f"FAILED: {e}")
                all_results['context_size_sensitivity'].append({
                    'context_size': int(ctx_size),
                    'seed': int(seed),
                    'error': str(e),
                })

            with open(output_path, 'w') as f:
                json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY: n_clusters sensitivity (5 seeds)")
    print("=" * 80)
    print(f"{'n_clusters':<12} {'Accuracy':<18} {'F1-Macro':<18} {'Sel Time':<12} {'N'}")
    for nc in N_CLUSTERS_VALUES:
        accs = [r['metrics']['accuracy'] for r in all_results['n_clusters_sensitivity']
                if r['n_clusters'] == nc and 'metrics' in r]
        f1s = [r['metrics']['f1_macro'] for r in all_results['n_clusters_sensitivity']
               if r['n_clusters'] == nc and 'metrics' in r]
        sels = [r['metrics']['selection_time'] for r in all_results['n_clusters_sensitivity']
                if r['n_clusters'] == nc and 'metrics' in r]
        if accs:
            print(f"{nc:<12} {np.mean(accs):.4f}±{np.std(accs, ddof=1):.4f}  "
                  f"{np.mean(f1s):.4f}±{np.std(f1s, ddof=1):.4f}  "
                  f"{np.mean(sels):.2f}s  n={len(accs)}")

    print("\n" + "=" * 80)
    print("SUMMARY: context_size sensitivity (5 seeds)")
    print("=" * 80)
    print(f"{'ctx_size':<12} {'Accuracy':<18} {'F1-Macro':<18} {'Sel Time':<12} {'N'}")
    for cs in CONTEXT_SIZE_VALUES:
        accs = [r['metrics']['accuracy'] for r in all_results['context_size_sensitivity']
                if r['context_size'] == cs and 'metrics' in r]
        f1s = [r['metrics']['f1_macro'] for r in all_results['context_size_sensitivity']
               if r['context_size'] == cs and 'metrics' in r]
        sels = [r['metrics']['selection_time'] for r in all_results['context_size_sensitivity']
                if r['context_size'] == cs and 'metrics' in r]
        if accs:
            print(f"{cs:<12} {np.mean(accs):.4f}±{np.std(accs, ddof=1):.4f}  "
                  f"{np.mean(f1s):.4f}±{np.std(f1s, ddof=1):.4f}  "
                  f"{np.mean(sels):.2f}s  n={len(accs)}")

    # Final save
    all_results['timestamp_end'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(output_path, 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    print("=" * 80)
    print("5-Seed Sensitivity Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
