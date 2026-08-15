"""5-seed Extended dataset experiments using LOCAL GPU TabPFN.

Re-runs the Bank and Telco dataset experiments with all 5 seeds
[42, 123, 456, 789, 2024] using LOCAL GPU TabPFN (device='cuda'),
addressing the reviewer concern that extended_dataset_results.json
only had 3 seeds.

Datasets:
  - bank: 11162 rows, context_size=5000, n_clusters=30
  - telco: 7043 rows, context_size=3000, n_clusters=20

Methods: DCS-Logistic, Random
Split: temporal (feature-ordered)
Seeds: 42, 123, 456, 789, 2024

Results saved to: results/extended_5seed_results.json
"""
import os
import sys
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
from config import RESULT_DIR
from splits import SPLIT_CONFIG, load_raw_dataframe, encode_features
from context_shield_methods import dcs_selection, random_context_selection, set_seed, json_safe

SEEDS = [42, 123, 456, 789, 2024]

# Dataset configs (context_size and n_clusters match extended_dataset_exp.py)
DATASETS_CONFIG = {
    'bank': {'context_size': 5000, 'n_clusters': 30},
    'telco': {'context_size': 3000, 'n_clusters': 20},
}

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


def main():
    print("=" * 80)
    print("5-Seed Extended Dataset Experiments using LOCAL GPU TabPFN")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Device: {DEVICE}, Max test: {N_TEST_MAX}")
    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {list(DATASETS_CONFIG.keys())}")
    print(f"Split: temporal, Methods: DCS-Logistic, Random")
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Check GPU
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    all_results = {
        'experiment': 'extended_dataset_5seed_local_gpu',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'datasets': list(DATASETS_CONFIG.keys()),
            'split': 'temporal',
            'seeds': SEEDS,
            'methods': ['TabPFN-DCS-Logistic', 'TabPFN-Random'],
            'tabpfn_mode': 'local_gpu',
            'device': DEVICE,
            'n_test_max': N_TEST_MAX,
        },
        'results': [],
    }

    output_path = os.path.join(RESULT_DIR, 'extended_5seed_results.json')

    for ds_name, ds_cfg in DATASETS_CONFIG.items():
        context_size = ds_cfg['context_size']
        n_clusters = ds_cfg['n_clusters']

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (ctx={context_size}, K={n_clusters})")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n  [{ds_name}/temporal/seed={seed}]")
            set_seed(seed)

            try:
                split_data = prepare_split(ds_name, seed)
            except Exception as e:
                print(f"    ERROR preparing split: {e}")
                all_results['results'].append({
                    'dataset': ds_name, 'split': 'temporal', 'seed': seed,
                    'method': 'ALL', 'error': str(e),
                })
                with open(output_path, 'w') as f:
                    json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
                continue

            X_train = split_data['X_train']
            y_train = split_data['y_train']
            X_test = split_data['X_test']
            y_test = split_data['y_test']
            n_train = split_data['n_train']
            n_test = split_data['n_test']
            print(f"    train={n_train}, test={n_test}")

            # Skip context selection if train <= context_size
            if n_train <= context_size:
                print(f"    Skip context selection (train={n_train} <= ctx={context_size})")
                # Run TabPFN with full data as Random
                try:
                    m = run_local_tabpfn(X_train, y_train, X_test, y_test)
                    m['selection_time'] = 0.0
                    print(f"    TabPFN-Random (full)  acc={m['accuracy']:.4f}")
                    all_results['results'].append({
                        'dataset': ds_name, 'split': 'temporal', 'seed': seed,
                        'method': 'TabPFN-Random', 'metrics': m,
                        'n_train': int(n_train), 'n_test': int(n_test),
                        'context_size': context_size,
                    })
                except Exception as e:
                    print(f"    TabPFN-Random FAILED: {e}")
                    all_results['results'].append({
                        'dataset': ds_name, 'split': 'temporal', 'seed': seed,
                        'method': 'TabPFN-Random', 'error': str(e),
                    })
                with open(output_path, 'w') as f:
                    json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
                continue

            # --- TabPFN-Random ---
            try:
                t0 = time.time()
                idx = random_context_selection(X_train, context_size, seed=seed)
                sel_time = time.time() - t0
                m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
                m['selection_time'] = float(sel_time)
                print(f"    TabPFN-Random         acc={m['accuracy']:.4f}")
                all_results['results'].append({
                    'dataset': ds_name, 'split': 'temporal', 'seed': seed,
                    'method': 'TabPFN-Random', 'metrics': m,
                    'n_train': int(n_train), 'n_test': int(n_test),
                    'context_size': context_size,
                })
            except Exception as e:
                print(f"    TabPFN-Random FAILED: {e}")
                all_results['results'].append({
                    'dataset': ds_name, 'split': 'temporal', 'seed': seed,
                    'method': 'TabPFN-Random', 'error': str(e),
                })

            # --- TabPFN-DCS-Logistic ---
            try:
                t0 = time.time()
                idx = dcs_selection(X_train, X_test, context_size,
                                    n_clusters=n_clusters, method='logistic', seed=seed)
                sel_time = time.time() - t0
                m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
                m['selection_time'] = float(sel_time)
                print(f"    TabPFN-DCS-Logistic   acc={m['accuracy']:.4f}")
                all_results['results'].append({
                    'dataset': ds_name, 'split': 'temporal', 'seed': seed,
                    'method': 'TabPFN-DCS-Logistic', 'metrics': m,
                    'n_train': int(n_train), 'n_test': int(n_test),
                    'context_size': context_size,
                })
            except Exception as e:
                print(f"    TabPFN-DCS-Logistic FAILED: {e}")
                all_results['results'].append({
                    'dataset': ds_name, 'split': 'temporal', 'seed': seed,
                    'method': 'TabPFN-DCS-Logistic', 'error': str(e),
                })

            # Save incrementally
            with open(output_path, 'w') as f:
                json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY: Mean ± Std over 5 seeds")
    print("=" * 80)
    print(f"{'Dataset':<10} {'Method':<25} {'Accuracy':<18} {'F1-Macro':<18} {'N'}")

    summary = {}
    for ds_name in DATASETS_CONFIG:
        for method in ['TabPFN-Random', 'TabPFN-DCS-Logistic']:
            accs = [r['metrics']['accuracy'] for r in all_results['results']
                    if r['dataset'] == ds_name and r.get('method') == method
                    and r.get('metrics')]
            f1s = [r['metrics']['f1_macro'] for r in all_results['results']
                   if r['dataset'] == ds_name and r.get('method') == method
                   and r.get('metrics')]
            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                mean_f1 = np.mean(f1s)
                std_f1 = np.std(f1s, ddof=1) if len(f1s) > 1 else 0.0
                key = f"{ds_name}_temporal_{method}"
                summary[key] = {
                    'accuracy_mean': float(mean_acc),
                    'accuracy_std': float(std_acc),
                    'f1_macro_mean': float(mean_f1),
                    'f1_macro_std': float(std_f1),
                    'n_seeds': len(accs),
                }
                print(f"{ds_name:<10} {method:<25} "
                      f"{mean_acc:.4f}±{std_acc:.4f}  "
                      f"{mean_f1:.4f}±{std_f1:.4f}  n={len(accs)}")

    # Improvement analysis
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS (DCS-Logistic vs Random)")
    print("=" * 80)
    for ds_name in DATASETS_CONFIG:
        random_key = f"{ds_name}_temporal_TabPFN-Random"
        dcs_key = f"{ds_name}_temporal_TabPFN-DCS-Logistic"
        r = summary.get(random_key, {})
        d = summary.get(dcs_key, {})
        if r and d:
            delta = (d['accuracy_mean'] - r['accuracy_mean']) * 100
            print(f"  {ds_name}: Random={r['accuracy_mean']:.4f}, "
                  f"DCS-Logistic={d['accuracy_mean']:.4f}, delta={delta:+.2f}pp")

    all_results['summary'] = summary
    all_results['timestamp_end'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(output_path, 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    print("=" * 80)
    print("5-Seed Extended Dataset Experiments Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
