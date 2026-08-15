"""Extract all TabReD datasets and run DCS experiments.
Run DCS+TabPFN vs Random+TabPFN on all 8 TabReD datasets using time-based splits.
Data source: E:\datasets\tabred_data\preprocessed\*.tabred
"""
import os, sys, json, time, tarfile, numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              mean_squared_error, r2_score)
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
from context_shield_methods import set_seed, json_safe

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

from tabpfn import TabPFNClassifier, TabPFNRegressor

DATA_DIR = r"E:\datasets\tabred_data\preprocessed"
EXTRACT_DIR = r"E:\datasets\tabred_extracted"
K_CLUSTERS = 50
N_CONTEXT = 10000  # TabPFN max context
N_TEST_MAX = 2000  # Subsample test set for speed
SEEDS = [42, 123, 456, 789, 2024]

DATASETS = {
    'homesite-insurance': 'classification',
    'ecom-offers': 'classification',
    'homecredit-default': 'classification',
    'sberbank-housing': 'regression',
    'cooking-time': 'regression',
    'delivery-eta': 'regression',
    'maps-routing': 'regression',
    'weather': 'regression',
}


def extract_all():
    """Extract all .tabred files."""
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    for name in DATASETS:
        out_dir = os.path.join(EXTRACT_DIR, name)
        if os.path.exists(os.path.join(out_dir, 'info.json')):
            print(f"  {name} already extracted")
            continue
        tabred_file = os.path.join(DATA_DIR, f"{name}.tabred")
        if not os.path.exists(tabred_file):
            print(f"  {name}: file not found!")
            continue
        print(f"  Extracting {name}...")
        with tarfile.open(tabred_file, 'r') as t:
            t.extractall(EXTRACT_DIR)


def load_dataset(name):
    """Load a TabReD dataset and return features, targets, and default split."""
    data_dir = os.path.join(EXTRACT_DIR, name)
    if not os.path.exists(data_dir):
        return None

    # Load info
    with open(os.path.join(data_dir, 'info.json'), 'r') as f:
        info = json.load(f)

    # Load features
    features = []
    for fname in ['x_num.npy', 'x_cat.npy', 'x_bin.npy']:
        fp = os.path.join(data_dir, fname)
        if os.path.exists(fp):
            arr = np.load(fp, allow_pickle=True)
            if arr.dtype.kind in ['U', 'O', 'S']:
                # Categorical strings - encode
                le = LabelEncoder()
                flat = arr.ravel()
                # Handle 2D arrays
                if arr.ndim == 2:
                    encoded = np.zeros_like(arr, dtype=np.float32)
                    for col in range(arr.shape[1]):
                        encoded[:, col] = le.fit_transform(arr[:, col].astype(str))
                    arr = encoded
                else:
                    arr = le.fit_transform(flat).reshape(arr.shape).astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            features.append(arr.astype(np.float32))

    if not features:
        return None

    X = np.concatenate(features, axis=1)
    y = np.load(os.path.join(data_dir, 'y.npy'), allow_pickle=True)

    # Load default split (time-based)
    split_dir = os.path.join(data_dir, 'splits', 'default')
    train_idx = np.load(os.path.join(split_dir, 'train.npy'))
    val_idx = np.load(os.path.join(split_dir, 'val.npy'))
    test_idx = np.load(os.path.join(split_dir, 'test.npy'))

    return {
        'X': X, 'y': y, 'info': info,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx
    }


def estimate_density_ratio(X_train, X_test, method='logistic', seed=42):
    """Estimate density ratios using logistic regression."""
    n_train = X_train.shape[0]
    n_test = min(X_test.shape[0], 10000)
    rng = np.random.RandomState(seed)
    test_sample = X_test[rng.choice(len(X_test), n_test, replace=False)]

    X_combined = np.vstack([X_train, test_sample])
    y_combined = np.concatenate([np.zeros(n_train), np.ones(n_test)])

    try:
        clf = LogisticRegression(max_iter=200, random_state=seed, n_jobs=None)
        clf.fit(X_combined, y_combined)
        log_ratios = clf.predict_proba(X_train)[:, 1] - clf.predict_proba(X_train)[:, 0]
        ratios = np.exp(log_ratios)
        ratios = np.clip(ratios, 1e-10, 1e10)
    except Exception:
        ratios = np.ones(n_train)

    return ratios / (ratios.sum() + 1e-10) * n_train


def dcs_selection(X_train, X_test, n_select, n_clusters=50, seed=42):
    """DCS: Density-ratio + k-means diversity selection."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)

    dr = estimate_density_ratio(X_train, X_test, method='logistic', seed=seed)
    n_clusters = min(n_clusters, n_train, n_select)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(X_train)

    # Compute cluster-level mean density ratio
    cluster_ratios = np.zeros(n_clusters)
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_ratios[c] = dr[mask].mean() if mask.sum() > 0 else 0
        cluster_sizes[c] = mask.sum()

    # Allocate samples per cluster proportional to density ratio
    cluster_weights = cluster_ratios * cluster_sizes
    total_weight = cluster_weights.sum()
    if total_weight == 0:
        allocation = np.full(n_clusters, n_select // n_clusters)
    else:
        allocation = np.floor(cluster_weights / total_weight * n_select).astype(int)
        allocation = np.maximum(1, allocation)
        allocation = np.minimum(allocation, cluster_sizes)
        # Fix remaining
        remaining = n_select - allocation.sum()
        if remaining > 0:
            fractional = cluster_weights / total_weight * n_select - np.floor(cluster_weights / total_weight * n_select)
            for idx in np.argsort(-fractional):
                if remaining <= 0:
                    break
                if allocation[idx] < cluster_sizes[idx]:
                    allocation[idx] += 1
                    remaining -= 1

    # Select top samples by density ratio within each cluster
    selected = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_indices = np.where(mask)[0]
        cluster_dr = dr[mask]
        n_from_cluster = min(allocation[c], len(cluster_indices))
        if n_from_cluster <= 0:
            continue
        top_indices = cluster_indices[np.argsort(-cluster_dr)[:n_from_cluster]]
        selected.extend(top_indices.tolist())

    return np.array(selected[:n_select])


def random_selection(X_train, n_select, seed=42):
    """Random context selection."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)
    rng = np.random.RandomState(seed)
    return rng.choice(n_train, n_select, replace=False)


def run_experiment(name, task_type, seed=42):
    """Run DCS vs Random on a TabReD dataset."""
    data = load_dataset(name)
    if data is None:
        return {'error': f'Could not load {name}'}

    X = data['X']
    y = data['y']

    # Get train and test sets using default (time-based) split
    train_idx = data['train_idx']
    test_idx = data['test_idx']

    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Subsample test set for speed
    if len(X_test) > N_TEST_MAX:
        rng = np.random.RandomState(seed)
        test_sample = rng.choice(len(X_test), N_TEST_MAX, replace=False)
        X_test = X_test[test_sample]
        y_test = y_test[test_sample]

    # Handle features > 500 (TabPFN limit)
    max_features = 500
    if X_train_full.shape[1] > max_features:
        # Use only first max_features columns
        X_train_full = X_train_full[:, :max_features]
        X_test = X_test[:, :max_features]

    # Impute NaN values and scale features
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_full = imputer.fit_transform(X_train_full)
    X_test = imputer.transform(X_test)
    # Fill any remaining NaN (e.g., from all-NaN columns) with 0
    X_train_full = np.nan_to_num(X_train_full, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    # Fill any NaN from zero-variance columns after scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

    # Handle target encoding for classification
    if task_type == 'classification':
        le = LabelEncoder()
        y_train_full = le.fit_transform(y_train_full)
        y_test = le.transform(y_test)
        n_classes = len(le.classes_)
        if n_classes > 2:
            return {'error': f'{name} has {n_classes} classes, only binary supported'}

    n_train = len(X_train_scaled)
    n_test = len(X_test_scaled)
    n_features = X_train_scaled.shape[1]
    print(f"    train={n_train}, test={n_test}, features={n_features}")

    results = {}

    # DCS selection
    t0 = time.time()
    dcs_idx = dcs_selection(X_train_scaled, X_test_scaled, min(N_CONTEXT, n_train),
                            n_clusters=min(K_CLUSTERS, n_train), seed=seed)
    selection_time = time.time() - t0

    X_ctx_dcs = X_train_scaled[dcs_idx]
    y_ctx_dcs = y_train_full[dcs_idx]

    # Random selection
    rnd_idx = random_selection(X_train_scaled, min(N_CONTEXT, n_train), seed=seed)
    X_ctx_rnd = X_train_scaled[rnd_idx]
    y_ctx_rnd = y_train_full[rnd_idx]

    # Run TabPFN
    if task_type == 'classification':
        clf = TabPFNClassifier(device='cuda')
    else:
        clf = TabPFNRegressor(device='cuda')

    # DCS
    t0 = time.time()
    clf.fit(X_ctx_dcs, y_ctx_dcs)
    fit_time = time.time() - t0
    t0 = time.time()
    y_pred = clf.predict(X_test_scaled)
    predict_time = time.time() - t0

    if task_type == 'classification':
        try:
            y_proba = clf.predict_proba(X_test_scaled)
            proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        except:
            proba_pos = y_pred.astype(float)
        results['dcs'] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            'n_context': int(len(dcs_idx)),
            'selection_time': float(selection_time),
            'fit_time': float(fit_time),
            'predict_time': float(predict_time),
        }
        try:
            results['dcs']['auc'] = float(roc_auc_score(y_test, proba_pos))
        except:
            pass
    else:
        results['dcs'] = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred)),
            'n_context': int(len(dcs_idx)),
            'selection_time': float(selection_time),
            'fit_time': float(fit_time),
            'predict_time': float(predict_time),
        }

    print(f"    DCS: {results['dcs']}")

    # Random
    if task_type == 'classification':
        clf = TabPFNClassifier(device='cuda')
    else:
        clf = TabPFNRegressor(device='cuda')

    t0 = time.time()
    clf.fit(X_ctx_rnd, y_ctx_rnd)
    fit_time = time.time() - t0
    t0 = time.time()
    y_pred = clf.predict(X_test_scaled)
    predict_time = time.time() - t0

    if task_type == 'classification':
        try:
            y_proba = clf.predict_proba(X_test_scaled)
            proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        except:
            proba_pos = y_pred.astype(float)
        results['random'] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
            'n_context': int(len(rnd_idx)),
            'fit_time': float(fit_time),
            'predict_time': float(predict_time),
        }
        try:
            results['random']['auc'] = float(roc_auc_score(y_test, proba_pos))
        except:
            pass
    else:
        results['random'] = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred)),
            'n_context': int(len(rnd_idx)),
            'fit_time': float(fit_time),
            'predict_time': float(predict_time),
        }

    print(f"    Random: {results['random']}")

    # Compute delta
    if task_type == 'classification':
        results['delta_accuracy'] = results['dcs']['accuracy'] - results['random']['accuracy']
    else:
        results['delta_rmse'] = results['random']['rmse'] - results['dcs']['rmse']  # positive = DCS better

    results['n_train'] = int(n_train)
    results['n_test'] = int(n_test)
    results['n_features'] = int(n_features)
    results['task_type'] = task_type

    return results


def main():
    print("=" * 80)
    print("TabReD DCS Experiments")
    print("=" * 80)

    # Step 1: Extract all datasets
    print("\nExtracting datasets...")
    extract_all()
    print("Done.\n")

    # Step 2: Run experiments
    output_path = os.path.join(RESULT_DIR, 'tabred_benchmark_results.json')

    # Load existing results to skip completed experiments
    try:
        with open(output_path, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {output_path}")
    except:
        all_results = {
            'experiment': 'tabred_dcs_vs_random',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'config': {
                'seeds': SEEDS,
                'n_context': N_CONTEXT,
                'n_test_max': N_TEST_MAX,
                'k_clusters': K_CLUSTERS,
            },
            'results': {}
        }

    for name, task_type in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {name} ({task_type})")
        print("=" * 60)

        if name not in all_results['results']:
            all_results['results'][name] = {'task_type': task_type, 'seeds': {}}

        for seed in SEEDS:
            # Skip if already completed
            existing = all_results['results'][name]['seeds'].get(str(seed), {})
            if existing and 'error' not in existing and 'dcs' in existing:
                print(f"\n  [seed={seed}] already done, skipping")
                continue

            print(f"\n  [seed={seed}]")
            set_seed(seed)
            try:
                result = run_experiment(name, task_type, seed=seed)
                all_results['results'][name]['seeds'][str(seed)] = result
                print(f"  Result: {result}")
            except Exception as e:
                print(f"  ERROR: {e}")
                all_results['results'][name]['seeds'][str(seed)] = {'error': str(e)}
                all_results['results'][name]['seeds'][str(seed)] = {'error': str(e)}

            # Save after each seed
            with open(output_path, 'w') as f:
                json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)

    for name, data in all_results['results'].items():
        task_type = data['task_type']
        seeds = data['seeds']
        valid_seeds = [s for s in SEEDS if str(s) in seeds and 'error' not in seeds[str(s)]]

        if not valid_seeds:
            print(f"\n{name}: ALL SEEDS FAILED")
            continue

        if task_type == 'classification':
            dcs_accs = [seeds[str(s)]['dcs']['accuracy'] for s in valid_seeds]
            rnd_accs = [seeds[str(s)]['random']['accuracy'] for s in valid_seeds]
            deltas = [d - r for d, r in zip(dcs_accs, rnd_accs)]
            print(f"\n{name} (classification, n={len(valid_seeds)} seeds):")
            print(f"  DCS:    {np.mean(dcs_accs):.4f}±{np.std(dcs_accs, ddof=1):.4f}")
            print(f"  Random: {np.mean(rnd_accs):.4f}±{np.std(rnd_accs, ddof=1):.4f}")
            print(f"  Δ:      {np.mean(deltas)*100:+.2f}pp")
        else:
            dcs_rmses = [seeds[str(s)]['dcs']['rmse'] for s in valid_seeds]
            rnd_rmses = [seeds[str(s)]['random']['rmse'] for s in valid_seeds]
            deltas = [r - d for d, r in zip(dcs_rmses, rnd_rmses)]
            print(f"\n{name} (regression, n={len(valid_seeds)} seeds):")
            print(f"  DCS:    RMSE={np.mean(dcs_rmses):.4f}±{np.std(dcs_rmses, ddof=1):.4f}")
            print(f"  Random: RMSE={np.mean(rnd_rmses):.4f}±{np.std(rnd_rmses, ddof=1):.4f}")
            print(f"  Δ:      {np.mean(deltas):+.4f} (positive = DCS better)")

    print(f"\nAll results saved to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific dataset only
        ds_name = sys.argv[1]
        if ds_name in DATASETS:
            DATASETS = {ds_name: DATASETS[ds_name]}
            print(f"Running only: {ds_name}")
        else:
            print(f"Unknown dataset: {ds_name}")
            print(f"Available: {list(DATASETS.keys())}")
            sys.exit(1)
    main()
