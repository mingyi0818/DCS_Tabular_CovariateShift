"""TabReD dataset loader: adapts TabReD .npy format to the DCS framework.

TabReD (ICLR 2025 Spotlight) provides 8 industry-grade tabular datasets with
real temporal timestamps.  This loader reads the preprocessed .npy files and
returns data in the same dict format as ``splits.prepare_split()`` so that
existing DCS / context selection code can be used without modification.

Supported datasets (after download via ``download_tabred.py``):
  - cooking-time: 192 features, regression -> median-binarized for classification
  - weather: 103 features, regression -> median-binarized for classification

TabReD data format (per dataset directory):
    data/tabred/<dataset>/
    ├── info.json          # task type, feature info
    ├── x_num.npy          # numerical features (N, F_num)
    ├── x_cat.npy          # categorical features (N, F_cat)  [optional]
    ├── x_bin.npy          # binary features (N, F_bin)       [optional]
    ├── y.npy              # targets (N,)
    └── splits/
        └── default/
            ├── train.npy  # row indices into the arrays
            ├── val.npy
            └── test.npy

Since TabReD's cooking-time and weather are regression tasks but the DCS
framework targets classification (TabPFN classifier), we convert the
regression target to binary via median split: y > median(y) -> 1, else 0.
This preserves the temporal distribution shift while enabling direct
comparison with the existing classification pipeline.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add code directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, DATA_DIR, RESULT_DIR

TABRED_DIR = DATA_DIR / 'tabred'

# Datasets to support
TABRED_DATASETS = {
    'cooking-time': {
        'name': 'TabReD-CookingTime',
        'task': 'regression',  # original task
        'n_features_expected': 192,
        'n_instances_expected': 319986,
    },
    'weather': {
        'name': 'TabReD-Weather',
        'task': 'regression',
        'n_features_expected': 103,
        'n_instances_expected': 423795,
    },
}


def load_tabred_dataset(dataset_name, split_type='default', max_train=10000,
                        max_test=5000, seed=42):
    """Load a TabReD dataset and convert to DCS framework format.

    Args:
        dataset_name: 'cooking-time' or 'weather'
        split_type: 'default' (temporal), 'random-0', 'random-1', 'random-2',
                    'sliding-window-0', etc.
        max_train: maximum training samples (subsample if larger, for TabPFN limit)
        max_test: maximum test samples (subsample if larger, for speed)
        seed: random seed for subsampling

    Returns:
        dict with keys matching splits.prepare_split() output:
            X_train, X_val, X_test (numpy arrays, scaled)
            y_train, y_val, y_test (numpy arrays, encoded)
            feature_names, n_features, n_classes
            split_info
        Or None if dataset not found / not downloaded.
    """
    ds_dir = TABRED_DIR / dataset_name
    if not ds_dir.exists():
        print(f"  [ERROR] TabReD dataset directory not found: {ds_dir}")
        print(f"          Run download_tabred.py first.")
        return None

    # Load info.json
    info_path = ds_dir / 'info.json'
    ds_info = {}
    if info_path.exists():
        with open(info_path) as f:
            ds_info = json.load(f)

    # Load features
    feature_arrays = []
    feature_names = []

    # Numerical features
    x_num_path = ds_dir / 'x_num.npy'
    if x_num_path.exists():
        x_num = np.load(x_num_path)
        # Handle NaN/Inf
        x_num = np.nan_to_num(x_num, nan=0.0, posinf=1e6, neginf=-1e6)
        feature_arrays.append(x_num)
        n_num = x_num.shape[1] if x_num.ndim > 1 else 1
        feature_names.extend([f'num_{i}' for i in range(n_num)])

    # Categorical features (encode as integers)
    x_cat_path = ds_dir / 'x_cat.npy'
    if x_cat_path.exists():
        x_cat = np.load(x_cat_path, allow_pickle=True)
        if x_cat.dtype == object or x_cat.dtype.kind in ('U', 'S'):
            # String/object categorical: label encode each column
            if x_cat.ndim == 1:
                x_cat = x_cat.reshape(-1, 1)
            n_cat = x_cat.shape[1]
            x_cat_encoded = np.zeros_like(x_cat, dtype=np.float64)
            for col in range(n_cat):
                le = LabelEncoder()
                # Handle NaN strings
                col_vals = pd.Series(x_cat[:, col]).fillna('__missing__').astype(str).values
                x_cat_encoded[:, col] = le.fit_transform(col_vals)
            feature_arrays.append(x_cat_encoded)
            feature_names.extend([f'cat_{i}' for i in range(n_cat)])
        else:
            # Already numeric
            feature_arrays.append(x_cat.astype(np.float64))
            n_cat = x_cat.shape[1] if x_cat.ndim > 1 else 1
            feature_names.extend([f'cat_{i}' for i in range(n_cat)])

    # Binary features
    x_bin_path = ds_dir / 'x_bin.npy'
    if x_bin_path.exists():
        x_bin = np.load(x_bin_path)
        x_bin = np.nan_to_num(x_bin, nan=0.0).astype(np.float64)
        feature_arrays.append(x_bin)
        n_bin = x_bin.shape[1] if x_bin.ndim > 1 else 1
        feature_names.extend([f'bin_{i}' for i in range(n_bin)])

    if not feature_arrays:
        print(f"  [ERROR] No feature arrays found in {ds_dir}")
        return None

    X = np.hstack(feature_arrays).astype(np.float64)

    # Load targets
    y_path = ds_dir / 'y.npy'
    if not y_path.exists():
        print(f"  [ERROR] y.npy not found in {ds_dir}")
        return None
    y_raw = np.load(y_path)
    y_raw = np.nan_to_num(y_raw, nan=0.0)

    # Convert regression to binary classification via median split
    # This enables use with the existing TabPFN classifier framework
    task = ds_info.get('task', 'regression')
    if task == 'regression' or y_raw.dtype.kind == 'f':
        median_val = np.median(y_raw)
        y = (y_raw > median_val).astype(int)
        conversion_info = {
            'original_task': 'regression',
            'median_value': float(median_val),
            'conversion': 'median_split (y > median -> 1)',
            'class_balance': {
                '0': int((y == 0).sum()),
                '1': int((y == 1).sum()),
            },
        }
    else:
        # Already classification
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        conversion_info = {
            'original_task': 'classification',
            'n_classes': int(len(le.classes_)),
        }

    # Load split indices
    split_dir = ds_dir / 'splits' / split_type
    if not split_dir.exists():
        print(f"  [WARN] Split '{split_type}' not found, trying 'default'")
        split_dir = ds_dir / 'splits' / 'default'
        split_type = 'default'

    if not split_dir.exists():
        print(f"  [ERROR] No splits directory found in {ds_dir}")
        return None

    train_idx = np.load(split_dir / 'train.npy')
    val_idx = np.load(split_dir / 'val.npy')
    test_idx = np.load(split_dir / 'test.npy')

    X_train_full = X[train_idx]
    y_train_full = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test_full = X[test_idx]
    y_test_full = y[test_idx]

    # Subsample if too large (TabPFN limit: 10,000 context samples)
    rng = np.random.RandomState(seed)
    if len(train_idx) > max_train:
        sub_idx = rng.choice(len(X_train_full), max_train, replace=False)
        X_train = X_train_full[sub_idx]
        y_train = y_train_full[sub_idx]
    else:
        X_train = X_train_full
        y_train = y_train_full

    if len(test_idx) > max_test:
        sub_idx = rng.choice(len(X_test_full), max_test, replace=False)
        X_test = X_test_full[sub_idx]
        y_test = y_test_full[sub_idx]
    else:
        X_test = X_test_full
        y_test = y_test_full

    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    split_info = {
        'dataset': dataset_name,
        'dataset_name': TABRED_DATASETS.get(dataset_name, {}).get('name', dataset_name),
        'source': 'TabReD (ICLR 2025)',
        'split_type': f'tabred_{split_type}',
        'seed': seed,
        'n_train': int(len(X_train)),
        'n_train_full': int(len(X_train_full)),
        'n_val': int(len(X_val)),
        'n_test': int(len(X_test)),
        'n_test_full': int(len(X_test_full)),
        'n_features': int(n_features),
        'n_classes': int(n_classes),
        'feature_names': feature_names,
        'task_conversion': conversion_info,
        'original_info': ds_info,
    }

    print(f"  TabReD [{dataset_name}] loaded:")
    print(f"    train: {len(X_train)} (of {len(X_train_full)} full)")
    print(f"    val:   {len(X_val)}")
    print(f"    test:  {len(X_test)} (of {len(X_test_full)} full)")
    print(f"    features: {n_features}")
    print(f"    classes: {n_classes}")
    if conversion_info.get('original_task') == 'regression':
        print(f"    task: regression -> binary (median={conversion_info['median_value']:.4f})")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'split_info': split_info,
    }


def check_tabred_available():
    """Check if TabReD datasets are available locally.

    Returns:
        dict mapping dataset_name -> bool (True if available)
    """
    availability = {}
    for ds_name in TABRED_DATASETS:
        ds_dir = TABRED_DIR / ds_name
        has_info = (ds_dir / 'info.json').exists()
        has_y = (ds_dir / 'y.npy').exists()
        has_x_num = (ds_dir / 'x_num.npy').exists()
        has_splits = (ds_dir / 'splits' / 'default' / 'train.npy').exists()
        availability[ds_name] = has_info and has_y and has_x_num and has_splits
    return availability


def get_tabred_status():
    """Get detailed status of TabReD datasets for logging."""
    status = {
        'tabred_dir': str(TABRED_DIR),
        'datasets': {},
    }
    for ds_name, ds_cfg in TABRED_DATASETS.items():
        ds_dir = TABRED_DIR / ds_name
        ds_status = {
            'expected_features': ds_cfg['n_features_expected'],
            'expected_instances': ds_cfg['n_instances_expected'],
            'available': ds_dir.exists(),
            'files': {},
        }
        if ds_dir.exists():
            for fname in ['info.json', 'x_num.npy', 'x_cat.npy', 'x_bin.npy',
                          'y.npy', 'splits/default/train.npy',
                          'splits/default/val.npy', 'splits/default/test.npy']:
                fpath = ds_dir / fname
                ds_status['files'][fname] = {
                    'exists': fpath.exists(),
                    'size': fpath.stat().st_size if fpath.exists() else 0,
                }
        status['datasets'][ds_name] = ds_status
    return status


if __name__ == '__main__':
    print("=" * 80)
    print("TabReD Loader Status Check")
    print("=" * 80)

    status = get_tabred_status()
    print(f"\nTabReD directory: {status['tabred_dir']}")

    for ds_name, ds_status in status['datasets'].items():
        print(f"\n  {ds_name}:")
        print(f"    Available: {ds_status['available']}")
        if ds_status['available']:
            for fname, finfo in ds_status['files'].items():
                status_str = "OK" if finfo['exists'] else "MISSING"
                size_str = f"({finfo['size']:,} bytes)" if finfo['exists'] else ""
                print(f"      {fname}: {status_str} {size_str}")

    # Try loading if available
    print("\n--- Testing load ---")
    availability = check_tabred_available()
    for ds_name, avail in availability.items():
        if avail:
            print(f"\n  Loading {ds_name}...")
            data = load_tabred_dataset(ds_name, max_train=5000, max_test=1000)
            if data is not None:
                print(f"  [OK] {ds_name} loaded successfully")
        else:
            print(f"\n  [SKIP] {ds_name} not available")
