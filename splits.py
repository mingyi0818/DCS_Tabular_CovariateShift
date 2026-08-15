"""
Data splitting protocols for distribution shift experiments.

Three split protocols:
1. IID: Random 70/15/15 train/val/test
2. Temporal: Sort by temporal column, first 70% train, next 15% val, last 15% test
3. Group: Hold out specific groups for test set

Each split returns (X_train, X_val, X_test, y_train, y_val, y_test, feature_names).
Categorical encoding and scaling are fit on train only, applied to val/test.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATASETS, CONFIG

# Split configuration per dataset
SPLIT_CONFIG = {
    'telco': {
        'temporal_col': 'tenure',
        'group_col': 'Contract',
        'group_test_values': None,  # Will be auto-selected
        'drop_cols': ['customerID'],
    },
    'bank': {
        'temporal_col': 'month',
        'temporal_order': ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                           'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'group_col': 'job',
        'group_test_values': None,
        'drop_cols': [],
    },
    'adult': {
        'temporal_col': 'age',
        'group_col': 'race',
        'group_test_values': None,
        'drop_cols': [],
    },
    'heart': {
        'temporal_col': 'age',
        'group_col': 'sex',
        'group_test_values': None,
        'drop_cols': [],
    },
    'mushroom': {
        'temporal_col': 'stem-height',
        'group_col': 'habitat',
        'group_test_values': None,
        'drop_cols': [],
    },
    'drybean': {
        'temporal_col': None,
        'group_col': None,
        'drop_cols': [],
    },
    'wine': {
        'temporal_col': None,
        'group_col': None,
        'drop_cols': [],
    },
    'sklearn_wine': {
        'temporal_col': None,
        'group_col': None,
        'drop_cols': [],
    },
}


def load_raw_dataframe(dataset_name):
    """Load raw dataframe for a dataset, applying minimal cleaning."""
    cfg = DATASETS[dataset_name]
    df = pd.read_csv(cfg['path'])
    split_cfg = SPLIT_CONFIG.get(dataset_name, {})

    # Drop specified columns
    for col in split_cfg.get('drop_cols', []):
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Dataset-specific cleaning
    if dataset_name == 'telco':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna()
    elif dataset_name == 'mushroom':
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('missing')
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
    else:
        df = df.dropna()

    return df, cfg['target']


def encode_features(df, target_col, fit_df=None):
    """
    Encode categorical columns and separate features from target.
    If fit_df is provided, use its encoders (for train/test consistency).
    Returns (X_df, y, encoders).
    """
    df = df.copy()
    if fit_df is not None:
        # Use encoders from fit_df
        cat_cols = fit_df.select_dtypes(include=['object']).columns
        cat_cols = [c for c in cat_cols if c != target_col]
    else:
        cat_cols = df.select_dtypes(include=['object']).columns
        cat_cols = [c for c in cat_cols if c != target_col]

    encoders = {}
    for col in cat_cols:
        if fit_df is not None:
            # Fit on fit_df, transform on df
            le = LabelEncoder()
            le.fit(fit_df[col].astype(str))
            # Handle unseen values
            df[col] = df[col].astype(str).map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])
            encoders[col] = le
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    # Encode target
    target_le = LabelEncoder()
    if fit_df is not None:
        target_le.fit(fit_df[target_col].astype(str))
        y = df[target_col].astype(str).map(
            lambda x: x if x in target_le.classes_ else target_le.classes_[0]
        )
        y = target_le.transform(y)
    else:
        y = target_le.fit_transform(df[target_col].astype(str))
    encoders['_target'] = target_le

    X_df = df.drop(target_col, axis=1)
    return X_df, y, encoders


def split_iid(df, target_col, seed=42):
    """IID random split: 70/15/15 train/val/test."""
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=df[target_col]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df[target_col]
    )
    return train_df, val_df, test_df


def split_temporal(df, target_col, dataset_name, seed=42):
    """Temporal split: sort by temporal column, first 70% train, last 15% test."""
    cfg = SPLIT_CONFIG[dataset_name]
    temporal_col = cfg['temporal_col']

    if temporal_col is None:
        raise ValueError(f"Dataset {dataset_name} has no temporal column")

    df_sorted = df.copy()

    # Handle categorical temporal columns (e.g., month names)
    if 'temporal_order' in cfg and cfg['temporal_order']:
        order_map = {m: i for i, m in enumerate(cfg['temporal_order'])}
        df_sorted['_temporal_order'] = df_sorted[temporal_col].map(
            lambda x: order_map.get(x, 0)
        )
        # Add random jitter within same temporal value to avoid deterministic ties
        rng = np.random.RandomState(seed)
        df_sorted['_temporal_jitter'] = rng.uniform(0, 0.5, size=len(df_sorted))
        df_sorted = df_sorted.sort_values(
            ['_temporal_order', '_temporal_jitter']
        )
        df_sorted = df_sorted.drop(['_temporal_order', '_temporal_jitter'], axis=1)
    else:
        # Numeric temporal column
        rng = np.random.RandomState(seed)
        df_sorted['_temporal_jitter'] = rng.uniform(
            0, 0.01 * df_sorted[temporal_col].std(), size=len(df_sorted)
        )
        df_sorted['_sort_key'] = df_sorted[temporal_col] + df_sorted['_temporal_jitter']
        df_sorted = df_sorted.sort_values('_sort_key')
        df_sorted = df_sorted.drop(['_temporal_jitter', '_sort_key'], axis=1)

    n = len(df_sorted)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_df = df_sorted.iloc[:n_train].copy()
    val_df = df_sorted.iloc[n_train:n_train + n_val].copy()
    test_df = df_sorted.iloc[n_train + n_val:].copy()

    return train_df, val_df, test_df


def split_group(df, target_col, dataset_name, seed=42):
    """
    Group split: hold out specific groups for test set.

    Selects test groups to achieve ~15-20% of total data as test set.
    Sorts groups by size and picks smallest groups for test (ensures
    majority of data stays in training).
    """
    cfg = SPLIT_CONFIG[dataset_name]
    group_col = cfg['group_col']

    if group_col is None:
        raise ValueError(f"Dataset {dataset_name} has no group column")

    # Count samples per group
    group_counts = df[group_col].value_counts()
    groups_sorted = group_counts.index.tolist()  # largest first

    n_groups = len(groups_sorted)
    n_total = len(df)

    if n_groups < 3:
        # If too few groups, split within each group
        return split_iid(df, target_col, seed)

    # Greedily assign smallest groups to test until ~15% of data
    test_groups = []
    test_size = 0
    target_test_size = n_total * 0.15

    # Start from smallest groups
    for g in reversed(groups_sorted):
        if test_size + group_counts[g] <= n_total * 0.30:
            test_groups.append(g)
            test_size += group_counts[g]
        if test_size >= target_test_size and len(test_groups) >= 1:
            break

    # Assign some remaining small groups to val (~7-10% of data)
    remaining_groups = [g for g in groups_sorted if g not in test_groups]
    val_groups = []
    val_size = 0
    target_val_size = n_total * 0.10

    for g in reversed(remaining_groups):
        if val_size + group_counts[g] <= n_total * 0.20:
            val_groups.append(g)
            val_size += group_counts[g]
        if val_size >= target_val_size and len(val_groups) >= 1:
            break

    train_groups = [g for g in groups_sorted
                    if g not in test_groups and g not in val_groups]

    # Ensure at least one group in train
    if len(train_groups) == 0:
        # Move largest test group to train
        train_groups = [test_groups[0]]
        test_groups = test_groups[1:]

    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy() if val_groups else pd.DataFrame()
    test_df = df[df[group_col].isin(test_groups)].copy()

    # If val is empty, sample from train
    if len(val_df) == 0:
        train_df, val_df = train_test_split(
            train_df, test_size=0.15, random_state=seed,
            stratify=train_df[target_col] if len(train_df) > 10 else None
        )

    return train_df, val_df, test_df


def prepare_split(dataset_name, split_type, seed=42):
    """
    Prepare a data split for a dataset.

    Args:
        dataset_name: Key in DATASETS
        split_type: 'iid', 'temporal', or 'group'
        seed: Random seed

    Returns:
        dict with keys:
            - X_train, X_val, X_test (numpy arrays, scaled)
            - y_train, y_val, y_test (numpy arrays, encoded)
            - feature_names (list of str)
            - scaler (fitted StandardScaler)
            - n_features, n_classes
            - split_info (dict with split details)
    """
    df, target_col = load_raw_dataframe(dataset_name)

    # Perform the split
    if split_type == 'iid':
        train_df, val_df, test_df = split_iid(df, target_col, seed)
    elif split_type == 'temporal':
        if SPLIT_CONFIG[dataset_name].get('temporal_col') is None:
            return None  # Dataset doesn't support temporal split
        train_df, val_df, test_df = split_temporal(df, target_col, dataset_name, seed)
    elif split_type == 'group':
        if SPLIT_CONFIG[dataset_name].get('group_col') is None:
            return None  # Dataset doesn't support group split
        train_df, val_df, test_df = split_group(df, target_col, dataset_name, seed)
    else:
        raise ValueError(f"Unknown split type: {split_type}")

    # Encode features (fit on train)
    X_train_df, y_train, encoders = encode_features(train_df, target_col)
    X_val_df, y_val, _ = encode_features(val_df, target_col, fit_df=train_df)
    X_test_df, y_test, _ = encode_features(test_df, target_col, fit_df=train_df)

    # Ensure all splits have same columns
    for col in X_train_df.columns:
        if col not in X_val_df.columns:
            X_val_df[col] = 0
        if col not in X_test_df.columns:
            X_test_df[col] = 0
    X_val_df = X_val_df[X_train_df.columns]
    X_test_df = X_test_df[X_train_df.columns]

    feature_names = list(X_train_df.columns)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.values)
    X_val = scaler.transform(X_val_df.values)
    X_test = scaler.transform(X_test_df.values)

    split_info = {
        'dataset': dataset_name,
        'split_type': split_type,
        'seed': seed,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features': len(feature_names),
        'n_classes': len(np.unique(y_train)),
        'feature_names': feature_names,
        'target_col': target_col,
    }

    # Add split-specific info
    if split_type == 'group':
        group_col = SPLIT_CONFIG[dataset_name]['group_col']
        split_info['group_col'] = group_col
        split_info['train_groups'] = sorted(train_df[group_col].unique().tolist())
        split_info['test_groups'] = sorted(test_df[group_col].unique().tolist())
    elif split_type == 'temporal':
        temporal_col = SPLIT_CONFIG[dataset_name]['temporal_col']
        split_info['temporal_col'] = temporal_col

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'encoders': encoders,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'df': df,
        'split_info': split_info,
    }


def get_supported_splits(dataset_name):
    """Return list of split types supported by a dataset."""
    supported = ['iid']
    cfg = SPLIT_CONFIG.get(dataset_name, {})
    if cfg.get('temporal_col') is not None:
        supported.append('temporal')
    if cfg.get('group_col') is not None:
        supported.append('group')
    return supported


if __name__ == '__main__':
    # Test all datasets and splits
    for ds_name in DATASETS:
        supported = get_supported_splits(ds_name)
        print(f"\n{ds_name} ({DATASETS[ds_name]['name']}):")
        print(f"  Supported splits: {supported}")
        for split_type in supported:
            result = prepare_split(ds_name, split_type, seed=42)
            if result is None:
                continue
            info = result['split_info']
            print(f"  {split_type}: train={info['n_train']}, val={info['n_val']}, "
                  f"test={info['n_test']}, features={info['n_features']}, "
                  f"classes={info['n_classes']}")
            if 'test_groups' in info:
                print(f"    test_groups: {info['test_groups']}")
