"""ContextShield: Advanced context selection strategies for TabPFN under distribution shift.

Implements three complementary strategies beyond simple KNN:
  1. Density-Ratio Weighted Selection (DRWS) — estimate p_test(x)/p_train(x) via
     a domain classifier, select train samples with highest density ratios.
  2. Diversity-Constrained Selection (DCS) — k-means clustering + per-cluster
     proportional allocation to ensure both distribution matching and diversity.
  3. Context Cleaning (CC) — remove harmful outliers from the selected context
     using Local Outlier Factor (LOF).

The full ContextShield pipeline: DRWS → DCS → CC.

Theoretical motivation:
  - TabPFN's ICL attends to context samples similar to the test query.
  - Under distribution shift, random context contains many samples far from
    the test distribution, wasting attention budget.
  - By selecting context samples with high p_test(x)/p_train(x), we ensure
    the context is "distribution-aware" and attention is well-spent.
  - Diversity constraint prevents selecting a narrow cluster of similar samples.
  - Cleaning removes noisy/mislabeled samples that could mislead attention.

Results saved to: results/context_shield_results.json
"""
import os
import sys
import json
import time
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
from splits import prepare_split

DATASETS_TO_TEST = ['adult']
SPLITS_TO_TEST = ['iid', 'temporal']
SEEDS = [42, 123, 456, 789, 2024]
CONTEXT_SIZE = 10000  # TabPFN limit


def set_seed(seed):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def json_safe(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
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


def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = 0.0
    if y_proba is not None:
        n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


# ============================================================================
# Strategy 1: Density-Ratio Weighted Selection (DRWS)
# ============================================================================

def estimate_density_ratio(X_train, X_test, method='logistic', seed=42):
    """Estimate density ratio r(x) = p_test(x) / p_train(x) for each train sample.

    Uses a domain classifier to distinguish train (label=0) vs test (label=1).
    By Bayes' rule: r(x) = p_test(x)/p_train(x) = [p(test|x) / p(train|x)] * [n_test / n_train]
    For selection purposes, the constant n_test/n_train doesn't matter, so we use:
        r(x) ≈ p(test|x) / (1 - p(test|x))

    Args:
        X_train, X_test: feature matrices
        method: 'logistic' (fast, linear) or 'lightgbm' (captures nonlinear shift)
        seed: random seed

    Returns:
        density_ratios: array of shape (n_train,), higher = more similar to test
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Subsample test if too large (for speed; domain classifier doesn't need all)
    if n_test > 5000:
        rng = np.random.RandomState(seed)
        test_idx = rng.choice(n_test, 5000, replace=False)
        X_test_sample = X_test[test_idx]
    else:
        X_test_sample = X_test

    # Construct domain classification problem
    X_domain = np.vstack([X_train, X_test_sample])
    y_domain = np.concatenate([np.zeros(n_train), np.ones(len(X_test_sample))])

    # Standardize for logistic regression
    scaler = StandardScaler()
    X_domain_s = scaler.fit_transform(X_domain)

    if method == 'lightgbm':
        try:
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=seed, verbose=-1, n_jobs=-1
            )
            clf.fit(X_domain_s, y_domain)
            p_test = clf.predict_proba(scaler.transform(X_train))[:, 1]
        except ImportError:
            clf = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
            clf.fit(X_domain_s, y_domain)
            p_test = clf.predict_proba(scaler.transform(X_train))[:, 1]
    else:
        clf = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
        clf.fit(X_domain_s, y_domain)
        p_test = clf.predict_proba(scaler.transform(X_train))[:, 1]

    # Avoid division by zero; clip probabilities
    p_test = np.clip(p_test, 1e-6, 1 - 1e-6)
    density_ratios = p_test / (1 - p_test)
    return density_ratios


def drws_selection(X_train, X_test, n_select, method='logistic', seed=42):
    """Density-Ratio Weighted Selection.

    Selects training samples with the highest estimated density ratios,
    i.e., those most similar to the test distribution.

    Args:
        X_train: training features
        X_test: test features (used ONLY to guide selection)
        n_select: number of samples to select
        method: 'logistic' or 'lightgbm' for domain classifier
        seed: random seed

    Returns:
        selected_indices: indices into X_train
        density_ratios: the estimated density ratios (for analysis)
    """
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train), np.ones(n_train)

    density_ratios = estimate_density_ratio(X_train, X_test, method=method, seed=seed)
    # Select top-n_select samples by density ratio
    selected = np.argsort(density_ratios)[-n_select:]
    return selected, density_ratios


# ============================================================================
# Strategy 2: Diversity-Constrained Selection (DCS)
# ============================================================================

def dcs_selection(X_train, X_test, n_select, n_clusters=50, method='logistic', seed=42):
    """Diversity-Constrained Selection.

    Combines density-ratio weighting with diversity constraint via k-means
    clustering. Ensures selected context covers diverse regions of the
    feature space, not just the single highest-density-ratio cluster.

    Algorithm:
      1. Cluster all training samples into n_clusters groups
      2. Compute average density ratio for each cluster
      3. Allocate selection budget via largest remainder method:
         q_k = (mu_k * n_k) / sum(mu_j * n_j) * n_select
         b_k = min(floor(q_k), n_k), then distribute remainder by r_k desc
      4. Within each cluster, select samples with highest individual density ratio

    Args:
        X_train: training features
        X_test: test features
        n_select: total number to select
        n_clusters: number of k-means clusters
        method: domain classifier method
        seed: random seed

    Returns:
        selected_indices: indices into X_train
    """
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)

    # Step 1: Estimate density ratios
    density_ratios = estimate_density_ratio(X_train, X_test, method=method, seed=seed)

    # Step 2: Cluster training data
    n_clusters = min(n_clusters, n_train)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train)

    # Step 3: Compute per-cluster stats
    cluster_ratios = np.zeros(n_clusters)
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_ratios[c] = density_ratios[mask].mean() if mask.sum() > 0 else 0
        cluster_sizes[c] = mask.sum()

    # Step 4: Allocate budget using largest remainder method
    # Weight = mu_k * n_k (cluster mean density ratio * cluster size)
    cluster_weights = cluster_ratios * cluster_sizes
    total_weight = cluster_weights.sum()

    if total_weight == 0:
        # Fallback: equal allocation by quota
        quotas = np.full(n_clusters, n_select / n_clusters, dtype=float)
    else:
        # Real-valued quotas: q_k = (mu_k * n_k) / sum(mu_j * n_j) * n_select
        quotas = (cluster_weights / total_weight) * n_select

    # Floor and cap at cluster size: b_k = min(floor(q_k), n_k)
    base_allocation = np.minimum(np.floor(quotas).astype(int), cluster_sizes)
    base_allocation = np.maximum(0, base_allocation)

    # Remainders: r_k = q_k - floor(q_k)
    remainders = quotas - np.floor(quotas)

    # Distribute remaining slots by largest remainder, capped at cluster size
    remaining = n_select - int(base_allocation.sum())
    if remaining > 0:
        # Sort by remainder desc, then by cluster weight desc for tie-breaking
        order = np.lexsort((-cluster_weights, -remainders))
        for i in range(len(order)):
            if remaining <= 0:
                break
            c = order[i]
            if base_allocation[c] < cluster_sizes[c]:
                base_allocation[c] += 1
                remaining -= 1

    # If still remaining (all clusters at capacity), distribute to any with room
    if remaining > 0:
        for c in range(n_clusters):
            if remaining <= 0:
                break
            if base_allocation[c] < cluster_sizes[c]:
                base_allocation[c] += 1
                remaining -= 1

    allocation = base_allocation

    # Step 5: Within each cluster, select top samples by density ratio
    selected = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_indices = np.where(mask)[0]
        cluster_dr = density_ratios[cluster_indices]
        n_from_cluster = min(allocation[c], len(cluster_indices))
        if n_from_cluster > 0:
            top_local = np.argsort(cluster_dr)[-n_from_cluster:]
            selected.extend(cluster_indices[top_local].tolist())

    selected = np.array(selected)

    # Safety: if fewer than n_select selected (edge case), top up by density ratio
    if len(selected) < n_select and n_train > len(selected):
        already = set(selected.tolist())
        remaining_pool = np.array(
            [i for i in range(n_train) if i not in already]
        )
        if len(remaining_pool) > 0:
            topup_dr = density_ratios[remaining_pool]
            n_topup = min(n_select - len(selected), len(remaining_pool))
            topup_idx = remaining_pool[np.argsort(topup_dr)[-n_topup:]]
            selected = np.concatenate([selected, topup_idx])

    return selected[:n_select]


# ============================================================================
# Strategy 3: Context Cleaning (CC)
# ============================================================================

def context_cleaning(X_selected, y_selected, contamination=0.05, seed=42):
    """Context Cleaning via Local Outlier Factor (LOF).

    Removes samples that are local outliers within the selected context.
    These are likely noisy/mislabeled samples that could mislead TabPFN's
    attention mechanism.

    Args:
        X_selected: features of selected context samples
        y_selected: labels of selected context samples
        contamination: fraction of samples to remove
        seed: random seed

    Returns:
        keep_mask: boolean array, True = keep this sample
    """
    n = X_selected.shape[0]
    if n < 20 or contamination <= 0:
        return np.ones(n, dtype=bool)

    n_neighbors = min(20, n - 1)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    # predict: 1 = inlier, -1 = outlier
    labels = lof.fit_predict(X_selected)
    keep_mask = labels == 1
    # Safety: ensure we don't remove too many
    if keep_mask.sum() < n * 0.8:
        keep_mask = np.ones(n, dtype=bool)
    return keep_mask


# ============================================================================
# Strategy 4: KNN (baseline from feasibility experiment, for comparison)
# ============================================================================

def knn_context_selection(X_train, y_train, X_test, n_select, k_neighbors=5, seed=42):
    """KNN context selection (baseline from feasibility experiment)."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)

    k = min(k_neighbors, n_train)
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    nn.fit(X_train)
    _, indices = nn.kneighbors(X_test)
    counter = Counter(indices.flatten())
    selected = [idx for idx, _ in counter.most_common(n_select)]
    if len(selected) < n_select:
        remaining = sorted(set(range(n_train)) - set(selected))
        rng = np.random.RandomState(seed)
        extra = rng.choice(remaining, n_select - len(selected), replace=False)
        selected.extend(extra.tolist())
    return np.array(selected)


def random_context_selection(X_train, n_select, seed=42):
    """Random subsampling (baseline)."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)
    rng = np.random.RandomState(seed)
    return rng.choice(n_train, n_select, replace=False)


# ============================================================================
# Strategy 5: Mixed Selection (density-ratio + random, for IID robustness)
# ============================================================================

def mixed_selection(X_train, X_test, n_select, alpha=0.5, method='lightgbm', seed=42):
    """Mixed selection: alpha fraction by density ratio, (1-alpha) fraction random.

    Mitigates IID bias by preserving some random diversity while still
    providing shift-aware benefit from density-ratio selection.

    Args:
        X_train: training features
        X_test: test features
        n_select: total number to select
        alpha: fraction selected by density ratio (0=all random, 1=all DRWS)
        method: domain classifier method
        seed: random seed

    Returns:
        selected_indices: indices into X_train
    """
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)

    n_drws = int(n_select * alpha)
    n_random = n_select - n_drws

    # Density-ratio selection
    drws_idx, _ = drws_selection(X_train, X_test, n_drws, method=method, seed=seed)

    # Random selection from remaining samples
    remaining = np.setdiff1d(np.arange(n_train), drws_idx)
    rng = np.random.RandomState(seed)
    random_idx = rng.choice(remaining, min(n_random, len(remaining)), replace=False)

    return np.concatenate([drws_idx, random_idx])


# ============================================================================
# ContextShield: Full Pipeline (DRWS + DCS + CC)
# ============================================================================

def context_shield_selection(X_train, y_train, X_test, n_select,
                              n_clusters=50, contamination=0.05,
                              method='logistic', seed=42):
    """ContextShield full pipeline: DCS selection + context cleaning.

    This is the complete ContextShield method combining:
      - Diversity-constrained density-ratio selection (DCS with density ratios)
      - Context cleaning (LOF-based outlier removal)

    Args:
        X_train, y_train: training data
        X_test: test data (guides selection only)
        n_select: target context size
        n_clusters: k-means clusters for diversity
        contamination: fraction to remove in cleaning
        method: domain classifier ('logistic' or 'lightgbm')
        seed: random seed

    Returns:
        selected_indices: final indices into X_train after selection + cleaning
        info: dict with selection metadata
    """
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train), {'n_before_cleaning': n_train, 'n_after_cleaning': n_train}

    # Step 1: Diversity-constrained density-ratio selection
    # Over-select to allow cleaning to remove some
    n_overselect = min(n_train, int(n_select * (1 + contamination * 2)))
    selected = dcs_selection(X_train, X_test, n_overselect,
                              n_clusters=n_clusters, method=method, seed=seed)

    # Step 2: Context cleaning
    X_sel = X_train[selected]
    y_sel = y_train[selected]
    keep_mask = context_cleaning(X_sel, y_sel, contamination=contamination, seed=seed)

    # Step 3: If cleaning removed too many, top up from remaining train samples
    cleaned = selected[keep_mask]
    if len(cleaned) < n_select:
        # Top up: select more from unselected by density ratio
        remaining = np.setdiff1d(np.arange(n_train), cleaned)
        if len(remaining) > 0:
            density_ratios = estimate_density_ratio(X_train, X_test, method=method, seed=seed)
            remaining_dr = density_ratios[remaining]
            n_topup = n_select - len(cleaned)
            topup_local = np.argsort(remaining_dr)[-n_topup:]
            cleaned = np.concatenate([cleaned, remaining[topup_local]])

    cleaned = cleaned[:n_select]
    info = {
        'n_before_cleaning': int(len(selected)),
        'n_after_cleaning': int(len(cleaned)),
        'cleaning_removed': int(len(selected) - keep_mask.sum()),
    }
    return cleaned, info


# ============================================================================
# TabPFN runner (reused from feasibility experiment)
# ============================================================================

def run_tabpfn(X_train, y_train, X_test, y_test, context_indices=None, label='TabPFN'):
    """Run TabPFN (cloud client) with optional pre-selected context."""
    import tabpfn_client
    if not getattr(run_tabpfn, '_initialized', False):
        tabpfn_client.init()
        run_tabpfn._initialized = True

    from tabpfn_client import TabPFNClassifier as _TabPFN

    if context_indices is not None:
        X_ctx = X_train[context_indices]
        y_ctx = y_train[context_indices]
    else:
        if X_train.shape[0] > CONTEXT_SIZE:
            idx = np.random.RandomState(42).choice(
                X_train.shape[0], CONTEXT_SIZE, replace=False
            )
            X_ctx = X_train[idx]
            y_ctx = y_train[idx]
        else:
            X_ctx = X_train
            y_ctx = y_train

    clf = _TabPFN()
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
    metrics['fit_time'] = float(fit_time)
    metrics['predict_time'] = float(predict_time)
    metrics['n_context'] = int(len(y_ctx))
    return metrics


def run_xgboost(X_train, y_train, X_test, y_test):
    """Run XGBoost as a reference baseline."""
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - t0

    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics['fit_time'] = float(fit_time)
    metrics['predict_time'] = float(predict_time)
    return metrics


# ============================================================================
# Main experiment runner
# ============================================================================

METHODS = [
    'XGBoost',
    'TabPFN-Random',
    'TabPFN-KNN',
    'TabPFN-DRWS-Logistic',
    'TabPFN-DRWS-LightGBM',
    'TabPFN-DCS-Logistic',
    'TabPFN-DCS-LightGBM',
    'TabPFN-Mixed-LightGBM',
    'TabPFN-ContextShield-Logistic',
    'TabPFN-ContextShield-LightGBM',
]


def run_all_methods(X_train, y_train, X_test, y_test, seed):
    """Run all context selection methods on one split."""
    results = {}
    n_train = X_train.shape[0]

    # --- XGBoost (reference) ---
    try:
        results['XGBoost'] = run_xgboost(X_train, y_train, X_test, y_test)
    except Exception as e:
        results['XGBoost'] = {'error': str(e)}

    # Only run context selection if train > context_size
    if n_train <= CONTEXT_SIZE:
        print(f"    Skip context selection (train={n_train} <= {CONTEXT_SIZE})")
        # Just run TabPFN with full data
        try:
            results['TabPFN-Random'] = run_tabpfn(X_train, y_train, X_test, y_test,
                                                    context_indices=None)
        except Exception as e:
            results['TabPFN-Random'] = {'error': str(e)}
        return results

    # --- TabPFN-Random ---
    try:
        idx = random_context_selection(X_train, CONTEXT_SIZE, seed=seed)
        results['TabPFN-Random'] = run_tabpfn(X_train, y_train, X_test, y_test,
                                               context_indices=idx)
    except Exception as e:
        results['TabPFN-Random'] = {'error': str(e)}

    # --- TabPFN-KNN (baseline) ---
    try:
        t0 = time.time()
        idx = knn_context_selection(X_train, y_train, X_test, CONTEXT_SIZE,
                                     k_neighbors=5, seed=seed)
        sel_time = time.time() - t0
        m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
        m['selection_time'] = float(sel_time)
        results['TabPFN-KNN'] = m
    except Exception as e:
        results['TabPFN-KNN'] = {'error': str(e)}

    # --- TabPFN-DRWS-Logistic ---
    try:
        t0 = time.time()
        idx, _ = drws_selection(X_train, X_test, CONTEXT_SIZE,
                                 method='logistic', seed=seed)
        sel_time = time.time() - t0
        m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
        m['selection_time'] = float(sel_time)
        results['TabPFN-DRWS-Logistic'] = m
    except Exception as e:
        results['TabPFN-DRWS-Logistic'] = {'error': str(e)}

    # --- TabPFN-DRWS-LightGBM ---
    try:
        t0 = time.time()
        idx, _ = drws_selection(X_train, X_test, CONTEXT_SIZE,
                                 method='lightgbm', seed=seed)
        sel_time = time.time() - t0
        m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
        m['selection_time'] = float(sel_time)
        results['TabPFN-DRWS-LightGBM'] = m
    except Exception as e:
        results['TabPFN-DRWS-LightGBM'] = {'error': str(e)}

    # --- TabPFN-DCS-Logistic ---
    try:
        t0 = time.time()
        idx = dcs_selection(X_train, X_test, CONTEXT_SIZE,
                            n_clusters=50, method='logistic', seed=seed)
        sel_time = time.time() - t0
        m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
        m['selection_time'] = float(sel_time)
        results['TabPFN-DCS-Logistic'] = m
    except Exception as e:
        results['TabPFN-DCS-Logistic'] = {'error': str(e)}

    # --- TabPFN-DCS-LightGBM ---
    try:
        t0 = time.time()
        idx = dcs_selection(X_train, X_test, CONTEXT_SIZE,
                            n_clusters=50, method='lightgbm', seed=seed)
        sel_time = time.time() - t0
        m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
        m['selection_time'] = float(sel_time)
        results['TabPFN-DCS-LightGBM'] = m
    except Exception as e:
        results['TabPFN-DCS-LightGBM'] = {'error': str(e)}

    # --- TabPFN-Mixed-LightGBM (50% DRWS + 50% Random) ---
    try:
        t0 = time.time()
        idx = mixed_selection(X_train, X_test, CONTEXT_SIZE,
                              alpha=0.5, method='lightgbm', seed=seed)
        sel_time = time.time() - t0
        m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
        m['selection_time'] = float(sel_time)
        results['TabPFN-Mixed-LightGBM'] = m
    except Exception as e:
        results['TabPFN-Mixed-LightGBM'] = {'error': str(e)}

    # --- TabPFN-ContextShield-Logistic (DRWS + DCS + CC) ---
    try:
        t0 = time.time()
        idx, info = context_shield_selection(
            X_train, y_train, X_test, CONTEXT_SIZE,
            n_clusters=50, contamination=0.05,
            method='logistic', seed=seed
        )
        sel_time = time.time() - t0
        m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
        m['selection_time'] = float(sel_time)
        m['cleaning_info'] = info
        results['TabPFN-ContextShield-Logistic'] = m
    except Exception as e:
        results['TabPFN-ContextShield-Logistic'] = {'error': str(e)}

    # --- TabPFN-ContextShield-LightGBM (DRWS + DCS + CC) ---
    try:
        t0 = time.time()
        idx, info = context_shield_selection(
            X_train, y_train, X_test, CONTEXT_SIZE,
            n_clusters=50, contamination=0.05,
            method='lightgbm', seed=seed
        )
        sel_time = time.time() - t0
        m = run_tabpfn(X_train, y_train, X_test, y_test, context_indices=idx)
        m['selection_time'] = float(sel_time)
        m['cleaning_info'] = info
        results['TabPFN-ContextShield-LightGBM'] = m
    except Exception as e:
        results['TabPFN-ContextShield-LightGBM'] = {'error': str(e)}

    return results


def main():
    print("=" * 80)
    print("ContextShield: Advanced Context Selection Strategies for TabPFN")
    print("=" * 80)
    print(f"Datasets: {DATASETS_TO_TEST}")
    print(f"Splits: {SPLITS_TO_TEST}")
    print(f"Seeds: {SEEDS}")
    print(f"Context size: {CONTEXT_SIZE}")
    print(f"Methods: {METHODS}")
    print("=" * 80)

    all_results = {
        'experiment': 'context_shield_advanced_selection',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'datasets': DATASETS_TO_TEST,
            'splits': SPLITS_TO_TEST,
            'seeds': SEEDS,
            'context_size': CONTEXT_SIZE,
            'methods': METHODS,
        },
        'results': [],
    }

    for ds_name in DATASETS_TO_TEST:
        for split_type in SPLITS_TO_TEST:
            for seed in SEEDS:
                print(f"\n[{ds_name}/{split_type}/seed={seed}]")
                set_seed(seed)

                try:
                    split_data = prepare_split(ds_name, split_type, seed=seed)
                except Exception as e:
                    print(f"  ERROR split failed: {e}")
                    continue

                if split_data is None:
                    print(f"  Skip: {ds_name} does not support {split_type} split")
                    continue

                X_train = split_data['X_train']
                y_train = split_data['y_train']
                X_test = split_data['X_test']
                y_test = split_data['y_test']
                info = split_data['split_info']
                print(f"  train={X_train.shape}, test={X_test.shape}, "
                      f"features={info['n_features']}, classes={info['n_classes']}")

                method_results = run_all_methods(X_train, y_train, X_test, y_test, seed)

                for method, metrics in method_results.items():
                    if 'error' in metrics:
                        print(f"  {method:<35} FAILED: {metrics['error'][:60]}")
                        all_results['results'].append({
                            'dataset': ds_name, 'split': split_type, 'seed': seed,
                            'method': method, 'metrics': None,
                            'error': metrics['error'],
                            'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                        })
                    else:
                        print(f"  {method:<35} acc={metrics['accuracy']:.4f} "
                              f"f1m={metrics['f1_macro']:.4f} "
                              f"auc={metrics.get('auc', 0):.4f} "
                              f"sel={metrics.get('selection_time', 0):.1f}s")
                        all_results['results'].append({
                            'dataset': ds_name, 'split': split_type, 'seed': seed,
                            'method': method, 'metrics': metrics,
                            'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                        })

                # Save incrementally
                with open(os.path.join(RESULT_DIR, 'context_shield_results.json'), 'w') as f:
                    json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY: Mean ± Std over seeds")
    print("=" * 80)
    print(f"{'Dataset':<10} {'Split':<10} {'Method':<37} {'Accuracy':<14} {'F1-Macro':<10}")
    print("-" * 85)

    summary = {}
    for ds_name in DATASETS_TO_TEST:
        for split_type in SPLITS_TO_TEST:
            for method in METHODS:
                accs = [r['metrics']['accuracy'] for r in all_results['results']
                        if r['dataset'] == ds_name and r['split'] == split_type
                        and r['method'] == method and r.get('metrics')]
                f1s = [r['metrics']['f1_macro'] for r in all_results['results']
                       if r['dataset'] == ds_name and r['split'] == split_type
                       and r['method'] == method and r.get('metrics')]
                if accs:
                    mean_acc = np.mean(accs)
                    mean_f1 = np.mean(f1s)
                    std_acc = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
                    print(f"{ds_name:<10} {split_type:<10} {method:<37} "
                          f"{mean_acc:.4f}±{std_acc:.4f}  {mean_f1:.4f}")
                    key = f"{ds_name}_{split_type}_{method}"
                    summary[key] = {
                        'accuracy_mean': float(mean_acc),
                        'accuracy_std': float(std_acc),
                        'f1_macro_mean': float(mean_f1),
                        'n_seeds': len(accs),
                    }

    # ---- Improvement analysis ----
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS (vs TabPFN-Random baseline)")
    print("=" * 80)
    for ds_name in DATASETS_TO_TEST:
        for split_type in SPLITS_TO_TEST:
            baseline_key = f"{ds_name}_{split_type}_TabPFN-Random"
            baseline = summary.get(baseline_key, {}).get('accuracy_mean')
            if baseline is None:
                continue
            print(f"\n  [{ds_name}/{split_type}] Baseline (TabPFN-Random) = {baseline:.4f}")
            for method in METHODS:
                if method == 'TabPFN-Random':
                    continue
                key = f"{ds_name}_{split_type}_{method}"
                m = summary.get(key)
                if m:
                    delta = m['accuracy_mean'] - baseline
                    delta_pp = delta * 100
                    print(f"    {method:<37} {m['accuracy_mean']:.4f}  "
                          f"Δ={delta_pp:+.2f}pp")

    all_results['summary'] = summary
    with open(os.path.join(RESULT_DIR, 'context_shield_results.json'), 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {os.path.join(RESULT_DIR, 'context_shield_results.json')}")
    print("=" * 80)
    print("ContextShield Advanced Selection Experiments Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
