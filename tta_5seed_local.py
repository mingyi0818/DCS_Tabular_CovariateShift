"""5-seed TTA Comparison using LOCAL GPU TabPFN.

Re-runs the DCS vs TTA Baselines comparison with 3 more seeds [456, 789, 2024]
using LOCAL GPU TabPFN (device='cuda'), to extend the existing 2-seed results
in tta_comparison_results.json to 5 seeds.

Methods compared:
  1. TabPFN-Random      — random context (lower bound)
  2. TabPFN-KNN         — KNN context selection
  3. TabPFN-DCS         — our Diversity-Constrained Density-Ratio Selection
  4. TabPFN-Tent        — entropy-minimizing context selection (Wang 2021)
  5. TabPFN-AdapTable   — shift-aware uncertainty calibration (Kim 2024)
  6. TabPFN-SelfTrain   — self-training with pseudo-labels

Dataset: Adult, splits: iid + temporal
Seeds to run: 456, 789, 2024 (seeds 42, 123 already done)

Results saved to: results/tta_5seed_results.json
  (separate file; can be merged with tta_comparison_results.json later)

NOTE: TTA methods (Tent, AdapTable, Self-Training) involve multiple TabPFN
calls per run. Using LOCAL GPU TabPFN should be faster than cloud API.
If the total runtime exceeds 30 minutes, results will be saved incrementally
and the script can report partial completion.
"""
import os
import sys
import json
import time
import traceback
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors

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
from splits import prepare_split, get_supported_splits
from context_shield_methods import (
    json_safe, dcs_selection, random_context_selection,
    knn_context_selection, estimate_density_ratio, set_seed,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Seeds to run (seeds 42, 123 already in tta_comparison_results.json)
SEEDS_TO_RUN = [456, 789, 2024]
DATASET = 'adult'
SPLITS = ['iid', 'temporal']
CONTEXT_SIZE = 10000
DEVICE = 'cuda'
N_TEST_MAX = 2000  # Subsample test for speed

# Time budget (seconds). If exceeded, save partial results and exit.
TIME_BUDGET_SEC = 30 * 60  # 30 minutes

METHODS = [
    'random',        # TabPFN-Random: random context (baseline)
    'knn',           # TabPFN-KNN: KNN context selection
    'dcs',           # TabPFN-DCS: our method
    'tent',          # TabPFN-Tent: entropy minimization
    'adaptable',     # TabPFN-AdapTable: uncertainty calibration
    'self_training', # TabPFN-SelfTrain: pseudo-label augmentation
]

METHOD_LABELS = {
    'random': 'TabPFN-Random',
    'knn': 'TabPFN-KNN',
    'dcs': 'TabPFN-DCS',
    'tent': 'TabPFN-Tent',
    'adaptable': 'TabPFN-AdapTable',
    'self_training': 'TabPFN-SelfTrain',
}


def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = 0.0
    if y_proba is not None:
        try:
            if y_proba.ndim > 1:
                proba_pos = y_proba[:, 1]
            else:
                proba_pos = y_proba
            auc = roc_auc_score(y_true, proba_pos)
        except Exception:
            auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


def _local_tabpfn_predict_proba(X_train_ctx, y_train_ctx, X_test):
    """Run LOCAL TabPFN and return predicted probabilities."""
    clf = TabPFNClassifier(device=DEVICE)
    clf.fit(X_train_ctx, y_train_ctx)
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        n_classes = len(np.unique(y_train_ctx))
        y_proba = np.zeros((len(X_test), n_classes))
        for i, p in enumerate(y_pred):
            if p < n_classes:
                y_proba[i, p] = 1.0
    return y_proba, y_pred


def _prediction_entropy(y_proba):
    """Per-sample prediction entropy."""
    p = np.clip(y_proba, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def run_local_tabpfn(X_ctx, y_ctx, X_test, y_test):
    """Run LOCAL TabPFN with given context and evaluate on test."""
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


# ============================================================================
# Tent: Entropy-Minimizing Context Selection (adapted for local TabPFN)
# ============================================================================

def tent_context_selection(X_train, y_train, X_test, n_select,
                           n_candidates=None, batch_size=200,
                           seed=42, max_tabpfn_evals=10):
    n_train = X_train.shape[0]
    rng = np.random.RandomState(seed)

    if n_train <= n_select:
        return np.arange(n_train), {'n_evals': 0, 'method': 'tent_full'}

    if n_candidates is None:
        n_candidates = min(n_train, 2 * n_select)

    # Generate candidates: density-ratio + random mix
    try:
        density_ratios = estimate_density_ratio(X_train, X_test, method='logistic', seed=seed)
        n_dr = int(n_candidates * 0.7)
        dr_candidates = np.argsort(density_ratios)[-n_dr:]
        n_rand = n_candidates - n_dr
        remaining = np.setdiff1d(np.arange(n_train), dr_candidates)
        rand_candidates = rng.choice(remaining, min(n_rand, len(remaining)), replace=False)
        all_candidates = np.concatenate([dr_candidates, rand_candidates])
    except Exception:
        all_candidates = rng.choice(n_train, n_candidates, replace=False)

    # Subsample test for entropy evaluation
    n_test_eval = min(100, len(X_test))
    test_eval_idx = rng.choice(len(X_test), n_test_eval, replace=False)
    X_test_eval = X_test[test_eval_idx]

    # Start with larger random seed context
    initial_size = min(2000, n_select // 4)
    seed_context = rng.choice(all_candidates, initial_size, replace=False)
    current_context = list(seed_context)
    remaining_candidates = np.setdiff1d(all_candidates, seed_context)

    n_evals = 0

    while len(current_context) < n_select and len(remaining_candidates) > 0:
        if n_evals >= max_tabpfn_evals:
            needed = n_select - len(current_context)
            extra = remaining_candidates[:needed]
            current_context.extend(extra.tolist())
            break

        batch = remaining_candidates[:batch_size]
        remaining_candidates = remaining_candidates[batch_size:]

        trial_context = np.array(current_context + batch.tolist())
        if len(trial_context) > n_select:
            trial_context = trial_context[:n_select]

        try:
            X_ctx = X_train[trial_context]
            y_ctx = y_train[trial_context]
            y_proba, _ = _local_tabpfn_predict_proba(X_ctx, y_ctx, X_test_eval)
            trial_entropy = np.mean(_prediction_entropy(y_proba))
            n_evals += 1
        except Exception:
            current_context.extend(batch.tolist())
            continue

        try:
            X_ctx_base = X_train[np.array(current_context)]
            y_ctx_base = y_train[np.array(current_context)]
            y_proba_base, _ = _local_tabpfn_predict_proba(X_ctx_base, y_ctx_base, X_test_eval)
            base_entropy = np.mean(_prediction_entropy(y_proba_base))
            n_evals += 1
        except Exception:
            base_entropy = float('inf')

        if trial_entropy <= base_entropy:
            current_context.extend(batch.tolist())

        if len(current_context) > n_select:
            current_context = current_context[:n_select]

    if len(current_context) < n_select:
        needed = n_select - len(current_context)
        pool = np.setdiff1d(np.arange(n_train), current_context)
        extra = rng.choice(pool, min(needed, len(pool)), replace=False)
        current_context.extend(extra.tolist())

    selected = np.array(current_context[:n_select])
    info = {
        'method': 'tent',
        'n_candidates': int(len(all_candidates)),
        'n_evals': int(n_evals),
        'n_test_eval': int(n_test_eval),
        'initial_seed_size': int(initial_size),
        'batch_size': int(batch_size),
        'max_tabpfn_evals': int(max_tabpfn_evals),
    }
    return selected, info


# ============================================================================
# AdapTable: Shift-aware Uncertainty Calibration (adapted for local TabPFN)
# ============================================================================

def adaptable_context_selection(X_train, y_train, X_test, n_select,
                                 k_neighbors=10, n_test_sample=500, seed=42):
    n_train = X_train.shape[0]
    rng = np.random.RandomState(seed)

    if n_train <= n_select:
        return np.arange(n_train), {'method': 'adaptable_full'}

    # Initial random context
    initial_ctx_size = min(2000, n_train, n_select)
    initial_ctx = rng.choice(n_train, initial_ctx_size, replace=False)

    n_test_use = min(n_test_sample, len(X_test))
    test_sample_idx = rng.choice(len(X_test), n_test_use, replace=False)
    X_test_sample = X_test[test_sample_idx]

    try:
        X_init_ctx = X_train[initial_ctx]
        y_init_ctx = y_train[initial_ctx]
        y_proba, _ = _local_tabpfn_predict_proba(X_init_ctx, y_init_ctx, X_test_sample)
        test_entropy = _prediction_entropy(y_proba)
    except Exception:
        test_entropy = np.ones(n_test_use)

    # KNN
    k = min(k_neighbors, n_train)
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    nn.fit(X_train)
    _, neighbor_indices = nn.kneighbors(X_test_sample)

    # Weight by test uncertainty
    weights = np.zeros(n_train)
    for i in range(n_test_use):
        for j in range(k):
            weights[neighbor_indices[i, j]] += test_entropy[i]

    selected = np.argsort(weights)[-n_select:]

    if len(set(selected.tolist())) < n_select:
        zero_weight = np.where(weights == 0)[0]
        if len(zero_weight) > 0:
            needed = n_select - len(set(selected.tolist()))
            extra = rng.choice(zero_weight, min(needed, len(zero_weight)), replace=False)
            selected = np.unique(np.concatenate([selected, extra]))[:n_select]

    info = {
        'method': 'adaptable',
        'initial_ctx_size': int(initial_ctx_size),
        'n_test_sample': int(n_test_use),
        'k_neighbors': int(k),
        'mean_test_entropy': float(np.mean(test_entropy)) if len(test_entropy) > 0 else 0.0,
        'std_test_entropy': float(np.std(test_entropy)) if len(test_entropy) > 0 else 0.0,
        'n_nonzero_weights': int((weights > 0).sum()),
    }
    return selected, info


# ============================================================================
# Self-Training (adapted for local TabPFN)
# ============================================================================

def self_training_context_selection(X_train, y_train, X_test, n_select,
                                     confidence_threshold=0.9,
                                     max_pseudo=2000, seed=42):
    n_train = X_train.shape[0]
    rng = np.random.RandomState(seed)

    if n_train <= n_select:
        return np.arange(n_train), np.array([]), np.array([]), {'method': 'self_training_full'}

    n_pseudo_budget = min(max_pseudo, n_select // 4)
    n_initial = n_select - n_pseudo_budget
    initial_ctx = rng.choice(n_train, n_initial, replace=False)

    try:
        X_init_ctx = X_train[initial_ctx]
        y_init_ctx = y_train[initial_ctx]
        y_proba, y_pred = _local_tabpfn_predict_proba(X_init_ctx, y_init_ctx, X_test)
        max_proba = np.max(y_proba, axis=1)
    except Exception as e:
        extra = rng.choice(np.setdiff1d(np.arange(n_train), initial_ctx),
                           n_pseudo_budget, replace=False)
        return np.concatenate([initial_ctx, extra]), np.array([]), np.array([]), {
            'method': 'self_training', 'error': str(e), 'n_pseudo': 0,
        }

    confident_mask = max_proba >= confidence_threshold
    n_confident = confident_mask.sum()

    if n_confident > 0:
        if n_confident > n_pseudo_budget:
            confident_idx = rng.choice(np.where(confident_mask)[0], n_pseudo_budget, replace=False)
        else:
            confident_idx = np.where(confident_mask)[0]
        pseudo_X = X_test[confident_idx]
        pseudo_y = y_pred[confident_idx]
        pseudo_confidence = max_proba[confident_idx]
    else:
        threshold_fallback = np.percentile(max_proba, 80)
        confident_mask = max_proba >= threshold_fallback
        n_confident = confident_mask.sum()
        if n_confident > 0:
            if n_confident > n_pseudo_budget:
                confident_idx = rng.choice(np.where(confident_mask)[0], n_pseudo_budget, replace=False)
            else:
                confident_idx = np.where(confident_mask)[0]
            pseudo_X = X_test[confident_idx]
            pseudo_y = y_pred[confident_idx]
            pseudo_confidence = max_proba[confident_idx]
        else:
            pseudo_X = np.array([])
            pseudo_y = np.array([])
            pseudo_confidence = np.array([])

    n_actual_pseudo = len(pseudo_X) if len(pseudo_X) > 0 else 0
    n_remaining = n_select - n_initial - n_actual_pseudo
    if n_remaining > 0:
        remaining_pool = np.setdiff1d(np.arange(n_train), initial_ctx)
        extra = rng.choice(remaining_pool, min(n_remaining, len(remaining_pool)), replace=False)
        initial_ctx = np.concatenate([initial_ctx, extra])

    info = {
        'method': 'self_training',
        'n_initial': int(n_initial),
        'n_pseudo': int(n_actual_pseudo),
        'confidence_threshold': float(confidence_threshold),
        'mean_pseudo_confidence': float(np.mean(pseudo_confidence)) if len(pseudo_confidence) > 0 else 0.0,
        'n_confident_total': int(n_confident),
    }
    return initial_ctx, pseudo_X, pseudo_y, info


def run_self_training_tabpfn(X_train, y_train, X_test, y_test, n_select,
                              confidence_threshold=0.9, max_pseudo=2000, seed=42):
    t0 = time.time()
    train_idx, pseudo_X, pseudo_y, st_info = self_training_context_selection(
        X_train, y_train, X_test, n_select,
        confidence_threshold=confidence_threshold,
        max_pseudo=max_pseudo, seed=seed
    )

    X_ctx = X_train[train_idx]
    y_ctx = y_train[train_idx]

    if len(pseudo_X) > 0:
        X_ctx = np.vstack([X_ctx, pseudo_X])
        y_ctx = np.concatenate([y_ctx, pseudo_y])

    sel_time = time.time() - t0

    m = run_local_tabpfn(X_ctx, y_ctx, X_test, y_test)
    m['n_pseudo'] = int(len(pseudo_X))
    m['selection_time'] = float(sel_time)
    m['self_training_info'] = st_info
    return m


def run_tta_method(method_name, X_train, y_train, X_test, y_test, n_select, seed=42):
    """Run a single TTA method using LOCAL TabPFN."""
    if method_name == 'random':
        idx = random_context_selection(X_train, n_select, seed=seed)
        return run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)

    elif method_name == 'knn':
        t0 = time.time()
        idx = knn_context_selection(X_train, y_train, X_test, n_select, k_neighbors=5, seed=seed)
        sel_time = time.time() - t0
        m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
        m['selection_time'] = float(sel_time)
        return m

    elif method_name == 'dcs':
        t0 = time.time()
        idx = dcs_selection(X_train, X_test, n_select, n_clusters=50, method='logistic', seed=seed)
        sel_time = time.time() - t0
        m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
        m['selection_time'] = float(sel_time)
        return m

    elif method_name == 'tent':
        t0 = time.time()
        idx, tent_info = tent_context_selection(X_train, y_train, X_test, n_select, seed=seed)
        sel_time = time.time() - t0
        m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
        m['selection_time'] = float(sel_time)
        m['tta_info'] = tent_info
        return m

    elif method_name == 'adaptable':
        t0 = time.time()
        idx, adapt_info = adaptable_context_selection(X_train, y_train, X_test, n_select, seed=seed)
        sel_time = time.time() - t0
        m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
        m['selection_time'] = float(sel_time)
        m['tta_info'] = adapt_info
        return m

    elif method_name == 'self_training':
        return run_self_training_tabpfn(X_train, y_train, X_test, y_test, n_select,
                                         confidence_threshold=0.9, max_pseudo=2000, seed=seed)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def subsample_test(X_test, y_test, seed):
    """Subsample test set to N_TEST_MAX for speed."""
    if len(X_test) > N_TEST_MAX:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_test), N_TEST_MAX, replace=False)
        return X_test[idx], y_test[idx]
    return X_test, y_test


def main():
    print("=" * 100)
    print("5-Seed TTA Comparison using LOCAL GPU TabPFN")
    print("=" * 100)
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Seeds to run: {SEEDS_TO_RUN}")
    print(f"Dataset: {DATASET}, Splits: {SPLITS}")
    print(f"Context size: {CONTEXT_SIZE}")
    print(f"Methods: {[METHOD_LABELS[m] for m in METHODS]}")
    print(f"Time budget: {TIME_BUDGET_SEC/60:.0f} minutes")
    print(f"Results file: {os.path.join(RESULT_DIR, 'tta_5seed_results.json')}")
    print("=" * 100)

    # Check GPU
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(RESULT_DIR, exist_ok=True)

    all_results = {
        'experiment': 'dcs_vs_tta_5seed_local_gpu',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'dataset': DATASET,
            'splits': SPLITS,
            'seeds_run': SEEDS_TO_RUN,
            'context_size': CONTEXT_SIZE,
            'methods': {m: METHOD_LABELS[m] for m in METHODS},
            'tabpfn_mode': 'local_gpu',
            'device': DEVICE,
            'n_test_max': N_TEST_MAX,
            'time_budget_sec': TIME_BUDGET_SEC,
        },
        'results': [],
        'summary': {},
        'errors': [],
    }

    output_path = os.path.join(RESULT_DIR, 'tta_5seed_results.json')
    start_time = time.time()

    for seed in SEEDS_TO_RUN:
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed > TIME_BUDGET_SEC:
            print(f"\n[TIME BUDGET EXCEEDED] {elapsed/60:.1f}min > {TIME_BUDGET_SEC/60:.0f}min")
            print(f"Stopping after {len(all_results['results'])} experiment runs.")
            all_results['time_budget_exceeded'] = True
            all_results['partial_completion'] = True
            break

        for split_type in SPLITS:
            elapsed = time.time() - start_time
            if elapsed > TIME_BUDGET_SEC:
                print(f"\n[TIME BUDGET EXCEEDED] {elapsed/60:.1f}min")
                all_results['time_budget_exceeded'] = True
                all_results['partial_completion'] = True
                break

            print(f"\n[{DATASET}/{split_type}/seed={seed}]")
            set_seed(seed)

            try:
                split_data = prepare_split(DATASET, split_type, seed=seed)
            except Exception as e:
                print(f"  ERROR preparing split: {e}")
                all_results['errors'].append({
                    'dataset': DATASET, 'split': split_type, 'seed': seed,
                    'phase': 'prepare_split', 'error': str(e),
                })
                continue

            if split_data is None:
                print(f"  Skip: {DATASET} does not support {split_type}")
                continue

            X_train = split_data['X_train']
            y_train = split_data['y_train']
            X_test = split_data['X_test']
            y_test = split_data['y_test']

            # Subsample test for speed
            X_test, y_test = subsample_test(X_test, y_test, seed)

            print(f"  train={X_train.shape}, test={X_test.shape}")

            results_for_combo = {}

            for method in METHODS:
                elapsed = time.time() - start_time
                if elapsed > TIME_BUDGET_SEC:
                    print(f"\n[TIME BUDGET EXCEEDED] {elapsed/60:.1f}min")
                    all_results['time_budget_exceeded'] = True
                    all_results['partial_completion'] = True
                    break

                label = METHOD_LABELS[method]
                print(f"    Running {label}...", end=' ', flush=True)
                t0 = time.time()

                try:
                    ctx = min(CONTEXT_SIZE, len(X_train))
                    metrics = run_tta_method(method, X_train, y_train, X_test, y_test,
                                             n_select=ctx, seed=seed)
                    elapsed_method = time.time() - t0
                    metrics['elapsed'] = float(elapsed_method)
                    print(f"acc={metrics['accuracy']:.4f} f1m={metrics['f1_macro']:.4f} "
                          f"({elapsed_method:.1f}s)")
                    results_for_combo[method] = metrics
                except Exception as e:
                    elapsed_method = time.time() - t0
                    error_msg = f"{type(e).__name__}: {str(e)[:200]}"
                    print(f"FAILED ({elapsed_method:.1f}s): {error_msg}")
                    results_for_combo[method] = {
                        'error': error_msg,
                        'elapsed': float(elapsed_method),
                    }
                    all_results['errors'].append({
                        'dataset': DATASET, 'split': split_type, 'seed': seed,
                        'method': method, 'error': error_msg,
                        'traceback': traceback.format_exc()[:500],
                    })

                # Save incrementally after each method
                all_results['results'].append({
                    'dataset': DATASET,
                    'split': split_type,
                    'seed': seed,
                    'split_info': {
                        'n_train': int(len(X_train)),
                        'n_test': int(len(X_test)),
                        'n_features': int(X_train.shape[1]),
                        'n_classes': int(len(np.unique(y_train))),
                    },
                    'context_size': int(min(CONTEXT_SIZE, len(X_train))),
                    'results': results_for_combo,
                })
                with open(output_path, 'w') as f:
                    json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

            # Check time budget after each split
            if elapsed > TIME_BUDGET_SEC:
                break

    # ---- Summary ----
    print("\n" + "=" * 100)
    print("SUMMARY: Mean ± Std over completed seeds")
    print("=" * 100)

    summary = {}
    for combo in [f"{DATASET}_{s}" for s in SPLITS]:
        print(f"\n  [{combo}]")
        print(f"  {'Method':<25} {'Accuracy':<18} {'F1-Macro':<18} {'N'}")
        for method in METHODS:
            label = METHOD_LABELS[method]
            metrics_list = []
            for exp in all_results['results']:
                if f"{exp['dataset']}_{exp['split']}" == combo:
                    m = exp['results'].get(method)
                    if m and 'error' not in m:
                        metrics_list.append(m)
            if metrics_list:
                accs = [m['accuracy'] for m in metrics_list]
                f1s = [m['f1_macro'] for m in metrics_list]
                summary_key = f"{combo}_{method}"
                summary[summary_key] = {
                    'accuracy_mean': float(np.mean(accs)),
                    'accuracy_std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                    'f1_macro_mean': float(np.mean(f1s)),
                    'f1_macro_std': float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                    'n_seeds': len(metrics_list),
                }
                print(f"  {label:<25} {np.mean(accs):.4f}±{np.std(accs, ddof=1) if len(accs)>1 else 0:.4f}  "
                      f"{np.mean(f1s):.4f}  n={len(metrics_list)}")

    all_results['summary'] = summary
    all_results['timestamp_end'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    all_results['total_elapsed_sec'] = float(time.time() - start_time)
    with open(output_path, 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    print(f"Total experiments: {len(all_results['results'])}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    if all_results.get('partial_completion'):
        print(f"[PARTIAL COMPLETION] Time budget exceeded. Some seeds/splits not completed.")
    print("=" * 100)
    print("5-Seed TTA Comparison Complete!")
    print("=" * 100)


if __name__ == '__main__':
    main()
