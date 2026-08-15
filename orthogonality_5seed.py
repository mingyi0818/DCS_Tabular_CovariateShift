"""Orthogonality Experiment (5-seed extension): runs the 2 additional seeds
[789, 2024] and combines them with the existing 3-seed results stored in
results/orthogonality_exp_results.json (seeds [42, 123, 456]).

Final combined file: results/orthogonality_5seed_results.json
Expected total entries: 5 seeds x 4 methods = 20.

The experimental setup (model loading, DCS-Logistic selection, metrics) is
identical to code/orthogonality_exp.py so the new seeds are directly
comparable to the existing 3 seeds.
"""
import os
import sys
import json
import time
import copy
import numpy as np
import torch

# Add Drift-Resilient TabPFN to path
DRIFT_TABPFN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'reference', 'Drift-Resilient_TabPFN-main',
)
sys.path.insert(0, DRIFT_TABPFN_PATH)

# Add our code to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULT_DIR
from splits import prepare_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Only run the 2 missing seeds. Existing file has [42, 123, 456].
NEW_SEEDS = [789, 2024]
EXISTING_SEEDS = [42, 123, 456]
ALL_SEEDS = EXISTING_SEEDS + NEW_SEEDS
CONTEXT_SIZE = 10000

EXISTING_RESULTS_PATH = os.path.join(RESULT_DIR, 'orthogonality_exp_results.json')
COMBINED_RESULTS_PATH = os.path.join(RESULT_DIR, 'orthogonality_5seed_results.json')
NEW_SEEDS_ONLY_PATH = os.path.join(RESULT_DIR, 'orthogonality_new_seeds_results.json')


def json_safe(obj):
    """Recursively convert numpy types to Python native types."""
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
        try:
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


def construct_dist_shift_domain(split_data, n_train_domains=5):
    """Construct dist_shift_domain from temporal ordering."""
    n_train = len(split_data['X_train'])
    n_test = len(split_data['X_test'])
    train_domain = np.zeros(n_train, dtype=np.int64)
    domain_size = n_train // n_train_domains
    for d in range(n_train_domains):
        start = d * domain_size
        end = (d + 1) * domain_size if d < n_train_domains - 1 else n_train
        train_domain[start:end] = d
    test_domain = np.full(n_test, n_train_domains, dtype=np.int64)
    return torch.LongTensor(train_domain), torch.LongTensor(test_domain)


def estimate_density_ratio(X_train, X_test, seed=42):
    """Estimate density ratio using logistic regression domain classifier."""
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    if n_test > 5000:
        rng = np.random.RandomState(seed)
        test_idx = rng.choice(n_test, 5000, replace=False)
        X_test_sample = X_test[test_idx]
    else:
        X_test_sample = X_test

    X_domain = np.vstack([X_train, X_test_sample])
    y_domain = np.concatenate([np.zeros(n_train), np.ones(len(X_test_sample))])

    scaler = StandardScaler()
    X_domain_s = scaler.fit_transform(X_domain)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_domain_s, y_domain)
    p_test = clf.predict_proba(scaler.transform(X_train))[:, 1]
    p_test = np.clip(p_test, 1e-6, 1 - 1e-6)
    return p_test / (1 - p_test)


def dcs_logistic_selection(X_train, X_test, n_select, n_clusters=50, seed=42):
    """DCS-Logistic selection (best method from context_shield experiments)."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)

    density_ratios = estimate_density_ratio(X_train, X_test, seed=seed)
    n_clusters = min(n_clusters, n_train)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train)

    cluster_ratios = np.zeros(n_clusters)
    cluster_sizes = np.zeros(n_clusters, dtype=int)
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_ratios[c] = density_ratios[mask].mean() if mask.sum() > 0 else 0
        cluster_sizes[c] = mask.sum()

    cluster_weights = cluster_ratios * cluster_sizes
    total_weight = cluster_weights.sum()
    if total_weight == 0:
        allocation = np.full(n_clusters, n_select // n_clusters)
    else:
        allocation = np.maximum(1, (cluster_weights / total_weight * n_select).astype(int))

    while allocation.sum() > n_select:
        c_min = np.argmin(cluster_weights * (allocation > 1))
        allocation[c_min] -= 1
    while allocation.sum() < n_select:
        c_max = np.argmax(cluster_weights)
        allocation[c_max] += 1

    selected = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_indices = np.where(mask)[0]
        cluster_dr = density_ratios[cluster_indices]
        n_from_cluster = min(allocation[c], len(cluster_indices))
        top_local = np.argsort(cluster_dr)[-n_from_cluster:]
        selected.extend(cluster_indices[top_local].tolist())

    return np.array(selected[:n_select])


def random_selection(X_train, n_select, seed=42):
    """Random context selection."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train)
    rng = np.random.RandomState(seed)
    return rng.choice(n_train, n_select, replace=False)


def load_drift_resilient_models():
    """Load dist and base models for Drift-Resilient TabPFN."""
    from importlib import resources
    import tabpfn
    from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig

    libpath = str(resources.files(tabpfn))

    def get_model(model_path, model_type):
        model_path_config = TabPFNModelPathsConfig(
            paths=[f"{libpath}/model_cache/{model_path}.cpkt"],
            task_type="dist_shift_multiclass"
        )
        model = get_best_tabpfn(
            task_type="dist_shift_multiclass",
            model_type=model_type,
            paths_config=model_path_config,
            debug=True,
            device="auto"
        )
        model.show_progress = False
        model.seed = 42
        return model

    dist_models = []
    base_models = []
    for i in [1, 2, 3]:
        print(f"  Loading tabpfn_dist_model_{i}...")
        dist_models.append(get_model(f"tabpfn_dist_model_{i}", "best_dist"))
        print(f"  Loading tabpfn_base_model_{i}...")
        base_models.append(get_model(f"tabpfn_base_model_{i}", "best_base"))

    return dist_models, base_models


def run_model_ensemble(models, X_train, y_train, X_test, train_domain, test_domain):
    """Run ensemble of models, average predicted probabilities."""
    all_preds = []
    for i, clf in enumerate(models):
        t0 = time.time()
        try:
            clf.fit(
                X_train, y_train,
                additional_x={"dist_shift_domain": train_domain}
            )
            fit_time = time.time() - t0

            t0 = time.time()
            preds = clf.predict_proba(
                X_test,
                additional_x={"dist_shift_domain": test_domain}
            )
            predict_time = time.time() - t0

            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
            all_preds.append(preds)
            print(f"    Model {i+1}/{len(models)}: fit={fit_time:.1f}s, predict={predict_time:.1f}s")
        except Exception as e:
            print(f"    Model {i+1}/{len(models)} FAILED: {e}")
            continue

    if not all_preds:
        return None, None

    avg_proba = np.mean(all_preds, axis=0)
    y_pred = np.argmax(avg_proba, axis=1)
    return y_pred, avg_proba


def run_one_seed(seed, dist_models, base_models):
    """Run all 4 methods for a single seed. Returns list of 4 result entries."""
    seed_results = []
    print(f"\n[seed={seed}]")
    np.random.seed(seed)
    torch.manual_seed(seed)

    split_data = prepare_split('adult', 'temporal', seed=seed)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    print(f"  train={X_train.shape}, test={X_test.shape}")

    train_domain_full, test_domain = construct_dist_shift_domain(split_data, n_train_domains=5)

    methods_to_run = [
        ('TabPFN-base-Random', base_models, 'random'),
        ('TabPFN-base-DCS-Logistic', base_models, 'dcs'),
        ('TabPFN-dist-Random', dist_models, 'random'),
        ('TabPFN-dist-DCS-Logistic', dist_models, 'dcs'),
    ]

    for method_name, models, sel_type in methods_to_run:
        print(f"  [{method_name}]")
        try:
            t0 = time.time()
            if sel_type == 'random':
                sel_idx = random_selection(X_train, CONTEXT_SIZE, seed=seed)
            else:
                sel_idx = dcs_logistic_selection(
                    X_train, X_test, CONTEXT_SIZE, n_clusters=50, seed=seed
                )
            sel_time = time.time() - t0
            X_ctx = X_train[sel_idx]
            y_ctx = y_train[sel_idx]
            train_domain_ctx = train_domain_full[sel_idx]
            y_pred, y_proba = run_model_ensemble(
                models, X_ctx, y_ctx, X_test, train_domain_ctx, test_domain
            )
            if y_pred is not None:
                metrics = compute_metrics(y_test, y_pred, y_proba)
                metrics['selection_time'] = float(sel_time)
                metrics['n_context'] = int(len(y_ctx))
                print(f"      acc={metrics['accuracy']:.4f} f1m={metrics['f1_macro']:.4f}")
                seed_results.append({
                    'dataset': 'adult', 'split': 'temporal', 'seed': seed,
                    'method': method_name, 'metrics': metrics,
                    'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                })
            else:
                print(f"      FAILED: no model produced predictions")
        except Exception as e:
            print(f"      FAILED: {e}")
            import traceback
            traceback.print_exc()

    return seed_results


def main():
    print("=" * 80)
    print("Orthogonality Experiment: 5-seed extension (new seeds [789, 2024])")
    print("=" * 80)

    # ---- Load existing results ----
    print(f"\n[1/4] Loading existing 3-seed results from {EXISTING_RESULTS_PATH}")
    if not os.path.exists(EXISTING_RESULTS_PATH):
        print(f"  FATAL: existing results file not found.")
        return
    with open(EXISTING_RESULTS_PATH, 'r') as f:
        existing_data = json.load(f)
    existing_results = existing_data.get('results', [])
    existing_seeds_seen = sorted(set(r['seed'] for r in existing_results))
    print(f"  Loaded {len(existing_results)} entries; seeds present: {existing_seeds_seen}")

    # Sanity: make sure none of the NEW_SEEDS are already in the existing file
    overlap = set(NEW_SEEDS) & set(existing_seeds_seen)
    if overlap:
        print(f"  WARNING: seeds {overlap} already present in existing file; "
              f"will skip re-running them and just keep existing entries.")
        seeds_to_run = [s for s in NEW_SEEDS if s not in existing_seeds_seen]
    else:
        seeds_to_run = list(NEW_SEEDS)
    print(f"  Seeds to run now: {seeds_to_run}")

    # ---- Load models ----
    print("\n[2/4] Loading Drift-Resilient TabPFN models...")
    if not seeds_to_run:
        print("  No new seeds to run; skipping model loading.")
        dist_models, base_models = None, None
    else:
        try:
            dist_models, base_models = load_drift_resilient_models()
            print(f"  Loaded {len(dist_models)} dist models, {len(base_models)} base models")
        except Exception as e:
            print(f"  FATAL: Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return

    # ---- Run experiments for new seeds ----
    print(f"\n[3/4] Running orthogonality experiments for new seeds {seeds_to_run}...")
    new_results = []
    for seed in seeds_to_run:
        seed_res = run_one_seed(seed, dist_models, base_models)
        new_results.extend(seed_res)
        # Incremental save of just the new seeds
        new_only = {
            'experiment': 'orthogonality_new_seeds',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'description': 'Orthogonality experiment for newly added seeds only',
            'new_seeds': seeds_to_run,
            'config': {
                'dataset': 'adult',
                'split': 'temporal',
                'seeds': seeds_to_run,
                'context_size': CONTEXT_SIZE,
                'methods': [
                    'TabPFN-base-Random',
                    'TabPFN-base-DCS-Logistic',
                    'TabPFN-dist-Random',
                    'TabPFN-dist-DCS-Logistic',
                ],
            },
            'results': new_results,
        }
        with open(NEW_SEEDS_ONLY_PATH, 'w') as f:
            json.dump(json_safe(new_only), f, indent=2, ensure_ascii=False)

    print(f"\n  New results collected: {len(new_results)} entries "
          f"(expected {len(seeds_to_run) * 4})")

    # ---- Combine with existing ----
    print("\n[4/4] Combining with existing 3-seed results...")
    combined_results = copy.deepcopy(existing_results)

    # Defensive: drop any duplicate (seed, method) entries from existing results
    # that would also appear in new_results. We key on (seed, method).
    seen_keys = set()
    deduped_existing = []
    for r in combined_results:
        key = (r['seed'], r['method'])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_existing.append(r)
    if len(deduped_existing) != len(combined_results):
        print(f"  Deduplicated existing entries: {len(combined_results)} -> {len(deduped_existing)}")
    combined_results = deduped_existing

    # Remove any existing entries whose (seed, method) is also produced by the
    # new run (prefer the freshly-run values).
    new_keys = {(r['seed'], r['method']) for r in new_results}
    combined_results = [r for r in combined_results
                        if (r['seed'], r['method']) not in new_keys]
    combined_results.extend(new_results)

    # Sort by seed then method for readability
    method_order = {
        'TabPFN-base-Random': 0,
        'TabPFN-base-DCS-Logistic': 1,
        'TabPFN-dist-Random': 2,
        'TabPFN-dist-DCS-Logistic': 3,
    }
    combined_results.sort(key=lambda r: (r['seed'], method_order.get(r['method'], 99)))

    # Build summary
    methods = ['TabPFN-base-Random', 'TabPFN-base-DCS-Logistic',
               'TabPFN-dist-Random', 'TabPFN-dist-DCS-Logistic']
    summary = {}
    for method in methods:
        accs = [r['metrics']['accuracy'] for r in combined_results
                if r['method'] == method and r.get('metrics')]
        f1s = [r['metrics']['f1_macro'] for r in combined_results
               if r['method'] == method and r.get('metrics')]
        aucs = [r['metrics']['auc'] for r in combined_results
                if r['method'] == method and r.get('metrics')]
        if accs:
            summary[method] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                'f1_macro_mean': float(np.mean(f1s)),
                'f1_macro_std': float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                'auc_mean': float(np.mean(aucs)),
                'auc_std': float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                'n_seeds': len(accs),
                'seeds': sorted(set(r['seed'] for r in combined_results
                                    if r['method'] == method)),
            }

    combined = {
        'experiment': 'orthogonality_exp_5seed',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'description': ('Test if DCS-Logistic context selection is orthogonal to '
                        'Drift-Resilient TabPFN (5-seed combined results)'),
        'config': {
            'dataset': 'adult',
            'split': 'temporal',
            'seeds': ALL_SEEDS,
            'existing_seeds': EXISTING_SEEDS,
            'new_seeds_added': NEW_SEEDS,
            'context_size': CONTEXT_SIZE,
            'methods': methods,
        },
        'source_files': {
            'existing_3seed': 'orthogonality_exp_results.json',
            'new_seeds_only': 'orthogonality_new_seeds_results.json',
        },
        'results': combined_results,
        'summary': summary,
    }

    with open(COMBINED_RESULTS_PATH, 'w') as f:
        json.dump(json_safe(combined), f, indent=2, ensure_ascii=False)

    print(f"\n  Combined results: {len(combined_results)} entries "
          f"(expected {len(ALL_SEEDS) * 4})")
    print(f"  Saved to: {COMBINED_RESULTS_PATH}")

    # ---- Print summary table ----
    print("\n" + "=" * 80)
    print("SUMMARY: 5-seed mean +/- std")
    print("=" * 80)
    print(f"{'Method':<32} {'Accuracy':<22} {'F1-Macro':<22} {'AUC':<22} {'N':<4}")
    print("-" * 100)
    base_random_acc = summary.get('TabPFN-base-Random', {}).get('accuracy_mean', 0)
    for method in methods:
        s = summary.get(method)
        if s:
            delta = (s['accuracy_mean'] - base_random_acc) * 100
            print(f"{method:<32} "
                  f"{s['accuracy_mean']:.4f}+/-{s['accuracy_std']:.4f}  "
                  f"{s['f1_macro_mean']:.4f}+/-{s['f1_macro_std']:.4f}  "
                  f"{s['auc_mean']:.4f}+/-{s['auc_std']:.4f}  "
                  f"{s['n_seeds']:<4d}  delta={delta:+.2f}pp")

    # ---- Orthogonality analysis ----
    print("\n" + "=" * 80)
    print("ORTHOGONALITY ANALYSIS (5 seeds)")
    print("=" * 80)
    base_random = summary.get('TabPFN-base-Random', {}).get('accuracy_mean')
    base_dcs = summary.get('TabPFN-base-DCS-Logistic', {}).get('accuracy_mean')
    dist_random = summary.get('TabPFN-dist-Random', {}).get('accuracy_mean')
    dist_dcs = summary.get('TabPFN-dist-DCS-Logistic', {}).get('accuracy_mean')

    if None not in (base_random, base_dcs, dist_random, dist_dcs):
        dcs_effect_on_base = (base_dcs - base_random) * 100
        dcs_effect_on_dist = (dist_dcs - dist_random) * 100
        dist_effect_on_base = (dist_random - base_random) * 100
        dist_effect_on_dcs = (dist_dcs - base_dcs) * 100

        print(f"\n  DCS-Logistic effect on base models:  {dcs_effect_on_base:+.2f}pp")
        print(f"  DCS-Logistic effect on dist models:  {dcs_effect_on_dist:+.2f}pp")
        print(f"  Drift-Resilient effect on random:    {dist_effect_on_base:+.2f}pp")
        print(f"  Drift-Resilient effect on DCS:       {dist_effect_on_dcs:+.2f}pp")
        print(f"\n  Combined gain over baseline:         "
              f"{(dist_dcs - base_random)*100:+.2f}pp")

        if dcs_effect_on_dist > 0:
            print(f"\n  ORTHOGONAL: DCS-Logistic improves dist models by "
                  f"{dcs_effect_on_dist:+.2f}pp")
        else:
            print(f"\n  NOT ORTHOGONAL: DCS-Logistic does not improve dist models")

    # ---- Verification ----
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    n_entries = len(combined_results)
    expected = len(ALL_SEEDS) * 4
    print(f"  Total entries:  {n_entries} (expected {expected})")
    seeds_in_combined = sorted(set(r['seed'] for r in combined_results))
    methods_in_combined = sorted(set(r['method'] for r in combined_results))
    print(f"  Seeds present:  {seeds_in_combined}")
    print(f"  Methods:        {len(methods_in_combined)} -> {methods_in_combined}")
    for seed in ALL_SEEDS:
        cnt = sum(1 for r in combined_results if r['seed'] == seed)
        print(f"    seed={seed}: {cnt} methods")
    if n_entries == expected:
        print("  PASS: 5 seeds x 4 methods = 20 entries verified.")
    else:
        print(f"  WARNING: entry count mismatch ({n_entries} vs {expected}).")

    print("\n" + "=" * 80)
    print("Orthogonality 5-seed experiment complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
