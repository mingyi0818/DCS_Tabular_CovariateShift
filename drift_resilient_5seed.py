"""5-seed Drift-Resilient TabPFN experiment on Adult/temporal.

Extends feasibility_exp2_drift_resilient.py from 1 seed to 5 seeds to
address reviewer issue M2: the abstract compared +1.21pp DCS (5-seed, cloud)
with +1.09pp Drift-Resilient (1-seed, local), which is apples-to-oranges.

This script runs Drift-Resilient TabPFN (dist) vs standard TabPFN (base) with
5 seeds (42, 123, 456, 789, 2024) on Adult/temporal using LOCAL models.

Results saved to: results/drift_resilient_5seed.json
"""
import os
import sys
import json
import time
import numpy as np
import torch

# Add Drift-Resilient TabPFN to path
DRIFT_TABPFN_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reference', 'Drift-Resilient_TabPFN-main')
sys.path.insert(0, DRIFT_TABPFN_PATH)

# Add our code to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULT_DIR
from splits import prepare_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEEDS = [42, 123, 456, 789, 2024]


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


def load_drift_resilient_models():
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
        m = get_model(f"tabpfn_dist_model_{i}", "best_dist")
        dist_models.append(m)
        print(f"  Loading tabpfn_base_model_{i}...")
        m = get_model(f"tabpfn_base_model_{i}", "best_base")
        base_models.append(m)

    return dist_models, base_models


def run_model_ensemble(models, X_train, y_train, X_test, train_domain, test_domain):
    all_preds = []
    for i, clf in enumerate(models):
        print(f"    Model {i+1}/{len(models)}...")
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
            print(f"      fit={fit_time:.1f}s, predict={predict_time:.1f}s")
        except Exception as e:
            print(f"      FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_preds:
        return None, None

    avg_proba = np.mean(all_preds, axis=0)
    y_pred = np.argmax(avg_proba, axis=1)
    return y_pred, avg_proba


def main():
    print("=" * 78)
    print("5-Seed Drift-Resilient TabPFN on Adult/temporal split")
    print("=" * 78)

    results = {
        'experiment': 'drift_resilient_5seed',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'description': '5-seed Drift-Resilient TabPFN vs base on Adult/temporal (local models)',
        'config': {
            'seeds': SEEDS,
            'n_dist_models': 3,
            'n_base_models': 3,
            'dataset': 'adult',
            'split': 'temporal',
        },
        'results': [],
    }

    # ---- Load models ----
    print("\n[1/2] Loading Drift-Resilient TabPFN models...")
    try:
        dist_models, base_models = load_drift_resilient_models()
        print(f"  Loaded {len(dist_models)} dist models, {len(base_models)} base models")
    except Exception as e:
        print(f"  FATAL: Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
        with open(os.path.join(RESULT_DIR, 'drift_resilient_5seed.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return

    # ---- Run 5 seeds ----
    print("\n[2/2] Running 5 seeds on Adult/temporal...")

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)

        try:
            split_data = prepare_split('adult', 'temporal', seed=seed)
        except Exception as e:
            print(f"  ERROR preparing split: {e}")
            continue

        X_train = split_data['X_train']
        y_train = split_data['y_train']
        X_test = split_data['X_test']
        y_test = split_data['y_test']
        info = split_data['split_info']
        print(f"  train={X_train.shape}, test={X_test.shape}")

        train_domain, test_domain = construct_dist_shift_domain(split_data, n_train_domains=5)

        # --- TabPFN-dist (Drift-Resilient) ---
        print(f"  [a] TabPFN-dist (Drift-Resilient)...")
        y_pred, y_proba = run_model_ensemble(
            dist_models, X_train, y_train, X_test, train_domain, test_domain
        )
        if y_pred is not None:
            metrics = compute_metrics(y_test, y_pred, y_proba)
            print(f"      acc={metrics['accuracy']:.4f}, f1m={metrics['f1_macro']:.4f}")
            results['results'].append({
                'dataset': 'adult', 'split': 'temporal', 'seed': seed,
                'method': 'TabPFN-dist', 'metrics': metrics,
                'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                'note': 'Drift-Resilient TabPFN with dist_shift_domain'
            })

        # --- TabPFN-base ---
        print(f"  [b] TabPFN-base (standard)...")
        y_pred, y_proba = run_model_ensemble(
            base_models, X_train, y_train, X_test, train_domain, test_domain
        )
        if y_pred is not None:
            metrics = compute_metrics(y_test, y_pred, y_proba)
            print(f"      acc={metrics['accuracy']:.4f}, f1m={metrics['f1_macro']:.4f}")
            results['results'].append({
                'dataset': 'adult', 'split': 'temporal', 'seed': seed,
                'method': 'TabPFN-base', 'metrics': metrics,
                'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                'note': 'Standard TabPFN base models (no dist-shift awareness)'
            })

        # Save incrementally
        with open(os.path.join(RESULT_DIR, 'drift_resilient_5seed.json'), 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("SUMMARY: 5-seed mean ± std")
    print("=" * 78)

    dist_accs = [r['metrics']['accuracy'] for r in results['results']
                 if r['method'] == 'TabPFN-dist']
    base_accs = [r['metrics']['accuracy'] for r in results['results']
                 if r['method'] == 'TabPFN-base']
    dist_f1s = [r['metrics']['f1_macro'] for r in results['results']
                if r['method'] == 'TabPFN-dist']
    base_f1s = [r['metrics']['f1_macro'] for r in results['results']
                if r['method'] == 'TabPFN-base']

    if dist_accs and base_accs:
        dist_mean = np.mean(dist_accs)
        dist_std = np.std(dist_accs, ddof=1) if len(dist_accs) > 1 else 0.0
        base_mean = np.mean(base_accs)
        base_std = np.std(base_accs, ddof=1) if len(base_accs) > 1 else 0.0
        delta_pp = (dist_mean - base_mean) * 100

        dist_f1_mean = np.mean(dist_f1s)
        base_f1_mean = np.mean(base_f1s)

        print(f"  TabPFN-dist (Drift-Resilient): acc={dist_mean:.4f}±{dist_std:.4f}, f1={dist_f1_mean:.4f} (n={len(dist_accs)})")
        print(f"  TabPFN-base (standard):        acc={base_mean:.4f}±{base_std:.4f}, f1={base_f1_mean:.4f} (n={len(base_accs)})")
        print(f"  Drift-Resilient improvement:    {delta_pp:+.2f}pp (accuracy)")

        results['summary'] = {
            'dist_acc_mean': float(dist_mean),
            'dist_acc_std': float(dist_std),
            'base_acc_mean': float(base_mean),
            'base_acc_std': float(base_std),
            'delta_pp': float(delta_pp),
            'dist_f1_mean': float(dist_f1_mean),
            'base_f1_mean': float(base_f1_mean),
            'n_seeds': len(dist_accs),
        }

    with open(os.path.join(RESULT_DIR, 'drift_resilient_5seed.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {os.path.join(RESULT_DIR, 'drift_resilient_5seed.json')}")
    print("=" * 78)
    print("5-Seed Drift-Resilient Experiment Complete!")
    print("=" * 78)


if __name__ == '__main__':
    main()
