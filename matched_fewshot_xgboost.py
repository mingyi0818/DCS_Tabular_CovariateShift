"""Matched few-shot XGBoost comparison: DCS-selected vs random-pool samples.

Addresses reviewer concern: the existing few-shot comparison trained XGBoost on
the FIRST N rows of the feature-ordered (temporal) split, while DCS+TabPFN
selected N samples from the FULL training pool — an unfair (unmatched)
information comparison.

This script runs a MATCHED comparison. Both arms train the SAME XGBoost
config (copied verbatim from fewshot_5seed_xgboost.py) on N samples drawn
from the SAME full training pool:
  - dcs_N:        N samples selected by DCS (transductive, guided by test
                  features, identical selection procedure as DCS+TabPFN)
  - randompool_N: N uniformly random samples from the full pool
                  (np.random.RandomState(seed).choice, without replacement)

Protocol (identical to fewshot_5seed_xgboost.py / fewshot_5seed_tabpfn.py):
  - Feature-ordered (temporal) split, seed-dependent tie-break jitter
  - Train pool = first 70% of rows (Adult: 34,189; Mushroom: 42,748)
  - Test = last 15% of rows (Adult: 7,327; Mushroom: 9,161), NOT subsampled
  - Encoding/StandardScaler fit on train only

Datasets/budgets:
  - Adult:   seeds x budgets [200, 500, 1000, 2000, 5000, 10000], both arms
  - Mushroom: seeds x budget 10000, DCS arm

Saves: results/matched_fewshot_xgboost.json
"""
import os, sys, json, time, warnings, numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
from splits import SPLIT_CONFIG, load_raw_dataframe, encode_features
from context_shield_methods import dcs_selection, set_seed, json_safe
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEEDS = [42, 123, 456, 789, 2024]
BUDGETS = [200, 500, 1000, 2000, 5000, 10000]
K_CLUSTERS = 50
DCS_METHOD = 'logistic'

# XGBoost config copied verbatim from fewshot_5seed_xgboost.py (run_xgboost):
# n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42,
# eval_metric='logloss', use_label_encoder=False (paper's baseline config).
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
}


def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = 0.0
    if y_proba is not None:
        try:
            if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except Exception:
            pass
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


def run_xgboost(X_train, y_train, X_test, y_test):
    """Identical to fewshot_5seed_xgboost.run_xgboost."""
    if len(np.unique(y_train)) < 2:
        return {'accuracy': float(np.mean(y_test == y_train[0])), 'f1_macro': 0.0,
                'auc': 0.0, 'error': 'Only one class in training subset'}
    clf = xgb.XGBClassifier(**XGB_PARAMS)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        y_proba = None
    return compute_metrics(y_test, y_pred, y_proba)


# ---------------------------------------------------------------------------
# Split protocol — copied verbatim from fewshot_5seed_xgboost.py (which is the
# same as fewshot_5seed_tabpfn.py WITHOUT the 2000-sample test subsampling).
# ---------------------------------------------------------------------------

def sort_temporal(df, dataset_name, seed):
    cfg = SPLIT_CONFIG.get(dataset_name, {})
    temporal_col = cfg.get('temporal_col')
    if temporal_col is None:
        return df
    df_sorted = df.copy()
    if 'temporal_order' in cfg and cfg['temporal_order']:
        order_map = {m: i for i, m in enumerate(cfg['temporal_order'])}
        df_sorted['_o'] = df_sorted[temporal_col].map(lambda x: order_map.get(x, 0))
        rng = np.random.RandomState(seed)
        df_sorted['_j'] = rng.uniform(0, 0.5, size=len(df_sorted))
        df_sorted = df_sorted.sort_values(['_o', '_j']).drop(['_o', '_j'], axis=1)
    else:
        rng = np.random.RandomState(seed)
        std_val = df_sorted[temporal_col].std()
        js = 0.01 * std_val if std_val > 0 else 0.01
        df_sorted['_j'] = rng.uniform(0, js, size=len(df_sorted))
        df_sorted['_k'] = df_sorted[temporal_col] + df_sorted['_j']
        df_sorted = df_sorted.sort_values('_k').drop(['_j', '_k'], axis=1)
    return df_sorted


def prepare_split(dataset_name, seed):
    df, target_col = load_raw_dataframe(dataset_name)
    df_sorted = sort_temporal(df, dataset_name, seed)
    n = len(df_sorted)
    n_train = int(n * 0.7)
    n_test_start = int(n * 0.85)
    train_df = df_sorted.iloc[:n_train].copy()
    test_df = df_sorted.iloc[n_test_start:].copy()
    X_train_df, y_train, _ = encode_features(train_df, target_col)
    X_test_df, y_test, _ = encode_features(test_df, target_col, fit_df=train_df)
    for col in X_train_df.columns:
        if col not in X_test_df.columns:
            X_test_df[col] = 0
    X_test_df = X_test_df[X_train_df.columns]
    scaler = StandardScaler()
    return {
        'X_train': scaler.fit_transform(X_train_df.values),
        'X_test': scaler.transform(X_test_df.values),
        'y_train': y_train, 'y_test': y_test,
        'n_train': len(X_train_df), 'n_test': len(X_test_df),
    }


def save(all_results, output_path):
    with open(output_path, 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)


def main():
    print("=" * 80)
    print("Matched Few-Shot XGBoost: DCS-selected vs random-pool (full training pool)")
    print("=" * 80)
    ts = time.strftime('%Y-%m-%dT%H:%M:%S')
    print(f"Timestamp: {ts}")
    print(f"Seeds: {SEEDS}")
    print(f"Budgets: {BUDGETS}")
    print(f"XGBoost params: {XGB_PARAMS}")
    print(f"DCS: n_clusters={K_CLUSTERS}, method={DCS_METHOD} (transductive)")
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_path = os.path.join(RESULT_DIR, 'matched_fewshot_xgboost.json')

    all_results = {
        'experiment': 'matched_fewshot_xgboost',
        'timestamp': ts,
        'config': {
            'seeds': SEEDS,
            'budgets': BUDGETS,
            'xgboost_params': XGB_PARAMS,
            'protocol': ('XGBoost trained on DCS-selected N samples vs random N samples '
                         'from full pool (matched information)'),
            'dcs': {'n_clusters': K_CLUSTERS, 'method': DCS_METHOD,
                    'transductive': True,
                    'selection': 'dcs_selection(X_train, X_test, N, n_clusters=50, '
                                 "method='logistic', seed=seed) from context_shield_methods.py"},
            'random_pool': 'np.random.RandomState(seed).choice(n_train, N, replace=False)',
            'split': 'feature-ordered (temporal) split; train pool = first 70% rows; '
                     'test = last 15% rows (fixed, not subsampled)',
            'note': ('XGBoost params copied verbatim from code/fewshot_5seed_xgboost.py '
                     '(random_state fixed at 42, as in the paper baseline); seed variation '
                     'enters through temporal-split jitter, DCS selection, and random sampling'),
        },
        'adult': {},
        'mushroom': {},
    }

    # ========================= Adult (both arms) =========================
    print(f"\n{'=' * 60}\nDataset: adult (temporal split, full pool)\n{'=' * 60}")
    for seed in SEEDS:
        print(f"\n  [seed={seed}] ", end='', flush=True)
        set_seed(seed)
        try:
            split = prepare_split('adult', seed)
        except Exception as e:
            print(f"ERROR split: {e}")
            all_results['adult'][str(seed)] = {'error': str(e)}
            save(all_results, output_path)
            continue
        X_train, y_train = split['X_train'], split['y_train']
        X_test, y_test = split['X_test'], split['y_test']
        n_train, n_test = split['n_train'], split['n_test']
        print(f"train={n_train}, test={n_test}")
        seed_data = {'n_train': int(n_train), 'n_test': int(n_test)}

        for budget in BUDGETS:
            if budget >= n_train:
                continue
            # --- Arm A: DCS-selected (transductive, uses test features) ---
            t0 = time.time()
            dcs_idx = dcs_selection(X_train, X_test, budget,
                                    n_clusters=K_CLUSTERS, method=DCS_METHOD, seed=seed)
            sel_time = time.time() - t0
            assert len(dcs_idx) == budget, f"DCS returned {len(dcs_idx)} != {budget}"
            m = run_xgboost(X_train[dcs_idx], y_train[dcs_idx], X_test, y_test)
            m['selection_time'] = float(sel_time)
            m['n_train_samples'] = int(budget)
            seed_data[f'dcs_{budget}'] = m

            # --- Arm B: random from full pool ---
            rng = np.random.RandomState(seed)
            rnd_idx = rng.choice(n_train, budget, replace=False)
            m2 = run_xgboost(X_train[rnd_idx], y_train[rnd_idx], X_test, y_test)
            m2['n_train_samples'] = int(budget)
            seed_data[f'randompool_{budget}'] = m2

            print(f"    b{budget}: dcs={m['accuracy']:.4f} rndpool={m2['accuracy']:.4f}"
                  f" (delta={m['accuracy'] - m2['accuracy']:+.4f})", flush=True)

        all_results['adult'][str(seed)] = seed_data
        save(all_results, output_path)  # incremental save after each seed

    # ==================== Mushroom (DCS, N=10000 only) ====================
    print(f"\n{'=' * 60}\nDataset: mushroom (temporal split, full pool, N=10000)\n{'=' * 60}")
    for seed in SEEDS:
        print(f"\n  [seed={seed}] ", end='', flush=True)
        set_seed(seed)
        try:
            split = prepare_split('mushroom', seed)
        except Exception as e:
            print(f"ERROR split: {e}")
            all_results['mushroom'][str(seed)] = {'error': str(e)}
            save(all_results, output_path)
            continue
        X_train, y_train = split['X_train'], split['y_train']
        X_test, y_test = split['X_test'], split['y_test']
        n_train, n_test = split['n_train'], split['n_test']
        print(f"train={n_train}, test={n_test}")
        seed_data = {'n_train': int(n_train), 'n_test': int(n_test)}

        budget = 10000
        if budget < n_train:
            t0 = time.time()
            dcs_idx = dcs_selection(X_train, X_test, budget,
                                    n_clusters=K_CLUSTERS, method=DCS_METHOD, seed=seed)
            sel_time = time.time() - t0
            assert len(dcs_idx) == budget, f"DCS returned {len(dcs_idx)} != {budget}"
            m = run_xgboost(X_train[dcs_idx], y_train[dcs_idx], X_test, y_test)
            m['selection_time'] = float(sel_time)
            m['n_train_samples'] = int(budget)
            seed_data[f'dcs_{budget}'] = m
            print(f"    dcs_10000={m['accuracy']:.4f} (sel {sel_time:.1f}s)", flush=True)
        else:
            seed_data['error'] = f'n_train={n_train} <= budget {budget}'

        all_results['mushroom'][str(seed)] = seed_data
        save(all_results, output_path)

    # ============================ Summary ============================
    print(f"\n{'=' * 80}")
    print("SUMMARY: accuracy, 5-seed mean ± std (matched comparison)")
    print("=" * 80)
    summary = {'adult': {}, 'mushroom': {}}

    print(f"\n--- Adult (temporal split; pool n_train="
          f"{all_results['adult'][str(SEEDS[0])].get('n_train', '?')}, "
          f"test n={all_results['adult'][str(SEEDS[0])].get('n_test', '?')}) ---")
    print(f"{'Budget':>8} | {'DCS-selected XGB':>22} | {'Random-pool XGB':>22} | {'Delta':>8}")
    print("-" * 70)
    for budget in BUDGETS:
        dcs_accs, rnd_accs = [], []
        for s in SEEDS:
            sd = all_results['adult'].get(str(s), {})
            if f'dcs_{budget}' in sd and 'accuracy' in sd[f'dcs_{budget}']:
                dcs_accs.append(sd[f'dcs_{budget}']['accuracy'])
            if f'randompool_{budget}' in sd and 'accuracy' in sd[f'randompool_{budget}']:
                rnd_accs.append(sd[f'randompool_{budget}']['accuracy'])
        if dcs_accs and rnd_accs:
            dm, ds = np.mean(dcs_accs), (np.std(dcs_accs, ddof=1) if len(dcs_accs) > 1 else 0.0)
            rm, rs = np.mean(rnd_accs), (np.std(rnd_accs, ddof=1) if len(rnd_accs) > 1 else 0.0)
            print(f"{budget:>8} | {dm:>12.4f} ± {ds:.4f} | {rm:>12.4f} ± {rs:.4f} | {dm - rm:>+8.4f}")
            summary['adult'][str(budget)] = {
                'dcs_mean': float(dm), 'dcs_std': float(ds),
                'randompool_mean': float(rm), 'randompool_std': float(rs),
                'delta_mean': float(dm - rm), 'n_seeds': len(dcs_accs),
                'dcs_values': dcs_accs, 'randompool_values': rnd_accs,
            }

    print(f"\n--- Mushroom (temporal split; pool n_train="
          f"{all_results['mushroom'][str(SEEDS[0])].get('n_train', '?')}, "
          f"test n={all_results['mushroom'][str(SEEDS[0])].get('n_test', '?')}), dcs_10000 ---")
    dcs_accs = []
    for s in SEEDS:
        sd = all_results['mushroom'].get(str(s), {})
        if 'dcs_10000' in sd and 'accuracy' in sd['dcs_10000']:
            dcs_accs.append(sd['dcs_10000']['accuracy'])
    if dcs_accs:
        dm, ds = np.mean(dcs_accs), (np.std(dcs_accs, ddof=1) if len(dcs_accs) > 1 else 0.0)
        print(f"  dcs_10000: {dm:.4f} ± {ds:.4f} (n={len(dcs_accs)})")
        summary['mushroom']['dcs_10000'] = {
            'mean': float(dm), 'std': float(ds), 'n_seeds': len(dcs_accs),
            'values': dcs_accs,
        }

    all_results['summary'] = summary
    save(all_results, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
