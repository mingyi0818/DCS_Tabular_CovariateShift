"""Categorical-encoding ablation for DCS on Mushroom (reviewer M6).

Separates the SELECTOR's feature representation from the downstream TabPFN input:
  - random:              uniform random context
  - dcs_logistic_le:     DCS, domain classifier + k-means on label-encoded matrix (main protocol)
  - dcs_logistic_oh:     DCS, domain classifier + k-means on one-hot + standardized matrix
  - dcs_lightgbm_le:     DCS, LightGBM domain score on label-encoded matrix

TabPFN always consumes the SAME label-encoded context rows; only the selection
pipeline's representation changes. 5 seeds, iid + temporal splits, local GPU
TabPFN, context=10,000, full test set (n=9,161).

Saves results/mushroom_encoding_ablation.json.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
from splits import load_raw_dataframe, prepare_split
from context_shield_methods import dcs_selection, random_context_selection, set_seed, json_safe

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

SEEDS = [42, 123, 456, 789, 2024]
SPLITS = ['iid', 'temporal']
CONTEXT = 10000
K_CLUSTERS = 50
DEVICE = 'cuda'


def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    except Exception:
        auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


def run_tabpfn(X_ctx, y_ctx, X_test, y_test):
    clf = TabPFNClassifier(device=DEVICE)
    t0 = time.time(); clf.fit(X_ctx, y_ctx); fit_t = time.time() - t0
    t0 = time.time(); y_pred = clf.predict(X_test); pred_t = time.time() - t0
    try:
        y_proba = clf.predict_proba(X_test)
    except Exception:
        y_proba = np.column_stack([1 - y_pred, y_pred])
    m = compute_metrics(y_test, y_pred, y_proba)
    m['fit_time'] = float(fit_t); m['predict_time'] = float(pred_t)
    m['n_context'] = int(len(y_ctx))
    return m


def one_hot_matrices(train_df, test_df, target_col):
    """One-hot encode categoricals (categories from train), keep numerics, scale all."""
    cat_cols = [c for c in train_df.select_dtypes(include=['object']).columns if c != target_col]
    num_cols = [c for c in train_df.columns if c not in cat_cols and c != target_col]
    tr_oh = pd.get_dummies(train_df[cat_cols].astype(str))
    te_oh = pd.get_dummies(test_df[cat_cols].astype(str))
    te_oh = te_oh.reindex(columns=tr_oh.columns, fill_value=0)
    Xtr = np.hstack([train_df[num_cols].values.astype(float), tr_oh.values.astype(float)])
    Xte = np.hstack([test_df[num_cols].values.astype(float), te_oh.values.astype(float)])
    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte), len(num_cols), tr_oh.shape[1]


def main():
    path = os.path.join(RESULT_DIR, 'mushroom_encoding_ablation.json')
    out = {
        'experiment': 'mushroom_encoding_ablation',
        'config': {'dataset': 'mushroom', 'seeds': SEEDS, 'splits': SPLITS,
                   'context_size': CONTEXT, 'n_clusters': K_CLUSTERS, 'device': DEVICE,
                   'note': 'selection-side representation ablation; TabPFN always consumes '
                           'the label-encoded rows; one-hot variant uses one-hot + standardized '
                           'matrix for domain classifier and k-means only'},
        'results': [],
    }
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
        out['results'] = prev.get('results', [])
    done = {(r['split'], r['seed'], r['method']) for r in out['results']}

    for split_type in SPLITS:
        for seed in SEEDS:
            print(f'\n[mushroom/{split_type}/seed={seed}]', flush=True)
            set_seed(seed)
            split = prepare_split('mushroom', split_type, seed=seed)
            X_le_tr, y_tr = split['X_train'], split['y_train']
            X_le_te, y_te = split['X_test'], split['y_test']
            tr_df, te_df = split['train_df'], split['test_df']
            target = split['split_info']['target_col']
            X_oh_tr, X_oh_te, n_num, n_ohcol = one_hot_matrices(tr_df, te_df, target)
            print(f'  train={len(y_tr)}, test={len(y_te)}, oh-dim={X_oh_tr.shape[1]} '
                  f'({n_num} num + {n_ohcol} one-hot)', flush=True)

            jobs = []
            if (split_type, seed, 'random') not in done:
                t0 = time.time()
                idx = random_context_selection(X_le_tr, CONTEXT, seed=seed)
                jobs.append(('random', X_le_tr[idx], y_tr[idx], time.time() - t0))
            if (split_type, seed, 'dcs_logistic_le') not in done:
                t0 = time.time()
                idx = dcs_selection(X_le_tr, X_le_te, CONTEXT, n_clusters=K_CLUSTERS,
                                    method='logistic', seed=seed)
                jobs.append(('dcs_logistic_le', X_le_tr[idx], y_tr[idx], time.time() - t0))
            if (split_type, seed, 'dcs_logistic_oh') not in done:
                t0 = time.time()
                idx = dcs_selection(X_oh_tr, X_oh_te, CONTEXT, n_clusters=K_CLUSTERS,
                                    method='logistic', seed=seed)
                jobs.append(('dcs_logistic_oh', X_le_tr[idx], y_tr[idx], time.time() - t0))
            if (split_type, seed, 'dcs_lightgbm_le') not in done:
                t0 = time.time()
                idx = dcs_selection(X_le_tr, X_le_te, CONTEXT, n_clusters=K_CLUSTERS,
                                    method='lightgbm', seed=seed)
                jobs.append(('dcs_lightgbm_le', X_le_tr[idx], y_tr[idx], time.time() - t0))

            for name, X_ctx, y_ctx, sel_t in jobs:
                m = run_tabpfn(X_ctx, y_ctx, X_le_te, y_te)
                m['selection_time'] = float(sel_t)
                out['results'].append({'split': split_type, 'seed': seed, 'method': name,
                                       'metrics': m})
                print(f"  {name}: acc={m['accuracy']:.4f} (sel {sel_t:.1f}s)", flush=True)
                with open(path, 'w') as f:
                    json.dump(json_safe(out), f, indent=2)

    # summary
    summ = {}
    for split_type in SPLITS:
        for name in ['random', 'dcs_logistic_le', 'dcs_logistic_oh', 'dcs_lightgbm_le']:
            accs = [r['metrics']['accuracy'] for r in out['results']
                    if r['split'] == split_type and r['method'] == name]
            if accs:
                summ[f'{split_type}_{name}'] = {
                    'accuracy_mean': float(np.mean(accs)),
                    'accuracy_std': float(np.std(accs, ddof=1)),
                    'n_seeds': len(accs)}
    out['summary'] = summ
    print('\nSummary:')
    for k, v in summ.items():
        print(f"  {k}: {v['accuracy_mean']:.4f}±{v['accuracy_std']:.4f} (n={v['n_seeds']})")

    with open(path, 'w') as f:
        json.dump(json_safe(out), f, indent=2)
    print('\nDone. Results in results/mushroom_encoding_ablation.json')


if __name__ == '__main__':
    main()
