"""Global importance-sampling context selection aligned with Theorem 1 (reviewer M1).

Compares four context-selection variants on Adult/feature-ordered, budgets
{200, 1000, 10000}, 5 seeds, local TabPFN on GPU, n_test subsampled to 2000
(protocol identical to the randomized-DCS section of comprehensive_local_tabpfn.json):

  - det:        deterministic DCS (Algorithm 1, top-b_k within cluster)
  - rnd:        randomized stratified DCS (sampling within fixed cluster quotas)
  - global_is:  GLOBAL draws WITH REPLACEMENT with probability proportional to the
                estimated density-ratio score s_i / sum_j s_j  (Theorem 1's law
                with estimated ratios); duplicates possible
  - random:     uniform random context (reference)

Saves results/global_is_dcs_results.json.
"""
import os, sys, json, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
from splits import SPLIT_CONFIG, load_raw_dataframe, encode_features
from context_shield_methods import dcs_selection, random_context_selection, \
    estimate_density_ratio, set_seed, json_safe

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
BUDGETS = [200, 1000, 10000]
K_CLUSTERS = 50
N_TEST_MAX = 2000
DEVICE = 'cuda'
N_BINS = 10


def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    brier = brier_score_loss(y_true, proba_pos)
    nll = log_loss(y_true, np.column_stack([1 - proba_pos, proba_pos]))
    bin_edges = np.linspace(0, 1, N_BINS + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(N_BINS):
        mask = (proba_pos >= bin_edges[i]) & (proba_pos < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - proba_pos[mask].mean())
    try:
        auc = roc_auc_score(y_true, proba_pos)
    except Exception:
        auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc),
            'brier_score': float(brier), 'ece': float(ece), 'nll': float(nll)}


def run_local_tabpfn(X_ctx, y_ctx, X_test, y_test):
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


def dcs_selection_randomized(X_train, X_test, n_select, n_clusters=50, method='logistic', seed=42):
    """Stratified randomized DCS: sampling w/o replacement within fixed cluster quotas."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train), n_train
    density_ratios = estimate_density_ratio(X_train, X_test, method=method, seed=seed)
    n_clusters = min(n_clusters, n_train)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train)
    cluster_ratios = np.zeros(n_clusters); cluster_sizes = np.zeros(n_clusters, dtype=int)
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_ratios[c] = density_ratios[mask].mean() if mask.sum() > 0 else 0
        cluster_sizes[c] = mask.sum()
    cluster_weights = cluster_ratios * cluster_sizes
    total_weight = cluster_weights.sum()
    if total_weight == 0:
        allocation = np.full(n_clusters, n_select // n_clusters)
    else:
        quotas = cluster_weights / total_weight * n_select
        allocation = np.minimum(np.floor(quotas).astype(int), cluster_sizes)
        allocation = np.maximum(1, allocation)
        remainders = quotas - np.floor(quotas)
        remaining = n_select - allocation.sum()
        if remaining > 0:
            for idx in np.argsort(-remainders):
                if remaining <= 0: break
                if allocation[idx] < cluster_sizes[idx]:
                    allocation[idx] += 1; remaining -= 1
        elif remaining < 0:
            for idx in np.argsort(remainders):
                if remaining >= 0: break
                if allocation[idx] > 1:
                    allocation[idx] -= 1; remaining += 1
    rng = np.random.RandomState(seed)
    selected = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_indices = np.where(mask)[0]
        cluster_dr = density_ratios[cluster_indices]
        n_from_cluster = min(allocation[c], len(cluster_indices))
        if n_from_cluster <= 0: continue
        dr_positive = np.maximum(cluster_dr, 1e-10)
        probs = dr_positive / dr_positive.sum()
        sampled = rng.choice(cluster_indices, size=n_from_cluster, replace=False, p=probs)
        selected.extend(sampled.tolist())
    idx = np.array(selected[:n_select])
    return idx, len(np.unique(idx))


def global_is_selection(X_train, X_test, n_select, method='logistic', seed=42):
    """Global importance sampling: WITH replacement, P(i) = s_i / sum_j s_j."""
    n_train = X_train.shape[0]
    if n_train <= n_select:
        return np.arange(n_train), n_train
    s = estimate_density_ratio(X_train, X_test, method=method, seed=seed)
    s_positive = np.maximum(s, 1e-10)
    probs = s_positive / s_positive.sum()
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_train, size=n_select, replace=True, p=probs)
    return idx, len(np.unique(idx))


def prepare_split(dataset_name, seed):
    df, tc = load_raw_dataframe(dataset_name)
    cfg = SPLIT_CONFIG.get(dataset_name, {})
    tcol = cfg.get('temporal_col')
    d = df.copy()
    if tcol:
        if 'temporal_order' in cfg and cfg['temporal_order']:
            om = {m: i for i, m in enumerate(cfg['temporal_order'])}
            d['_o'] = d[tcol].map(lambda x: om.get(x, 0))
            rng = np.random.RandomState(seed)
            d['_j'] = rng.uniform(0, 0.5, size=len(d))
            d = d.sort_values(['_o', '_j']).drop(['_o', '_j'], axis=1)
        else:
            rng = np.random.RandomState(seed)
            sv = d[tcol].std()
            js = 0.01 * sv if sv > 0 else 0.01
            d['_j'] = rng.uniform(0, js, size=len(d))
            d['_k'] = d[tcol] + d['_j']
            d = d.sort_values('_k').drop(['_j', '_k'], axis=1)
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
    if len(X_test) > N_TEST_MAX:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_test), N_TEST_MAX, replace=False)
        X_test = X_test[idx]; yte = yte[idx]
    return X_train, ytr, X_test, yte


def main():
    out = {
        'experiment': 'global_is_dcs_comparison',
        'config': {'seeds': SEEDS, 'budgets': BUDGETS, 'K': K_CLUSTERS,
                   'device': DEVICE, 'n_test_max': N_TEST_MAX,
                   'dataset': 'adult', 'split': 'feature-ordered',
                   'variants': ['deterministic_dcs', 'randomized_stratified_dcs',
                                'global_is', 'random']},
        'results': {},
    }
    path = os.path.join(RESULT_DIR, 'global_is_dcs_results.json')
    for seed in SEEDS:
        if str(seed) in out['results']:
            continue
        print(f'\n[seed={seed}]')
        set_seed(seed)
        X_train, y_train, X_test, y_test = prepare_split('adult', seed)
        sd = {}
        for budget in BUDGETS:
            print(f'  b={budget}:', end=' ', flush=True)
            # deterministic DCS
            idx = dcs_selection(X_train, X_test, budget, n_clusters=min(K_CLUSTERS, budget),
                                method='logistic', seed=seed)
            sd[f'det_{budget}'] = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
            sd[f'det_{budget}']['n_unique'] = int(len(np.unique(idx)))
            print(f"det={sd[f'det_{budget}']['accuracy']:.4f}", end=' ', flush=True)
            # randomized stratified DCS
            idx, nu = dcs_selection_randomized(X_train, X_test, budget,
                                               n_clusters=min(K_CLUSTERS, budget),
                                               method='logistic', seed=seed)
            sd[f'rnd_{budget}'] = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
            sd[f'rnd_{budget}']['n_unique'] = int(nu)
            print(f"rnd={sd[f'rnd_{budget}']['accuracy']:.4f}", end=' ', flush=True)
            # global importance sampling (with replacement)
            idx, nu = global_is_selection(X_train, X_test, budget, method='logistic', seed=seed)
            sd[f'gis_{budget}'] = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
            sd[f'gis_{budget}']['n_unique'] = int(nu)
            print(f"gis={sd[f'gis_{budget}']['accuracy']:.4f}(u={nu})", end=' ', flush=True)
            # random reference
            idx = random_context_selection(X_train, budget, seed=seed)
            sd[f'ran_{budget}'] = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
            sd[f'ran_{budget}']['n_unique'] = int(len(np.unique(idx)))
            print(f"ran={sd[f'ran_{budget}']['accuracy']:.4f}", flush=True)
        out['results'][str(seed)] = sd
        with open(path, 'w') as f:
            json.dump(json_safe(out), f, indent=2)
    # summary
    summ = {}
    for v in ['det', 'rnd', 'gis', 'ran']:
        for b in BUDGETS:
            accs = [out['results'][str(s)][f'{v}_{b}']['accuracy'] for s in SEEDS]
            unqs = [out['results'][str(s)][f'{v}_{b}']['n_unique'] for s in SEEDS]
            summ[f'{v}_{b}'] = {'acc_mean': float(np.mean(accs)),
                                'acc_std': float(np.std(accs, ddof=1)),
                                'n_unique_mean': float(np.mean(unqs))}
    out['summary'] = summ
    with open(path, 'w') as f:
        json.dump(json_safe(out), f, indent=2)
    print('\nSummary:')
    for k, v in summ.items():
        print(f"  {k}: {v['acc_mean']:.4f}±{v['acc_std']:.4f} (unique={v['n_unique_mean']:.0f})")


if __name__ == '__main__':
    main()
