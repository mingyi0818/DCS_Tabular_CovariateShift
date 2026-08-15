"""Optimized comprehensive experiments using LOCAL TabPFN on GPU.

Key optimizations:
- Use GPU (RTX Pro 2000 16GB) instead of CPU
- Subsample test set to 2000 for large budgets
- Only run key budgets: 200, 1000, 5000, 10000
"""
import os, sys, json, time, numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              brier_score_loss, log_loss)
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR, DATASETS
from splits import SPLIT_CONFIG, load_raw_dataframe, encode_features
from context_shield_methods import (
    dcs_selection, random_context_selection, estimate_density_ratio,
    set_seed, json_safe,
)

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
except:
    pass

from tabpfn import TabPFNClassifier

SEEDS = [42, 123, 456, 789, 2024]
BUDGETS = [200, 500, 1000, 2000, 5000, 10000]
K_CLUSTERS = 50
N_BINS = 10
N_TEST_MAX = 2000  # Subsample test for speed
DEVICE = 'cuda'    # Use GPU


def compute_full_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if y_proba.ndim > 1:
        proba_pos = y_proba[:, 1]
    else:
        proba_pos = y_proba
    brier = brier_score_loss(y_true, proba_pos)
    nll = log_loss(y_true, np.column_stack([1-proba_pos, proba_pos]))
    bin_edges = np.linspace(0, 1, N_BINS + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(N_BINS):
        mask = (proba_pos >= bin_edges[i]) & (proba_pos < bin_edges[i+1])
        if mask.sum() == 0: continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - proba_pos[mask].mean())
    try: auc = roc_auc_score(y_true, proba_pos)
    except: auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc),
            'brier_score': float(brier), 'ece': float(ece), 'nll': float(nll)}


def run_local_tabpfn(X_ctx, y_ctx, X_test, y_test):
    t0 = time.time()
    clf = TabPFNClassifier(device=DEVICE)
    clf.fit(X_ctx, y_ctx)
    fit_time = time.time() - t0
    t0 = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - t0
    try: y_proba = clf.predict_proba(X_test)
    except: y_proba = np.column_stack([1-y_pred, y_pred])
    m = compute_full_metrics(y_test, y_pred, y_proba)
    m['fit_time'] = float(fit_time)
    m['predict_time'] = float(predict_time)
    m['n_context'] = int(len(y_ctx))
    return m


def dcs_selection_randomized(X_train, X_test, n_select, n_clusters=50, method='logistic', seed=42):
    n_train = X_train.shape[0]
    if n_train <= n_select: return np.arange(n_train)
    density_ratios = estimate_density_ratio(X_train, X_test, method=method, seed=seed)
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
        quotas = cluster_weights / total_weight * n_select
        allocation = np.minimum(np.floor(quotas).astype(int), cluster_sizes)
        allocation = np.maximum(1, allocation)
        remainders = quotas - np.floor(quotas)
        remaining = n_select - allocation.sum()
        if remaining > 0:
            order = np.argsort(-remainders)
            for idx in order:
                if remaining <= 0: break
                if allocation[idx] < cluster_sizes[idx]:
                    allocation[idx] += 1; remaining -= 1
        elif remaining < 0:
            order = np.argsort(remainders)
            for idx in order:
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
    return np.array(selected[:n_select])


def compute_kl(X_context, X_test, n_bins=20):
    n_features = X_context.shape[1]
    kls = []
    for f in range(n_features):
        cv, tv = X_context[:, f], X_test[:, f]
        av = np.concatenate([cv, tv])
        be = np.linspace(av.min(), av.max() + 1e-8, n_bins + 1)
        ch, _ = np.histogram(cv, bins=be, density=True)
        th, _ = np.histogram(tv, bins=be, density=True)
        eps = 1e-10
        ch = ch + eps; th = th + eps
        ch = ch / ch.sum(); th = th / th.sum()
        kls.append(float(np.sum(th * np.log(th / ch))))
    return float(np.mean(kls))


def sort_temporal(df, dataset_name, seed):
    cfg = SPLIT_CONFIG.get(dataset_name, {})
    tc = cfg.get('temporal_col')
    if tc is None: return df
    d = df.copy()
    if 'temporal_order' in cfg and cfg['temporal_order']:
        om = {m: i for i, m in enumerate(cfg['temporal_order'])}
        d['_o'] = d[tc].map(lambda x: om.get(x, 0))
        rng = np.random.RandomState(seed)
        d['_j'] = rng.uniform(0, 0.5, size=len(d))
        d = d.sort_values(['_o', '_j']).drop(['_o', '_j'], axis=1)
    else:
        rng = np.random.RandomState(seed)
        sv = d[tc].std()
        js = 0.01 * sv if sv > 0 else 0.01
        d['_j'] = rng.uniform(0, js, size=len(d))
        d['_k'] = d[tc] + d['_j']
        d = d.sort_values('_k').drop(['_j', '_k'], axis=1)
    return d


def prepare_split(dataset_name, seed):
    df, tc = load_raw_dataframe(dataset_name)
    d = sort_temporal(df, dataset_name, seed)
    n = len(d)
    nt = int(n * 0.7); nts = int(n * 0.85)
    tr = d.iloc[:nt].copy(); te = d.iloc[nts:].copy()
    Xtr_df, ytr, _ = encode_features(tr, tc)
    Xte_df, yte, _ = encode_features(te, tc, fit_df=tr)
    for c in Xtr_df.columns:
        if c not in Xte_df.columns: Xte_df[c] = 0
    Xte_df = Xte_df[Xtr_df.columns]
    sc = StandardScaler()
    X_train = sc.fit_transform(Xtr_df.values)
    X_test = sc.transform(Xte_df.values)
    y_train, y_test = ytr, yte
    # Subsample test
    if len(X_test) > N_TEST_MAX:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X_test), N_TEST_MAX, replace=False)
        X_test = X_test[idx]; y_test = y_test[idx]
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'n_train': len(X_train), 'n_test': len(X_test)}


def main():
    print("="*80)
    print("Comprehensive Local TabPFN on GPU (No API Limit)")
    print("="*80)
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Device: {DEVICE}, Max test: {N_TEST_MAX}")
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Check GPU
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    all_results = {
        'experiment': 'comprehensive_local_tabpfn_gpu',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {'seeds': SEEDS, 'budgets': BUDGETS, 'K': K_CLUSTERS,
                   'tabpfn_mode': 'local_gpu', 'device': DEVICE, 'n_test_max': N_TEST_MAX},
        'fewshot_5seed': {},
        'kl_accuracy_tabpfn': {},
        'randomized_dcs': {},
        'calibration': {},
    }
    
    # === Task 1+2: 5-seed few-shot ===
    for ds_name in ['adult', 'mushroom']:
        print(f"\n{'='*60}\n{ds_name} few-shot (5 seeds)\n{'='*60}")
        all_results['fewshot_5seed'][ds_name] = {}
        
        for seed in SEEDS:
            print(f"\n  [seed={seed}]")
            set_seed(seed)
            try: split = prepare_split(ds_name, seed)
            except Exception as e:
                print(f"    ERROR: {e}"); continue
            
            X_train, y_train = split['X_train'], split['y_train']
            X_test, y_test = split['X_test'], split['y_test']
            n_train = split['n_train']
            print(f"    train={n_train}, test={split['n_test']}")
            
            sd = {'n_train': n_train, 'n_test': split['n_test'], 'budgets': {}}
            
            for budget in BUDGETS:
                if budget >= n_train: continue
                ctx = min(budget, 10000)
                print(f"    b={budget}...", end=' ', flush=True)
                
                try:
                    idx = dcs_selection(X_train, X_test, ctx, n_clusters=min(K_CLUSTERS, ctx),
                                        method='logistic', seed=seed)
                    m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
                    sd['budgets'][f'dcs_{budget}'] = m
                    print(f"D={m['accuracy']:.4f} ", end=' ', flush=True)
                except Exception as e:
                    sd['budgets'][f'dcs_{budget}'] = {'error': str(e)[:80]}
                    print(f"D=FAIL ", end=' ', flush=True)
                
                try:
                    idx = random_context_selection(X_train, ctx, seed=seed)
                    m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
                    sd['budgets'][f'random_{budget}'] = m
                    print(f"R={m['accuracy']:.4f}", flush=True)
                except Exception as e:
                    sd['budgets'][f'random_{budget}'] = {'error': str(e)[:80]}
                    print(f"R=FAIL", flush=True)
            
            all_results['fewshot_5seed'][ds_name][str(seed)] = sd
            # Save incrementally
            with open(os.path.join(RESULT_DIR, 'comprehensive_local_tabpfn.json'), 'w') as f:
                json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
    
    # === Task 3: KL-accuracy on TabPFN ===
    print(f"\n{'='*60}\nKL-accuracy on TabPFN\n{'='*60}")
    set_seed(42)
    split = prepare_split('adult', 42)
    X_train, y_train = split['X_train'], split['y_train']
    X_test, y_test = split['X_test'], split['y_test']
    
    kl_data = []
    for budget in BUDGETS:
        if budget >= split['n_train']: continue
        ctx = min(budget, 10000)
        for mn, sf in [('DCS', dcs_selection), ('Random', random_context_selection)]:
            print(f"  {mn} b={budget}...", end=' ', flush=True)
            try:
                if mn == 'DCS':
                    idx = sf(X_train, X_test, ctx, n_clusters=min(K_CLUSTERS, ctx), method='logistic', seed=42)
                else:
                    idx = sf(X_train, ctx, seed=42)
                kl = compute_kl(X_train[idx], X_test)
                m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
                kl_data.append({'method': mn, 'budget': budget, 'kl': kl,
                                'accuracy': m['accuracy'], 'brier': m['brier_score'],
                                'ece': m['ece'], 'nll': m['nll']})
                print(f"KL={kl:.4f} acc={m['accuracy']:.4f}")
            except Exception as e:
                print(f"FAIL: {str(e)[:60]}")
    
    from scipy.stats import pearsonr, spearmanr
    kls = [d['kl'] for d in kl_data]
    accs = [d['accuracy'] for d in kl_data]
    r_p, p_p = pearsonr(kls, accs)
    r_s, p_s = spearmanr(kls, accs)
    all_results['kl_accuracy_tabpfn'] = {
        'data': kl_data, 'pearson_r': float(r_p), 'pearson_p': float(p_p),
        'spearman_r': float(r_s), 'spearman_p': float(p_s)}
    print(f"\n  KL vs Acc (TabPFN): r={r_p:.4f} (p={p_p:.4f}), rho={r_s:.4f} (p={p_s:.4f})")
    
    # === Task 4: Randomized DCS ===
    print(f"\n{'='*60}\nRandomized DCS variant\n{'='*60}")
    for seed in SEEDS:
        print(f"\n  [seed={seed}]")
        set_seed(seed)
        split = prepare_split('adult', seed)
        X_train, y_train = split['X_train'], split['y_train']
        X_test, y_test = split['X_test'], split['y_test']
        sd = {}
        for budget in [200, 1000, 10000]:
            if budget >= split['n_train']: continue
            ctx = min(budget, 10000)
            print(f"    b={budget}...", end=' ', flush=True)
            try:
                idx = dcs_selection(X_train, X_test, ctx, n_clusters=min(K_CLUSTERS, ctx), method='logistic', seed=seed)
                m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
                sd[f'det_{budget}'] = m
                print(f"D={m['accuracy']:.4f} ", end=' ', flush=True)
            except: print(f"D=FAIL ", end=' ', flush=True)
            try:
                idx = dcs_selection_randomized(X_train, X_test, ctx, n_clusters=min(K_CLUSTERS, ctx), method='logistic', seed=seed)
                m = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
                sd[f'rnd_{budget}'] = m
                print(f"R={m['accuracy']:.4f}", flush=True)
            except: print(f"R=FAIL", flush=True)
        all_results['randomized_dcs'][str(seed)] = sd
        with open(os.path.join(RESULT_DIR, 'comprehensive_local_tabpfn.json'), 'w') as f:
            json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
    
    # === Task 5: Calibration ===
    print(f"\n{'='*60}\nCalibration\n{'='*60}")
    set_seed(42)
    split = prepare_split('adult', 42)
    X_train, y_train = split['X_train'], split['y_train']
    X_test, y_test = split['X_test'], split['y_test']
    
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                            random_state=42, eval_metric='logloss', use_label_encoder=False)
    clf.fit(X_train, y_train)
    xp = clf.predict_proba(X_test)[:, 1]
    xd = clf.predict(X_test)
    xc = compute_full_metrics(y_test, xd, np.column_stack([1-xp, xp]))
    all_results['calibration']['XGBoost'] = xc
    print(f"  XGBoost: Brier={xc['brier_score']:.6f}, ECE={xc['ece']:.6f}")
    
    idx = dcs_selection(X_train, X_test, 10000, n_clusters=K_CLUSTERS, method='logistic', seed=42)
    dc = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
    all_results['calibration']['TabPFN-DCS'] = dc
    print(f"  TabPFN-DCS: Brier={dc['brier_score']:.6f}, ECE={dc['ece']:.6f}")
    
    idx = random_context_selection(X_train, 10000, seed=42)
    rc = run_local_tabpfn(X_train[idx], y_train[idx], X_test, y_test)
    all_results['calibration']['TabPFN-Random'] = rc
    print(f"  TabPFN-Random: Brier={rc['brier_score']:.6f}, ECE={rc['ece']:.6f}")
    
    # === Summary ===
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    for ds in all_results['fewshot_5seed']:
        print(f"\n--- {ds} ---")
        for b in BUDGETS:
            da, ra = [], []
            for s in SEEDS:
                if str(s) in all_results['fewshot_5seed'][ds]:
                    bd = all_results['fewshot_5seed'][ds][str(s)].get('budgets', {})
                    if f'dcs_{b}' in bd and 'accuracy' in bd[f'dcs_{b}']: da.append(bd[f'dcs_{b}']['accuracy'])
                    if f'random_{b}' in bd and 'accuracy' in bd[f'random_{b}']: ra.append(bd[f'random_{b}']['accuracy'])
            if da:
                print(f"  b={b}: DCS={np.mean(da):.4f}±{np.std(da,ddof=1):.4f}(n={len(da)}), "
                      f"Rnd={np.mean(ra):.4f}±{np.std(ra,ddof=1):.4f}(n={len(ra)})")
    
    kl = all_results['kl_accuracy_tabpfn']
    print(f"\n--- KL vs Acc (TabPFN) ---")
    print(f"  r={kl['pearson_r']:.4f} (p={kl['pearson_p']:.4f}), rho={kl['spearman_r']:.4f} (p={kl['spearman_p']:.4f})")
    
    print(f"\n--- Calibration ---")
    for m, c in all_results['calibration'].items():
        if 'brier_score' in c:
            print(f"  {m}: Brier={c['brier_score']:.6f}, ECE={c['ece']:.6f}, NLL={c['nll']:.6f}")
    
    with open(os.path.join(RESULT_DIR, 'comprehensive_local_tabpfn.json'), 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nSaved to comprehensive_local_tabpfn.json")


if __name__ == '__main__':
    main()
