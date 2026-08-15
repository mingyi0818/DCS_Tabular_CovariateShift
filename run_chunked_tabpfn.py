"""Chunked TabPFN baseline: use ALL training data by chunking into 10K blocks.
Compares DCS (10K selected) vs Random (10K selected) vs Chunked (all data, chunked).
Reference: Sergazinov & Yin (2025), arXiv:2509.00326
"""
import os, sys, json, time, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
from splits import SPLIT_CONFIG, load_raw_dataframe, encode_features
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
K_CLUSTERS = 50
N_CONTEXT = 10000  # TabPFN max context per chunk
N_TEST_MAX = 2000
DEVICE = 'cuda'


def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    proba_pos = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    try:
        auc = roc_auc_score(y_true, proba_pos)
    except:
        auc = 0.0
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}


def run_chunked_tabpfn(X_train, y_train, X_test, y_test, chunk_size=10000):
    """Run TabPFN on all training data by chunking.
    Average predicted probabilities across chunks (log-sum-exp merge)."""
    n_train = len(X_train)
    n_chunks = int(np.ceil(n_train / chunk_size))

    print(f"    Chunked TabPFN: {n_train} samples in {n_chunks} chunks")

    all_probas = []
    total_fit_time = 0
    total_predict_time = 0

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n_train)
        X_chunk = X_train[start:end]
        y_chunk = y_train[start:end]

        if len(np.unique(y_chunk)) < 2:
            # Skip chunks with only one class
            continue

        clf = TabPFNClassifier(device=DEVICE)
        t0 = time.time()
        clf.fit(X_chunk, y_chunk)
        total_fit_time += time.time() - t0

        t0 = time.time()
        try:
            proba = clf.predict_proba(X_test)
            all_probas.append(proba)
        except Exception as e:
            print(f"      Chunk {i+1}/{n_chunks}: predict failed ({e})")
            continue
        total_predict_time += time.time() - t0

        if (i + 1) % 5 == 0:
            print(f"      Chunk {i+1}/{n_chunks} done")

    if not all_probas:
        return {'error': 'All chunks failed'}

    # Average probabilities across chunks
    avg_proba = np.mean(all_probas, axis=0)
    y_pred = np.argmax(avg_proba, axis=1)

    m = compute_metrics(y_test, y_pred, avg_proba)
    m['fit_time'] = float(total_fit_time)
    m['predict_time'] = float(total_predict_time)
    m['n_chunks'] = len(all_probas)
    m['n_train_total'] = int(n_train)
    return m


def run_tabpfn(X_ctx, y_ctx, X_test, y_test):
    """Standard TabPFN with 10K context."""
    clf = TabPFNClassifier(device=DEVICE)
    t0 = time.time()
    clf.fit(X_ctx, y_ctx)
    ft = time.time() - t0
    t0 = time.time()
    y_pred = clf.predict(X_test)
    pt = time.time() - t0
    try:
        y_proba = clf.predict_proba(X_test)
    except:
        y_proba = np.column_stack([1-y_pred, y_pred])
    m = compute_metrics(y_test, y_pred, y_proba)
    m['fit_time'] = float(ft)
    m['predict_time'] = float(pt)
    m['n_context'] = int(len(y_ctx))
    return m


def prepare_split(ds, seed):
    """Prepare Adult dataset split."""
    df, tc = load_raw_dataframe(ds)
    cfg = SPLIT_CONFIG.get(ds, {})
    tcol = cfg.get('temporal_col')
    if tcol:
        d = df.copy()
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
    else:
        d = df

    n = len(d)
    nt = int(n * 0.7)
    nts = int(n * 0.85)
    tr = d.iloc[:nt].copy()
    te = d.iloc[nts:].copy()
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
        X_test = X_test[idx]
        yte = yte[idx]
    return {'X_train': X_train, 'X_test': X_test,
            'y_train': ytr, 'y_test': yte}


def main():
    print("=" * 80)
    print("Chunked TabPFN Baseline Experiment")
    print("=" * 80)

    output_path = os.path.join(RESULT_DIR, 'chunked_tabpfn_results.json')

    # Load existing results
    try:
        with open(output_path, 'r') as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {output_path}")
    except:
        all_results = {
            'experiment': 'chunked_tabpfn_baseline',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'config': {'seeds': SEEDS, 'n_context': N_CONTEXT,
                       'n_test_max': N_TEST_MAX, 'chunk_size': N_CONTEXT},
            'results': {}
        }

    for seed in SEEDS:
        if str(seed) in all_results['results']:
            print(f"\n[seed={seed}] already done, skipping")
            continue

        print(f"\n[seed={seed}]")
        set_seed(seed)
        split = prepare_split('adult', seed)
        X_train, y_train = split['X_train'], split['y_train']
        X_test, y_test = split['X_test'], split['y_test']
        print(f"  train={len(X_train)}, test={len(X_test)}")

        result = {}

        # DCS (10K selected)
        print("  Running DCS...")
        t0 = time.time()
        dcs_idx = dcs_selection(X_train, X_test, N_CONTEXT,
                                n_clusters=K_CLUSTERS, method='logistic', seed=seed)
        result['dcs'] = run_tabpfn(X_train[dcs_idx], y_train[dcs_idx], X_test, y_test)
        result['dcs']['selection_time'] = float(time.time() - t0)
        print(f"    DCS: acc={result['dcs']['accuracy']:.4f}")

        # Random (10K selected)
        print("  Running Random...")
        rnd_idx = random_context_selection(X_train, N_CONTEXT, seed=seed)
        result['random'] = run_tabpfn(X_train[rnd_idx], y_train[rnd_idx], X_test, y_test)
        print(f"    Random: acc={result['random']['accuracy']:.4f}")

        # Chunked (all data)
        print("  Running Chunked TabPFN...")
        result['chunked'] = run_chunked_tabpfn(X_train, y_train, X_test, y_test)
        if 'accuracy' in result['chunked']:
            print(f"    Chunked: acc={result['chunked']['accuracy']:.4f}")
        else:
            print(f"    Chunked: {result['chunked']}")

        all_results['results'][str(seed)] = result

        # Save after each seed
        with open(output_path, 'w') as f:
            json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
        print(f"  Saved.")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("=" * 80)

    valid_seeds = [s for s in SEEDS if str(s) in all_results['results']
                   and 'accuracy' in all_results['results'][str(s)].get('dcs', {})]

    if valid_seeds:
        dcs_accs = [all_results['results'][str(s)]['dcs']['accuracy'] for s in valid_seeds]
        rnd_accs = [all_results['results'][str(s)]['random']['accuracy'] for s in valid_seeds]

        print(f"\nAdult (n={len(valid_seeds)} seeds):")
        print(f"  DCS (10K selected):     {np.mean(dcs_accs):.4f}±{np.std(dcs_accs, ddof=1):.4f}")
        print(f"  Random (10K selected):  {np.mean(rnd_accs):.4f}±{np.std(rnd_accs, ddof=1):.4f}")

        chunked_accs = []
        for s in valid_seeds:
            ch = all_results['results'][str(s)].get('chunked', {})
            if 'accuracy' in ch:
                chunked_accs.append(ch['accuracy'])

        if chunked_accs:
            print(f"  Chunked (all {34189} train): {np.mean(chunked_accs):.4f}±{np.std(chunked_accs, ddof=1):.4f}")
            print(f"\n  DCS vs Chunked:     {np.mean(dcs_accs) - np.mean(chunked_accs):+.4f}")
            print(f"  Random vs Chunked:  {np.mean(rnd_accs) - np.mean(chunked_accs):+.4f}")

    print(f"\nAll results saved to {output_path}")


if __name__ == '__main__':
    main()
