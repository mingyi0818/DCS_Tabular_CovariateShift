"""Corrected DCS selection timing for the chunked-TabPFN comparison (reviewer M4).

The original run_chunked_tabpfn.py wrapped TabPFN fit+predict inside the
selection timer, so its 'selection_time' field (107-272s) actually measured
selection + TabPFN inference. This script re-measures ONLY the DCS selection
stage (domain classifier + k-means + allocation) under the identical split
protocol, for the same 5 seeds.
"""
import os, sys, json, time
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
from splits import SPLIT_CONFIG, load_raw_dataframe, encode_features
from context_shield_methods import dcs_selection, set_seed, json_safe

SEEDS = [42, 123, 456, 789, 2024]
K_CLUSTERS = 50
N_CONTEXT = 10000
N_TEST_MAX = 2000


def prepare_split(ds, seed):
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
    return X_train, X_test


def main():
    out = {
        'experiment': 'chunked_dcs_selection_timing_correction',
        'description': 'True DCS selection-stage timing (domain fit + scoring + k-means + allocation), '
                       'same split protocol as chunked_tabpfn_results.json (n_test_max=2000); '
                       'the original file mislabeled selection+inference as selection.',
        'config': {'seeds': SEEDS, 'n_clusters': K_CLUSTERS, 'n_context': N_CONTEXT,
                   'n_test_max': N_TEST_MAX, 'dataset': 'adult', 'split': 'feature-ordered'},
        'results': {},
    }
    for seed in SEEDS:
        set_seed(seed)
        X_train, X_test = prepare_split('adult', seed)
        t0 = time.time()
        idx = dcs_selection(X_train, X_test, N_CONTEXT, n_clusters=K_CLUSTERS,
                            method='logistic', seed=seed)
        sel_t = time.time() - t0
        out['results'][str(seed)] = {
            'selection_time_s': float(sel_t), 'n_selected': int(len(idx)),
            'n_train': int(X_train.shape[0]), 'n_test_sel': int(X_test.shape[0]),
        }
        print(f'seed={seed}: selection={sel_t:.2f}s (n={len(idx)})')
    times = [v['selection_time_s'] for v in out['results'].values()]
    out['summary'] = {'mean_selection_time_s': float(np.mean(times)),
                      'std_selection_time_s': float(np.std(times, ddof=1))}
    print('mean = %.2f +/- %.2f s' % (out['summary']['mean_selection_time_s'],
                                      out['summary']['std_selection_time_s']))
    with open(os.path.join(RESULT_DIR, 'chunked_timing_correction.json'), 'w') as f:
        json.dump(json_safe(out), f, indent=2)
    print('saved results/chunked_timing_correction.json')


if __name__ == '__main__':
    main()
