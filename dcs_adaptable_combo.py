"""DCS + AdapTable combination experiment: DCS-seeded AdapTable.

Motivation: Table 16 shows AdapTable (uncertainty-calibrated context
selection) exceeds DCS by 0.20pp on Adult/feature-ordered, and the paper
states their combination is untested. Because TabPFN's parameters cannot be
updated at test time, all TTA baselines in this framework are context-
selection methods. AdapTable selects context by weighting training samples
with the prediction uncertainty (entropy) of test samples, where the
entropies are estimated with an initial RANDOM context of 2,000 samples.
The combination tested here replaces that random seed context with a
DCS-selected 2,000-sample context, so the uncertainty estimates come from
a distribution-matched context instead. Everything else in AdapTable
(KNN k=10, 500 test samples for entropy, top-10,000 by weight) is kept
identical.

Protocol: identical to tta_unified_fulltest.py (Table 16):
  - Adult, splits: iid + temporal (feature-ordered)
  - Seeds: [42, 123, 456, 789, 2024]
  - Context size: 10,000
  - FULL test set (n_test=7,327), local GPU TabPFN

References (Random, DCS, AdapTable per-seed values) are read from
results/tta_unified_5seed_results.json for paired statistics.

Output: results/dcs_adaptable_combo_results.json
"""
import os
import sys
import json
import time
import traceback
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import stats as sps

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
from splits import prepare_split
from context_shield_methods import json_safe, dcs_selection, set_seed
from sklearn.metrics import accuracy_score, f1_score

SEEDS = [42, 123, 456, 789, 2024]
DATASET = 'adult'
SPLITS = ['iid', 'temporal']
CONTEXT_SIZE = 10000
DEVICE = 'cuda'
OUTPUT_NAME = 'dcs_adaptable_combo_results.json'
REF_FILE = 'tta_unified_5seed_results.json'


def _tabpfn_predict_proba(X_train_ctx, y_train_ctx, X_test):
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
    p = np.clip(y_proba, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def dcs_seeded_adaptable(X_train, y_train, X_test, n_select,
                         k_neighbors=10, n_test_sample=500, seed=42,
                         init_ctx_size=2000):
    """AdapTable with DCS-selected initial context.

    Identical to adaptable_context_selection in tta_5seed_local.py except
    that the initial 2,000-sample context (used only for entropy
    estimation on 500 test samples) is selected by DCS instead of random
    sampling. Selection of the final n_select context rows follows the
    same KNN-uncertainty weighting.
    """
    n_train = X_train.shape[0]
    rng = np.random.RandomState(seed)

    if n_train <= n_select:
        return np.arange(n_train), {'method': 'dcs_adaptable_full'}

    # Step 1: DCS-selected initial context (replaces random initial context)
    t_dcs = time.time()
    init_size = min(init_ctx_size, n_train, n_select)
    initial_ctx = dcs_selection(X_train, X_test, init_size,
                                n_clusters=50, method='logistic', seed=seed)
    dcs_time = time.time() - t_dcs

    # Step 2: entropy of test samples under the DCS-seeded context
    n_test_use = min(n_test_sample, len(X_test))
    test_sample_idx = rng.choice(len(X_test), n_test_use, replace=False)
    X_test_sample = X_test[test_sample_idx]

    try:
        X_init_ctx = X_train[initial_ctx]
        y_init_ctx = y_train[initial_ctx]
        y_proba, _ = _tabpfn_predict_proba(X_init_ctx, y_init_ctx, X_test_sample)
        test_entropy = _prediction_entropy(y_proba)
        entropy_src = 'dcs_ctx'
    except Exception:
        test_entropy = np.ones(n_test_use)
        entropy_src = 'uniform_fallback'

    # Step 3: KNN neighbours of the test samples in the training pool
    k = min(k_neighbors, n_train)
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    nn.fit(X_train)
    _, neighbor_indices = nn.kneighbors(X_test_sample)

    # Step 4: weight training candidates by accumulated test uncertainty
    weights = np.zeros(n_train)
    for i in range(n_test_use):
        for j in range(k):
            weights[neighbor_indices[i, j]] += test_entropy[i]

    selected = np.argsort(weights)[-n_select:]

    if len(set(selected.tolist())) < n_select:
        zero_weight = np.where(weights == 0)[0]
        if len(zero_weight) > 0:
            needed = n_select - len(set(selected.tolist()))
            extra = rng.choice(zero_weight, min(needed, len(zero_weight)),
                               replace=False)
            selected = np.unique(np.concatenate([selected, extra]))[:n_select]

    info = {
        'method': 'dcs_adaptable',
        'initial_ctx_size': int(init_size),
        'initial_ctx_source': 'dcs_logistic_K50',
        'entropy_source': entropy_src,
        'k_neighbors': int(k),
        'n_test_sample': int(n_test_use),
        'dcs_init_time': float(dcs_time),
    }
    return selected, info


def evaluate(X_train, y_train, X_test, y_test, idx, seed):
    """Run TabPFN with selected context on the FULL test set."""
    m = {}
    t0 = time.time()
    clf = TabPFNClassifier(device=DEVICE)
    clf.fit(X_train[idx], y_train[idx])
    m['fit_time'] = float(time.time() - t0)
    t0 = time.time()
    y_pred = clf.predict(X_test)
    m['predict_time'] = float(time.time() - t0)
    m['accuracy'] = float(accuracy_score(y_test, y_pred))
    m['f1_macro'] = float(f1_score(y_test, y_pred, average='macro',
                                   zero_division=0))
    m['n_context'] = int(len(idx))
    return m


def paired_t(a, b):
    """Paired t-test of a minus b over 5 seeds."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1))
    if std_d == 0:
        t_stat, p_val = float('nan'), float('nan')
    else:
        t_stat, p_val = sps.ttest_rel(a, b)
    sd_diff = std_d
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    cohens_d = float(mean_d / sd_diff) if sd_diff > 0 else float('nan')
    se = sd_diff / np.sqrt(n)
    tcrit = sps.t.ppf(0.975, n - 1)
    ci = [float(mean_d - tcrit * se), float(mean_d + tcrit * se)]
    return {
        'mean_diff': mean_d, 'std_diff': std_d, 'delta_pp': mean_d * 100,
        't_statistic': float(t_stat), 'p_value': float(p_val),
        'df': n - 1, 'ci_95': ci, 'cohens_d': cohens_d,
        'pooled_sd': float(pooled),
    }


def load_reference(path):
    """Read per-seed Random/DCS/AdapTable values from the unified TTA file."""
    method_map = {
        'random': 'TabPFN-Random',
        'dcs': 'TabPFN-DCS',
        'adaptable': 'TabPFN-AdapTable',
    }
    with open(path) as f:
        ref = json.load(f)
    out = {}
    for blk in ref.get('results', []):
        split = blk['split']
        seed = blk['seed']
        methods = blk.get('results', {})
        for code, label in method_map.items():
            if code in methods:
                r = methods[code]
                out.setdefault((split, label), {})[seed] = {
                    'accuracy': r['accuracy'],
                    'f1_macro': r['f1_macro'],
                    'selection_time': r.get('selection_time', None),
                }
    return out


def main():
    output_path = os.path.join(RESULT_DIR, OUTPUT_NAME)
    ref_path = os.path.join(RESULT_DIR, REF_FILE)

    all_results = {
        'experiment': 'dcs_adaptable_combination_5seed_fulltest',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {
            'dataset': DATASET,
            'splits': SPLITS,
            'seeds': SEEDS,
            'context_size': CONTEXT_SIZE,
            'method': 'DCS-seeded AdapTable (initial context via DCS-Logistic K=50, '
                      '2,000 rows; rest of AdapTable identical: kNN k=10, '
                      '500 test samples for entropy, top-10,000 by weight)',
            'tabpfn_mode': 'local_gpu',
            'device': DEVICE,
            'n_test': 'full (no subsampling)',
            'protocol': 'unified: identical to Table 16 protocol (n_test=7,327)',
            'reference_file': REF_FILE,
        },
        'results': [],
        'errors': [],
    }

    # Resume support
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            prev = json.load(f)
        if prev.get('experiment') == all_results['experiment']:
            for blk in prev.get('results', []):
                if 'metrics' in blk:
                    all_results['results'].append(blk)
                    done.add((blk['split'], blk['seed']))
            print(f"Resuming: {len(done)} blocks already complete")

    def save():
        with open(output_path, 'w') as f:
            json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)

    print(f"Running DCS-seeded AdapTable: {DATASET}, full test set")
    t_start = time.time()

    for seed in SEEDS:
        for split_type in SPLITS:
            if (split_type, seed) in done:
                continue
            print(f"\n[{DATASET}/{split_type}/seed={seed}] "
                  f"elapsed={time.time()-t_start:.0f}s")
            set_seed(seed)
            try:
                split_data = prepare_split(DATASET, split_type, seed=seed)
            except Exception as e:
                all_results['errors'].append({'split': split_type,
                                              'seed': seed,
                                              'phase': 'prepare_split',
                                              'error': str(e)})
                continue

            X_train, y_train = split_data['X_train'], split_data['y_train']
            X_test, y_test = split_data['X_test'], split_data['y_test']
            print(f"  train={X_train.shape}, test={X_test.shape}")

            try:
                t0 = time.time()
                idx, info = dcs_seeded_adaptable(
                    X_train, y_train, X_test, CONTEXT_SIZE, seed=seed)
                sel_time = time.time() - t0
                print(f"  selection done in {sel_time:.1f}s "
                      f"(DCS init {info.get('dcs_init_time', 0):.1f}s)")
                metrics = evaluate(X_train, y_train, X_test, y_test, idx, seed)
                metrics['selection_time'] = float(sel_time)
                metrics['dcs_init_time'] = info.get('dcs_init_time')
                print(f"  acc={metrics['accuracy']:.4f} "
                      f"f1={metrics['f1_macro']:.4f}")
                all_results['results'].append({
                    'split': split_type, 'seed': seed,
                    'method': 'TabPFN-DCS+AdapTable',
                    'metrics': metrics, 'info': info,
                })
                save()
            except Exception as e:
                traceback.print_exc()
                all_results['errors'].append({'split': split_type,
                                              'seed': seed,
                                              'phase': 'run',
                                              'error': str(e)})

    # ---- Paired statistics vs references ----
    stats_block = {}
    if os.path.exists(ref_path):
        ref = load_reference(ref_path)
        for split_type in SPLITS:
            combo_acc, combo_f1 = [], []
            seeds_used = []
            for blk in all_results['results']:
                if blk['split'] == split_type:
                    combo_acc.append(blk['metrics']['accuracy'])
                    combo_f1.append(blk['metrics']['f1_macro'])
                    seeds_used.append(blk['seed'])
            if len(combo_acc) < len(SEEDS):
                print(f"[{split_type}] incomplete: {len(combo_acc)}/{len(SEEDS)} "
                      f"seeds, statistics skipped")
                continue
            entry = {'n_seeds': len(combo_acc), 'seeds': seeds_used}
            for label in ('TabPFN-Random', 'TabPFN-DCS', 'TabPFN-AdapTable'):
                key = (split_type, label)
                if key not in ref:
                    continue
                ref_acc = [ref[key][s]['accuracy'] for s in seeds_used]
                ref_f1 = [ref[key][s]['f1_macro'] for s in seeds_used]
                entry[label] = {
                    'combo_acc_mean': float(np.mean(combo_acc)),
                    'ref_acc_mean': float(np.mean(ref_acc)),
                    'accuracy': paired_t(combo_acc, ref_acc),
                    'f1_macro': paired_t(combo_f1, ref_f1),
                }
            entry['combo_acc'] = [float(a) for a in combo_acc]
            entry['combo_f1'] = [float(a) for a in combo_f1]
            stats_block[split_type] = entry
        all_results['paired_statistics'] = stats_block
        save()
        print("\n===== Paired statistics summary =====")
        for split_type, entry in stats_block.items():
            print(f"\n[{split_type}] combo acc = "
                  f"{np.mean(entry['combo_acc']):.4f}")
            for label in ('TabPFN-Random', 'TabPFN-DCS', 'TabPFN-AdapTable'):
                if label in entry:
                    st = entry[label]['accuracy']
                    print(f"  vs {label}: ref={entry[label]['ref_acc_mean']:.4f} "
                          f"delta={st['delta_pp']:+.4f}pp t={st['t_statistic']:.2f} "
                          f"p={st['p_value']:.4f}")
    else:
        print(f"Reference file not found: {ref_path}")

    save()
    print(f"\nDone. Results saved to {output_path}")
    print(f"Total elapsed: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
