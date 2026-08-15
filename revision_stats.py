"""Recompute statistics requested by reviewer feedback (M2, M4, M5, m4, m6).

Reads ONLY existing result JSONs; writes results/revision_stats.json.
No new model inference is performed here (new experiments are separate scripts).
"""
import json, os
import numpy as np
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.normpath(os.path.join(BASE, '..', 'results'))


def load(name):
    with open(os.path.join(RD, name), 'r', encoding='utf-8') as f:
        return json.load(f)


def paired_t(a, b):
    """Paired t-test of a-b (a, b: per-seed values)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    mean = float(d.mean())
    if d.std(ddof=1) == 0:
        return {'mean_diff': mean, 'std_diff': 0.0, 't_stat': None, 'p_value': 1.0,
                'ci95': [mean, mean], 'n': n}
    t, p = stats.ttest_rel(a, b)
    se = d.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    return {'mean_diff': mean, 'std_diff': float(d.std(ddof=1)), 't_stat': float(t),
            'p_value': float(p), 'ci95': [float(mean - tcrit * se), float(mean + tcrit * se)],
            'n': n}


def holm(pvals):
    """Holm-Bonferroni: returns adjusted p-values and significance at 0.05."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)  # enforce monotonicity
        adj[idx] = min(1.0, running)
    return adj


def partial_spearman(x, y, z):
    """Partial Spearman correlation of x,y controlling z (rank-residual method)."""
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    X = np.column_stack([rz, np.ones(len(rz))])
    res_x = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    res_y = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    r, p = stats.pearsonr(res_x, res_y)
    return float(r), float(p)


def main():
    out = {}

    # ============ M2: KL-accuracy, budget-controlled ============
    comp = load('comprehensive_local_tabpfn.json')
    kl_data = comp['kl_accuracy_tabpfn']['data']
    budgets = np.array([d['budget'] for d in kl_data], dtype=float)
    kls = np.array([d['kl'] for d in kl_data])
    accs = np.array([d['accuracy'] for d in kl_data])

    r_s, p_s = stats.spearmanr(kls, accs)
    r_p, p_p = stats.pearsonr(kls, accs)
    r_kl_b, p_kl_b = stats.spearmanr(kls, np.log10(budgets))
    r_acc_b, p_acc_b = stats.spearmanr(accs, np.log10(budgets))
    r_part, p_part = partial_spearman(kls, accs, np.log10(budgets))

    # Paired within-budget differences (6 budgets): dKL vs dAcc
    b_list = [int(b) for b in sorted(set(budgets.astype(int)))]
    d_kl, d_acc = [], []
    for b in b_list:
        dcs = next(d for d in kl_data if d['budget'] == b and d['method'] == 'DCS')
        rnd = next(d for d in kl_data if d['budget'] == b and d['method'] == 'Random')
        d_kl.append(dcs['kl'] - rnd['kl'])
        d_acc.append(dcs['accuracy'] - rnd['accuracy'])
    r_pair, p_pair = stats.spearmanr(d_kl, d_acc)
    signs_agree = int(np.sum(np.sign(d_kl) == np.sign(d_acc)))

    out['M2_kl_accuracy'] = {
        'n_cells': len(kl_data),
        'spearman_r': float(r_s), 'spearman_p': float(p_s),
        'pearson_r': float(r_p), 'pearson_p': float(p_p),
        'spearman_kl_vs_logbudget': {'r': float(r_kl_b), 'p': float(p_kl_b)},
        'spearman_acc_vs_logbudget': {'r': float(r_acc_b), 'p': float(p_acc_b)},
        'partial_spearman_ctrl_logbudget': {'r': r_part, 'p': p_part, 'df': len(kl_data) - 3},
        'paired_within_budget': {
            'budgets': b_list,
            'delta_kl': [float(v) for v in d_kl],
            'delta_acc': [float(v) for v in d_acc],
            'spearman_r': float(r_pair), 'spearman_p': float(p_pair),
            'sign_agreement': f'{signs_agree}/6',
        },
    }

    # ============ M5: matched-budget paired tests ============
    SEEDS = ['42', '123', '456', '789', '2024']
    BUDGETS = [200, 500, 1000, 2000, 5000, 10000]

    # XGBoost-DCS vs XGBoost-RandPool (matched_fewshot_xgboost.json)
    mf = load('matched_fewshot_xgboost.json')['adult']
    fam1 = {}
    for b in BUDGETS:
        a = [mf[s][f'dcs_{b}']['accuracy'] for s in SEEDS]
        c = [mf[s][f'randompool_{b}']['accuracy'] for s in SEEDS]
        fam1[b] = paired_t(a, c)
    pv = [fam1[b]['p_value'] for b in BUDGETS]
    adj = holm(pv)
    for b, a_ in zip(BUDGETS, adj):
        fam1[b]['p_holm'] = float(a_)
        fam1[b]['sig_holm_05'] = bool(a_ < 0.05)

    # TabPFN-DCS vs TabPFN-Random (comprehensive_local_tabpfn.json fewshot)
    fa = comp['fewshot_5seed']['adult']
    fam2 = {}
    for b in BUDGETS:
        a = [fa[s]['budgets'][f'dcs_{b}']['accuracy'] for s in SEEDS]
        c = [fa[s]['budgets'][f'random_{b}']['accuracy'] for s in SEEDS]
        fam2[b] = paired_t(a, c)
    pv = [fam2[b]['p_value'] for b in BUDGETS]
    adj = holm(pv)
    for b, a_ in zip(BUDGETS, adj):
        fam2[b]['p_holm'] = float(a_)
        fam2[b]['sig_holm_05'] = bool(a_ < 0.05)

    # TabPFN-DCS vs XGBoost-DCS (cross-file, paired by seed; same split logic, full test)
    fam3 = {}
    for b in BUDGETS:
        a = [fa[s]['budgets'][f'dcs_{b}']['accuracy'] for s in SEEDS]
        c = [mf[s][f'dcs_{b}']['accuracy'] for s in SEEDS]
        fam3[b] = paired_t(a, c)
    pv = [fam3[b]['p_value'] for b in BUDGETS]
    adj = holm(pv)
    for b, a_ in zip(BUDGETS, adj):
        fam3[b]['p_holm'] = float(a_)
        fam3[b]['sig_holm_05'] = bool(a_ < 0.05)

    out['M5_matched_budget_paired_tests'] = {
        'xgb_dcs_vs_randpool': fam1,
        'tabpfn_dcs_vs_random': fam2,
        'tabpfn_dcs_vs_xgb_dcs': fam3,
        'note': 'paired by seed (same feature-ordered split per seed); Holm across 6 budgets within each family',
    }

    # ============ m4: TabReD Holm correction ============
    tp = load('tabred_paired_tests.json')['results']
    names = list(tp.keys())
    pvals = [tp[n]['p_ttest'] if tp[n]['p_ttest'] == tp[n]['p_ttest'] else 1.0 for n in names]
    adj = holm(pvals)
    m4 = {}
    for n, p, a_ in zip(names, pvals, adj):
        m4[n] = {'p_ttest': tp[n]['p_ttest'], 'p_holm': float(a_),
                 'sig_holm_05': bool(a_ < 0.05),
                 'bonferroni_alpha_05_over_8_sig': bool(p < 0.05 / 8)}
    out['m4_tabred_holm'] = m4

    # ============ m6: TTA timing means (5 seeds) ============
    tta = load('tta_combined_5seed_results.json')
    timing = {}
    for split in ['iid', 'temporal']:
        for meth in ['random', 'knn', 'dcs', 'tent', 'adaptable', 'self_training']:
            sels, ntests = [], []
            for entry in tta['results']:
                if entry['split'] != split:
                    continue
                r = entry['results'].get(meth, {})
                if 'selection_time' in r:
                    sels.append(r['selection_time'])
                    ntests.append(entry['split_info']['n_test'])
                elif meth == 'dcs' and 'elapsed' in r:
                    # legacy 2-seed entries lack selection_time: estimate from elapsed-fit-predict
                    sels.append(r['elapsed'] - r['fit_time'] - r['predict_time'])
                    ntests.append(entry['split_info']['n_test'])
            if sels:
                timing[f'{split}_{meth}'] = {
                    'mean_overhead_s': float(np.mean(sels)),
                    'per_seed': [round(v, 2) for v in sels],
                    'n_seeds': len(sels),
                    'n_test_values': ntests,
                    'note': 'dcs legacy seeds (42,123) overhead estimated as elapsed-fit-predict'
                            if meth == 'dcs' else 'selection_time/adaptation overhead as recorded',
                }
    out['m6_tta_timing_5seed'] = timing

    # ============ M4: chunked table totals recomputed (fit+predict only) ============
    ch = load('chunked_tabpfn_results.json')['results']
    tot = {}
    for meth in ['dcs', 'random', 'chunked']:
        f = [ch[s][meth]['fit_time'] for s in SEEDS]
        p_ = [ch[s][meth]['predict_time'] for s in SEEDS]
        t = np.array(f) + np.array(p_)
        tot[meth] = {'fit_mean': float(np.mean(f)), 'predict_mean': float(np.mean(p_)),
                     'total_mean_fit_predict': float(np.mean(t)),
                     'total_std_fit_predict': float(np.std(t, ddof=1)),
                     'mislabeled_selection_time_mean': float(np.mean(
                         [ch[s]['dcs']['selection_time'] for s in SEEDS])) if meth == 'dcs' else None,
                     'accuracy_mean': float(np.mean([ch[s][meth]['accuracy'] for s in SEEDS])),
                     'accuracy_std': float(np.std([ch[s][meth]['accuracy'] for s in SEEDS], ddof=1))}
    out['M4_chunked_totals_fit_predict_only'] = tot

    with open(os.path.join(RD, 'revision_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == '__main__':
    main()
