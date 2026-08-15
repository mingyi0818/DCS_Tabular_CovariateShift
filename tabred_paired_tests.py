"""M3: Paired significance tests for TabReD benchmark from existing seed-level data.

For each dataset: 5 paired (DCS, Random) values -> paired t-test + Wilcoxon signed-rank.
Also compute 95% CI of the paired difference.
"""
import json
import numpy as np
from scipy import stats

RESULTS = r'd:\ResearchPaperPrepare\67_DCS_Tabular_CovariateShift\results\tabred_benchmark_results.json'
SEEDS = ['42', '123', '456', '789', '2024']

with open(RESULTS, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"{'Dataset':<22} {'Metric':<5} {'DCS-Random':>12} {'t':>7} {'p(t)':>8} {'W':>4} {'p(Wil)':>8} {'95% CI':>20} {'Sig?'}")
print('-' * 100)

output = {}
for ds, d in data['results'].items():
    task = d['task_type']
    metric = 'accuracy' if task == 'classification' else 'rmse'
    # For regression, "better" = lower RMSE -> diff = Random - DCS (positive = DCS better)
    # For classification, diff = DCS - Random (positive = DCS better)
    dcs_vals, rnd_vals = [], []
    for s in SEEDS:
        dcs_vals.append(d['seeds'][s]['dcs'][metric])
        rnd_vals.append(d['seeds'][s]['random'][metric])
    dcs, rnd = np.array(dcs_vals), np.array(rnd_vals)
    if task == 'classification':
        diff = dcs - rnd
    else:
        diff = rnd - dcs  # positive = DCS better (lower RMSE)
    m = diff.mean()
    sd = diff.std(ddof=1)
    t_stat, p_t = stats.ttest_1samp(diff, 0)
    try:
        w_stat, p_w = stats.wilcoxon(diff)
    except ValueError:
        w_stat, p_w = float('nan'), float('nan')
    # 95% CI (t-based, df=4)
    ci_half = stats.t.ppf(0.975, 4) * sd / np.sqrt(5)
    lo, hi = m - ci_half, m + ci_half
    sig = 'YES' if (p_t < 0.05 and lo > 0) else ('p<.05,CI incl 0' if p_t < 0.05 else 'no')
    unit = 'pp' if task == 'classification' else 'RMSE'
    print(f"{ds:<22} {metric[:4]:<5} {m:>+10.4f} {t_stat:>7.2f} {p_t:>8.4f} {w_stat:>4.0f} {p_w:>8.4f} [{lo:>+9.4f},{hi:>+9.4f}] {sig}")
    output[ds] = {
        'metric': metric, 'unit': unit, 'mean_diff': float(m), 'std_diff': float(sd),
        't_stat': float(t_stat), 'p_ttest': float(p_t),
        'wilcoxon_W': float(w_stat) if w_stat == w_stat else None, 'p_wilcoxon': float(p_w) if p_w == p_w else None,
        'ci95': [float(lo), float(hi)],
        'significant_p05_and_ci_excludes_0': bool(p_t < 0.05 and lo > 0),
    }

with open(r'd:\ResearchPaperPrepare\67_DCS_Tabular_CovariateShift\results\tabred_paired_tests.json', 'w') as f:
    json.dump({'experiment': 'tabred_paired_tests', 'n_seeds': 5,
               'note': 'diff direction: positive = DCS better; CI is t-based df=4',
               'results': output}, f, indent=2)
print('\nSaved to results/tabred_paired_tests.json')
