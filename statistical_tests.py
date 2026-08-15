"""Statistical significance tests for DCS experiments.

Performs paired t-tests, Wilcoxon signed-rank tests, Cohen's d effect sizes,
and 95% confidence intervals for all method comparisons.

Data sources:
  - results/context_shield_results.json  (5-seed, Adult IID+Temporal)
  - results/orthogonality_exp_results.json (3-seed, Adult Temporal)

Results saved to: results/statistical_test_results.json
"""
import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def cohens_d(x, y):
    """Cohen's d effect size for paired samples."""
    diff = np.array(x) - np.array(y)
    if len(diff) < 2:
        return 0.0
    d = diff.mean() / diff.std(ddof=1)
    return float(d)


def confidence_interval(x, confidence=0.95):
    """95% confidence interval for the mean of x."""
    n = len(x)
    if n < 2:
        return (float(x[0]), float(x[0]))
    mean = np.mean(x)
    sem = stats.sem(x)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (float(mean - h), float(mean + h))


def paired_t_test(x, y):
    """Paired t-test: x vs y (same seeds)."""
    x, y = np.array(x), np.array(y)
    diff = x - y
    n = len(diff)
    if n < 2:
        return {'error': 'Need at least 2 paired samples'}
    t_stat, p_value = stats.ttest_rel(x, y)
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    ci_low, ci_high = confidence_interval(diff.tolist())
    return {
        'test': 'paired t-test',
        'n': int(n),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'df': int(n - 1),
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'cohens_d': cohens_d(x.tolist(), y.tolist()),
        'significant_at_0.05': bool(p_value < 0.05),
    }


def wilcoxon_test(x, y):
    """Wilcoxon signed-rank test (non-parametric)."""
    try:
        stat, p_value = stats.wilcoxon(x, y)
        return {
            'test': 'Wilcoxon signed-rank',
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant_at_0.05': bool(p_value < 0.05),
        }
    except Exception as e:
        return {'test': 'Wilcoxon signed-rank', 'error': str(e)}


def extract_per_seed(results, dataset, split, method, metric='accuracy'):
    """Extract per-seed metric values for a specific method."""
    values = []
    for r in results:
        if (r['dataset'] == dataset and r['split'] == split
                and r['method'] == method and r.get('metrics')):
            values.append(r['metrics'][metric])
    return values


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple comparisons."""
    m = len(p_values)
    corrected_alpha = alpha / m
    return {
        'n_comparisons': m,
        'corrected_alpha': float(corrected_alpha),
        'significant': [bool(p < corrected_alpha) for p in p_values],
    }


def analyze_context_shield_results():
    """Analyze context_shield_results.json (5-seed, Adult)."""
    filepath = os.path.join(RESULT_DIR, 'context_shield_results.json')
    with open(filepath, 'r') as f:
        data = json.load(f)

    all_results = data['results']
    methods_to_compare = [
        'TabPFN-KNN',
        'TabPFN-DRWS-Logistic',
        'TabPFN-DRWS-LightGBM',
        'TabPFN-DCS-Logistic',
        'TabPFN-DCS-LightGBM',
        'TabPFN-Mixed-LightGBM',
        'TabPFN-ContextShield-Logistic',
        'TabPFN-ContextShield-LightGBM',
    ]

    analysis = {}
    all_p_values = []

    for split in ['iid', 'temporal']:
        for metric in ['accuracy', 'f1_macro']:
            baseline = extract_per_seed(all_results, 'adult', split, 'TabPFN-Random', metric)
            if not baseline:
                continue

            split_key = f'adult_{split}_{metric}'
            analysis[split_key] = {
                'baseline': 'TabPFN-Random',
                'baseline_values': baseline,
                'baseline_mean': float(np.mean(baseline)),
                'n_seeds': len(baseline),
                'comparisons': {},
            }

            for method in methods_to_compare:
                method_vals = extract_per_seed(all_results, 'adult', split, method, metric)
                if len(method_vals) != len(baseline):
                    continue

                t_test = paired_t_test(method_vals, baseline)
                wilcoxon = wilcoxon_test(method_vals, baseline)

                analysis[split_key]['comparisons'][method] = {
                    'values': method_vals,
                    'mean': float(np.mean(method_vals)),
                    'delta_mean': float(np.mean(method_vals) - np.mean(baseline)),
                    'delta_pp': float((np.mean(method_vals) - np.mean(baseline)) * 100),
                    'paired_t_test': t_test,
                    'wilcoxon': wilcoxon,
                }

                if 'p_value' in t_test:
                    all_p_values.append(t_test['p_value'])

    # Bonferroni correction across all comparisons
    bonferroni = bonferroni_correction(all_p_values)
    analysis['_bonferroni_correction'] = bonferroni

    return analysis


def analyze_orthogonality_results():
    """Analyze orthogonality_exp_results.json (3-seed, Adult Temporal)."""
    filepath = os.path.join(RESULT_DIR, 'orthogonality_exp_results.json')
    with open(filepath, 'r') as f:
        data = json.load(f)

    all_results = data['results']

    # Extract per-seed accuracy for each method
    methods = ['TabPFN-base-Random', 'TabPFN-base-DCS-Logistic',
               'TabPFN-dist-Random', 'TabPFN-dist-DCS-Logistic']
    method_vals = {}
    for m in methods:
        vals = [r['metrics']['accuracy'] for r in all_results
                if r['method'] == m and r.get('metrics')]
        method_vals[m] = vals

    analysis = {
        'experiment': 'orthogonality',
        'n_seeds': len(method_vals['TabPFN-base-Random']),
        'method_values': method_vals,
        'comparisons': {},
    }

    # Comparison 1: DCS effect on base model
    analysis['comparisons']['DCS_vs_Random_on_base'] = paired_t_test(
        method_vals['TabPFN-base-DCS-Logistic'],
        method_vals['TabPFN-base-Random']
    )

    # Comparison 2: DCS effect on dist model
    analysis['comparisons']['DCS_vs_Random_on_dist'] = paired_t_test(
        method_vals['TabPFN-dist-DCS-Logistic'],
        method_vals['TabPFN-dist-Random']
    )

    # Comparison 3: Drift-Resilient effect (dist vs base) with Random context
    analysis['comparisons']['dist_vs_base_with_Random'] = paired_t_test(
        method_vals['TabPFN-dist-Random'],
        method_vals['TabPFN-base-Random']
    )

    # Comparison 4: Drift-Resilient effect (dist vs base) with DCS context
    analysis['comparisons']['dist_vs_base_with_DCS'] = paired_t_test(
        method_vals['TabPFN-dist-DCS-Logistic'],
        method_vals['TabPFN-base-DCS-Logistic']
    )

    # Summary table
    analysis['summary'] = {
        'base_Random_mean': float(np.mean(method_vals['TabPFN-base-Random'])),
        'base_DCS_mean': float(np.mean(method_vals['TabPFN-base-DCS-Logistic'])),
        'dist_Random_mean': float(np.mean(method_vals['TabPFN-dist-Random'])),
        'dist_DCS_mean': float(np.mean(method_vals['TabPFN-dist-DCS-Logistic'])),
        'DCS_effect_on_base_pp': float(
            (np.mean(method_vals['TabPFN-base-DCS-Logistic']) -
             np.mean(method_vals['TabPFN-base-Random'])) * 100),
        'DCS_effect_on_dist_pp': float(
            (np.mean(method_vals['TabPFN-dist-DCS-Logistic']) -
             np.mean(method_vals['TabPFN-dist-Random'])) * 100),
        'dist_effect_on_Random_pp': float(
            (np.mean(method_vals['TabPFN-dist-Random']) -
             np.mean(method_vals['TabPFN-base-Random'])) * 100),
        'dist_effect_on_DCS_pp': float(
            (np.mean(method_vals['TabPFN-dist-DCS-Logistic']) -
             np.mean(method_vals['TabPFN-base-DCS-Logistic'])) * 100),
    }

    return analysis


def main():
    print("=" * 80)
    print("Statistical Significance Tests for DCS Experiments")
    print("=" * 80)

    output = {
        'experiment': 'statistical_tests',
        'timestamp': __import__('time').strftime('%Y-%m-%dT%H:%M:%S'),
    }

    # 1. Context Shield results (5-seed)
    print("\n[1/2] Analyzing context_shield_results.json (5-seed, Adult)...")
    cs_analysis = analyze_context_shield_results()
    output['context_shield_analysis'] = cs_analysis

    # Print summary
    for split in ['iid', 'temporal']:
        for metric in ['accuracy', 'f1_macro']:
            key = f'adult_{split}_{metric}'
            if key not in cs_analysis:
                continue
            print(f"\n  --- {key} (baseline: TabPFN-Random, n={cs_analysis[key]['n_seeds']}) ---")
            print(f"  {'Method':<37} {'Mean':<10} {'Δ(pp)':<10} {'t-stat':<10} {'p-value':<12} {'Cohen d':<10} {'Sig?'}")
            for method, comp in cs_analysis[key]['comparisons'].items():
                t = comp['paired_t_test']
                sig = '***' if t.get('p_value', 1) < 0.001 else ('**' if t.get('p_value', 1) < 0.01 else ('*' if t.get('p_value', 1) < 0.05 else 'ns'))
                print(f"  {method:<37} {comp['mean']:<10.4f} {comp['delta_pp']:<+10.2f} "
                      f"{t.get('t_statistic', 0):<10.4f} {t.get('p_value', 1):<12.6f} "
                      f"{t.get('cohens_d', 0):<10.4f} {sig}")

    bonf = cs_analysis['_bonferroni_correction']
    print(f"\n  Bonferroni correction: {bonf['n_comparisons']} comparisons, "
          f"corrected alpha={bonf['corrected_alpha']:.5f}")

    # 2. Orthogonality results (3-seed)
    print("\n[2/2] Analyzing orthogonality_exp_results.json (3-seed, Adult Temporal)...")
    orth_analysis = analyze_orthogonality_results()
    output['orthogonality_analysis'] = orth_analysis

    print(f"\n  --- Orthogonality (n={orth_analysis['n_seeds']}) ---")
    print(f"  {'Comparison':<35} {'Δ(pp)':<10} {'t-stat':<10} {'p-value':<12} {'Cohen d':<10} {'Sig?'}")
    for comp_name, comp in orth_analysis['comparisons'].items():
        if 'error' in comp:
            print(f"  {comp_name:<35} ERROR: {comp['error']}")
            continue
        sig = '***' if comp.get('p_value', 1) < 0.001 else ('**' if comp.get('p_value', 1) < 0.01 else ('*' if comp.get('p_value', 1) < 0.05 else 'ns'))
        delta_pp = comp['mean_diff'] * 100
        print(f"  {comp_name:<35} {delta_pp:<+10.2f} {comp.get('t_statistic', 0):<10.4f} "
              f"{comp.get('p_value', 1):<12.6f} {comp.get('cohens_d', 0):<10.4f} {sig}")

    print(f"\n  Summary:")
    s = orth_analysis['summary']
    print(f"    base+Random:  {s['base_Random_mean']:.4f}")
    print(f"    base+DCS:     {s['base_DCS_mean']:.4f}  (Δ={s['DCS_effect_on_base_pp']:+.2f}pp)")
    print(f"    dist+Random:  {s['dist_Random_mean']:.4f}  (Δ={s['dist_effect_on_Random_pp']:+.2f}pp)")
    print(f"    dist+DCS:     {s['dist_DCS_mean']:.4f}  (Δ={s['dist_effect_on_DCS_pp']:+.2f}pp vs base+DCS)")

    # Save
    output_path = os.path.join(RESULT_DIR, 'statistical_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(json_safe(output), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
