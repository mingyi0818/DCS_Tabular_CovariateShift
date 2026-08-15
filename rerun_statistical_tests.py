"""Re-run statistical significance tests from the fixed DCS results.

Reads the NEW (largest-remainder-method) result files and computes:
  1. DCS-Logistic vs TabPFN-Random paired t-test (5 seeds, Adult/temporal)
  2. DCS-Logistic vs DRWS-Logistic paired t-test
  3. All methods vs Random paired t-test
  4. Bonferroni correction
  5. Interaction test (difference-in-differences from orthogonality_exp_results.json)

Data sources:
  - results/context_shield_results.json   (5-seed, Adult IID+Temporal)
  - results/orthogonality_exp_results.json (3-seed, Adult Temporal)

Results saved to: results/statistical_test_results.json (overwrites old file)
"""
import os
import sys
import json
import time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------

def json_safe(obj):
    """Recursively convert numpy types to Python native types."""
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


def cohens_d_paired(x, y):
    """Cohen's d effect size for paired samples."""
    diff = np.array(x) - np.array(y)
    if len(diff) < 2:
        return 0.0
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def confidence_interval(x, confidence=0.95):
    """95% confidence interval for the mean of x."""
    n = len(x)
    if n < 2:
        return (float(x[0]) if n == 1 else 0.0, float(x[0]) if n == 1 else 0.0)
    mean = np.mean(x)
    sem = stats.sem(x)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (float(mean - h), float(mean + h))


def paired_t_test(x, y):
    """Paired t-test: x vs y (same seeds)."""
    x_arr, y_arr = np.array(x), np.array(y)
    diff = x_arr - y_arr
    n = len(diff)
    if n < 2:
        return {'error': 'Need at least 2 paired samples', 'n': int(n)}
    t_stat, p_value = stats.ttest_rel(x_arr, y_arr)
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    ci_low, ci_high = confidence_interval(diff.tolist())
    return {
        'test': 'paired t-test',
        'n': int(n),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'delta_pp': float(mean_diff * 100),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'df': int(n - 1),
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'cohens_d': cohens_d_paired(x, y),
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


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction for multiple comparisons."""
    m = len(p_values)
    corrected_alpha = alpha / m if m > 0 else alpha
    return {
        'n_comparisons': m,
        'corrected_alpha': float(corrected_alpha),
        'significant': [bool(p < corrected_alpha) for p in p_values],
        'original_p_values': [float(p) for p in p_values],
    }


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    m = len(p_values)
    if m == 0:
        return {'n_comparisons': 0, 'significant': []}
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    significant_sorted = sorted_p <= thresholds
    # Make it cumulative from the largest
    if significant_sorted.any():
        max_sig = np.max(np.where(significant_sorted))
        significant_sorted[:max_sig + 1] = True
    significant = np.zeros(m, dtype=bool)
    significant[sorted_indices] = significant_sorted
    return {
        'n_comparisons': m,
        'alpha': float(alpha),
        'significant': significant.tolist(),
    }


def extract_per_seed(results, dataset, split, method, metric='accuracy'):
    """Extract per-seed metric values for a specific method."""
    values = []
    for r in results:
        if (r.get('dataset') == dataset and r.get('split') == split
                and r.get('method') == method and r.get('metrics')):
            values.append(r['metrics'][metric])
    return values


# ----------------------------------------------------------------------------
# 1. Context Shield analysis (5-seed, Adult)
# ----------------------------------------------------------------------------

def analyze_context_shield_results():
    """Analyze context_shield_results.json (5-seed, Adult).

    Computes:
      - All methods vs TabPFN-Random paired t-test
      - DCS-Logistic vs DRWS-Logistic paired t-test
      - Bonferroni correction
    """
    filepath = os.path.join(RESULT_DIR, 'context_shield_results.json')
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found")
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    all_results = data['results']

    # All methods that should be compared against Random
    methods_to_compare = [
        'XGBoost',
        'TabPFN-KNN',
        'TabPFN-DRWS-Logistic',
        'TabPFN-DRWS-LightGBM',
        'TabPFN-DCS-Logistic',
        'TabPFN-DCS-LightGBM',
        'TabPFN-Mixed-LightGBM',
        'TabPFN-ContextShield-Logistic',
        'TabPFN-ContextShield-LightGBM',
    ]

    analysis = {
        'data_source': 'context_shield_results.json',
        'n_seeds_expected': 5,
        'splits': {},
    }

    all_p_values = []

    for split in ['iid', 'temporal']:
        for metric in ['accuracy', 'f1_macro']:
            baseline = extract_per_seed(all_results, 'adult', split,
                                        'TabPFN-Random', metric)
            if not baseline:
                continue

            split_key = f'adult_{split}_{metric}'
            split_analysis = {
                'baseline_method': 'TabPFN-Random',
                'baseline_values': baseline,
                'baseline_mean': float(np.mean(baseline)),
                'baseline_std': float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 0.0,
                'n_seeds': len(baseline),
                'comparisons': {},
            }

            for method in methods_to_compare:
                method_vals = extract_per_seed(all_results, 'adult', split,
                                               method, metric)
                if len(method_vals) != len(baseline):
                    continue

                t_test = paired_t_test(method_vals, baseline)
                wilcoxon = wilcoxon_test(method_vals, baseline)

                split_analysis['comparisons'][method] = {
                    'values': method_vals,
                    'mean': float(np.mean(method_vals)),
                    'std': float(np.std(method_vals, ddof=1)) if len(method_vals) > 1 else 0.0,
                    'delta_mean': float(np.mean(method_vals) - np.mean(baseline)),
                    'delta_pp': float((np.mean(method_vals) - np.mean(baseline)) * 100),
                    'paired_t_test': t_test,
                    'wilcoxon': wilcoxon,
                }

                if 'p_value' in t_test:
                    all_p_values.append(t_test['p_value'])

            analysis['splits'][split_key] = split_analysis

    # --- DCS-Logistic vs DRWS-Logistic head-to-head ---
    dcs_vs_drws = {}
    for split in ['iid', 'temporal']:
        for metric in ['accuracy', 'f1_macro']:
            dcs_vals = extract_per_seed(all_results, 'adult', split,
                                        'TabPFN-DCS-Logistic', metric)
            drws_vals = extract_per_seed(all_results, 'adult', split,
                                         'TabPFN-DRWS-Logistic', metric)
            if dcs_vals and drws_vals and len(dcs_vals) == len(drws_vals):
                key = f'adult_{split}_{metric}'
                t_test = paired_t_test(dcs_vals, drws_vals)
                wilcoxon = wilcoxon_test(dcs_vals, drws_vals)
                dcs_vs_drws[key] = {
                    'dcs_logistic_values': dcs_vals,
                    'dcs_logistic_mean': float(np.mean(dcs_vals)),
                    'dcs_logistic_std': float(np.std(dcs_vals, ddof=1)) if len(dcs_vals) > 1 else 0.0,
                    'drws_logistic_values': drws_vals,
                    'drws_logistic_mean': float(np.mean(drws_vals)),
                    'drws_logistic_std': float(np.std(drws_vals, ddof=1)) if len(drws_vals) > 1 else 0.0,
                    'paired_t_test': t_test,
                    'wilcoxon': wilcoxon,
                }
                if 'p_value' in t_test:
                    all_p_values.append(t_test['p_value'])

    analysis['dcs_vs_drws_logistic'] = dcs_vs_drws

    # --- Bonferroni correction across all comparisons ---
    analysis['bonferroni_correction'] = bonferroni_correction(all_p_values)
    analysis['benjamini_hochberg'] = benjamini_hochberg(all_p_values)
    analysis['total_comparisons'] = len(all_p_values)

    return analysis


# ----------------------------------------------------------------------------
# 2. Interaction test: Difference-in-Differences (from orthogonality)
# ----------------------------------------------------------------------------

def analyze_interaction_did():
    """Analyze orthogonality_exp_results.json using difference-in-differences.

    The 2x2 design:
                    Random context    DCS-Logistic context
      base models      A                  B
      dist models      C                  D

    DCS effect on base:   B - A
    DCS effect on dist:   D - C
    DiD = (D - C) - (B - A)

    If DiD ≈ 0 (not significant), the two methods are orthogonal.
    If DiD ≠ 0 (significant), there is an interaction.
    """
    filepath = os.path.join(RESULT_DIR, 'orthogonality_exp_results.json')
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found")
        return None

    with open(filepath, 'r') as f:
        data = json.load(f)

    all_results = data['results']

    methods = ['TabPFN-base-Random', 'TabPFN-base-DCS-Logistic',
               'TabPFN-dist-Random', 'TabPFN-dist-DCS-Logistic']

    method_vals = {}
    for m in methods:
        vals = [r['metrics']['accuracy'] for r in all_results
                if r.get('method') == m and r.get('metrics')]
        method_vals[m] = vals

    n = len(method_vals.get('TabPFN-base-Random', []))
    if n < 2:
        return {'error': 'Not enough seeds for DiD analysis', 'n_seeds': n}

    base_random = np.array(method_vals['TabPFN-base-Random'])
    base_dcs = np.array(method_vals['TabPFN-base-DCS-Logistic'])
    dist_random = np.array(method_vals['TabPFN-dist-Random'])
    dist_dcs = np.array(method_vals['TabPFN-dist-DCS-Logistic'])

    # Per-seed effects
    dcs_effect_base = base_dcs - base_random   # B - A
    dcs_effect_dist = dist_dcs - dist_random    # D - C
    dist_effect_random = dist_random - base_random  # C - A
    dist_effect_dcs = dist_dcs - base_dcs           # D - B

    # Difference-in-differences
    did_values = dcs_effect_dist - dcs_effect_base  # (D-C) - (B-A)

    # One-sample t-test: H0: DiD = 0
    t_stat, p_value = stats.ttest_1samp(did_values, 0)
    ci_low, ci_high = confidence_interval(did_values.tolist())

    # Cohen's d for the DiD (one-sample)
    did_std = did_values.std(ddof=1) if n > 1 else 0.0
    did_cohens_d = float(did_values.mean() / did_std) if did_std > 0 else 0.0

    # Also compute the 2x2 paired t-tests
    comparisons = {
        'DCS_vs_Random_on_base': paired_t_test(base_dcs.tolist(), base_random.tolist()),
        'DCS_vs_Random_on_dist': paired_t_test(dist_dcs.tolist(), dist_random.tolist()),
        'dist_vs_base_with_Random': paired_t_test(dist_random.tolist(), base_random.tolist()),
        'dist_vs_base_with_DCS': paired_t_test(dist_dcs.tolist(), base_dcs.tolist()),
    }

    analysis = {
        'data_source': 'orthogonality_exp_results.json',
        'experiment': 'difference_in_differences',
        'n_seeds': int(n),
        'design': '2x2 factorial (model_type x context_selection)',
        'method_values': {
            'TabPFN-base-Random': base_random.tolist(),
            'TabPFN-base-DCS-Logistic': base_dcs.tolist(),
            'TabPFN-dist-Random': dist_random.tolist(),
            'TabPFN-dist-DCS-Logistic': dist_dcs.tolist(),
        },
        'means': {
            'base_Random': float(base_random.mean()),
            'base_DCS': float(base_dcs.mean()),
            'dist_Random': float(dist_random.mean()),
            'dist_DCS': float(dist_dcs.mean()),
        },
        'effects': {
            'DCS_effect_on_base_pp': float(dcs_effect_base.mean() * 100),
            'DCS_effect_on_dist_pp': float(dcs_effect_dist.mean() * 100),
            'dist_effect_on_Random_pp': float(dist_effect_random.mean() * 100),
            'dist_effect_on_DCS_pp': float(dist_effect_dcs.mean() * 100),
        },
        'difference_in_differences': {
            'description': 'DiD = (dist_DCS - dist_Random) - (base_DCS - base_Random)',
            'did_values': did_values.tolist(),
            'did_mean': float(did_values.mean()),
            'did_std': float(did_std),
            'did_mean_pp': float(did_values.mean() * 100),
            'one_sample_t_test': {
                'test': 'one-sample t-test (H0: DiD=0)',
                'n': int(n),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'df': int(n - 1),
                'ci_95_low': ci_low,
                'ci_95_high': ci_high,
                'cohens_d': did_cohens_d,
                'significant_at_0.05': bool(p_value < 0.05),
            },
            'interpretation': (
                'orthogonal (no interaction)' if p_value >= 0.05
                else 'interaction detected (not orthogonal)'
            ),
        },
        'pairwise_comparisons': comparisons,
    }

    # Summary
    s = analysis['means']
    e = analysis['effects']
    did = analysis['difference_in_differences']
    analysis['summary'] = {
        'base_Random_mean': s['base_Random'],
        'base_DCS_mean': s['base_DCS'],
        'dist_Random_mean': s['dist_Random'],
        'dist_DCS_mean': s['dist_DCS'],
        'DCS_effect_on_base_pp': e['DCS_effect_on_base_pp'],
        'DCS_effect_on_dist_pp': e['DCS_effect_on_dist_pp'],
        'dist_effect_on_Random_pp': e['dist_effect_on_Random_pp'],
        'dist_effect_on_DCS_pp': e['dist_effect_on_DCS_pp'],
        'did_pp': did['did_mean_pp'],
        'did_p_value': did['one_sample_t_test']['p_value'],
        'did_significant': did['one_sample_t_test']['significant_at_0.05'],
        'orthogonal': not did['one_sample_t_test']['significant_at_0.05'],
    }

    return analysis


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Re-run Statistical Significance Tests (Fixed DCS Results)")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Result directory: {RESULT_DIR}")

    output = {
        'experiment': 'statistical_tests_rerun',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'description': (
            'Statistical tests computed from fixed DCS results '
            '(largest remainder method, n_context=10000).'
        ),
    }

    # --- 1. Context Shield analysis (5-seed) ---
    print("\n[1/2] Analyzing context_shield_results.json (5-seed, Adult)...")
    cs_analysis = analyze_context_shield_results()
    if cs_analysis is None:
        print("  ERROR: Could not load context_shield_results.json")
        output['context_shield_analysis'] = {'error': 'File not found'}
    else:
        output['context_shield_analysis'] = cs_analysis

        # Print summary for temporal accuracy (primary metric)
        key = 'adult_temporal_accuracy'
        if key in cs_analysis.get('splits', {}):
            sa = cs_analysis['splits'][key]
            print(f"\n  --- {key} (baseline: TabPFN-Random, n={sa['n_seeds']}) ---")
            print(f"  {'Method':<37} {'Mean':<10} {'Δ(pp)':<10} {'t-stat':<10} "
                  f"{'p-value':<12} {'Cohen d':<10} {'Sig?'}")
            for method, comp in sa['comparisons'].items():
                t = comp['paired_t_test']
                if 'error' in t:
                    continue
                p = t.get('p_value', 1)
                sig = ('***' if p < 0.001 else '**' if p < 0.01
                       else '*' if p < 0.05 else 'ns')
                print(f"  {method:<37} {comp['mean']:<10.4f} {comp['delta_pp']:<+10.2f} "
                      f"{t.get('t_statistic', 0):<10.4f} {p:<12.6f} "
                      f"{t.get('cohens_d', 0):<10.4f} {sig}")

        # DCS vs DRWS head-to-head
        dcs_key = 'adult_temporal_accuracy'
        if dcs_key in cs_analysis.get('dcs_vs_drws_logistic', {}):
            dv = cs_analysis['dcs_vs_drws_logistic'][dcs_key]
            t = dv['paired_t_test']
            print(f"\n  --- DCS-Logistic vs DRWS-Logistic (temporal, accuracy) ---")
            print(f"  DCS mean:  {dv['dcs_logistic_mean']:.4f} ± {dv['dcs_logistic_std']:.4f}")
            print(f"  DRWS mean: {dv['drws_logistic_mean']:.4f} ± {dv['drws_logistic_std']:.4f}")
            if 't_statistic' in t:
                print(f"  t={t['t_statistic']:.4f}, p={t['p_value']:.6f}, "
                      f"df={t['df']}, Cohen's d={t['cohens_d']:.4f}")
                print(f"  95% CI: [{t['ci_95_low']:.6f}, {t['ci_95_high']:.6f}]")
                print(f"  Δ = {t['delta_pp']:+.2f}pp, significant: {t['significant_at_0.05']}")

        # Bonferroni
        bonf = cs_analysis.get('bonferroni_correction', {})
        print(f"\n  Bonferroni correction: {bonf.get('n_comparisons', 0)} comparisons, "
              f"corrected α={bonf.get('corrected_alpha', 0):.5f}")

    # --- 2. Interaction / DiD analysis (3-seed) ---
    print("\n[2/2] Analyzing orthogonality_exp_results.json (DiD, 3-seed)...")
    did_analysis = analyze_interaction_did()
    if did_analysis is None:
        print("  ERROR: Could not load orthogonality_exp_results.json")
        output['interaction_did_analysis'] = {'error': 'File not found'}
    elif 'error' in did_analysis:
        print(f"  ERROR: {did_analysis['error']}")
        output['interaction_did_analysis'] = did_analysis
    else:
        output['interaction_did_analysis'] = did_analysis

        s = did_analysis['summary']
        did = did_analysis['difference_in_differences']
        print(f"\n  --- Difference-in-Differences (n={did_analysis['n_seeds']} seeds) ---")
        print(f"  base+Random:  {s['base_Random_mean']:.4f}")
        print(f"  base+DCS:     {s['base_DCS_mean']:.4f}  "
              f"(DCS effect: {s['DCS_effect_on_base_pp']:+.2f}pp)")
        print(f"  dist+Random:  {s['dist_Random_mean']:.4f}  "
              f"(dist effect: {s['dist_effect_on_Random_pp']:+.2f}pp)")
        print(f"  dist+DCS:     {s['dist_DCS_mean']:.4f}  "
              f"(combined: {(s['dist_DCS_mean']-s['base_Random_mean'])*100:+.2f}pp)")
        print(f"\n  DiD = {s['did_pp']:+.2f}pp")
        print(f"  t={did['one_sample_t_test']['t_statistic']:.4f}, "
              f"p={did['one_sample_t_test']['p_value']:.6f}, "
              f"df={did['one_sample_t_test']['df']}")
        print(f"  Cohen's d={did['one_sample_t_test']['cohens_d']:.4f}")
        print(f"  95% CI: [{did['one_sample_t_test']['ci_95_low']:.6f}, "
              f"{did['one_sample_t_test']['ci_95_high']:.6f}]")
        print(f"  Interpretation: {did['interpretation']}")

    # --- Save ---
    os.makedirs(RESULT_DIR, exist_ok=True)
    output_path = os.path.join(RESULT_DIR, 'statistical_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(json_safe(output), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
