"""5-seed XGBoost few-shot experiment on all datasets.

Seeds: 42, 123, 456, 789, 2024
Budgets: 200, 500, 1000, 2000, 5000, 10000
Datasets: Adult, Mushroom, Bank, Telco, Bike Sharing

No TabPFN API needed - only XGBoost.
"""
import os, sys, json, time, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR
from splits import SPLIT_CONFIG, load_raw_dataframe, encode_features
from context_shield_methods import set_seed, json_safe

SEEDS = [42, 123, 456, 789, 2024]
BUDGETS = [200, 500, 1000, 2000, 5000, 10000]

def compute_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    auc = 0.0
    if y_proba is not None:
        try:
            if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except: pass
    return {'accuracy': float(acc), 'f1_macro': float(f1m), 'auc': float(auc)}

def sort_temporal(df, dataset_name, seed):
    cfg = SPLIT_CONFIG.get(dataset_name, {})
    temporal_col = cfg.get('temporal_col')
    if temporal_col is None: return df
    df_sorted = df.copy()
    if 'temporal_order' in cfg and cfg['temporal_order']:
        order_map = {m: i for i, m in enumerate(cfg['temporal_order'])}
        df_sorted['_o'] = df_sorted[temporal_col].map(lambda x: order_map.get(x, 0))
        rng = np.random.RandomState(seed)
        df_sorted['_j'] = rng.uniform(0, 0.5, size=len(df_sorted))
        df_sorted = df_sorted.sort_values(['_o', '_j']).drop(['_o', '_j'], axis=1)
    else:
        rng = np.random.RandomState(seed)
        std_val = df_sorted[temporal_col].std()
        js = 0.01 * std_val if std_val > 0 else 0.01
        df_sorted['_j'] = rng.uniform(0, js, size=len(df_sorted))
        df_sorted['_k'] = df_sorted[temporal_col] + df_sorted['_j']
        df_sorted = df_sorted.sort_values('_k').drop(['_j', '_k'], axis=1)
    return df_sorted

def prepare_split(dataset_name, seed):
    df, target_col = load_raw_dataframe(dataset_name)
    df_sorted = sort_temporal(df, dataset_name, seed)
    n = len(df_sorted)
    n_train = int(n * 0.7)
    n_test_start = int(n * 0.85)
    train_df = df_sorted.iloc[:n_train].copy()
    test_df = df_sorted.iloc[n_test_start:].copy()
    X_train_df, y_train, _ = encode_features(train_df, target_col)
    X_test_df, y_test, _ = encode_features(test_df, target_col, fit_df=train_df)
    for col in X_train_df.columns:
        if col not in X_test_df.columns: X_test_df[col] = 0
    X_test_df = X_test_df[X_train_df.columns]
    scaler = StandardScaler()
    return {
        'X_train': scaler.fit_transform(X_train_df.values),
        'X_test': scaler.transform(X_test_df.values),
        'y_train': y_train, 'y_test': y_test,
        'n_train': len(X_train_df), 'n_test': len(X_test_df),
    }

def load_bike_sharing(seed=42):
    import zipfile, urllib.request
    data_dir = os.path.join(os.path.dirname(RESULT_DIR), 'data', 'raw', 'bike_sharing')
    csv_path = os.path.join(data_dir, 'hour.csv')
    if not os.path.exists(csv_path):
        os.makedirs(data_dir, exist_ok=True)
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip'
        zip_path = os.path.join(data_dir, 'Bike-Sharing-Dataset.zip')
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf: zf.extract('hour.csv', data_dir)
        os.remove(zip_path)
    df = pd.read_csv(csv_path)
    df = df.drop(['instant', 'casual', 'registered', 'dteday'], axis=1).dropna()
    target = 'cnt'
    median_val = df[target].median()
    df[target] = (df[target] > median_val).astype(int)
    rng = np.random.RandomState(seed)
    df['_j'] = rng.uniform(0, 0.01, size=len(df))
    df = df.sort_values(['yr', 'mnth', 'hr', '_j']).drop('_j', axis=1)
    n = len(df); n_train = int(n * 0.7); n_test_start = int(n * 0.85)
    train_df = df.iloc[:n_train].copy(); test_df = df.iloc[n_test_start:].copy()
    cat_cols = [c for c in train_df.select_dtypes(include=['object']).columns if c != target]
    for col in cat_cols:
        le = LabelEncoder(); train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = test_df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
        test_df[col] = le.transform(test_df[col])
    y_train = train_df[target].values; y_test = test_df[target].values
    X_train_df = train_df.drop(target, axis=1); X_test_df = test_df.drop(target, axis=1)
    for col in X_train_df.columns:
        if col not in X_test_df.columns: X_test_df[col] = 0
    X_test_df = X_test_df[X_train_df.columns]
    scaler = StandardScaler()
    return {
        'X_train': scaler.fit_transform(X_train_df.values),
        'X_test': scaler.transform(X_test_df.values),
        'y_train': y_train, 'y_test': y_test,
        'n_train': len(X_train_df), 'n_test': len(X_test_df),
    }

def run_xgboost(X_train, y_train, X_test, y_test):
    if len(np.unique(y_train)) < 2:
        return {'accuracy': float(np.mean(y_test == y_train[0])), 'f1_macro': 0.0, 'auc': 0.0,
                'error': 'Only one class in training subset'}
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                            random_state=42, eval_metric='logloss', use_label_encoder=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try: y_proba = clf.predict_proba(X_test)
    except: y_proba = None
    return compute_metrics(y_test, y_pred, y_proba)

def main():
    print("="*80)
    print("5-Seed XGBoost Few-Shot Experiment (All Datasets)")
    print("="*80)
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"Seeds: {SEEDS}")
    print(f"Budgets: {BUDGETS}")
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    all_results = {
        'experiment': 'fewshot_5seed_xgboost',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'config': {'seeds': SEEDS, 'budgets': BUDGETS},
        'results': {},
    }
    
    datasets = ['adult', 'mushroom', 'bank', 'telco']
    
    for ds_name in datasets:
        print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")
        all_results['results'][ds_name] = {}
        
        for seed in SEEDS:
            print(f"  [seed={seed}] ", end='', flush=True)
            set_seed(seed)
            try:
                split = prepare_split(ds_name, seed)
            except Exception as e:
                print(f"ERROR: {e}")
                all_results['results'][ds_name][str(seed)] = {'error': str(e)}
                continue
            
            X_train, y_train = split['X_train'], split['y_train']
            X_test, y_test = split['X_test'], split['y_test']
            n_train = split['n_train']
            
            seed_data = {'n_train': n_train, 'n_test': split['n_test']}
            
            # XGBoost full
            m = run_xgboost(X_train, y_train, X_test, y_test)
            seed_data['xgboost_full'] = m
            print(f"full={m['accuracy']:.4f} ", end='', flush=True)
            
            # XGBoost budgets
            for budget in BUDGETS:
                if budget >= n_train: continue
                m = run_xgboost(X_train[:budget], y_train[:budget], X_test, y_test)
                seed_data[f'xgboost_{budget}'] = m
                print(f"b{budget}={m['accuracy']:.4f} ", end='', flush=True)
            
            print()
            all_results['results'][ds_name][str(seed)] = seed_data
    
    # Bike Sharing
    print(f"\n{'='*60}\nDataset: bike_sharing\n{'='*60}")
    ds_name = 'bike_sharing'
    all_results['results'][ds_name] = {}
    
    for seed in SEEDS:
        print(f"  [seed={seed}] ", end='', flush=True)
        set_seed(seed)
        try:
            split = load_bike_sharing(seed=seed)
        except Exception as e:
            print(f"ERROR: {e}")
            all_results['results'][ds_name][str(seed)] = {'error': str(e)}
            continue
        
        X_train, y_train = split['X_train'], split['y_train']
        X_test, y_test = split['X_test'], split['y_test']
        n_train = split['n_train']
        
        seed_data = {'n_train': n_train, 'n_test': split['n_test']}
        
        m = run_xgboost(X_train, y_train, X_test, y_test)
        seed_data['xgboost_full'] = m
        print(f"full={m['accuracy']:.4f} ", end='', flush=True)
        
        for budget in BUDGETS:
            if budget >= n_train: continue
            m = run_xgboost(X_train[:budget], y_train[:budget], X_test, y_test)
            seed_data[f'xgboost_{budget}'] = m
            print(f"b{budget}={m['accuracy']:.4f} ", end='', flush=True)
        
        print()
        all_results['results'][ds_name][str(seed)] = seed_data
    
    # Summary
    print(f"\n{'='*80}\nSUMMARY: XGBoost accuracy (5-seed mean ± std)\n{'='*80}")
    
    summary = {}
    for ds_name in all_results['results']:
        print(f"\n--- {ds_name} ---")
        summary[ds_name] = {}
        seeds_data = all_results['results'][ds_name]
        
        # Collect all metrics
        keys = set()
        for s in SEEDS:
            if str(s) in seeds_data:
                keys.update(seeds_data[str(s)].keys())
        keys.discard('n_train'); keys.discard('n_test'); keys.discard('error')
        
        for key in sorted(keys):
            accs = []
            for s in SEEDS:
                if str(s) in seeds_data and key in seeds_data[str(s)] and 'accuracy' in seeds_data[str(s)][key]:
                    accs.append(seeds_data[str(s)][key]['accuracy'])
            if accs:
                mean = np.mean(accs)
                std = np.std(accs, ddof=1) if len(accs) > 1 else 0
                print(f"  {key}: {mean:.4f} ± {std:.4f} (n={len(accs)})")
                summary[ds_name][key] = {'mean': float(mean), 'std': float(std), 'n': len(accs), 'values': accs}
    
    all_results['summary'] = summary
    
    output_path = os.path.join(RESULT_DIR, 'fewshot_5seed_xgboost.json')
    with open(output_path, 'w') as f:
        json.dump(json_safe(all_results), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()
