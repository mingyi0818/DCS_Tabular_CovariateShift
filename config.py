"""Configuration for DCS: Diversity-Constrained Density-Ratio Selection
for Test-Time Context Optimization of Tabular Foundation Models.

Paper: DCS for Tabular Foundation Models under Covariate Shift
Target journal: International Journal of Machine Learning and Cybernetics (Springer, SCIE)
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# === Datasets used in experiments ===
DATASETS = {
    'adult': {
        'name': 'Adult-Income',
        'path': os.path.join(RAW_DATA_DIR, 'adult', 'adult.csv'),
        'target': 'income',
        'n_way': 2
    },
    'bank': {
        'name': 'Bank-Marketing',
        'path': os.path.join(RAW_DATA_DIR, 'bank', 'bank.csv'),
        'target': 'deposit',
        'n_way': 2
    },
    'telco': {
        'name': 'Telco-Customer-Churn',
        'path': os.path.join(RAW_DATA_DIR, 'telco', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        'target': 'Churn',
        'n_way': 2
    },
    'mushroom': {
        'name': 'Secondary-Mushroom',
        'path': os.path.join(RAW_DATA_DIR, 'mushroom', 'mushroom.csv'),
        'target': 'class',
        'n_way': 2
    },
}

# === Experiment configuration ===
CONFIG = {
    'seed': 42,
    'seeds': [42, 123, 456, 789, 2024],
    'context_size': 10000,
    'n_clusters': 50,
    'contamination': 0.05,
    'metrics': ['accuracy', 'f1_macro', 'auc'],
}

# === Device ===
DEVICE = 'cuda'

# === Drift-Resilient TabPFN path ===
DRIFT_TABPFN_PATH = os.path.join(BASE_DIR, 'reference', 'Drift-Resilient_TabPFN-main')

if __name__ == '__main__':
    print("DCS Tabular CovariateShift Config Loaded")
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Base dir: {BASE_DIR}")
    print(f"Result dir: {RESULT_DIR}")
