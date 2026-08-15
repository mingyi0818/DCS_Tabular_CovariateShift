"""Download the Adult Income dataset from UCI ML Repository.

The Adult dataset is required for the DCS vs TTA comparison experiment.
If it's not already present in data/raw/adult/adult.csv, this script
downloads it from UCI and saves it in the expected format.

UCI ML Repository: https://archive.ics.uci.edu/dataset/2/adult
"""
import os
import sys
import time
import urllib.request
import ssl
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
ADULT_DIR = RAW_DATA_DIR / 'adult'
ADULT_CSV = ADULT_DIR / 'adult.csv'

# URLs for the Adult dataset
ADULT_DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
ADULT_TEST_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

# Column names for the Adult dataset (no header in the raw files)
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]


def download_file(url, dst_path, timeout=60):
    """Download a file with SSL context that allows unverified connections."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
        data = response.read()
    with open(dst_path, 'wb') as f:
        f.write(data)
    return len(data)


def download_adult():
    """Download and prepare the Adult dataset."""
    if ADULT_CSV.exists():
        # Check if file is non-empty and has expected columns
        import pandas as pd
        try:
            df = pd.read_csv(ADULT_CSV, nrows=5)
            if 'income' in df.columns and len(df) > 0:
                print(f"  [OK] Adult dataset already exists: {ADULT_CSV}")
                print(f"       Size: {ADULT_CSV.stat().st_size:,} bytes")
                return True
        except Exception:
            pass  # File might be corrupted, re-download

    print(f"\n--- Downloading Adult dataset ---")
    ADULT_DIR.mkdir(parents=True, exist_ok=True)

    # Download training data
    train_path = ADULT_DIR / 'adult.data'
    print(f"  Downloading from: {ADULT_DATA_URL}")
    try:
        size = download_file(ADULT_DATA_URL, train_path)
        print(f"  [OK] Downloaded adult.data: {size:,} bytes")
    except Exception as e:
        print(f"  [FAIL] Could not download adult.data: {e}")
        return False

    # Download test data
    test_path = ADULT_DIR / 'adult.test'
    print(f"  Downloading from: {ADULT_TEST_URL}")
    try:
        size = download_file(ADULT_TEST_URL, test_path)
        print(f"  [OK] Downloaded adult.test: {size:,} bytes")
    except Exception as e:
        print(f"  [FAIL] Could not download adult.test: {e}")
        return False

    # Combine into a single CSV with headers
    import pandas as pd
    print(f"  Combining train + test into adult.csv...")

    try:
        # Read training data (no header, skip blank lines)
        df_train = pd.read_csv(train_path, header=None, names=COLUMNS,
                               skipinitialspace=True, na_values='?')
        df_train = df_train.dropna()

        # Read test data (skip first row which is "|1x3 Cross validator")
        df_test = pd.read_csv(test_path, header=None, names=COLUMNS,
                              skipinitialspace=True, na_values='?', skiprows=1)
        df_test = df_test.dropna()

        # Clean income column (test data has periods at the end)
        df_train['income'] = df_train['income'].str.replace('.', '', regex=False)
        df_test['income'] = df_test['income'].str.replace('.', '', regex=False)

        # Combine
        df = pd.concat([df_train, df_test], ignore_index=True)

        # Save as adult.csv
        df.to_csv(ADULT_CSV, index=False)
        print(f"  [OK] Saved adult.csv: {len(df)} rows, {len(df.columns)} columns")
        print(f"       Path: {ADULT_CSV}")
        print(f"       Target column: 'income'")
        print(f"       Classes: {df['income'].unique().tolist()}")
        return True

    except Exception as e:
        print(f"  [FAIL] Could not process Adult data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Adult Dataset Downloader")
    print("=" * 60)
    print(f"Target path: {ADULT_CSV}")

    success = download_adult()

    if success:
        print(f"\n  [SUCCESS] Adult dataset ready at {ADULT_CSV}")
    else:
        print(f"\n  [FAILED] Could not download Adult dataset.")
        print(f"  Manual download instructions:")
        print(f"  1. Go to: https://archive.ics.uci.edu/dataset/2/adult")
        print(f"  2. Download adult.data and adult.test")
        print(f"  3. Combine them into a CSV with columns: {COLUMNS}")
        print(f"  4. Save as: {ADULT_CSV}")

    return success


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
