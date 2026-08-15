"""Download TabReD datasets from Kaggle."""
import os, sys, json

# Ensure kaggle credentials are set
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
creds = {"username": "zengjy08", "key": "KGAT_9baa88bab0148843c89d4e936b33af85"}
with open(kaggle_json, "w") as f:
    json.dump(creds, f)
os.environ["KAGGLE_USERNAME"] = "zengjy08"
os.environ["KAGGLE_KEY"] = "KGAT_9baa88bab0148843c89d4e936b33af85"

# Try to authenticate and download
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Kaggle authentication successful!")

    # Download the TabReD dataset (irubachev/tabred)
    # This contains Cooking Time, Delivery ETA, Maps Routing, Weather
    output_dir = r"E:\datasets\tabred_raw"
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading irubachev/tabred dataset...")
    api.dataset_download_files("irubachev/tabred", path=output_dir, unzip=True)
    print(f"Downloaded to {output_dir}")

    # List downloaded files
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp) / 1e6
            print(f"  {os.path.relpath(fp, output_dir)} ({sz:.1f} MB)")

except Exception as e:
    print(f"Error: {e}")

    # Try alternative: use kaggle CLI directly
    print("\nTrying kaggle CLI...")
    os.system('kaggle datasets download -d irubachev/tabred -p E:/datasets/tabred_raw --unzip')
