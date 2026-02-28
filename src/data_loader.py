import os
import gdown
import pandas as pd
import numpy as np


# ---------------------------------------------------------------
# Paste your actual Google Drive file IDs here after uploading
# ---------------------------------------------------------------
GDRIVE_FILES = {
    "data/raw/DataCoSupplyChainDataset.csv":          "1SO_hfDduPSVI8awI9rXpQgtk6rXcg_XG",
    "data/raw/commodity_trade_statistics_data.csv":   "1_Zek9da59HZ_9bSK-nfbqsgPt8oe-83W",
}


def download_datasets():
    """Download large datasets from Google Drive if not already present."""
    os.makedirs("data/raw", exist_ok=True)

    for local_path, file_id in GDRIVE_FILES.items():
        if not os.path.exists(local_path):
            print(f"Downloading {local_path} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, local_path, quiet=False)
            print(f"Saved to {local_path}")
        else:
            print(f"Already exists, skipping: {local_path}")


def generate_synthetic_data(n=10000):
    """
    Fallback synthetic data if Google Drive download fails.
    Mirrors the structure and statistics of the real DataCo dataset.
    """
    np.random.seed(42)

    markets     = ["Europe", "LATAM", "Pacific Asia", "USCA", "Africa"]
    ship_modes  = ["Standard Class", "Second Class", "First Class", "Same Day"]
    categories  = ["Office Supplies", "Furniture", "Technology"]
    departments = ["Fan Shop", "Apparel", "Golf", "Footwear", "Outdoors"]
    statuses    = ["Complete", "Pending", "Shipped", "Cancelled"]

    df = pd.DataFrame({
        "market":               np.random.choice(markets, n),
        "shipping_mode":        np.random.choice(ship_modes, n),
        "category_name":        np.random.choice(categories, n),
        "department_name":      np.random.choice(departments, n),
        "order_status":         np.random.choice(statuses, n),
        "sales":                np.random.lognormal(5, 1, n).round(2),
        "profit":               np.random.normal(20, 15, n).round(2),
        "order_quantity":       np.random.randint(1, 10, n),
        "discount":             np.random.uniform(0, 0.5, n).round(2),
        "is_delayed":           np.random.choice([0, 1], n, p=[0.45, 0.55]),
        "delay_days":           np.random.randint(-2, 15, n),
        "order_date":           pd.date_range(
                                    start="2021-01-01",
                                    periods=n,
                                    freq="1h"
                                ).strftime("%Y-%m-%d"),
        "profit_margin_pct":    np.random.uniform(-10, 40, n).round(2),
        "delay_rate_market":    np.random.uniform(0.3, 0.8, n).round(3),
        "delay_rate_ship_mode": np.random.uniform(0.2, 0.7, n).round(3),
    })

    return df


def load_data():
    """
    Main data loader.
    1. Try to load already-cleaned data
    2. If not found, try downloading raw data from Google Drive
    3. If download fails, fall back to synthetic data
    """
    cleaned_path = "data/cleaned/dataco_cleaned.csv"

    # --- Option 1: cleaned data already exists (local run) ---
    if os.path.exists(cleaned_path):
        print("Loading cleaned dataset...")
        return pd.read_csv(cleaned_path, low_memory=False)

    # --- Option 2: try downloading from Google Drive ---
    try:
        download_datasets()
        if os.path.exists("data/raw/DataCoSupplyChainDataset.csv"):
            print("Loading raw dataset from Google Drive download...")
            return pd.read_csv(
                "data/raw/DataCoSupplyChainDataset.csv",
                encoding="latin-1",
                low_memory=False
            )
    except Exception as e:
        print(f"Google Drive download failed: {e}")

    # --- Option 3: synthetic fallback ---
    print("Using synthetic demo data...")
    return generate_synthetic_data()