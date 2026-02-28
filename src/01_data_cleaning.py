"""
╔══════════════════════════════════════════════════════════════╗
║   SUPPLY CHAIN DISRUPTION ANALYSIS                           ║
║   SCRIPT 1 OF 5 — DATA CLEANING                             ║
╠══════════════════════════════════════════════════════════════╣
║  DATASETS USED (download from Kaggle):                       ║
║  1. DataCo Smart Supply Chain (main dataset)                 ║
║     → kaggle.com/datasets/shashwatwork/                      ║
║       dataco-smart-supply-chain-for-big-data-analysis        ║
║  2. Supply Chain Analysis                                     ║
║     → kaggle.com/datasets/harshsingh2209/                    ║
║       supply-chain-analysis                                   ║
║  3. Global Commodity Trade Statistics                         ║
║     → kaggle.com/datasets/unitednations/                     ║
║       global-commodity-trade-statistics                       ║
║                                                              ║
║  HOW TO RUN IN PYCHARM:                                      ║
║  1. Open PyCharm → File → Open → supply_chain_project folder ║
║  2. Terminal → pip install -r requirements.txt               ║
║  3. Right-click this file → Run '01_data_cleaning'           ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── Standard library ─────────────────────────────────────────
import os
import warnings

# ── Third-party ───────────────────────────────────────────────
import numpy  as np
import pandas as pd
from colorama import Fore, Style, init

warnings.filterwarnings("ignore")
init(autoreset=True)   # makes Fore/Style work on Windows too


# ════════════════════════════════════════════════════════════════
#  CONSOLE HELPERS  (makes terminal output look professional)
# ════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    """Print a bold cyan section banner."""
    print(f"\n{Fore.CYAN}{'═' * 62}")
    print(f"  {title}")
    print(f"{'═' * 62}{Style.RESET_ALL}")

def ok(msg: str)   -> None: print(f"{Fore.GREEN}  ✔  {msg}{Style.RESET_ALL}")
def note(msg: str) -> None: print(f"{Fore.YELLOW}  ℹ  {msg}{Style.RESET_ALL}")
def warn(msg: str) -> None: print(f"{Fore.RED}  ✘  {msg}{Style.RESET_ALL}")


# ════════════════════════════════════════════════════════════════
#  STEP 0 — CREATE PROJECT FOLDER STRUCTURE
# ════════════════════════════════════════════════════════════════

def make_dirs() -> None:
    """
    Build the folder tree once so every other script can rely on it.
    Calling os.makedirs(..., exist_ok=True) is safe — it does nothing
    if the folder already exists.
    """
    folders = [
        "data/raw",           # put your Kaggle CSV files here
        "data/cleaned",       # output of this script
        "data/final",         # output of script 03
        "outputs/charts",     # all PNG charts
        "outputs/reports",    # CSV summary tables
        "src",                # all Python scripts live here
        "notebooks",          # optional Jupyter notebooks
    ]
    for f in folders:
        os.makedirs(f, exist_ok=True)
    ok("Project folder structure ready.")


# ════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD RAW DATA
#  • If the real Kaggle file is present → load it
#  • If not → generate a synthetic version so the whole pipeline
#    still runs end-to-end (swap in the real file later)
# ════════════════════════════════════════════════════════════════

# ── 1a  DataCo Smart Supply Chain ───────────────────────────────

def load_dataco() -> pd.DataFrame:
    """
    Main dataset: 180 000+ orders with shipping mode, delay flag,
    profit, customer location, product category, etc.
    """
    section("Loading Dataset 1 — DataCo Supply Chain")
    path = "data/raw/DataCoSupplyChainDataset.csv"

    if os.path.exists(path):
        df = pd.read_csv(path, encoding="latin-1")
        ok(f"Real file loaded  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df

    note("Kaggle file not found — generating synthetic version (50 000 rows).")
    note("Place the real CSV at:  data/raw/DataCoSupplyChainDataset.csv")

    rng = np.random.default_rng(42)
    n   = 50_000

    categories  = ["Electronics","Clothing","Furniture","Food","Sports","Automotive"]
    markets     = ["Europe","LATAM","Pacific Asia","USCA","Africa"]
    ship_modes  = ["Standard Class","Second Class","First Class","Same Day"]
    departments = ["Fan Shop","Apparel","Golf","Outdoors","Technology","Pet Shop"]
    countries   = ["United States","Mexico","France","Germany","Brazil",
                   "China","India","Australia","Nigeria","Argentina"]
    statuses    = ["Complete","Pending","Shipped","Closed","Processing",
                   "On Hold","Payment Review","Canceled","Suspected_Fraud"]
    status_p    = [0.35, 0.10, 0.15, 0.10, 0.09, 0.06, 0.06, 0.05, 0.04]

    sched = rng.integers(1, 8, n)
    real  = np.clip(sched + rng.integers(-1, 10, n), 0, 30)

    df = pd.DataFrame({
        "Order Id"                     : range(1, n + 1),
        "Order Date"                   : pd.date_range("2015-01-01", periods=n, freq="6H"),
        "Customer Country"             : rng.choice(countries,   n),
        "Market"                       : rng.choice(markets,     n),
        "Order Region"                 : rng.choice(markets,     n),
        "Category Name"                : rng.choice(categories,  n),
        "Department Name"              : rng.choice(departments, n),
        "Shipping Mode"                : rng.choice(ship_modes,  n),
        "Order Status"                 : rng.choice(statuses,    n, p=status_p),
        "Sales"                        : rng.uniform(10, 5000, n).round(2),
        "Order Item Quantity"          : rng.integers(1, 20, n),
        "Order Item Discount Rate"     : rng.uniform(0, 0.4,  n).round(2),
        "Order Item Profit Ratio"      : rng.uniform(-0.2, 0.5, n).round(2),
        "Days for shipping (real)"     : real,
        "Days for shipment (scheduled)": sched,
        "Late_delivery_risk"           : (real > sched).astype(int),
        "Benefit per order"            : rng.uniform(-200, 800, n).round(2),
        "Order Item Total"             : rng.uniform(10, 5000, n).round(2),
        "Product Price"                : rng.uniform(5, 2000, n).round(2),
    })

    # Inject 5 % missing values to make cleaning realistic
    for col in ["Customer Country", "Days for shipping (real)", "Order Item Discount Rate"]:
        df.loc[rng.random(n) < 0.05, col] = np.nan

    df.to_csv(path, index=False)
    ok(f"Synthetic DataCo saved → {path}")
    return df


# ── 1b  Supply Chain Analysis (products & inventory) ────────────

def load_product_chain() -> pd.DataFrame:
    """
    Secondary dataset: product-level supply chain with stock levels,
    lead times, defect rates, and supplier info.
    """
    section("Loading Dataset 2 — Product Supply Chain")
    path = "data/raw/supply_chain_data.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        ok(f"Real file loaded  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df

    note("File not found — generating synthetic product supply chain.")
    note("Place the real CSV at:  data/raw/supply_chain_data.csv")

    rng = np.random.default_rng(7)
    n   = 1_000

    product_types = ["haircare","skincare","cosmetics","accessories"]
    suppliers     = [f"Supplier_{i}" for i in range(1, 6)]

    df = pd.DataFrame({
        "Product type"          : rng.choice(product_types, n),
        "SKU"                   : [f"SKU{i:05d}" for i in range(n)],
        "Price"                 : rng.uniform(5, 300, n).round(2),
        "Availability"          : rng.integers(0, 500, n),
        "Number of products sold": rng.integers(0, 2000, n),
        "Revenue generated"     : rng.uniform(100, 50000, n).round(2),
        "Stock levels"          : rng.integers(0, 1000, n),
        "Lead times"            : rng.integers(1, 30,  n),
        "Order quantities"      : rng.integers(10, 500, n),
        "Shipping times"        : rng.integers(1, 14,  n),
        "Shipping carriers"     : rng.choice(["Carrier A","Carrier B","Carrier C"], n),
        "Shipping costs"        : rng.uniform(5, 200, n).round(2),
        "Supplier name"         : rng.choice(suppliers, n),
        "Location"              : rng.choice(["Mumbai","Delhi","Bangalore","Chennai","Kolkata"], n),
        "Production volumes"    : rng.integers(100, 5000, n),
        "Manufacturing lead time": rng.integers(1, 20, n),
        "Manufacturing costs"   : rng.uniform(20, 500, n).round(2),
        "Inspection results"    : rng.choice(["Pass","Fail","Pending"], n, p=[0.80, 0.10, 0.10]),
        "Defect rates"          : rng.uniform(0, 0.15, n).round(4),
        "Transportation modes"  : rng.choice(["Road","Air","Sea","Rail"], n),
        "Routes"                : rng.choice(["Route A","Route B","Route C"], n),
        "Costs"                 : rng.uniform(50, 2000, n).round(2),
    })

    df.to_csv(path, index=False)
    ok(f"Synthetic product chain saved → {path}")
    return df


# ── 1c  Global Trade Statistics ─────────────────────────────────

def load_global_trade() -> pd.DataFrame:
    """
    Tertiary dataset: country-level import/export values by commodity.
    Used for macroeconomic context and regional risk scoring.
    """
    section("Loading Dataset 3 — Global Trade Statistics")
    path = "data/raw/global_commodity_trade_statistics.csv"

    if os.path.exists(path):
        df = pd.read_csv(path, encoding="latin-1")
        ok(f"Real file loaded  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df

    note("File not found — generating synthetic global trade data.")
    note("Place the real CSV at:  data/raw/global_commodity_trade_statistics.csv")

    rng = np.random.default_rng(99)
    n   = 5_000

    countries    = ["USA","Germany","China","India","Brazil","France",
                    "UK","Japan","Australia","Mexico","Nigeria","Argentina"]
    commodities  = ["Machinery","Vehicles","Electronics","Chemicals",
                    "Textiles","Food","Fuel","Metals","Plastics","Pharmaceuticals"]
    flows        = ["Export","Import"]

    df = pd.DataFrame({
        "country_or_area" : rng.choice(countries,   n),
        "year"            : rng.integers(2010, 2024, n),
        "commodity"       : rng.choice(commodities,  n),
        "flow"            : rng.choice(flows,         n),
        "trade_usd"       : rng.uniform(1e6, 1e10,   n).round(0),
        "weight_kg"       : rng.uniform(1e4, 1e8,    n).round(0),
        "quantity_name"   : rng.choice(["Number of items","Weight in kg","Volume in liters"], n),
        "quantity"        : rng.uniform(1e3, 1e7,    n).round(0),
        "category"        : rng.choice(commodities,  n),
    })

    df.to_csv(path, index=False)
    ok(f"Synthetic global trade saved → {path}")
    return df


# ════════════════════════════════════════════════════════════════
#  STEP 2 — CLEAN EACH DATASET
# ════════════════════════════════════════════════════════════════

def _find_col(df: pd.DataFrame, candidates: list) -> str | None:
    """
    Find a column by trying a list of possible names.
    Also does a loose substring match as fallback.
    """
    for name in candidates:
        if name in df.columns:
            return name
    for name in candidates:
        hits = [c for c in df.columns
                if name.replace("_","") in c.lower().replace(" ","").replace("_","")]
        if hits:
            return hits[0]
    return None


def _standardise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip spaces, replace separators with underscore."""
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[\s\(\)\/\-]+", "_", regex=True)
          .str.replace(r"_+", "_", regex=True)
          .str.strip("_")
    )
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    • Numeric columns  → median  (robust to outliers)
    • String columns   → mode    (most frequent category)
    """
    before = df.isnull().sum().sum()

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=["object","category"]).columns
    for col in cat_cols:
        mode = df[col].mode()
        df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")

    after = df.isnull().sum().sum()
    note(f"  Missing values: {before:,} → {after:,}")
    return df


def clean_dataco(df: pd.DataFrame) -> pd.DataFrame:
    section("Cleaning DataCo Dataset")
    note(f"  Raw shape: {df.shape}")

    df = _standardise_cols(df)

    # Parse all date-like columns
    for col in [c for c in df.columns if "date" in c]:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")

    df = _impute(df)
    dups = df.duplicated().sum()
    df   = df.drop_duplicates()
    note(f"  Removed {dups:,} duplicate rows")

    # ── Derive delay features ──────────────────────────────────
    real  = _find_col(df, ["days_for_shipping_real"])
    sched = _find_col(df, ["days_for_shipment_scheduled"])

    if real and sched:
        df["delay_days"] = df[real] - df[sched]
        df["is_delayed"] = (df["delay_days"] > 0).astype(int)
        df["delay_severity"] = pd.cut(
            df["delay_days"],
            bins   = [-np.inf, 0, 2, 5, np.inf],
            labels = ["On Time", "Minor", "Moderate", "Severe"]
        )
        ok("  Derived: delay_days, is_delayed, delay_severity")

    # ── Derive time features ───────────────────────────────────
    order_col = _find_col(df, ["order_date"])
    if order_col:
        df["order_year"]     = df[order_col].dt.year
        df["order_month"]    = df[order_col].dt.month
        df["order_quarter"]  = df[order_col].dt.quarter
        df["order_dayofweek"]= df[order_col].dt.dayofweek
        ok("  Derived: order_year, order_month, order_quarter, order_dayofweek")

    # ── Profit margin % ────────────────────────────────────────
    sales  = _find_col(df, ["sales"])
    profit = _find_col(df, ["benefit_per_order"])
    if sales and profit:
        df["profit_margin_pct"] = np.where(
            df[sales] != 0,
            (df[profit] / df[sales] * 100).round(2),
            0
        )
        ok("  Derived: profit_margin_pct")

    # ── Remove extreme 1%-99% outliers on sales ───────────────
    if sales:
        q_lo, q_hi = df[sales].quantile([0.01, 0.99])
        before = len(df)
        df = df[(df[sales] >= q_lo) & (df[sales] <= q_hi)]
        note(f"  Removed {before - len(df):,} outlier rows on sales")

    ok(f"  Clean shape: {df.shape}")
    return df


def clean_product_chain(df: pd.DataFrame) -> pd.DataFrame:
    section("Cleaning Product Supply Chain Dataset")
    note(f"  Raw shape: {df.shape}")

    df = _standardise_cols(df)
    df = _impute(df)
    df = df.drop_duplicates()

    # Convert defect_rates to percentage
    dr = _find_col(df, ["defect_rates"])
    if dr and df[dr].max() <= 1.0:
        df["defect_rate_pct"] = (df[dr] * 100).round(2)
        ok("  Derived: defect_rate_pct")

    # Supplier risk score: combo of defect rate + lead time (normalised 0–100)
    lt = _find_col(df, ["lead_times", "manufacturing_lead_time"])
    if dr and lt:
        dr_norm = (df[dr]  - df[dr].min())  / (df[dr].max()  - df[dr].min() + 1e-9)
        lt_norm = (df[lt]  - df[lt].min())  / (df[lt].max()  - df[lt].min() + 1e-9)
        df["supplier_risk_score"] = ((0.6 * dr_norm + 0.4 * lt_norm) * 100).round(2)
        ok("  Derived: supplier_risk_score (0–100, higher = riskier)")

    ok(f"  Clean shape: {df.shape}")
    return df


def clean_global_trade(df: pd.DataFrame) -> pd.DataFrame:
    section("Cleaning Global Trade Dataset")
    note(f"  Raw shape: {df.shape}")

    df = _standardise_cols(df)
    df = _impute(df)
    df = df.drop_duplicates()

    # Convert trade_usd to billions for readability in charts
    trade = _find_col(df, ["trade_usd"])
    if trade:
        df["trade_usd_billions"] = (df[trade] / 1e9).round(4)
        ok("  Derived: trade_usd_billions")

    # Year as int
    yr = _find_col(df, ["year"])
    if yr:
        df[yr] = pd.to_numeric(df[yr], errors="coerce").astype("Int64")

    ok(f"  Clean shape: {df.shape}")
    return df


# ════════════════════════════════════════════════════════════════
#  STEP 3 — SAVE CLEANED FILES
# ════════════════════════════════════════════════════════════════

def save(df: pd.DataFrame, filename: str) -> None:
    path = f"data/cleaned/{filename}"
    df.to_csv(path, index=False)
    ok(f"Saved → {path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{Fore.MAGENTA}{'▓' * 62}")
    print("  SUPPLY CHAIN PROJECT  ·  STEP 1: DATA CLEANING")
    print(f"{'▓' * 62}{Style.RESET_ALL}")

    make_dirs()

    # Load
    dataco  = load_dataco()
    product = load_product_chain()
    trade   = load_global_trade()

    # Clean
    dataco_clean  = clean_dataco(dataco)
    product_clean = clean_product_chain(product)
    trade_clean   = clean_global_trade(trade)

    # Save
    save(dataco_clean,  "dataco_clean.csv")
    save(product_clean, "product_chain_clean.csv")
    save(trade_clean,   "global_trade_clean.csv")

    section("✅ STEP 1 COMPLETE")
    print(f"""
{Fore.GREEN}  Cleaned files saved to  data/cleaned/
    • dataco_clean.csv
    • product_chain_clean.csv
    • global_trade_clean.csv

  ▶  Next step → run:  src/02_exploratory_analysis.py
{Style.RESET_ALL}""")


if __name__ == "__main__":
    main()