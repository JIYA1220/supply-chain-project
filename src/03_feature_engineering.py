"""
╔══════════════════════════════════════════════════════════════╗
║   SUPPLY CHAIN DISRUPTION ANALYSIS                           ║
║   SCRIPT 3 OF 5 — FEATURE ENGINEERING                       ║
╠══════════════════════════════════════════════════════════════╣
║  What this script does:                                      ║
║  • Build 30+ ML-ready features from the cleaned data         ║
║  • Rolling statistics (7-day, 30-day)                        ║
║  • Lag features (previous order delay)                       ║
║  • Risk scores per region, supplier, category                ║
║  • One-hot encoding of categorical variables                  ║
║  • Final dataset saved to data/final/ml_ready.csv            ║
║                                                              ║
║  Run after: 02_exploratory_analysis.py                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
import pandas as pd
import numpy  as np
from sklearn.preprocessing import LabelEncoder
from colorama import Fore, Style, init

warnings.filterwarnings("ignore")
init(autoreset=True)


def section(t): print(f"\n{Fore.CYAN}{'═'*62}\n  {t}\n{'═'*62}{Style.RESET_ALL}")
def ok(m):      print(f"{Fore.GREEN}  ✔  {m}{Style.RESET_ALL}")
def note(m):    print(f"{Fore.YELLOW}  ℹ  {m}{Style.RESET_ALL}")

def find(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    for c in candidates:
        hits = [col for col in df.columns
                if c.replace("_","") in col.lower().replace("_","").replace(" ","")]
        if hits: return hits[0]
    return None


# ════════════════════════════════════════════════════════════════
#  LOAD
# ════════════════════════════════════════════════════════════════

def load_data():
    section("Loading Cleaned DataCo Dataset")
    df = pd.read_csv("../data/cleaned/dataco_clean.csv", low_memory=False)
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    ok(f"Loaded  {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ════════════════════════════════════════════════════════════════
#  FEATURE GROUP A — TIME FEATURES
# ════════════════════════════════════════════════════════════════

def time_features(df):
    section("Feature Group A — Time Features")
    date_col = find(df, ["order_date"])
    if not date_col:
        note("No date column found — skipping time features."); return df

    df = df.sort_values(date_col).reset_index(drop=True)

    df["is_weekend"]      = df[date_col].dt.dayofweek.isin([5,6]).astype(int)
    df["is_month_end"]    = df[date_col].dt.is_month_end.astype(int)
    df["is_month_start"]  = df[date_col].dt.is_month_start.astype(int)
    df["is_quarter_end"]  = df[date_col].dt.is_quarter_end.astype(int)
    df["week_of_year"]    = df[date_col].dt.isocalendar().week.astype(int)
    df["day_of_year"]     = df[date_col].dt.dayofyear

    # Cyclical encoding — tells the model Jan and Dec are close
    df["month_sin"] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7)

    ok("Added: is_weekend, is_month_end, is_month_start, is_quarter_end,")
    ok("       week_of_year, day_of_year, month_sin/cos, dow_sin/cos")
    return df


# ════════════════════════════════════════════════════════════════
#  FEATURE GROUP B — ROLLING STATISTICS
#  (captures short-term trends in delay & sales)
# ════════════════════════════════════════════════════════════════

def rolling_features(df):
    section("Feature Group B — Rolling Statistics")
    delay  = find(df, ["delay_days"])
    sales  = find(df, ["sales"])
    profit = find(df, ["profit_margin_pct"])
    date   = find(df, ["order_date"])

    if not (delay and date):
        note("Delay/date columns not found — skipping rolling features."); return df

    df = df.sort_values(date).reset_index(drop=True)

    for window, label in [(7,"7d"), (30,"30d")]:
        df[f"delay_rolling_mean_{label}"]   = (
            df[delay].rolling(window, min_periods=1).mean().round(3))
        df[f"delay_rolling_std_{label}"]    = (
            df[delay].rolling(window, min_periods=1).std().fillna(0).round(3))
        if sales:
            df[f"sales_rolling_sum_{label}"] = (
                df[sales].rolling(window, min_periods=1).sum().round(2))
            df[f"sales_rolling_mean_{label}"] = (
                df[sales].rolling(window, min_periods=1).mean().round(2))
        if profit:
            df[f"profit_rolling_mean_{label}"] = (
                df[profit].rolling(window, min_periods=1).mean().round(3))

    ok("Added 7-day and 30-day rolling: mean_delay, std_delay, sum_sales, mean_sales, mean_profit")
    return df


# ════════════════════════════════════════════════════════════════
#  FEATURE GROUP C — LAG FEATURES
#  (what happened in the previous order — great for ML)
# ════════════════════════════════════════════════════════════════

def lag_features(df):
    section("Feature Group C — Lag Features")
    delay  = find(df, ["delay_days"])
    sales  = find(df, ["sales"])
    profit = find(df, ["profit_margin_pct"])

    if not delay:
        note("Delay column not found — skipping lag features."); return df

    for lag in [1, 3, 7]:
        df[f"delay_lag_{lag}"]  = df[delay].shift(lag).fillna(0)
        if sales:
            df[f"sales_lag_{lag}"] = df[sales].shift(lag).fillna(0)

    # Delay acceleration: rate of change
    df["delay_change"]      = df[delay] - df[f"delay_lag_1"]
    df["delay_acceleration"]= df["delay_change"] - df["delay_change"].shift(1).fillna(0)

    ok("Added lag features (1, 3, 7) for delay and sales")
    ok("Added delay_change, delay_acceleration")
    return df


# ════════════════════════════════════════════════════════════════
#  FEATURE GROUP D — RISK SCORES
#  (aggregate delay rate per group → business-meaningful feature)
# ════════════════════════════════════════════════════════════════

def risk_score_features(df):
    section("Feature Group D — Risk Scores by Group")
    is_del = find(df, ["is_delayed"])
    delay  = find(df, ["delay_days"])

    for group_col, suffix in [
        (find(df, ["market"]),             "market"),
        (find(df, ["shipping_mode"]),      "ship_mode"),
        (find(df, ["category_name"]),      "category"),
        (find(df, ["customer_country"]),   "country"),
        (find(df, ["order_region"]),       "region"),
    ]:
        if not group_col: continue

        if is_del:
            rate = df.groupby(group_col)[is_del].transform("mean")
            df[f"delay_rate_{suffix}"] = rate.round(4)

        if delay:
            avg = df.groupby(group_col)[delay].transform("mean")
            df[f"avg_delay_{suffix}"] = avg.round(3)

        ok(f"  Added delay_rate_{suffix}, avg_delay_{suffix}")

    return df


# ════════════════════════════════════════════════════════════════
#  FEATURE GROUP E — ENCODING CATEGORICAL VARIABLES
#  (ML models need numbers, not text)
# ════════════════════════════════════════════════════════════════

def encode_categoricals(df):
    section("Feature Group E — Encoding Categorical Variables")

    # One-hot encode low-cardinality columns (≤ 10 unique values)
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()

    ohe_cols    = []
    label_cols  = []

    for col in cat_cols:
        nunique = df[col].nunique()
        if nunique <= 10:
            ohe_cols.append(col)
        elif nunique <= 100:
            label_cols.append(col)
        # > 100 unique → drop for ML (too high cardinality, e.g. customer name)

    if ohe_cols:
        ohe_df = pd.get_dummies(df[ohe_cols], prefix=ohe_cols, drop_first=True, dtype=int)
        df = pd.concat([df, ohe_df], axis=1)
        ok(f"One-hot encoded {len(ohe_cols)} columns → {ohe_df.shape[1]} new binary columns")

    le = LabelEncoder()
    for col in label_cols:
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        ok(f"Label encoded: {col}")

    # Drop original string columns (keep encoded versions)
    df.drop(columns=cat_cols, inplace=True, errors="ignore")

    return df


# ════════════════════════════════════════════════════════════════
#  FEATURE GROUP F — BUSINESS KPIs
# ════════════════════════════════════════════════════════════════

def business_kpis(df):
    section("Feature Group F — Business KPI Features")

    sales   = find(df, ["sales"])
    qty     = find(df, ["order_item_quantity"])
    profit  = find(df, ["benefit_per_order"])
    product = find(df, ["product_price"])
    disc    = find(df, ["order_item_discount_rate"])

    if sales and qty:
        df["revenue_per_unit"] = (df[sales] / df[qty].replace(0, 1)).round(2)
        ok("Added: revenue_per_unit")

    if profit and sales:
        df["is_profitable"] = (df[profit] > 0).astype(int)
        ok("Added: is_profitable (1 = positive profit)")

    if disc:
        df["heavy_discount"] = (df[disc] > 0.2).astype(int)
        ok("Added: heavy_discount (1 = discount > 20%)")

    if product and sales:
        df["price_ratio"] = (df[sales] / df[product].replace(0, 1)).round(3)
        ok("Added: price_ratio (actual sales / product price)")

    return df


# ════════════════════════════════════════════════════════════════
#  CLEAN UP & SAVE
# ════════════════════════════════════════════════════════════════

def finalise(df):
    section("Finalising ML-Ready Dataset")

    # Drop date columns (not useful for ML directly)
    date_cols = [c for c in df.columns if "date" in c]
    df.drop(columns=date_cols, inplace=True, errors="ignore")

    # Drop columns with > 50 % NaN
    thresh = len(df) * 0.5
    df.dropna(axis=1, thresh=thresh, inplace=True)

    # Fill any remaining NaNs with 0
    df.fillna(0, inplace=True)

    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    ok(f"Final ML dataset shape: {df.shape[0]:,} rows × {df.shape[1]} features")

    path = "../data/final/ml_ready.csv"
    df.to_csv(path, index=False)
    ok(f"Saved → {path}")

    # Feature summary report
    summary = pd.DataFrame({
        "feature"   : df.columns,
        "dtype"     : df.dtypes.values,
        "null_count": df.isnull().sum().values,
        "nunique"   : df.nunique().values,
        "mean"      : df.mean(numeric_only=True).reindex(df.columns).values,
    })
    summary.to_csv("outputs/reports/feature_summary.csv", index=False)
    ok("Feature summary saved → outputs/reports/feature_summary.csv")

    return df


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print(f"\n{Fore.MAGENTA}{'▓'*62}")
    print("  SUPPLY CHAIN PROJECT  ·  STEP 3: FEATURE ENGINEERING")
    print(f"{'▓'*62}{Style.RESET_ALL}")

    df = load_data()
    df = time_features(df)
    df = rolling_features(df)
    df = lag_features(df)
    df = risk_score_features(df)
    df = business_kpis(df)
    df = encode_categoricals(df)
    df = finalise(df)

    section("✅ STEP 3 COMPLETE")
    print(f"""
{Fore.GREEN}  Final ML dataset:  data/final/ml_ready.csv
  Feature summary:   outputs/reports/feature_summary.csv

  Total features engineered: {df.shape[1]}

  ▶  Next step → run:  src/04_predictive_model.py
{Style.RESET_ALL}""")


if __name__ == "__main__":
    main()