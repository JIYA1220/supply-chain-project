"""
╔══════════════════════════════════════════════════════════════╗
║   SUPPLY CHAIN DISRUPTION ANALYSIS                           ║
║   SCRIPT 2 OF 5 — EXPLORATORY DATA ANALYSIS (EDA)           ║
╠══════════════════════════════════════════════════════════════╣
║  What this script does:                                      ║
║  • Summary statistics for every dataset                      ║
║  • Distribution plots (sales, delays, profit)                ║
║  • Delay analysis by shipping mode, market, category         ║
║  • Monthly & yearly order trends                             ║
║  • Correlation heatmap                                       ║
║  • Supplier risk analysis                                     ║
║  • Global trade volume by country & commodity                ║
║  • All charts saved to  outputs/charts/                      ║
║                                                              ║
║  Run after: 01_data_cleaning.py                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import warnings
import pandas as pd
import numpy  as np
import matplotlib.pyplot    as plt
import matplotlib.ticker    as mtick
import seaborn              as sns
from colorama import Fore, Style, init

warnings.filterwarnings("ignore")
init(autoreset=True)

# ── Path Handling ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(rel_path: str) -> str:
    """
    Ensure paths work from both the project root and the src/ folder.
    - If it's a ../ path, go up one level from script directory.
    - If not, try both the current directory and the script's parent.
    """
    if rel_path.startswith("../"):
        clean_rel = rel_path[3:]
        base_dir  = os.path.dirname(SCRIPT_DIR)
        return os.path.join(base_dir, clean_rel)

    # If it's a relative path starting from root, but we're in src/
    alt_path = os.path.join(os.path.dirname(SCRIPT_DIR), rel_path)
    if os.path.exists(alt_path) or not os.path.exists(rel_path):
        return alt_path

    return rel_path

# ════════════════════════════════════════════════════════════════
#  GLOBAL CHART STYLE  (dark professional theme)
# ════════════════════════════════════════════════════════════════

BG      = "#0d1117"
PANEL   = "#161b22"
GRID    = "#21262d"
TEXT    = "#c9d1d9"
ACCENT  = "#58a6ff"
PALETTE = ["#58a6ff","#3fb950","#f78166","#d2a8ff",
           "#ffa657","#79c0ff","#56d364","#ff7b72"]

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : PANEL,
    "axes.edgecolor"   : GRID,
    "axes.labelcolor"  : TEXT,
    "axes.titlecolor"  : TEXT,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "xtick.color"      : TEXT,
    "ytick.color"      : TEXT,
    "text.color"       : TEXT,
    "grid.color"       : GRID,
    "grid.linestyle"   : "--",
    "grid.alpha"       : 0.6,
    "legend.facecolor" : PANEL,
    "legend.edgecolor" : GRID,
    "figure.dpi"       : 130,
    "font.family"      : "DejaVu Sans",
})


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════

def section(t): print(f"\n{Fore.CYAN}{'═'*62}\n  {t}\n{'═'*62}{Style.RESET_ALL}")
def ok(m):      print(f"{Fore.GREEN}  ✔  {m}{Style.RESET_ALL}")
def note(m):    print(f"{Fore.YELLOW}  ℹ  {m}{Style.RESET_ALL}")

CHART_DIR = "outputs/charts"

def save(name: str) -> None:
    path = get_path(f"{CHART_DIR}/{name}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close()
    ok(f"Chart saved → {path}")

def find(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    for c in candidates:
        hits = [col for col in df.columns
                if c.replace("_","") in col.lower().replace("_","").replace(" ","")]
        if hits: return hits[0]
    return None


# ════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════

def load_all():
    section("Loading Cleaned Datasets")

    def read(path, label):
        path = get_path(path)
        df = pd.read_csv(path, low_memory=False)
        for col in df.columns:
            if "date" in col:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        ok(f"{label}  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df

    dc = read("../data/cleaned/dataco_clean.csv", "DataCo Supply Chain")
    pr = read("../data/cleaned/product_chain_clean.csv", "Product Chain     ")
    tr = read("../data/cleaned/global_trade_clean.csv", "Global Trade      ")
    return dc, pr, tr


# ════════════════════════════════════════════════════════════════
#  CHART 1 — SUMMARY STATISTICS TABLE  (printed, not plotted)
# ════════════════════════════════════════════════════════════════

def summary_stats(dc, pr, tr):
    section("Summary Statistics")
    for label, df in [("DataCo", dc), ("Products", pr), ("Global Trade", tr)]:
        print(f"\n{Fore.MAGENTA}  ── {label} ──{Style.RESET_ALL}")
        num = df.select_dtypes(include=np.number)
        print(num.describe().round(2).to_string())

    # Save as CSV reports
    path_dc = get_path("outputs/reports/dataco_stats.csv")
    path_pr = get_path("outputs/reports/product_stats.csv")
    path_tr = get_path("outputs/reports/trade_stats.csv")

    os.makedirs(os.path.dirname(path_dc), exist_ok=True)
    dc.describe().round(2).to_csv(path_dc)
    pr.describe().round(2).to_csv(path_pr)
    tr.describe().round(2).to_csv(path_tr)
    ok(f"Summary CSVs saved to outputs/reports/")


# ════════════════════════════════════════════════════════════════
#  CHART 2 — SALES DISTRIBUTION
# ════════════════════════════════════════════════════════════════

def chart_sales_distribution(dc):
    section("Chart: Sales Distribution")
    sales = find(dc, ["sales"])
    if not sales:
        note("Sales column not found — skipping."); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sales Distribution", fontsize=16, color=TEXT, fontweight="bold")

    # Histogram
    ax = axes[0]
    ax.hist(dc[sales].dropna(), bins=60, color=ACCENT, edgecolor=BG, alpha=0.85)
    ax.set_title("Sales — Histogram")
    ax.set_xlabel("Sales (USD)")
    ax.set_ylabel("Order Count")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{x:,.0f}"))
    ax.grid(True)

    # Box plot by market
    ax = axes[1]
    market = find(dc, ["market", "order_region"])
    if market:
        markets = dc[market].dropna().unique()
        data    = [dc.loc[dc[market]==m, sales].dropna() for m in markets]
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticklabels(markets, rotation=25, ha="right")
        ax.set_title("Sales by Market")
        ax.set_ylabel("Sales (USD)")
    ax.grid(True)

    plt.tight_layout()
    save("02_sales_distribution")


# ════════════════════════════════════════════════════════════════
#  CHART 3 — DELAY ANALYSIS
# ════════════════════════════════════════════════════════════════

def chart_delay_analysis(dc):
    section("Chart: Delay Analysis")
    delay = find(dc, ["delay_days"])
    is_del = find(dc, ["is_delayed", "late_delivery_risk"])
    ship   = find(dc, ["shipping_mode"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("Shipping Delay Deep Dive", fontsize=17, color=TEXT, fontweight="bold")

    # ── 3a  Delay distribution ────────────────────────────────
    ax = axes[0, 0]
    if delay:
        d = dc[delay].dropna()
        ax.hist(d, bins=40, color="#f78166", edgecolor=BG, alpha=0.85)
        ax.axvline(0, color="white", linewidth=1.5, linestyle="--", label="On-time line")
        ax.set_title("Distribution of Delay Days")
        ax.set_xlabel("Delay (days)")
        ax.set_ylabel("Count")
        ax.legend()
    ax.grid(True)

    # ── 3b  Late delivery % by Shipping Mode ─────────────────
    ax = axes[0, 1]
    if ship and is_del:
        grp = dc.groupby(ship)[is_del].mean().mul(100).sort_values(ascending=False)
        bars = ax.bar(grp.index, grp.values, color=PALETTE[:len(grp)], edgecolor=BG, alpha=0.85)
        ax.bar_label(bars, fmt="%.1f%%", color=TEXT, fontsize=9, padding=3)
        ax.set_title("Late Delivery % by Shipping Mode")
        ax.set_ylabel("Late Delivery Rate (%)")
        ax.set_ylim(0, grp.max() * 1.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(True)

    # ── 3c  Average delay by Category ────────────────────────
    ax = axes[1, 0]
    cat = find(dc, ["category_name", "department_name"])
    if cat and delay:
        grp = dc.groupby(cat)[delay].mean().sort_values(ascending=False).head(10)
        bars = ax.barh(grp.index, grp.values, color=ACCENT, edgecolor=BG, alpha=0.85)
        ax.bar_label(bars, fmt="%.1f d", color=TEXT, fontsize=9, padding=3)
        ax.set_title("Avg Delay by Product Category (Top 10)")
        ax.set_xlabel("Average Delay (days)")
        ax.invert_yaxis()
    ax.grid(True)

    # ── 3d  Delay severity pie ────────────────────────────────
    ax = axes[1, 1]
    sev = find(dc, ["delay_severity"])
    if sev:
        counts = dc[sev].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels      = counts.index,
            autopct     = "%1.1f%%",
            colors      = PALETTE[:len(counts)],
            startangle  = 140,
            wedgeprops  = dict(edgecolor=BG, linewidth=2),
        )
        for at in autotexts:
            at.set_color("white"); at.set_fontsize(10)
        ax.set_title("Delay Severity Distribution")

    plt.tight_layout()
    save("03_delay_analysis")


# ════════════════════════════════════════════════════════════════
#  CHART 4 — MONTHLY ORDER TRENDS
# ════════════════════════════════════════════════════════════════

def chart_monthly_trends(dc):
    section("Chart: Monthly Order Trends")
    yr  = find(dc, ["order_year"])
    mo  = find(dc, ["order_month"])
    sal = find(dc, ["sales"])

    if not (yr and mo and sal):
        note("Required columns missing — skipping."); return

    monthly = (
        dc.groupby([yr, mo])[sal]
          .agg(["sum","count"])
          .reset_index()
          .rename(columns={"sum":"total_sales","count":"order_count"})
    )
    monthly["period"] = pd.to_datetime(
        monthly[[yr, mo]].rename(columns={yr:"year", mo:"month"}).assign(day=1)
    )
    monthly = monthly.sort_values("period")

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.suptitle("Monthly Supply Chain Trends", fontsize=17, color=TEXT, fontweight="bold")

    ax = axes[0]
    ax.fill_between(monthly["period"], monthly["total_sales"], alpha=0.3, color=ACCENT)
    ax.plot(monthly["period"], monthly["total_sales"], color=ACCENT, linewidth=2)
    ax.set_title("Total Monthly Revenue (USD)")
    ax.set_ylabel("Revenue (USD)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x/1e6:.1f}M"))
    ax.grid(True)

    ax = axes[1]
    ax.fill_between(monthly["period"], monthly["order_count"], alpha=0.3, color="#3fb950")
    ax.plot(monthly["period"], monthly["order_count"], color="#3fb950", linewidth=2)
    ax.set_title("Monthly Order Volume")
    ax.set_ylabel("Number of Orders")
    ax.grid(True)

    plt.tight_layout()
    save("04_monthly_trends")


# ════════════════════════════════════════════════════════════════
#  CHART 5 — CORRELATION HEATMAP
# ════════════════════════════════════════════════════════════════

def chart_correlation(dc):
    section("Chart: Correlation Heatmap")

    num_cols = dc.select_dtypes(include=np.number).columns.tolist()
    # Keep only meaningful columns
    keep = [c for c in num_cols
            if dc[c].nunique() > 5 and "id" not in c.lower()][:14]

    corr = dc[keep].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask      = mask,
        cmap      = cmap,
        center    = 0,
        annot     = True,
        fmt       = ".2f",
        linewidths= 0.5,
        linecolor = BG,
        ax        = ax,
        annot_kws = {"size": 8, "color": TEXT},
        cbar_kws  = {"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=15, pad=15)
    ax.set_facecolor(PANEL)
    fig.patch.set_facecolor(BG)
    plt.xticks(rotation=40, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save("05_correlation_heatmap")


# ════════════════════════════════════════════════════════════════
#  CHART 6 — SUPPLIER RISK ANALYSIS
# ════════════════════════════════════════════════════════════════

def chart_supplier_risk(pr):
    section("Chart: Supplier Risk Analysis")
    supplier = find(pr, ["supplier_name"])
    risk     = find(pr, ["supplier_risk_score"])
    defect   = find(pr, ["defect_rates","defect_rate_pct"])
    lead     = find(pr, ["lead_times","manufacturing_lead_time"])

    if not (supplier and risk):
        note("Supplier columns not found — skipping."); return

    grp = pr.groupby(supplier).agg(
        avg_risk   = (risk,   "mean"),
        avg_defect = (defect, "mean") if defect else (risk, "mean"),
        avg_lead   = (lead,   "mean") if lead   else (risk, "mean"),
        sku_count  = (supplier, "count"),
    ).reset_index().sort_values("avg_risk", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Supplier Risk Dashboard", fontsize=16, color=TEXT, fontweight="bold")

    # ── Bar chart: risk score ──────────────────────────────────
    ax = axes[0]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(grp))]
    bars = ax.barh(grp[supplier], grp["avg_risk"], color=colors, edgecolor=BG, alpha=0.85)
    ax.bar_label(bars, fmt="%.1f", color=TEXT, fontsize=9, padding=3)
    ax.set_title("Avg Supplier Risk Score (0–100)")
    ax.set_xlabel("Risk Score")
    ax.invert_yaxis()
    ax.grid(True)

    # ── Scatter: defect vs lead time, sized by SKU count ───────
    ax = axes[1]
    if defect and lead:
        scatter = ax.scatter(
            grp["avg_defect"],
            grp["avg_lead"],
            s       = grp["sku_count"] * 8,
            c       = grp["avg_risk"],
            cmap    = "RdYlGn_r",
            alpha   = 0.85,
            edgecolors = "white",
            linewidths  = 0.5,
        )
        for _, row in grp.iterrows():
            ax.annotate(row[supplier],
                        (row["avg_defect"], row["avg_lead"]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=TEXT)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Risk Score", color=TEXT)
        cbar.ax.yaxis.set_tick_params(color=TEXT)
        ax.set_xlabel("Avg Defect Rate")
        ax.set_ylabel("Avg Lead Time (days)")
        ax.set_title("Defect Rate vs Lead Time\n(bubble = SKU count, colour = risk)")
    ax.grid(True)

    plt.tight_layout()
    save("06_supplier_risk")


# ════════════════════════════════════════════════════════════════
#  CHART 7 — GLOBAL TRADE VOLUME
# ════════════════════════════════════════════════════════════════

def chart_global_trade(tr):
    section("Chart: Global Trade Analysis")
    country   = find(tr, ["country_or_area", "country"])
    commodity = find(tr, ["commodity", "category"])
    trade     = find(tr, ["trade_usd_billions", "trade_usd"])
    year_col  = find(tr, ["year"])
    flow_col  = find(tr, ["flow"])

    if not (country and trade):
        note("Trade columns missing — skipping."); return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Global Trade Analysis", fontsize=16, color=TEXT, fontweight="bold")

    # ── Top 10 trading countries ───────────────────────────────
    ax = axes[0]
    top_countries = (
        tr.groupby(country)[trade].sum()
          .sort_values(ascending=False)
          .head(10)
    )
    bars = ax.bar(top_countries.index, top_countries.values,
                  color=PALETTE[:10], edgecolor=BG, alpha=0.85)
    ax.bar_label(bars, fmt="%.1f", color=TEXT, fontsize=8, padding=3)
    ax.set_title("Top 10 Countries by Trade Volume")
    ax.set_ylabel("Trade Value (Billions USD)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(True)

    # ── Trade volume over time by flow ─────────────────────────
    ax = axes[1]
    if year_col and flow_col:
        pivot = tr.groupby([year_col, flow_col])[trade].sum().reset_index()
        for i, flow in enumerate(pivot[flow_col].unique()):
            sub = pivot[pivot[flow_col]==flow].sort_values(year_col)
            ax.plot(sub[year_col], sub[trade],
                    marker="o", label=flow,
                    color=PALETTE[i], linewidth=2.5, markersize=5)
        ax.set_title("Export vs Import Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Trade Value (Billions USD)")
        ax.legend()
    ax.grid(True)

    plt.tight_layout()
    save("07_global_trade")


# ════════════════════════════════════════════════════════════════
#  CHART 8 — PROFIT MARGIN ANALYSIS
# ════════════════════════════════════════════════════════════════

def chart_profit(dc):
    section("Chart: Profit Margin Analysis")
    margin = find(dc, ["profit_margin_pct"])
    cat    = find(dc, ["category_name","department_name"])
    market = find(dc, ["market"])

    if not margin:
        note("Profit margin column not found — skipping."); return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Profit Margin Analysis", fontsize=16, color=TEXT, fontweight="bold")

    # ── By category ───────────────────────────────────────────
    ax = axes[0]
    if cat:
        grp = dc.groupby(cat)[margin].median().sort_values(ascending=False).head(8)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(grp))]
        bars = ax.bar(grp.index, grp.values, color=colors, edgecolor=BG, alpha=0.85)
        ax.bar_label(bars, fmt="%.1f%%", color=TEXT, fontsize=9, padding=3)
        ax.axhline(0, color="white", linewidth=1, linestyle="--")
        ax.set_title("Median Profit Margin by Category")
        ax.set_ylabel("Profit Margin (%)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True)

    # ── By market violin ──────────────────────────────────────
    ax = axes[1]
    if market:
        markets = dc[market].dropna().unique()
        data    = [dc.loc[dc[market]==m, margin].dropna() for m in markets]
        vp = ax.violinplot(data, positions=range(len(markets)),
                           showmedians=True, showmeans=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(PALETTE[i % len(PALETTE)])
            body.set_alpha(0.7)
        vp["cmedians"].set_color("white")
        ax.set_xticks(range(len(markets)))
        ax.set_xticklabels(markets, rotation=25, ha="right")
        ax.axhline(0, color="white", linewidth=1, linestyle="--")
        ax.set_title("Profit Margin Distribution by Market")
        ax.set_ylabel("Profit Margin (%)")
    ax.grid(True)

    plt.tight_layout()
    save("08_profit_margin")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print(f"\n{Fore.MAGENTA}{'▓'*62}")
    print("  SUPPLY CHAIN PROJECT  ·  STEP 2: EXPLORATORY ANALYSIS")
    print(f"{'▓'*62}{Style.RESET_ALL}")

    dc, pr, tr = load_all()

    summary_stats(dc, pr, tr)
    chart_sales_distribution(dc)
    chart_delay_analysis(dc)
    chart_monthly_trends(dc)
    chart_correlation(dc)
    chart_supplier_risk(pr)
    chart_global_trade(tr)
    chart_profit(dc)

    section("✅ STEP 2 COMPLETE")
    print(f"""
{Fore.GREEN}  7 charts saved to  outputs/charts/
    02_sales_distribution.png
    03_delay_analysis.png
    04_monthly_trends.png
    05_correlation_heatmap.png
    06_supplier_risk.png
    07_global_trade.png
    08_profit_margin.png

  ▶  Next step → run:  src/03_feature_engineering.py
{Style.RESET_ALL}""")


if __name__ == "__main__":
    main()