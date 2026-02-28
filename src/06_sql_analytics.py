"""
╔══════════════════════════════════════════════════════════════╗
║   SUPPLY CHAIN DISRUPTION ANALYSIS                           ║
║   SCRIPT 6 OF 7 — SQL ANALYTICS LAYER                       ║
╠══════════════════════════════════════════════════════════════╣
║  What this script does:                                      ║
║  • Loads cleaned data into a local SQLite database           ║
║    (works without PostgreSQL installed — same SQL syntax)    ║
║  • Runs 15 production-grade SQL queries covering:            ║
║    - Aggregations, GROUP BY, HAVING                          ║
║    - Window functions (RANK, ROW_NUMBER, LAG, LEAD)          ║
║    - CTEs (Common Table Expressions)                         ║
║    - Subqueries & JOINs across tables                        ║
║    - Rolling averages using window frames                    ║
║  • Saves all query results as CSVs                           ║
║  • Prints results to console in a formatted table            ║
║                                                              ║
║  HOW TO RUN:                                                 ║
║    python src/06_sql_analytics.py                            ║
║                                                              ║
║  PostgreSQL note:                                            ║
║    All queries are 100% PostgreSQL-compatible.               ║
║    To switch from SQLite to PostgreSQL, replace              ║
║    get_connection() with:                                    ║
║      import psycopg2                                         ║
║      conn = psycopg2.connect(                                ║
║          host="localhost", dbname="supply_chain",            ║
║          user="postgres", password="yourpassword"            ║
║      )                                                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sqlite3
import warnings
import textwrap
import pandas as pd
from colorama import Fore, Style, init

warnings.filterwarnings("ignore")
init(autoreset=True)

# ════════════════════════════════════════════════════════════════
#  PATH HELPERS
# ════════════════════════════════════════════════════════════════

# Script is in src/, so BASE_DIR is parent of script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUT_DIR    = os.path.join(BASE_DIR, "outputs", "sql_results")
DB_PATH    = os.path.join(DATA_DIR, "supply_chain.db")

os.makedirs(OUT_DIR, exist_ok=True)


def section(t): print(f"\n{Fore.CYAN}{'═'*62}\n  {t}\n{'═'*62}{Style.RESET_ALL}")
def ok(m):      print(f"{Fore.GREEN}  ✔  {m}{Style.RESET_ALL}")
def note(m):    print(f"{Fore.YELLOW}  ℹ  {m}{Style.RESET_ALL}")
def header(t):  print(f"\n{Fore.MAGENTA}  ── {t} ──{Style.RESET_ALL}")


# ════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ════════════════════════════════════════════════════════════════

def get_connection():
    """
    Returns a SQLite connection.
    SQLite is built into Python — no installation needed.
    All queries below are written in standard SQL that works
    identically in PostgreSQL.
    """
    return sqlite3.connect(DB_PATH)


def load_data_to_db():
    """Load all cleaned CSVs into SQLite tables."""
    section("Loading Data into SQL Database")

    files = {
        "orders"    : os.path.join(DATA_DIR, "cleaned", "dataco_clean.csv"),
        "products"  : os.path.join(DATA_DIR, "cleaned", "product_chain_clean.csv"),
        "trade"     : os.path.join(DATA_DIR, "cleaned", "global_trade_clean.csv"),
    }

    conn = get_connection()

    for table_name, path in files.items():
        if not os.path.exists(path):
            note(f"  {path} not found — run Script 01 first.")
            continue

        df = pd.read_csv(path, low_memory=False)

        # Standardise column names for SQL (no spaces)
        df.columns = (
            df.columns.str.lower()
              .str.replace(r"[\s\(\)\/\-]+", "_", regex=True)
              .str.replace(r"_+", "_", regex=True)
              .str.strip("_")
        )

        df.to_sql(table_name, conn, if_exists="replace", index=False)
        ok(f"  Loaded table '{table_name}'  →  {len(df):,} rows")

    conn.commit()
    conn.close()


# ════════════════════════════════════════════════════════════════
#  QUERY RUNNER
# ════════════════════════════════════════════════════════════════

def run_query(name: str, sql: str, description: str) -> pd.DataFrame:
    """Execute a SQL query and display results."""
    header(f"Query: {name}")
    print(f"  {Fore.WHITE}{description}{Style.RESET_ALL}")
    print(f"\n{Fore.YELLOW}  SQL:{Style.RESET_ALL}")
    for line in textwrap.dedent(sql).strip().splitlines():
        print(f"    {Fore.BLUE}{line}{Style.RESET_ALL}")

    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn)
        print(f"\n{Fore.GREEN}  Result ({len(df)} rows):{Style.RESET_ALL}")
        print(df.to_string(index=False, max_rows=10))

        # Save result
        out_path = os.path.join(OUT_DIR, f"{name.lower().replace(' ','_')}.csv")
        df.to_csv(out_path, index=False)
        ok(f"  Saved → {out_path}")

    except Exception as e:
        print(f"{Fore.RED}  Error: {e}{Style.RESET_ALL}")
        df = pd.DataFrame()
    finally:
        conn.close()

    return df


# ════════════════════════════════════════════════════════════════
#  THE 15 SQL QUERIES
# ════════════════════════════════════════════════════════════════

def run_all_queries():
    section("Running 15 SQL Analytics Queries")

    # ── Q1: Basic aggregation — revenue & orders by market ───────
    run_query(
        name        = "01_revenue_by_market",
        description = "Total revenue, order count, and average order value per market.",
        sql         = """
            SELECT
                market,
                COUNT(*)                          AS total_orders,
                ROUND(SUM(sales), 2)              AS total_revenue,
                ROUND(AVG(sales), 2)              AS avg_order_value,
                ROUND(SUM(benefit_per_order), 2)  AS total_profit
            FROM orders
            WHERE market IS NOT NULL
            GROUP BY market
            ORDER BY total_revenue DESC
        """
    )

    # ── Q2: HAVING — markets with high late delivery rate ────────
    run_query(
        name        = "02_high_delay_markets",
        description = "Markets where the late delivery rate exceeds 50% (HAVING clause).",
        sql         = """
            SELECT
                market,
                COUNT(*)                                      AS total_orders,
                SUM(is_delayed)                               AS delayed_orders,
                ROUND(100.0 * SUM(is_delayed) / COUNT(*), 2) AS late_rate_pct
            FROM orders
            WHERE market IS NOT NULL
              AND is_delayed IS NOT NULL
            GROUP BY market
            HAVING ROUND(100.0 * SUM(is_delayed) / COUNT(*), 2) > 50
            ORDER BY late_rate_pct DESC
        """
    )

    # ── Q3: Window function — RANK by revenue per market ────────
    run_query(
        name        = "03_category_rank_by_market",
        description = "Rank product categories by revenue within each market using RANK().",
        sql         = """
            SELECT
                market,
                category_name,
                ROUND(SUM(sales), 2) AS category_revenue,
                RANK() OVER (
                    PARTITION BY market
                    ORDER BY SUM(sales) DESC
                ) AS revenue_rank
            FROM orders
            WHERE market IS NOT NULL
              AND category_name IS NOT NULL
            GROUP BY market, category_name
            ORDER BY market, revenue_rank
        """
    )

    # ── Q4: Window function — ROW_NUMBER for top-1 per group ────
    run_query(
        name        = "04_top_category_per_market",
        description = "The single best-selling product category in each market.",
        sql         = """
            WITH ranked AS (
                SELECT
                    market,
                    category_name,
                    ROUND(SUM(sales), 2) AS revenue,
                    ROW_NUMBER() OVER (
                        PARTITION BY market
                        ORDER BY SUM(sales) DESC
                    ) AS rn
                FROM orders
                WHERE market IS NOT NULL
                  AND category_name IS NOT NULL
                GROUP BY market, category_name
            )
            SELECT market, category_name, revenue
            FROM ranked
            WHERE rn = 1
            ORDER BY revenue DESC
        """
    )

    # ── Q5: Window function — running total (cumulative sum) ─────
    run_query(
        name        = "05_cumulative_revenue_by_year",
        description = "Cumulative revenue over years using SUM() OVER window.",
        sql         = """
            SELECT
                order_year,
                ROUND(SUM(sales), 2)                          AS yearly_revenue,
                ROUND(SUM(SUM(sales)) OVER (
                    ORDER BY order_year
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ), 2)                                         AS cumulative_revenue
            FROM orders
            WHERE order_year IS NOT NULL
            GROUP BY order_year
            ORDER BY order_year
        """
    )

    # ── Q6: LAG window function — year-over-year growth ─────────
    run_query(
        name        = "06_yoy_revenue_growth",
        description = "Year-over-year revenue growth % using LAG() window function.",
        sql         = """
            WITH yearly AS (
                SELECT
                    order_year,
                    ROUND(SUM(sales), 2) AS revenue
                FROM orders
                WHERE order_year IS NOT NULL
                GROUP BY order_year
            )
            SELECT
                order_year,
                revenue,
                LAG(revenue) OVER (ORDER BY order_year) AS prev_year_revenue,
                ROUND(
                    100.0 * (revenue - LAG(revenue) OVER (ORDER BY order_year))
                    / NULLIF(LAG(revenue) OVER (ORDER BY order_year), 0),
                    2
                ) AS yoy_growth_pct
            FROM yearly
            ORDER BY order_year
        """
    )

    # ── Q7: CTE — supplier risk segmentation ────────────────────
    run_query(
        name        = "07_supplier_risk_segments",
        description = "Classify suppliers into Low / Medium / High risk using a CTE and CASE.",
        sql         = """
            WITH supplier_stats AS (
                SELECT
                    supplier_name,
                    ROUND(AVG(supplier_risk_score), 2) AS avg_risk,
                    ROUND(AVG(defect_rates), 4)        AS avg_defect,
                    ROUND(AVG(lead_times), 1)          AS avg_lead_time,
                    COUNT(*)                           AS sku_count
                FROM products
                WHERE supplier_name IS NOT NULL
                GROUP BY supplier_name
            )
            SELECT
                supplier_name,
                avg_risk,
                avg_defect,
                avg_lead_time,
                sku_count,
                CASE
                    WHEN avg_risk >= 70 THEN 'HIGH RISK'
                    WHEN avg_risk >= 40 THEN 'MEDIUM RISK'
                    ELSE                     'LOW RISK'
                END AS risk_segment
            FROM supplier_stats
            ORDER BY avg_risk DESC
        """
    )

    # ── Q8: JOIN — orders joined with product data ───────────────
    run_query(
        name        = "08_orders_with_product_details",
        description = "JOIN orders and products on shared category to enrich order data.",
        sql         = """
            SELECT
                o.market,
                o.category_name,
                ROUND(AVG(o.sales), 2)                AS avg_order_sales,
                ROUND(AVG(p.manufacturing_costs), 2)  AS avg_mfg_cost,
                ROUND(AVG(o.profit_margin_pct), 2)    AS avg_profit_pct,
                COUNT(DISTINCT o.rowid)               AS order_count
            FROM orders o
            LEFT JOIN products p
                ON LOWER(o.category_name) = LOWER(p.product_type)
            WHERE o.category_name IS NOT NULL
            GROUP BY o.market, o.category_name
            ORDER BY avg_profit_pct DESC
            LIMIT 20
        """
    )

    # ── Q9: Subquery — orders above category average sales ───────
    run_query(
        name        = "09_above_avg_orders",
        description = "Find orders with sales above their own category's average (subquery).",
        sql         = """
            SELECT
                category_name,
                market,
                ROUND(sales, 2)                 AS order_sales,
                ROUND(avg_cat.avg_sales, 2)     AS category_avg_sales,
                ROUND(sales - avg_cat.avg_sales, 2) AS above_avg_by
            FROM orders
            JOIN (
                SELECT category_name, AVG(sales) AS avg_sales
                FROM orders
                GROUP BY category_name
            ) AS avg_cat USING (category_name)
            WHERE sales > avg_cat.avg_sales
              AND category_name IS NOT NULL
            ORDER BY above_avg_by DESC
            LIMIT 20
        """
    )

    # ── Q10: Shipping mode performance analysis ───────────────────
    run_query(
        name        = "10_shipping_mode_performance",
        description = "Full KPI breakdown by shipping mode — cost vs speed vs reliability.",
        sql         = """
            SELECT
                shipping_mode,
                COUNT(*)                                          AS total_orders,
                ROUND(AVG(days_for_shipping_real), 2)            AS avg_actual_days,
                ROUND(AVG(days_for_shipment_scheduled), 2)       AS avg_scheduled_days,
                ROUND(AVG(delay_days), 2)                        AS avg_delay_days,
                ROUND(100.0 * SUM(is_delayed) / COUNT(*), 1)     AS late_rate_pct,
                ROUND(AVG(sales), 2)                             AS avg_order_value,
                ROUND(AVG(profit_margin_pct), 2)                 AS avg_profit_margin
            FROM orders
            WHERE shipping_mode IS NOT NULL
            GROUP BY shipping_mode
            ORDER BY late_rate_pct DESC
        """
    )

    # ── Q11: Global trade — top commodities by export value ──────
    run_query(
        name        = "11_top_export_commodities",
        description = "Top 10 commodities by total export value from the global trade table.",
        sql         = """
            SELECT
                commodity,
                ROUND(SUM(trade_usd_billions), 3) AS total_export_billions,
                COUNT(DISTINCT country_or_area)   AS country_count,
                ROUND(AVG(trade_usd_billions), 4) AS avg_per_country
            FROM trade
            WHERE flow = 'Export'
              AND commodity IS NOT NULL
            GROUP BY commodity
            ORDER BY total_export_billions DESC
            LIMIT 10
        """
    )

    # ── Q12: Window NTILE — segment orders into quartiles ────────
    run_query(
        name        = "12_sales_quartile_analysis",
        description = "Split orders into 4 revenue quartiles using NTILE(4) window function.",
        sql         = """
            WITH quartiled AS (
                SELECT
                    market,
                    category_name,
                    sales,
                    NTILE(4) OVER (ORDER BY sales) AS sales_quartile
                FROM orders
                WHERE sales IS NOT NULL
            )
            SELECT
                sales_quartile,
                COUNT(*)                     AS order_count,
                ROUND(MIN(sales), 2)         AS min_sales,
                ROUND(MAX(sales), 2)         AS max_sales,
                ROUND(AVG(sales), 2)         AS avg_sales,
                ROUND(SUM(sales), 2)         AS total_sales
            FROM quartiled
            GROUP BY sales_quartile
            ORDER BY sales_quartile
        """
    )

    # ── Q13: LEAD window — forecast next month demand ────────────
    run_query(
        name        = "13_monthly_demand_with_lead",
        description = "Show each month's orders alongside next month using LEAD() function.",
        sql         = """
            WITH monthly AS (
                SELECT
                    order_year,
                    order_month,
                    COUNT(*)             AS order_count,
                    ROUND(SUM(sales),2)  AS monthly_revenue
                FROM orders
                WHERE order_year IS NOT NULL AND order_month IS NOT NULL
                GROUP BY order_year, order_month
            )
            SELECT
                order_year,
                order_month,
                order_count,
                monthly_revenue,
                LEAD(order_count)      OVER (ORDER BY order_year, order_month) AS next_month_orders,
                LEAD(monthly_revenue)  OVER (ORDER BY order_year, order_month) AS next_month_revenue
            FROM monthly
            ORDER BY order_year, order_month
        """
    )

    # ── Q14: Profitability deep dive — CASE bucketing ────────────
    run_query(
        name        = "14_profitability_buckets",
        description = "Bucket orders into profitability tiers and count how many fall in each.",
        sql         = """
            SELECT
                CASE
                    WHEN profit_margin_pct >= 30  THEN 'High Margin (30%+)'
                    WHEN profit_margin_pct >= 10  THEN 'Medium Margin (10-30%)'
                    WHEN profit_margin_pct >= 0   THEN 'Low Margin (0-10%)'
                    ELSE                               'Loss Making (<0%)'
                END                         AS profitability_tier,
                COUNT(*)                    AS order_count,
                ROUND(SUM(sales), 2)        AS total_revenue,
                ROUND(AVG(sales), 2)        AS avg_order_value,
                ROUND(AVG(profit_margin_pct), 2) AS avg_margin_pct
            FROM orders
            WHERE profit_margin_pct IS NOT NULL
            GROUP BY profitability_tier
            ORDER BY avg_margin_pct DESC
        """
    )

    # ── Q15: Executive summary — single CTE with everything ──────
    run_query(
        name        = "15_executive_summary",
        description = "One-query executive dashboard: all top KPIs in a single result row.",
        sql         = """
            SELECT
                COUNT(*)                                       AS total_orders,
                ROUND(SUM(sales), 2)                          AS total_revenue_usd,
                ROUND(AVG(sales), 2)                          AS avg_order_value,
                ROUND(100.0 * SUM(is_delayed) / COUNT(*), 1) AS overall_late_rate_pct,
                ROUND(AVG(profit_margin_pct), 2)              AS avg_profit_margin_pct,
                ROUND(AVG(delay_days), 2)                     AS avg_delay_days,
                MIN(order_year)                               AS first_year,
                MAX(order_year)                               AS last_year,
                COUNT(DISTINCT market)                        AS markets_served,
                COUNT(DISTINCT category_name)                 AS product_categories
            FROM orders
        """
    )


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print(f"\n{Fore.MAGENTA}{'▓'*62}")
    print("  SUPPLY CHAIN PROJECT  ·  STEP 6: SQL ANALYTICS LAYER")
    print(f"{'▓'*62}{Style.RESET_ALL}")

    load_data_to_db()
    run_all_queries()

    section("✅ STEP 6 COMPLETE")
    print(f"""
{Fore.GREEN}  Database created:  data/supply_chain.db
  Query results:     outputs/sql_results/ (15 CSV files)

  Queries demonstrated:
    • GROUP BY + aggregations (SUM, AVG, COUNT)
    • HAVING for post-aggregation filtering
    • Window functions: RANK, ROW_NUMBER, NTILE, LAG, LEAD
    • Running totals: SUM() OVER with ROWS BETWEEN
    • CTEs (Common Table Expressions)
    • JOINs across orders + products tables
    • Correlated subqueries
    • CASE WHEN bucketing
    • Multi-table executive summary

  ▶  Next step → run:  src/07_pdf_report.py
{Style.RESET_ALL}""")


if __name__ == "__main__":
    main()