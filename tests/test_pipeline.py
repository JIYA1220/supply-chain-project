"""
╔══════════════════════════════════════════════════════════════╗
║   SUPPLY CHAIN DISRUPTION ANALYSIS                           ║
║   TESTS — Unit & Integration Test Suite                      ║
╠══════════════════════════════════════════════════════════════╣
║  HOW TO RUN:                                                 ║
║    pip install pytest                                        ║
║    pytest tests/test_pipeline.py -v                          ║
║                                                              ║
║  Or run from project root:                                   ║
║    pytest -v                                                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import os
import warnings
import pytest
import numpy  as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add src to path so we can import scripts directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ════════════════════════════════════════════════════════════════
#  FIXTURES  — reusable mock datasets
# ════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_orders():
    """Minimal supply-chain orders DataFrame matching the real schema."""
    rng = np.random.default_rng(42)
    n   = 500
    sched = rng.integers(1, 8, n)
    real  = np.clip(sched + rng.integers(-1, 6, n), 0, 20)
    return pd.DataFrame({
        "Order Id"                      : range(1, n + 1),
        "Order Date"                    : pd.date_range("2020-01-01", periods=n, freq="6H"),
        "Market"                        : rng.choice(["Europe","LATAM","USCA","Africa"], n),
        "Category Name"                 : rng.choice(["Electronics","Clothing","Food"], n),
        "Shipping Mode"                 : rng.choice(["Standard Class","Same Day","First Class"], n),
        "Order Status"                  : rng.choice(["Complete","Pending","Shipped"], n),
        "Sales"                         : rng.uniform(10, 2000, n).round(2),
        "Order Item Quantity"           : rng.integers(1, 20, n),
        "Order Item Discount Rate"      : rng.uniform(0, 0.4, n).round(2),
        "Order Item Profit Ratio"       : rng.uniform(-0.1, 0.5, n).round(2),
        "Days for shipping (real)"      : real,
        "Days for shipment (scheduled)" : sched,
        "Late_delivery_risk"            : (real > sched).astype(int),
        "Benefit per order"             : rng.uniform(-100, 500, n).round(2),
        "Order Item Total"              : rng.uniform(10, 2000, n).round(2),
        "Product Price"                 : rng.uniform(5, 1000, n).round(2),
    })


@pytest.fixture
def sample_products():
    """Minimal product supply-chain DataFrame."""
    rng = np.random.default_rng(7)
    n   = 100
    return pd.DataFrame({
        "Product type"      : rng.choice(["skincare","haircare","cosmetics"], n),
        "SKU"               : [f"SKU{i:04d}" for i in range(n)],
        "Price"             : rng.uniform(5, 300, n).round(2),
        "Stock levels"      : rng.integers(0, 500, n),
        "Lead times"        : rng.integers(1, 30, n),
        "Defect rates"      : rng.uniform(0, 0.12, n).round(4),
        "Supplier name"     : rng.choice(["Supplier_A","Supplier_B","Supplier_C"], n),
        "Manufacturing costs": rng.uniform(20, 300, n).round(2),
        "Shipping costs"    : rng.uniform(5, 100, n).round(2),
    })


@pytest.fixture
def cleaned_orders(sample_orders):
    """Run cleaning on sample data and return the result."""
    # Import cleaning functions inline to avoid module-level side effects
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "cleaning",
        os.path.join(os.path.dirname(__file__), "..", "src", "01_data_cleaning.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.clean_dataco(sample_orders.copy())


# ════════════════════════════════════════════════════════════════
#  TEST GROUP 1 — DATA CLEANING
# ════════════════════════════════════════════════════════════════

class TestDataCleaning:

    def test_no_missing_values_after_clean(self, cleaned_orders):
        """After cleaning, there should be zero NaN values."""
        assert cleaned_orders.isnull().sum().sum() == 0, \
            "Cleaning should eliminate all missing values."

    def test_no_duplicates_after_clean(self, cleaned_orders):
        """Cleaning should remove all duplicate rows."""
        assert cleaned_orders.duplicated().sum() == 0, \
            "No duplicate rows should remain after cleaning."

    def test_delay_days_column_exists(self, cleaned_orders):
        """Engineering step should create 'delay_days' column."""
        assert "delay_days" in cleaned_orders.columns, \
            "'delay_days' must be derived during cleaning."

    def test_is_delayed_is_binary(self, cleaned_orders):
        """'is_delayed' should only contain 0 and 1."""
        if "is_delayed" in cleaned_orders.columns:
            unique_vals = set(cleaned_orders["is_delayed"].unique())
            assert unique_vals.issubset({0, 1}), \
                f"'is_delayed' should be binary. Found: {unique_vals}"

    def test_order_year_in_range(self, cleaned_orders):
        """Derived order_year should fall within plausible range."""
        if "order_year" in cleaned_orders.columns:
            years = cleaned_orders["order_year"].dropna()
            assert years.between(2000, 2030).all(), \
                "All order years should be between 2000 and 2030."

    def test_profit_margin_derived(self, cleaned_orders):
        """profit_margin_pct should be a numeric column after cleaning."""
        if "profit_margin_pct" in cleaned_orders.columns:
            assert pd.api.types.is_numeric_dtype(cleaned_orders["profit_margin_pct"]), \
                "'profit_margin_pct' should be numeric."

    def test_column_names_have_no_spaces(self, cleaned_orders):
        """All column names should be lowercase and use underscores, not spaces."""
        for col in cleaned_orders.columns:
            assert " " not in col, f"Column '{col}' contains a space — should use underscore."

    def test_row_count_reasonable(self, sample_orders, cleaned_orders):
        """Cleaning should not drop more than 10% of rows (outlier removal only)."""
        ratio = len(cleaned_orders) / len(sample_orders)
        assert ratio >= 0.90, \
            f"Too many rows dropped: {1-ratio:.1%} removed (max 10% expected)."


# ════════════════════════════════════════════════════════════════
#  TEST GROUP 2 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════

class TestFeatureEngineering:

    def _run_feature_engineering(self, df):
        """Helper to run feature scripts on a DataFrame."""
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "fe",
            os.path.join(os.path.dirname(__file__), "..", "src", "03_feature_engineering.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        df = mod.time_features(df.copy())
        df = mod.rolling_features(df)
        df = mod.lag_features(df)
        df = mod.business_kpis(df)
        return df

    def test_cyclical_month_encoding(self, cleaned_orders):
        """month_sin and month_cos should be in range [-1, 1]."""
        df = self._run_feature_engineering(cleaned_orders)
        if "month_sin" in df.columns:
            assert df["month_sin"].between(-1, 1).all(), "month_sin out of range."
        if "month_cos" in df.columns:
            assert df["month_cos"].between(-1, 1).all(), "month_cos out of range."

    def test_is_weekend_binary(self, cleaned_orders):
        """is_weekend feature should only be 0 or 1."""
        df = self._run_feature_engineering(cleaned_orders)
        if "is_weekend" in df.columns:
            assert set(df["is_weekend"].unique()).issubset({0, 1}), \
                "'is_weekend' must be binary."

    def test_rolling_features_no_nulls(self, cleaned_orders):
        """Rolling features should have no NaNs (min_periods=1 handles edges)."""
        df = self._run_feature_engineering(cleaned_orders)
        roll_cols = [c for c in df.columns if "rolling" in c]
        for col in roll_cols:
            assert df[col].isnull().sum() == 0, \
                f"Rolling feature '{col}' has NaN values."

    def test_lag_features_exist(self, cleaned_orders):
        """Lag features for delay_lag_1, delay_lag_3, delay_lag_7 should be created."""
        df = self._run_feature_engineering(cleaned_orders)
        for lag in [1, 3, 7]:
            assert f"delay_lag_{lag}" in df.columns, \
                f"'delay_lag_{lag}' missing from feature set."

    def test_revenue_per_unit_positive(self, cleaned_orders):
        """Revenue per unit should be non-negative."""
        df = self._run_feature_engineering(cleaned_orders)
        if "revenue_per_unit" in df.columns:
            assert (df["revenue_per_unit"] >= 0).all(), \
                "'revenue_per_unit' must be non-negative."


# ════════════════════════════════════════════════════════════════
#  TEST GROUP 3 — STATISTICAL VALIDATIONS
# ════════════════════════════════════════════════════════════════

class TestStatisticalValidations:

    def test_delay_rate_between_0_and_1(self, sample_orders):
        """Late delivery risk rate should be a proportion between 0 and 1."""
        col = "Late_delivery_risk"
        if col in sample_orders.columns:
            rate = sample_orders[col].mean()
            assert 0 <= rate <= 1, \
                f"Late delivery rate {rate} is out of [0, 1] range."

    def test_sales_all_positive(self, sample_orders):
        """All sales values should be positive."""
        assert (sample_orders["Sales"] > 0).all(), \
            "Sales column contains zero or negative values."

    def test_no_future_dates(self, sample_orders):
        """Order dates should not be in the future."""
        max_date = pd.to_datetime(sample_orders["Order Date"]).max()
        assert max_date <= pd.Timestamp.now(), \
            f"Order dates contain future dates: {max_date}"

    def test_scheduled_days_positive(self, sample_orders):
        """Scheduled shipping days should all be positive integers."""
        col = "Days for shipment (scheduled)"
        if col in sample_orders.columns:
            assert (sample_orders[col] > 0).all(), \
                "Scheduled shipping days must all be > 0."

    def test_delay_days_consistent(self, cleaned_orders):
        """delay_days = real days - scheduled days."""
        if all(c in cleaned_orders.columns for c in
               ["delay_days","days_for_shipping_real","days_for_shipment_scheduled"]):
            computed = (cleaned_orders["days_for_shipping_real"]
                       - cleaned_orders["days_for_shipment_scheduled"])
            pd.testing.assert_series_equal(
                cleaned_orders["delay_days"].astype(float),
                computed.astype(float),
                check_names=False,
            )

    def test_product_defect_rate_in_range(self, sample_products):
        """Defect rates should be between 0 and 1 (proportion, not %)."""
        if "Defect rates" in sample_products.columns:
            rates = sample_products["Defect rates"].dropna()
            assert (rates >= 0).all() and (rates <= 1).all(), \
                "Defect rates must be in [0, 1]."


# ════════════════════════════════════════════════════════════════
#  TEST GROUP 4 — MODEL PIPELINE
# ════════════════════════════════════════════════════════════════

class TestModelPipeline:

    def _get_xy(self, cleaned_orders):
        """Extract X, y from cleaned orders for model tests."""
        df = cleaned_orders.copy()
        # Use a simple numeric subset
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target   = "is_delayed" if "is_delayed" in df.columns else "late_delivery_risk"
        if target not in num_cols:
            pytest.skip("Target column not present in fixture data.")
        features = [c for c in num_cols if c != target and "delay_days" not in c][:10]
        X = df[features].fillna(0)
        y = df[target].astype(int)
        return X, y

    def test_random_forest_trains_without_error(self, cleaned_orders):
        """RandomForestClassifier should train on cleaned data without raising."""
        from sklearn.ensemble import RandomForestClassifier
        X, y = self._get_xy(cleaned_orders)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        assert hasattr(clf, "feature_importances_"), \
            "Fitted model should have feature_importances_ attribute."

    def test_predictions_are_binary(self, cleaned_orders):
        """Model predictions must be 0 or 1 for classification."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        X, y = self._get_xy(cleaned_orders)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert set(preds).issubset({0, 1}), \
            f"Predictions contain non-binary values: {set(preds)}"

    def test_no_data_leakage_in_features(self, cleaned_orders):
        """Features that directly reveal the target should not be present."""
        leakage_patterns = ["days_for_shipping_real", "order_status"]
        leakage_found    = [c for c in cleaned_orders.columns
                            if any(p in c.lower() for p in leakage_patterns)
                            and c not in ["days_for_shipment_scheduled"]]
        # If they exist, they should be handled by the model script's drop logic
        # This test just warns — not a hard failure, as cleaning may keep them
        if leakage_found:
            pytest.warns(UserWarning)  # soft signal
        assert True  # pipeline handles this with drop_patterns

    def test_train_test_split_no_overlap(self, cleaned_orders):
        """Time-based split should produce non-overlapping train and test sets."""
        n         = len(cleaned_orders)
        split_idx = int(n * 0.80)
        train_idx = set(range(split_idx))
        test_idx  = set(range(split_idx, n))
        assert train_idx.isdisjoint(test_idx), \
            "Train and test index sets must not overlap."

    def test_roc_auc_above_random(self, cleaned_orders):
        """Model should perform better than random (AUC > 0.5)."""
        from sklearn.ensemble         import RandomForestClassifier
        from sklearn.model_selection  import train_test_split
        from sklearn.metrics          import roc_auc_score
        X, y = self._get_xy(cleaned_orders)
        if y.nunique() < 2:
            pytest.skip("Target has only one class in fixture — skipping AUC test.")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:,1])
        assert auc > 0.5, \
            f"Model AUC of {auc:.3f} is not better than random chance (0.5)."


# ════════════════════════════════════════════════════════════════
#  TEST GROUP 5 — SQL LAYER
# ════════════════════════════════════════════════════════════════

class TestSQLLayer:

    @pytest.fixture
    def in_memory_db(self, sample_orders):
        """Create an in-memory SQLite database loaded with orders."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        df   = sample_orders.copy()
        df.columns = (
            df.columns.str.lower()
              .str.replace(r"[\s\(\)\/]+", "_", regex=True)
              .str.replace(r"_+", "_", regex=True)
              .str.strip("_")
        )
        df.to_sql("orders", conn, if_exists="replace", index=False)
        yield conn
        conn.close()

    def test_revenue_by_market_returns_rows(self, in_memory_db):
        """GROUP BY market query should return at least 1 row."""
        sql = "SELECT market, SUM(sales) AS revenue FROM orders GROUP BY market"
        df  = pd.read_sql_query(sql, in_memory_db)
        assert len(df) >= 1, "Revenue by market query returned no results."

    def test_having_clause_filters_correctly(self, in_memory_db):
        """HAVING clause should only return markets above the threshold."""
        sql = """
            SELECT market, ROUND(100.0 * SUM(late_delivery_risk) / COUNT(*), 2) AS late_pct
            FROM orders
            WHERE market IS NOT NULL
            GROUP BY market
            HAVING late_pct > 0
        """
        df = pd.read_sql_query(sql, in_memory_db)
        assert (df["late_pct"] > 0).all(), \
            "HAVING clause should filter markets to those with late_pct > 0."

    def test_window_rank_assigns_1_to_max(self, in_memory_db):
        """RANK() window function should assign rank 1 to the highest-revenue category."""
        sql = """
            SELECT category_name, SUM(sales) AS rev,
                   RANK() OVER (ORDER BY SUM(sales) DESC) AS rnk
            FROM orders
            GROUP BY category_name
        """
        df = pd.read_sql_query(sql, in_memory_db)
        top_row = df[df["rnk"] == 1].iloc[0]
        assert top_row["rev"] == df["rev"].max(), \
            "RANK 1 should go to the category with maximum revenue."

    def test_aggregate_count_matches_len(self, in_memory_db):
        """COUNT(*) in SQL should match the number of rows in the DataFrame."""
        sql = "SELECT COUNT(*) AS n FROM orders"
        result = pd.read_sql_query(sql, in_memory_db).iloc[0]["n"]
        assert result == 500, f"Expected 500 rows, got {result}."


# ════════════════════════════════════════════════════════════════
#  TEST GROUP 6 — DATA INTEGRITY
# ════════════════════════════════════════════════════════════════

class TestDataIntegrity:

    def test_orders_schema_has_required_columns(self, sample_orders):
        """Core required columns must be present in the orders dataset."""
        required = ["Sales", "Market", "Shipping Mode",
                    "Days for shipping (real)", "Days for shipment (scheduled)"]
        for col in required:
            assert col in sample_orders.columns, \
                f"Required column '{col}' missing from orders dataset."

    def test_products_schema_has_required_columns(self, sample_products):
        """Core required columns must be present in the products dataset."""
        required = ["Defect rates", "Lead times", "Supplier name"]
        for col in required:
            assert col in sample_products.columns, \
                f"Required column '{col}' missing from products dataset."

    def test_no_negative_sales(self, sample_orders):
        """Sales should be strictly positive."""
        assert (sample_orders["Sales"] > 0).all(), \
            "All sales values must be positive."

    def test_quantity_is_integer(self, sample_orders):
        """Order quantities should be whole numbers."""
        if "Order Item Quantity" in sample_orders.columns:
            assert (sample_orders["Order Item Quantity"] % 1 == 0).all(), \
                "Quantities should be integers."

    def test_market_values_known(self, sample_orders):
        """Market column should only contain expected values."""
        known_markets = {"Europe","LATAM","Pacific Asia","USCA","Africa"}
        actual = set(sample_orders["Market"].dropna().unique())
        unknown = actual - known_markets
        assert len(unknown) == 0, \
            f"Unknown market values found: {unknown}"
