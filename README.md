<div align="center">

# Supply Chain Intelligence & Risk Analytics

### A Production-Grade, End-to-End Data Analytics System

**180,000+ Orders · 3 Datasets · 30+ Features · 3 ML Models · 90% AUC · 15 SQL Queries · Live Dashboard · Automated PDF Reports · Full CI/CD**

<br/>

[Quick Start](#quick-start) &nbsp;·&nbsp; [Results](#key-results) &nbsp;·&nbsp; [Architecture](#architecture) &nbsp;·&nbsp; [Project Structure](#project-structure) &nbsp;·&nbsp; [Tests](#tests) &nbsp;·&nbsp; [Resume Bullets](#resume-bullet-points)

---

</div>


## Overview

**Business Question:** *Which supply chain factors cause delivery delays — and can we predict them before they happen?*

This project is a complete, production-grade analytics system built to answer that question. It ingests raw operational data and — through a fully automated pipeline — cleans it, engineers features, trains machine learning models, runs SQL analytics, generates an interactive dashboard, and produces a 10-page executive PDF report.

This is not a notebook. It is a **7-stage pipeline** that mirrors how real data engineering and analytics teams operate at scale.

---

## Key Results

<div align="center">

| Metric | Value |
|:---|:---:|
| Total Orders Analysed | **180,000+** |
| Late Delivery Rate Discovered | **54.8%** |
| Best Model ROC-AUC (Gradient Boosting) | **~90%** |
| Features Engineered | **30+** |
| SQL Queries Written | **15** |
| Charts Generated | **13** |
| Forecast Horizon | **180 days** |
| Unit Tests | **35** |
| PDF Report Pages | **10** |

</div>

---

## Architecture

```
+------------------------------------------------------------------+
|                         RAW DATA SOURCES                         |
|    DataCo Orders (180K+)  ·  Supplier Metrics  ·  UN Trade       |
+------------------------------+-----------------------------------+
                               |
                               v
              +-----------------------------+
              |   Stage 01  DATA CLEANING   |
              |   - Schema normalisation    |
              |   - Missing value imputation|
              |   - Outlier removal         |
              |   - Date parsing            |
              +--------------+--------------+
                             |
                             v
              +-----------------------------+
              |   Stage 02  EDA             |
              |   - 7 publication charts    |
              |   - Summary statistics      |
              |   - Correlation heatmap     |
              +--------------+--------------+
                             |
                             v
              +-----------------------------+
              |   Stage 03  FEATURE ENG.    |
              |   - Cyclical sin/cos encod. |
              |   - Lag features (1, 3, 7)  |
              |   - Rolling 7d / 30d stats  |
              |   - Risk scores per group   |
              +--------------+--------------+
                             |
                             v
              +-----------------------------+
              |   Stage 04  ML MODELLING    |
              |   - Logistic Regression     |
              |   - Random Forest           |
              |   - Gradient Boosting       |
              |   - Prophet 180-day fcst.   |
              +--------------+--------------+
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
+-----------------+  +----------------+  +--------------+
|   Stage 05      |  |   Stage 06     |  |  Stage 07    |
|   STREAMLIT     |  |   SQL LAYER    |  |  PDF REPORT  |
|   DASHBOARD     |  |   15 Queries   |  |  10 Pages    |
|   5 tabs        |  |   CTEs, Windows|  |  Auto-gen.   |
|   Live predict. |  |   JOINs, LAG   |  |  ReportLab   |
+-----------------+  +----------------+  +--------------+
```

---

## Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:---|:---|:---|
| Language | Python 3.11 | Core pipeline |
| Data Processing | Pandas, NumPy | Cleaning, transformation, feature engineering |
| Machine Learning | Scikit-learn | Pipelines, Random Forest, Gradient Boosting |
| Forecasting | Facebook Prophet | 180-day time-series revenue forecast |
| Static Visualisation | Matplotlib, Seaborn | EDA charts exported as PNG |
| Interactive Visualisation | Plotly | Dashboard charts, 3D market trajectory |
| Dashboard | Streamlit | Live web interface with real-time predictor |
| Database | SQLite (PostgreSQL-compatible) | SQL analytics warehouse |
| PDF Reporting | ReportLab | Automated 10-page executive report |
| Testing | Pytest + pytest-cov | 35 unit and integration tests |
| CI/CD | GitHub Actions | Automated testing, linting, pipeline, and PDF on every push |

</div>

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip
- Git

### 1. Clone and Install

```bash
git clone https://github.com/JIYA1220/supply-chain-project.git
cd supply-chain-project
pip install -r requirements.txt
```

### 2. Add Datasets (Optional)

Download from Kaggle and place in `data/raw/`:

| Dataset | Source | Filename |
|:---|:---|:---|
| DataCo Smart Supply Chain | [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) | `DataCoSupplyChainDataset.csv` |
| Supply Chain Analysis | [Kaggle](https://www.kaggle.com/datasets/harshsingh2209/supply-chain-analysis) | `supply_chain_data.csv` |
| Global Commodity Trade | [Kaggle](https://www.kaggle.com/datasets/unitednations/global-commodity-trade-statistics) | `commodity_trade_statistics_data.csv` |

> **Note:** Datasets are optional. All scripts auto-generate realistic synthetic data if the files are absent, so the full pipeline can be executed immediately without a Kaggle account.

### 3. Run the Pipeline

```bash
# Stage 1 — Clean and merge all 3 datasets
python src/01_data_cleaning.py

# Stage 2 — Generate 7 EDA charts
python src/02_exploratory_analysis.py

# Stage 3 — Engineer 30+ ML features
python src/03_feature_engineering.py

# Stage 4 — Train 3 models and generate 180-day forecast
python src/04_predictive_model.py

# Stage 5 — Run 15 SQL analytics queries
python src/06_sql_analytics.py

# Stage 6 — Generate 10-page PDF report
python src/07_pdf_report.py

# Stage 7 — Launch interactive dashboard
streamlit run src/05_dashboard.py
```

Open **http://localhost:8501** in your browser to access the dashboard.

### 4. Run Tests

```bash
# Run all 35 tests with verbose output
pytest tests/ -v

# With code coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Project Structure

```
supply-chain-project/
|
+-- .github/
|   +-- workflows/
|       +-- ci.yml                    <- GitHub Actions CI/CD (4 jobs)
|
+-- data/
|   +-- raw/                          <- Place Kaggle CSVs here
|   +-- cleaned/                      <- Auto-generated after Stage 01
|   +-- final/                        <- ML-ready dataset after Stage 03
|
+-- outputs/
|   +-- charts/                       <- 13 high-quality PNG charts
|   +-- reports/                      <- Model pkl, forecast CSV, PDF report
|   +-- sql_results/                  <- 15 query result CSVs
|
+-- src/
|   +-- 01_data_cleaning.py           <- Load, clean and merge 3 datasets
|   +-- 02_exploratory_analysis.py    <- 7-chart deep EDA
|   +-- 03_feature_engineering.py     <- 30+ features (rolling, lag, risk)
|   +-- 04_predictive_model.py        <- 3 ML models + Prophet forecast
|   +-- 05_dashboard.py               <- 5-tab Streamlit dashboard
|   +-- 06_sql_analytics.py           <- 15 SQL queries (CTEs, windows, joins)
|   +-- 07_pdf_report.py              <- 10-page automated PDF report
|
+-- tests/
|   +-- test_pipeline.py              <- 35 unit and integration tests
|
+-- requirements.txt
+-- README.md
```

---

## Pipeline Deep Dive

<details>
<summary><b>Stage 01 — Data Cleaning</b></summary>

<br/>

- Loads 3 raw datasets: DataCo orders, product supply chain metrics, and global trade statistics
- Standardises all column names to `snake_case` via regex substitution
- Parses all date columns and normalises to ISO 8601 format
- Imputes missing values using **median** for numeric fields and **mode** for categorical fields
- Removes duplicate rows and extreme outliers (1st–99th percentile)
- Derives core business features: `delay_days`, `is_delayed`, `delay_severity`, `profit_margin_pct`
- Saves 3 cleaned CSVs to `data/cleaned/`

</details>

<details>
<summary><b>Stage 02 — Exploratory Data Analysis</b></summary>

<br/>

Generates **7 publication-quality charts** saved to `outputs/charts/`:

| Chart | Description |
|:---|:---|
| `02_sales_distribution` | Histogram and box plot segmented by market |
| `03_delay_analysis` | 4-panel delay deep dive |
| `04_monthly_trends` | Revenue and order volume over time |
| `05_correlation_heatmap` | Feature correlation matrix |
| `06_supplier_risk` | Risk scores and defect-vs-lead-time scatter |
| `07_global_trade` | Country rankings and export/import trends |
| `08_profit_margin` | Margin by product category and market (violin plot) |

</details>

<details>
<summary><b>Stage 03 — Feature Engineering</b></summary>

<br/>

Builds **30+ ML-ready features** across 6 groups:

**Group A — Temporal Features**
`is_weekend`, `is_month_end`, `is_quarter_end`, `week_of_year`, `day_of_year`

**Group B — Cyclical Encoding** *(prevents January/December discontinuity)*
`month_sin`, `month_cos`, `dow_sin`, `dow_cos`

**Group C — Rolling Statistics**
7-day and 30-day windows: `mean_delay`, `std_delay`, `sum_sales`, `mean_profit`

**Group D — Lag Features** *(historical context for the model)*
`delay_lag_1`, `delay_lag_3`, `delay_lag_7`, `delay_change`, `delay_acceleration`

**Group E — Risk Scores**
`delay_rate_market`, `delay_rate_ship_mode`, `avg_delay_category`, `avg_delay_region`

**Group F — Business KPIs**
`revenue_per_unit`, `is_profitable`, `heavy_discount`, `price_ratio`

</details>

<details>
<summary><b>Stage 04 — Predictive Modelling</b></summary>

<br/>

**Anti-Leakage Protocol:** Future-peek variables such as `days_for_shipping_real` and `order_status_Shipped` were removed. Without this, accuracy inflates to 1.0. With rigorous features only, the real AUC is **~90%**.

**Time-Based Split:** The first 80% of chronologically ordered data is used for training; the last 20% for testing. No shuffling is applied — this prevents future data from leaking into the training set.

**3 Models trained within Scikit-learn Pipelines:**

| Model | Accuracy | F1 | ROC-AUC | CV-AUC |
|:---|:---:|:---:|:---:|:---:|
| Logistic Regression | ~72% | ~71% | ~78% | ~77% |
| Random Forest | ~83% | ~82% | ~89% | ~88% |
| **Gradient Boosting** | **~85%** | **~84%** | **~90%** | **~90%** |

**Prophet Forecast:** Multiplicative seasonality (yearly and weekly components), 180-day horizon with 95% confidence intervals.

**Outputs:** 6 charts, classification report TXT, serialised model PKL, and forecast CSV.

</details>

<details>
<summary><b>Stage 06 — SQL Analytics Layer</b></summary>

<br/>

15 production-grade SQL queries loaded into a SQLite warehouse (`supply_chain.db`), fully PostgreSQL-compatible:

| # | Query | Concepts Used |
|:---|:---|:---|
| 01 | Revenue and profit by market | `GROUP BY`, `SUM`, `AVG` |
| 02 | High late-delivery markets | `HAVING` |
| 03 | Category rank within market | `RANK() OVER (PARTITION BY)` |
| 04 | Top category per market | `ROW_NUMBER()`, CTE |
| 05 | Cumulative revenue by year | `SUM() OVER (ROWS BETWEEN)` |
| 06 | Year-over-year growth | `LAG()` window function |
| 07 | Supplier risk segments | CTE + `CASE WHEN` |
| 08 | Orders joined with products | `LEFT JOIN` across tables |
| 09 | Above-average orders | Correlated subquery |
| 10 | Shipping mode KPIs | Multi-metric aggregation |
| 11 | Top export commodities | Trade table aggregation |
| 12 | Sales quartile analysis | `NTILE(4)` window function |
| 13 | Monthly demand with LEAD | `LEAD()` window function |
| 14 | Profitability buckets | `CASE WHEN` tiering |
| 15 | Executive KPI summary | Single-row full KPI summary |

</details>

<details>
<summary><b>Stage 07 — Automated PDF Report</b></summary>

<br/>

**10-page executive business report** auto-generated with ReportLab:

1. Cover Page
2. Executive Summary with 5 KPI Cards
3. Delay and Shipping Analysis (embedded charts)
4. Supplier Risk Analysis and leaderboard table
5. Revenue and Profitability by market and category
6. Global Trade Context
7. Predictive Model Performance (embedded charts and metrics table)
8. 180-Day Revenue Forecast
9. Strategic Recommendations (4 priority-ranked action cards)
10. Methodology and Tech Stack

The report includes dynamic narrative logic: it reads the processed data and auto-generates contextual text, for example automatically naming the worst-performing market by late delivery rate.

</details>

---

## Tests

**35 tests across 6 groups** — run with `pytest tests/ -v`

<div align="center">

| Group | Tests | What Is Validated |
|:---|:---:|:---|
| Data Cleaning | 8 | No nulls, no duplicates, column derivation, row count preservation |
| Feature Engineering | 5 | Cyclical range [-1, 1], binary flags, rolling NaN-free, lag existence |
| Statistical Validation | 6 | Delay rate in [0, 1], positive sales, no future dates, type consistency |
| Model Pipeline | 5 | Training completes, binary predictions, no leakage, AUC > 0.5 |
| SQL Layer | 4 | GROUP BY rows, HAVING filter, RANK() correctness, COUNT() accuracy |
| Data Integrity | 6 | Schema presence, value ranges, integer quantities, known categories |

</div>

---

## CI/CD Pipeline

Every push to `main` or `develop` triggers 4 automated jobs:

```
Push to GitHub
      |
      +-- Job 1: Unit and Integration Tests (35 tests + coverage report)
      |
      +-- Job 2: Code Quality Check (flake8 linting)
      |
      +-- Job 3: Pipeline Smoke Test (runs all 7 scripts end-to-end)
      |
      +-- Job 4: PDF Report Generation (uploads as build artifact)
```

The live pipeline status is available under the **Actions** tab of this repository.

---


## Outputs Reference

After executing the full pipeline, the following files will be generated:

```
outputs/
|
+-- charts/
|   +-- 02_sales_distribution.png
|   +-- 03_delay_analysis.png
|   +-- 04_monthly_trends.png
|   +-- 05_correlation_heatmap.png
|   +-- 06_supplier_risk.png
|   +-- 07_global_trade.png
|   +-- 08_profit_margin.png
|   +-- 09_model_comparison.png
|   +-- 10_roc_curves.png
|   +-- 11_confusion_matrices.png
|   +-- 12_feature_importance.png
|   +-- 13_sales_forecast.png
|
+-- reports/
|   +-- best_model.pkl
|   +-- best_model_report.txt
|   +-- sales_forecast.csv
|   +-- feature_summary.csv
|   +-- dataco_stats.csv
|   +-- Supply_Chain_Analysis_Report.pdf
|
+-- sql_results/
    +-- 01_revenue_by_market.csv
    +-- 02_high_delay_markets.csv
    +-- ...
    +-- 15_executive_summary.csv
```

---

## Connect

Built by **Jiya** 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR_PROFILE)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/JIYA1220)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)

*If this project was useful to you, consider starring the repository.*

---

## License

MIT License — free to use, fork, learn from, and build upon.

```
MIT License
Copyright (c) 2026 Jiya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software.
```
