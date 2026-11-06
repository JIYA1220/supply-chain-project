# Supply Chain Delivery Delay Analytics

A seven-script Python pipeline that cleans order data, explores it, engineers features, trains delay classifiers, runs SQL analytics, builds a Streamlit dashboard and writes a PDF report.

## Read this first: the data in this repo is synthetic

The pipeline expects the Kaggle DataCo supply chain dataset in `data/raw/`. That file is not committed. When it is missing, `src/01_data_cleaning.py` generates a **synthetic** stand-in (49,000 orders, random values with a fixed seed), and every number, chart and report committed here comes from that synthetic data.

Because the synthetic labels are random, the models cannot learn anything. That is the honest result of the committed run (see below). Results on the real dataset have not been reproduced in this repository.

## Business question

Which supply chain factors are associated with late deliveries, and can late orders be flagged before they happen?

## Data sources

| Dataset | Source | Used for |
|:---|:---|:---|
| DataCo Smart Supply Chain | [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis) (check the dataset page for its license) | Orders and delays (not committed) |
| Supply Chain Analysis | [Kaggle](https://www.kaggle.com/datasets/harshsingh2209/supply-chain-analysis) | Product and supplier metrics (100 rows, committed in `data/raw/`) |
| Global Commodity Trade Statistics | [Kaggle](https://www.kaggle.com/datasets/unitednations/global-commodity-trade-statistics) | Trade context (not committed) |

If the large files are absent, the cleaning script falls back to synthetic data, and `src/data_loader.py` can try to fetch them from Google Drive.

## What was done

1. `01_data_cleaning.py`: normalise column names, parse dates, impute missing values, drop duplicates and outliers, derive `delay_days`, `is_delayed`, `profit_margin_pct`.
2. `02_exploratory_analysis.py`: 7 EDA charts.
3. `03_feature_engineering.py`: time, rolling, lag, per-group delay-rate, business KPI features and one-hot encoding (101 columns in the ML table).
4. `04_predictive_model.py`: Logistic Regression, Random Forest and Gradient Boosting with a time-ordered 80/20 split. Identifier and leakage columns (delay days, real shipping days, order status, totals, lag and rolling delay features) are dropped before training (72 numeric features remain). Optional Prophet sales forecast.
5. `05_dashboard.py`: Streamlit dashboard with 6 tabs (Performance, Disruption, Suppliers, Risk Engine, SQL, Executive Report).
6. `06_sql_analytics.py`: 16 SQL queries on a SQLite database (GROUP BY, HAVING, CTEs, window functions, joins, subqueries). Results are in `outputs/sql_results/`.
7. `07_pdf_report.py`: generates an 11-page PDF report with ReportLab.

## Results of the committed (synthetic) run

| Item | Value | Produced by |
|:---|:---|:---|
| Orders | 49,000 | `data/cleaned/dataco_clean.csv` |
| Share of orders flagged late | 82.7% | `outputs/sql_results/15_executive_summary.csv` |
| Logistic Regression ROC-AUC | 0.49 | `src/04_predictive_model.py` |
| Random Forest ROC-AUC | 0.50 | `src/04_predictive_model.py` |
| Gradient Boosting ROC-AUC | 0.52 | `src/04_predictive_model.py` |
| Unit tests | 33 passing | `pytest tests` |

Late rates are about 82 to 83% in every market and shipping mode, and AUC is about 0.5 for every model. This is expected for randomly generated labels and is not evidence about real supply chains. The Gradient Boosting accuracy of 0.83 only reflects the 83% majority class (see `outputs/reports/best_model_report.txt`: recall on on-time orders is 0).

## How to run (Windows)

```bat
git clone https://github.com/JIYA1220/supply-chain-project.git
cd supply-chain-project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python src\01_data_cleaning.py
python src\02_exploratory_analysis.py
python src\03_feature_engineering.py
python src\04_predictive_model.py
python src\06_sql_analytics.py
python src\07_pdf_report.py
streamlit run src\05_dashboard.py
pytest tests -q
```

To use the real data, download the DataCo CSV from Kaggle into `data/raw/DataCoSupplyChainDataset.csv` and re-run the scripts in order. Then report whatever numbers you actually get.

## Limitations

- Committed outputs come from synthetic data; the models have no predictive signal on it.
- The pipeline scores orders by delay probability, but there is no scheduler, no live data feed and no "advance warning" window. Nothing here measures how many hours ahead a risky shipment is flagged.
- The synthetic order dates run into the year 2049 because of how they are generated.
- Prophet needs `cmdstanpy==1.1.0` with the pinned versions; the forecast is only an illustration on synthetic sales.
- `main.py` is an unused IDE template.
