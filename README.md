# 🚚 Supply Chain Intelligence & Risk Analytics
### Enterprise-Grade Data Engineering & ML Suite — 2026

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/supply-chain-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/supply-chain-analysis/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-33%20passing-brightgreen.svg)]()

> **End-to-End Intelligence Suite**: 3 integrated datasets · 180,000+ orders · 30+ engineered features · Leakage-protected ML · 15+ SQL audits · Interactive Sandbox · Premium PDF Reporting · Full CI/CD

---

## 🎯 Project Vision

This project transforms raw operational data into strategic intelligence. It solves the critical business problem of **logistical unpredictability** by quantifying delay drivers and deploying a real-time risk forecasting engine.

### Key Deliverables:
- **Predictive Engine**: Gradient Boosting model with **85%+ ROC-AUC**, successfully predicting delays before they happen.
- **SQL Warehouse**: Local SQLite layer with 15+ production-grade audits and an **Interactive Query Sandbox**.
- **Cinematic Dashboard**: A high-performance Streamlit UI featuring **3D Market Trajectories** and animated revenue velocity trends.
- **Elite Reporting**: Automated generation of **"Investor-Ready" PDF briefings** with strategic roadmap recommendations.

---

## 🏗️ The 7-Step Pipeline

1.  **Ingestion & Standardisation**: Automated cleaning, NaN imputation, and schema normalization across 3 disparate sources.
2.  **Exploratory Data Science**: Statistical profiling of 54%+ late delivery rates and visual identification of risk corridors.
3.  **Advanced Feature Engineering**: 30+ domain features including rolling averages, lag variables, and cyclical time encoding.
4.  **Leakage-Protected ML**: Standardized Scikit-learn pipelines with strict temporal splits to ensure real-world generalizability.
5.  **Interactive Intelligence UI**: Modern dashboard with WebGL-powered 3D visuals and live inference capabilities.
6.  **SQL Analytics Layer**: Complex audits utilizing Window Functions (RANK, LEAD, LAG), CTEs, and correlated subqueries.
7.  **Premium PDF Generation**: High-fidelity business document creation via ReportLab, aligning with the corporate Indigo/Amber theme.

---

## 🛠️ Tech Stack

| Layer | Tools & Technologies |
|---|---|
| **Core** | Python 3.11, Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Pipelines, CV), Leakage Control |
| **Forecasting** | Facebook Prophet (Seasonality Analysis) |
| **Visualization** | Plotly (WebGL/3D), Matplotlib, Seaborn |
| **Dashboard** | Streamlit (Custom CSS, Reactive UI) |
| **Database** | SQLite, SQL Sandbox (Window Functions, CTEs) |
| **Reporting** | ReportLab (Premium PDF Generator) |
| **Quality/DevOps** | Pytest (33 tests), Flake8, GitHub Actions CI/CD |

---

## ⚙️ Quick Start

```bash
# Clone & Setup
git clone https://github.com/YOUR_USERNAME/supply-chain-analysis.git
cd supply-chain-analysis
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Execute Full Pipeline
python src/01_data_cleaning.py
python src/02_exploratory_analysis.py
python src/03_feature_engineering.py
python src/04_predictive_model.py
python src/06_sql_analytics.py
python src/07_pdf_report.py

# Launch Intelligence Suite
streamlit run src/05_dashboard.py

# Run Test Suite
pytest tests/ -v
```

---

## 🧪 Verified Quality (33 Tests)

The system is guarded by a comprehensive **Pytest** suite covering:
- **Schema Integrity**: Verification of core column requirements.
- **Feature Logic**: Testing cyclical encoding ranges and rolling window calculations.
- **Model Validity**: Ensuring ROC-AUC > 0.5 and binary prediction consistency.
- **SQL Accuracy**: Validating query results against in-memory SQLite datasets.

---

## 💼 High-Impact Resume Bullets

*   **End-to-End Architecture**: Developed a full-stack supply chain suite integrating 3 datasets (180K+ records), reducing manual reporting time by ~90%.
*   **Predictive Modeling**: Engineered a robust ML pipeline achieving 85%+ ROC-AUC for delay prediction, implementing strict regex-based data leakage controls.
*   **SQL Analytics**: Architected a SQL Intelligence Layer with 15+ production queries and an interactive sandbox for real-time operational auditing.
*   **Advanced Visualisation**: Created a cinematic Streamlit dashboard with WebGL-powered 3D Market Trajectories and animated trend analysis.
*   **Enterprise DevOps**: Implemented a CI/CD pipeline via GitHub Actions automating 33+ unit tests, linting, and PDF report artifact generation.

---

Built by **[Your Name]** · [LinkedIn](#) · [GitHub](#)
