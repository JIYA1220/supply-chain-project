"""
╔══════════════════════════════════════════════════════════════╗
║   SUPPLY CHAIN DISRUPTION ANALYSIS                           ║
║   SCRIPT 4 OF 5 — PREDICTIVE MODELLING                      ║
╠══════════════════════════════════════════════════════════════╣
║  What this script does:                                      ║
║  • Train/test split (time-based — no data leakage)           ║
║  • Model 1: Logistic Regression (baseline)                   ║
║  • Model 2: Random Forest Classifier                         ║
║  • Model 3: Gradient Boosting (XGBoost-style)               ║
║  • Compare all models with full metrics                      ║
║  • SHAP feature importance (explainability)                  ║
║  • Confusion matrix & ROC curve charts                       ║
║  • Sales forecasting with Prophet                            ║
║  • Save best model to outputs/best_model.pkl                 ║
║                                                              ║
║  Run after: 03_feature_engineering.py                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import warnings
import pickle
import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from colorama import Fore, Style, init

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score, f1_score
)
from sklearn.pipeline          import Pipeline
from sklearn.inspection        import permutation_importance

warnings.filterwarnings("ignore")
init(autoreset=True)


# ════════════════════════════════════════════════════════════════
#  STYLE
# ════════════════════════════════════════════════════════════════

BG     = "#0d1117"
PANEL  = "#161b22"
GRID   = "#21262d"
TEXT   = "#c9d1d9"
ACCENT = "#58a6ff"
PAL    = ["#58a6ff","#3fb950","#f78166","#d2a8ff","#ffa657","#56d364"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor":  GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "text.color": TEXT,
    "grid.color": GRID, "grid.linestyle": "--", "grid.alpha": 0.6,
    "legend.facecolor": PANEL, "legend.edgecolor": GRID,
    "figure.dpi": 130, "font.family": "DejaVu Sans",
})

# Path helper
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == 'src' else SCRIPT_DIR

def get_path(rel_path):
    # Try relative to CWD first, then relative to script dir
    if os.path.exists(rel_path):
        return rel_path
    
    # If running from root, and file is in src/data/...
    src_path = os.path.join(SCRIPT_DIR, rel_path)
    if os.path.exists(src_path):
        return src_path
    
    return rel_path

def section(t): print(f"\n{Fore.CYAN}{'═'*62}\n  {t}\n{'═'*62}{Style.RESET_ALL}")
def ok(m):      print(f"{Fore.GREEN}  ✔  {m}{Style.RESET_ALL}")
def note(m):    print(f"{Fore.YELLOW}  ℹ  {m}{Style.RESET_ALL}")

def save(name):
    path = get_path(f"outputs/charts/{name}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close()
    ok(f"Chart saved → {path}")

def find(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    for c in candidates:
        hits = [col for col in df.columns
                if c.replace("_","") in col.lower().replace("_","")]
        if hits: return hits[0]
    return None


# ════════════════════════════════════════════════════════════════
#  LOAD & PREPARE
# ════════════════════════════════════════════════════════════════

def load_and_prepare():
    section("Loading ML-Ready Dataset")
    data_path = get_path("../data/final/ml_ready.csv")
    df = pd.read_csv(data_path, low_memory=False)
    df.fillna(0, inplace=True)
    ok(f"Loaded  {df.shape[0]:,} rows × {df.shape[1]} features")

    # Target: predict whether a delivery will be late
    target = find(df, ["is_delayed","late_delivery_risk"])
    if not target:
        raise ValueError("Target column 'is_delayed' not found. Re-run script 01 and 03.")

    note(f"  Target column: '{target}'")
    note(f"  Class balance:\n{df[target].value_counts().to_string()}")

    # Drop identifiers and leakage columns (THESE CAUSE 1.0 ACCURACY)
    drop_patterns = [
        "order_id","id","delay_days","delay_severity","benefit",
        "shipping_real", "order_status", "total", "profit_ratio", "is_profitable",
        "late_delivery_risk", "delay_rolling", "delay_lag", "delay_change", "delay_acceleration"
    ]
    drop_cols = [c for c in df.columns
                 if any(p in c.lower() for p in drop_patterns)]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    X = df.drop(columns=[target])
    y = df[target].astype(int)

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)

    ok(f"Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
    return X, y, target


# ════════════════════════════════════════════════════════════════
#  TRAIN / TEST SPLIT
#  Using first 80% of data as train, last 20% as test
#  (time-based split → prevents future data leaking into training)
# ════════════════════════════════════════════════════════════════

def split(X, y):
    section("Train / Test Split (Time-Based — 80 / 20)")
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    note(f"  Train: {X_train.shape[0]:,} samples")
    note(f"  Test:  {X_test.shape[0]:,} samples")
    return X_train, X_test, y_train, y_test


# ════════════════════════════════════════════════════════════════
#  TRAIN THREE MODELS
# ════════════════════════════════════════════════════════════════

def train_models(X_train, X_test, y_train, y_test):
    section("Training 3 Models")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=50, max_depth=10, min_samples_leaf=5,
                random_state=42, class_weight="balanced", n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42
            )),
        ]),
    }

    results = {}
    for name, model in models.items():
        note(f"  Training {name} ...")
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted")
        auc  = roc_auc_score(y_test, y_proba)
        cv   = cross_val_score(model, X_train, y_train, cv=2, scoring="roc_auc")

        results[name] = {
            "model"  : model,
            "y_pred" : y_pred,
            "y_proba": y_proba,
            "acc"    : acc,
            "f1"     : f1,
            "auc"    : auc,
            "cv_mean": cv.mean(),
            "cv_std" : cv.std(),
        }
        ok(f"  {name:<25}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  CV-AUC={cv.mean():.3f}±{cv.std():.3f}")

    return results


# ════════════════════════════════════════════════════════════════
#  CHART — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════

def chart_model_comparison(results):
    section("Chart: Model Comparison")

    names   = list(results.keys())
    metrics = ["acc","f1","auc","cv_mean"]
    labels  = ["Accuracy","F1 Score","ROC-AUC","CV AUC (5-fold)"]

    x  = np.arange(len(names))
    w  = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Model Performance Comparison", fontsize=16, color=TEXT, fontweight="bold")

    for i, (m, lbl) in enumerate(zip(metrics, labels)):
        vals = [results[n][m] for n in names]
        bars = ax.bar(x + i*w, vals, width=w, label=lbl,
                      color=PAL[i], edgecolor=BG, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", color=TEXT, fontsize=8, padding=2)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y")

    plt.tight_layout()
    save("09_model_comparison")


# ════════════════════════════════════════════════════════════════
#  CHART — ROC CURVES
# ════════════════════════════════════════════════════════════════

def chart_roc(results, y_test):
    section("Chart: ROC Curves")

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("ROC Curves — All Models", fontsize=15, color=TEXT, fontweight="bold")

    ax.plot([0,1],[0,1], "w--", linewidth=1, label="Random (AUC=0.50)")

    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        ax.plot(fpr, tpr, color=PAL[i], linewidth=2.5,
                label=f"{name}  (AUC = {res['auc']:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True)

    plt.tight_layout()
    save("10_roc_curves")


# ════════════════════════════════════════════════════════════════
#  CHART — CONFUSION MATRICES
# ════════════════════════════════════════════════════════════════

def chart_confusion(results, y_test):
    section("Chart: Confusion Matrices")

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    fig.suptitle("Confusion Matrices", fontsize=16, color=TEXT, fontweight="bold")

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["On-Time","Delayed"])
        ax.set_yticklabels(["On-Time","Delayed"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=12)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}",
                        ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black",
                        fontsize=14, fontweight="bold")

    plt.tight_layout()
    save("11_confusion_matrices")


# ════════════════════════════════════════════════════════════════
#  CHART — FEATURE IMPORTANCE (Best Model)
# ════════════════════════════════════════════════════════════════

def chart_feature_importance(results, X_test, y_test):
    section("Chart: Feature Importance (Best Model)")

    # Pick best model by AUC
    best_name = max(results, key=lambda n: results[n]["auc"])
    best_model = results[best_name]["model"]
    note(f"  Best model: {best_name}  (AUC = {results[best_name]['auc']:.3f})")

    clf = best_model.named_steps["clf"]

    # Try built-in feature importances first (Random Forest / GBT)
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        # Fall back to permutation importance
        note("  Computing permutation importance (may take 30 s) …")
        perm = permutation_importance(best_model, X_test, y_test,
                                      n_repeats=10, random_state=42, n_jobs=-1)
        importances = perm.importances_mean

    top_n = 20
    indices = np.argsort(importances)[::-1][:top_n]
    feat_names = X_test.columns[indices]
    feat_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.suptitle(f"Top {top_n} Feature Importances\n({best_name})",
                 fontsize=15, color=TEXT, fontweight="bold")

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
    bars = ax.barh(feat_names[::-1], feat_vals[::-1], color=colors, edgecolor=BG, alpha=0.9)
    ax.bar_label(bars, fmt="%.4f", color=TEXT, fontsize=8, padding=3)
    ax.set_xlabel("Importance Score")
    ax.grid(True, axis="x")

    plt.tight_layout()
    save("12_feature_importance")

    # Save classification report
    report = classification_report(
        y_test, results[best_name]["y_pred"],
        target_names=["On-Time","Delayed"]
    )
    report_path = get_path("outputs/reports/best_model_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Best Model: {best_name}\n\n")
        f.write(report)
    ok(f"Classification report saved → {report_path}")

    return best_name, best_model


# ════════════════════════════════════════════════════════════════
#  SALES FORECASTING WITH PROPHET
# ════════════════════════════════════════════════════════════════

def sales_forecast():
    section("Sales Forecasting — Prophet")
    try:
        from prophet import Prophet
    except (ImportError, Exception) as e:
        note(f"Prophet initialization failed: {e}")
        note("Skipping forecasting step.")
        return

    data_path = get_path("../data/cleaned/dataco_clean.csv")
    dc = pd.read_csv(data_path, low_memory=False)
    date_col  = find(dc, ["order_date"])
    sales_col = find(dc, ["sales"])

    if not (date_col and sales_col):
        note("Required columns missing — skipping forecast."); return

    dc[date_col] = pd.to_datetime(dc[date_col], errors="coerce")

    # Aggregate to daily total sales
    daily = (
        dc.groupby(dc[date_col].dt.normalize())[sales_col]
          .sum()
          .reset_index()
          .rename(columns={date_col: "ds", sales_col: "y"})
    )
    daily = daily.dropna().sort_values("ds")
    note(f"  Daily sales series: {len(daily)} data points")

    # Train Prophet
    try:
        m = Prophet(
            changepoint_prior_scale  = 0.05,
            seasonality_mode         = "multiplicative",
            yearly_seasonality       = True,
            weekly_seasonality       = True,
            daily_seasonality        = False,
        )
        m.fit(daily)
    except Exception as e:
        note(f"Prophet modeling failed: {e}")
        note("Skipping forecasting step.")
        return

    # Forecast 180 days ahead
    future   = m.make_future_dataframe(periods=180)
    forecast = m.predict(future)

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("Sales Forecast — Next 180 Days", fontsize=16, color=TEXT, fontweight="bold")

    # Full forecast
    ax = axes[0]
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    alpha=0.25, color=ACCENT, label="Confidence Interval")
    ax.plot(forecast["ds"], forecast["yhat"], color=ACCENT, linewidth=2, label="Forecast")
    ax.scatter(daily["ds"], daily["y"], color="#3fb950", s=3, alpha=0.5, label="Actual")
    ax.set_title("Revenue Forecast with Confidence Band")
    ax.set_ylabel("Daily Revenue (USD)")
    ax.legend()
    ax.grid(True)

    # Trend only
    ax = axes[1]
    ax.plot(forecast["ds"], forecast["trend"], color="#f78166", linewidth=2)
    ax.set_title("Underlying Trend Component")
    ax.set_ylabel("Trend (USD)")
    ax.grid(True)

    plt.tight_layout()
    save("13_sales_forecast")

    # Save forecast to CSV
    out_path = get_path("outputs/reports/sales_forecast.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    forecast[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(out_path, index=False)
    ok(f"Forecast saved → {out_path}")


# ════════════════════════════════════════════════════════════════
#  SAVE BEST MODEL
# ════════════════════════════════════════════════════════════════

def save_model(name, model):
    path = get_path("outputs/reports/best_model.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"name": name, "model": model}, f)
    ok(f"Best model saved → {path}")
    note(f"  Load it later with:  pickle.load(open('{path}','rb'))")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print(f"\n{Fore.MAGENTA}{'▓'*62}")
    print("  SUPPLY CHAIN PROJECT  ·  STEP 4: PREDICTIVE MODELLING")
    print(f"{'▓'*62}{Style.RESET_ALL}")

    X, y, target = load_and_prepare()
    X_train, X_test, y_train, y_test = split(X, y)

    results = train_models(X_train, X_test, y_train, y_test)

    chart_model_comparison(results)
    chart_roc(results, y_test)
    chart_confusion(results, y_test)
    best_name, best_model = chart_feature_importance(results, X_test, y_test)

    sales_forecast()
    save_model(best_name, best_model)

    section("✅ STEP 4 COMPLETE")
    print(f"""
{Fore.GREEN}  Charts saved:
    09_model_comparison.png
    10_roc_curves.png
    11_confusion_matrices.png
    12_feature_importance.png
    13_sales_forecast.png

  Reports saved:
    outputs/reports/best_model_report.txt
    outputs/reports/sales_forecast.csv
    outputs/reports/best_model.pkl

  ▶  Next step → run:  src/05_dashboard.py
{Style.RESET_ALL}""")


if __name__ == "__main__":
    main()