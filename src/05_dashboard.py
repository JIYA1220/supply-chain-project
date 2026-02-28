"""
╔══════════════════════════════════════════════════════════════╗
║   SUPPLY CHAIN DISRUPTION ANALYSIS                           ║
║   SCRIPT 5 OF 5 — INTERACTIVE DASHBOARD (Streamlit)         ║
╠══════════════════════════════════════════════════════════════╣
║  What this script does:                                      ║
║  • Full interactive web dashboard (runs in browser)          ║
║  • KPI cards (revenue, delays, orders, profit)               ║
║  • Filter by market, category, year                          ║
║  • Delay heatmap by market × shipping mode                   ║
║  • Supplier risk leaderboard                                  ║
║  • Live sales trend chart                                     ║
║  • Model prediction on custom input                          ║
║                                                              ║
║  HOW TO RUN:                                                 ║
║  pip install streamlit plotly                                ║
║  streamlit run src/05_dashboard.py                           ║
╚══════════════════════════════════════════════════════════════╝
"""

import pickle
import warnings
import os
import numpy  as np
import pandas as pd
import plotly.express       as px
import plotly.graph_objects as go
import streamlit            as st
import sys

sys.path.append(os.path.dirname(__file__))
from data_loader import load_data

# Show banner if running on Streamlit Cloud (no real data)
if not os.path.exists("data/cleaned/dataco_cleaned.csv"):
    st.info(
        "Demo Mode: Real datasets are being loaded from Google Drive. "
        "This may take a moment on first launch.",
        icon="ℹ️"
    )

# Load data (auto-downloads from Drive if needed)
df = load_data()

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
#  PATH HELPERS
# ════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_path(rel_path):
    if rel_path.startswith("../"):
        clean_rel = rel_path[3:]
        base_dir = os.path.dirname(SCRIPT_DIR)
        path = os.path.join(base_dir, clean_rel)
    else:
        if os.path.exists(rel_path):
            return rel_path
        path = os.path.join(SCRIPT_DIR, rel_path)
    return path

# ════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Supply Chain Intelligence",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ════════════════════════════════════════════════════════════════
#  MINIMALISTIC LIGHT CSS (Indigo & Amber)
# ════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp { background-color: #FFFFFF; color: #1F2937; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #F9FAFB;
        border-right: 1px solid #F3F4F6;
    }

    /* KPI metric cards */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #F3F4F6;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    div[data-testid="metric-container"] label { color: #6B7280 !important; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="metric-container"] div   { color: #111827 !important; font-size: 28px; font-weight: 600; }

    /* Buttons */
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
    }

    /* Clean headers */
    h1 { color: #111827 !important; font-weight: 600 !important; font-size: 2.2rem !important; }
    h2, h3 { color: #1F2937 !important; font-weight: 600 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F9FAFB;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        color: #6B7280;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4F46E5 !important;
        font-weight: 600;
    }

    hr { border-color: #F3F4F6; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Plotly Theme
PLOT_BG = "rgba(0,0,0,0)"
PLOT_TEXT = "#374151"
PLOT_GRID = "#F3F4F6"
COLOR_MAIN = "#4F46E5" # Indigo
COLOR_SEC  = "#F59E0B" # Amber
COLOR_PAL  = ["#4F46E5", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6"]

# ════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════

@st.cache_data
def load_dataco():
    path = get_path("../data/cleaned/dataco_clean.csv")
    df = pd.read_csv(path, low_memory=False)
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data
def load_products():
    path = get_path("../data/cleaned/product_chain_clean.csv")
    return pd.read_csv(path, low_memory=False)

@st.cache_data
def load_trade():
    path = get_path("../data/cleaned/global_trade_clean.csv")
    return pd.read_csv(path, low_memory=False)

@st.cache_resource
def load_model():
    try:
        path = get_path("outputs/reports/best_model.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def find(df, candidates):
    for c in candidates:
        if c in df.columns: return c
    for c in candidates:
        hits = [col for col in df.columns
                if c.replace("_","") in col.lower().replace("_","")]
        if hits: return hits[0]
    return None

# ════════════════════════════════════════════════════════════════
#  SIDEBAR FILTERS
# ════════════════════════════════════════════════════════════════

def sidebar_filters(dc):
    st.sidebar.subheader("Analytics Scope")
    
    market_col = find(dc, ["market"])
    cat_col    = find(dc, ["category_name"])
    yr_col     = find(dc, ["order_year"])

    filters = {}

    if market_col:
        markets = ["Global View"] + sorted(dc[market_col].dropna().unique().tolist())
        filters["market"] = st.sidebar.selectbox("Market Region", markets)

    if cat_col:
        cats = ["All Categories"] + sorted(dc[cat_col].dropna().unique().tolist())
        filters["category"] = st.sidebar.selectbox("Product Category", cats)

    if yr_col:
        years = sorted(dc[yr_col].dropna().astype(int).unique().tolist())
        if len(years) >= 2:
            filters["year_range"] = st.sidebar.select_slider(
                "Timeframe", options=years, value=(min(years), max(years))
            )

    return filters

def apply_filters(dc, filters):
    df = dc.copy()
    market_col = find(df, ["market"])
    cat_col    = find(df, ["category_name"])
    yr_col     = find(df, ["order_year"])

    if market_col and filters.get("market","Global View") != "Global View":
        df = df[df[market_col] == filters["market"]]
    if cat_col and filters.get("category","All Categories") != "All Categories":
        df = df[df[cat_col] == filters["category"]]
    if yr_col and "year_range" in filters:
        lo, hi = filters["year_range"]
        df = df[(df[yr_col] >= lo) & (df[yr_col] <= hi)]

    return df

# ════════════════════════════════════════════════════════════════
#  KPI TAB
# ════════════════════════════════════════════════════════════════

def tab_kpis(df):
    sales_col  = find(df, ["sales"])
    del_col    = find(df, ["is_delayed","late_delivery_risk"])
    profit_col = find(df, ["profit_margin_pct"])
    qty_col    = find(df, ["order_item_quantity"])
    yr_col     = find(df, ["order_year"])
    market_col = find(df, ["market"])
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total = df[sales_col].sum() if sales_col else 0
        st.metric("Total Revenue", f"${total/1e6:.1f}M")
    with c2:
        st.metric("Operations Count", f"{len(df):,}")
    with c3:
        rate = df[del_col].mean() * 100 if del_col else 0
        st.metric("Service Delay Rate", f"{rate:.1f}%")
    with c4:
        avg_p = df[profit_col].mean() if profit_col else 0
        st.metric("Net Profit Margin", f"{avg_p:.1f}%")

    st.markdown("---")

    # 1. Professional Revenue Velocity (Full Width)
    date_col  = find(df, ["order_date"])
    if sales_col and date_col:
        # Aggregating and calculating rolling average for professional look
        monthly = df.set_index(date_col).resample('M')[sales_col].sum().reset_index()
        monthly.columns = ["Date", "Revenue"]
        monthly["Trend (3mo Avg)"] = monthly["Revenue"].rolling(window=3).mean()
        
        fig_rev = go.Figure()
        # Area for actual revenue
        fig_rev.add_trace(go.Scatter(
            x=monthly["Date"], y=monthly["Revenue"],
            fill='tozeroy', name="Actual Revenue",
            line=dict(color='rgba(79, 70, 229, 0.3)', width=0.5),
            fillcolor='rgba(79, 70, 229, 0.1)'
        ))
        # Bold line for Trend
        fig_rev.add_trace(go.Scatter(
            x=monthly["Date"], y=monthly["Trend (3mo Avg)"],
            name="Strategic Trend",
            line=dict(color='#4F46E5', width=4, dash='solid')
        ))
        
        fig_rev.update_layout(
            title="Revenue Analytics: Actuals vs Strategic Trend",
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=PLOT_TEXT, family="Inter"),
            xaxis=dict(gridcolor=PLOT_GRID, title="", showgrid=False),
            yaxis=dict(gridcolor=PLOT_GRID, title="Revenue ($)", tickformat="$,.0f"),
            height=380,
            margin=dict(t=60, b=40, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    st.markdown("---")

    # 2. Vibrant & Smooth Animated 3D Market Trajectory (Full Width)
    st.subheader("Market Intelligence: 3D Strategic Evolution")
    
    if all([sales_col, profit_col, qty_col, yr_col, market_col]):
        # Data aggregation
        traj_df = df.groupby([market_col, yr_col]).agg({
            sales_col: "sum",
            profit_col: "mean",
            qty_col: "sum"
        }).reset_index().sort_values(yr_col)

        # Using a much more vibrant and high-contrast color palette
        fig_3d = px.scatter_3d(
            traj_df, x=sales_col, y=profit_col, z=qty_col,
            color=market_col,
            size=sales_col,
            animation_frame=yr_col,
            animation_group=market_col,
            labels={sales_col: "Revenue ($)", profit_col: "Margin (%)", qty_col: "Orders"},
            title="Strategic Market Flight-Paths (High Velocity)",
            color_discrete_sequence=px.colors.qualitative.Vivid, # Vibrant palette
            size_max=65, # Larger bubbles for more "pop"
            opacity=0.9
        )

        # Ultra-Smooth Cinematic Styling
        fig_3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=50),
            scene=dict(
                xaxis=dict(backgroundcolor="#F1F5F9", gridcolor="#CBD5E1", showbackground=True),
                yaxis=dict(backgroundcolor="#F1F5F9", gridcolor="#CBD5E1", showbackground=True),
                zaxis=dict(backgroundcolor="#F1F5F9", gridcolor="#CBD5E1", showbackground=True),
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.4)) # Slightly pulled back for better view
            ),
            height=700,
            paper_bgcolor=PLOT_BG,
            font=dict(family="Inter", color=PLOT_TEXT),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Super-smooth animation settings
        fig_3d.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1200 # Longer duration
        fig_3d.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 800 # Smooth transition
        fig_3d.layout.updatemenus[0].buttons[0].args[1]['transition']['easing'] = "cubic-in-out"
        
        st.plotly_chart(fig_3d, use_container_width=True)

# ════════════════════════════════════════════════════════════════
#  DELAY TAB
# ════════════════════════════════════════════════════════════════

def tab_delays(df):
    is_del_col = find(df, ["is_delayed","late_delivery_risk"])
    ship_col   = find(df, ["shipping_mode"])
    market_col = find(df, ["market"])
    
    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        if ship_col and is_del_col:
            grp = df.groupby(ship_col)[is_del_col].mean().mul(100).reset_index()
            grp.columns = ["Method","Rate"]
            fig = px.bar(grp, x="Method", y="Rate",
                         title="Delay Probability by Shipping Protocol",
                         color="Rate", color_continuous_scale="Viridis")
            fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                              font=dict(color=PLOT_TEXT), height=400,
                              xaxis=dict(title="", gridcolor=PLOT_GRID),
                              yaxis=dict(title="Probability (%)", gridcolor=PLOT_GRID))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if is_del_col:
            counts = df[is_del_col].value_counts().reset_index()
            counts.columns = ["Status","Count"]
            counts["Status"] = counts["Status"].map({1: "Delayed", 0: "On-Time"})
            fig = px.pie(counts, names="Status", values="Count",
                         hole=0.6, title="Service Reliability",
                         color_discrete_sequence=[COLOR_SEC, COLOR_MAIN])
            fig.update_layout(paper_bgcolor=PLOT_BG, font=dict(color=PLOT_TEXT), height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Restored Disruption Heatmap
    st.markdown("---")
    st.subheader("Global Disruption Heatmap")
    if market_col and ship_col and is_del_col:
        pivot = df.pivot_table(
            values=is_del_col, index=market_col,
            columns=ship_col, aggfunc="mean"
        ).mul(100).round(1)

        fig = px.imshow(
            pivot,
            color_continuous_scale="Greys",
            title="Late Delivery Rate (%) — Market × Shipping Mode",
            text_auto=".1f",
        )
        fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                          font=dict(color=PLOT_TEXT), height=450)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
#  SUPPLIER TAB
# ════════════════════════════════════════════════════════════════

def tab_suppliers(pr):
    st.subheader("Supplier Assessment")

    sup_col    = find(pr, ["supplier_name"])
    risk_col   = find(pr, ["supplier_risk_score"])
    defect_col = find(pr, ["defect_rates","defect_rate_pct"])

    if not (sup_col and risk_col):
        st.info("Supplier analytics unavailable."); return

    grp = pr.groupby(sup_col).agg(
        Risk=(risk_col, "mean"),
        Defects=(defect_col, "mean") if defect_col else (risk_col, "mean")
    ).reset_index().round(2).sort_values("Risk", ascending=False).head(15)

    # Circle size increased (size_max increased for better visibility)
    fig = px.scatter(grp, x="Defects", y="Risk", text=sup_col,
                     size="Risk", color="Risk",
                     title="Supplier Integrity Matrix (Top 15 Risk Profiles)",
                     color_continuous_scale="YlOrRd",
                     size_max=60) # Increased size from default ~30 to 60
    fig.update_traces(textposition='top center')
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      font=dict(color=PLOT_TEXT), height=500,
                      xaxis=dict(gridcolor=PLOT_GRID, title="Defect Rate"),
                      yaxis=dict(gridcolor=PLOT_GRID, title="Risk Index"))
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
#  GLOBAL TRADE TAB (with animation)
# ════════════════════════════════════════════════════════════════

def tab_trade(tr):
    st.subheader("Animated Global Trade Flux")

    country_col = find(tr, ["country_or_area","country"])
    trade_col   = find(tr, ["trade_usd_billions","trade_usd"])
    yr_col      = find(tr, ["year"])
    flow_col    = find(tr, ["flow"])

    if not (country_col and trade_col and yr_col):
        st.info("Insufficient trade data."); return

    # Animated bubble chart of trade volume over years
    # Prepare top countries for cleaner animation
    top_countries = tr.groupby(country_col)[trade_col].sum().sort_values(ascending=False).head(15).index
    tr_sub = tr[tr[country_col].isin(top_countries)].sort_values(yr_col)

    fig = px.scatter(tr_sub, x=country_col, y=trade_col, animation_frame=yr_col,
                     size=trade_col, color=flow_col,
                     title="Trade Value Evolution (Top 15 Countries)",
                     labels={trade_col: "Trade Value", yr_col: "Year"},
                     color_discrete_sequence=[COLOR_MAIN, COLOR_SEC],
                     size_max=50, range_y=[0, tr_sub[trade_col].max() * 1.1])
    
    fig.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                      font=dict(color=PLOT_TEXT), height=500,
                      xaxis=dict(title="", gridcolor=PLOT_GRID),
                      yaxis=dict(gridcolor=PLOT_GRID))
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════
#  PREDICTOR TAB
# ════════════════════════════════════════════════════════════════

def tab_predictor():
    st.subheader("Intelligence Engine: Risk Forecasting")
    
    model_data = load_model()
    if model_data is None:
        st.warning("Prediction engine offline."); return

    model = model_data["model"]
    
    # Try to extract required feature names from model steps
    try:
        if hasattr(model, "feature_names_in_"):
            expected_feats = model.feature_names_in_
        elif hasattr(model.named_steps.get("clf"), "feature_names_in_"):
            expected_feats = model.named_steps["clf"].feature_names_in_
        elif hasattr(model.named_steps.get("scaler"), "feature_names_in_"):
            expected_feats = model.named_steps["scaler"].feature_names_in_
        else:
            expected_feats = None
    except:
        expected_feats = None

    c1, c2, c3 = st.columns(3)
    with c1:
        sched = st.slider("Target Delivery Window (Days)", 1, 7, 3)
        qty   = st.number_input("Unit Quantity", 1, 100, 5)
    with c2:
        price = st.number_input("Item Valuation ($)", 10, 1000, 150)
        sales = st.number_input("Gross Sale ($)", 10, 5000, 300)
    with c3:
        month = st.slider("Seasonality (Month)", 1, 12, 6)
        wkend = st.checkbox("Processing on Weekend")

    if st.button("RUN RISK ANALYSIS", use_container_width=True):
        # Feature Mapping
        input_data = {
            "days_for_shipment_scheduled": sched,
            "order_item_quantity": qty,
            "product_price": price,
            "sales": sales,
            "order_month": month,
            "is_weekend": int(wkend),
            "order_year": 2022, 
            "profit_margin_pct": 10, 
            "order_item_discount_rate": 0.1
        }
        
        X_df = pd.DataFrame([input_data])
        
        # Robust Feature Alignment to avoid ValueError
        if expected_feats is not None:
            # Add missing features with default 0
            for f in expected_feats:
                if f not in X_df.columns:
                    X_df[f] = 0.0
            # Ensure correct order and drop extra columns
            X_df = X_df.reindex(columns=expected_feats)
        
        try:
            proba = model.predict_proba(X_df)[0][1]
            
            # High-Quality Animated Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Delay Risk Index", 'font': {'size': 24, 'color': COLOR_MAIN}},
                number = {'suffix': "%", 'font': {'color': PLOT_TEXT}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': PLOT_TEXT},
                    'bar': {'color': COLOR_MAIN},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': PLOT_GRID,
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.1)'},
                        {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor=PLOT_BG,
                font={'color': PLOT_TEXT, 'family': 'Inter'},
                height=400,
                margin=dict(t=100, b=20, l=50, r=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if proba > 0.6:
                st.error("### HIGH RISK: Logistic Disruption Likely")
            elif proba > 0.3:
                st.warning("### MODERATE RISK: Potential Scheduling Conflict")
            else:
                st.success("### LOW RISK: Optimal Shipping Pipeline")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("The model features have changed. Please ensure Script 04 has been fully executed with the latest code.")

# ════════════════════════════════════════════════════════════════
#  SQL ANALYTICS TAB
# ════════════════════════════════════════════════════════════════

def tab_sql():
    st.subheader("SQL Intelligence Layer")
    st.markdown("Query the centralized SQLite data warehouse directly or view pre-computed strategic insights.")
    
    tab_pre, tab_custom = st.tabs(["Pre-Computed Insights", "Interactive SQL Sandbox"])
    
    with tab_pre:
        sql_dir = get_path("../outputs/sql_results")
        if not os.path.exists(sql_dir):
            st.warning("SQL results not found. Please run Script 06 first.")
        else:
            files = [f for f in os.listdir(sql_dir) if f.endswith(".csv")]
            if files:
                # Clean names for dropdown
                file_map = {f.replace(".csv","").replace("_"," ").title(): f for f in sorted(files)}
                selected_query = st.selectbox("Select SQL Query Result", list(file_map.keys()))
                
                if selected_query:
                    df_sql = pd.read_csv(os.path.join(sql_dir, file_map[selected_query]))
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.dataframe(df_sql, use_container_width=True, hide_index=True)
                    with col2:
                        st.metric("Total Records", len(df_sql))
                        st.download_button(
                            label=f"Export {selected_query} to CSV",
                            data=df_sql.to_csv(index=False),
                            file_name=file_map[selected_query],
                            mime="text/csv",
                        )
            else:
                st.info("No SQL query results available.")
                
    with tab_custom:
        st.markdown("**Write your own SQL query against the database.**")
        st.caption("Available Tables: `orders`, `products`, `trade`")
        
        custom_query = st.text_area(
            "SQL Editor", 
            height=150, 
            value="SELECT \n    market, \n    COUNT(*) as total_orders, \n    ROUND(AVG(profit_margin_pct), 2) as avg_margin\nFROM orders\nGROUP BY market\nORDER BY total_orders DESC;"
        )
        
        if st.button("Execute Query"):
            db_path = get_path("../data/supply_chain.db")
            if not os.path.exists(db_path):
                st.error("Database not found. Please run Script 06 to generate the SQLite database first.")
            else:
                import sqlite3
                try:
                    conn = sqlite3.connect(db_path)
                    df_custom = pd.read_sql_query(custom_query, conn)
                    conn.close()
                    
                    st.success(f"✅ Query executed successfully. Returned {len(df_custom)} rows.")
                    st.dataframe(df_custom, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"SQL Error: {e}")

# ════════════════════════════════════════════════════════════════
#  EXECUTIVE REPORT TAB
# ════════════════════════════════════════════════════════════════

def tab_report():
    st.subheader("Automated Executive Briefing")
    st.markdown("Download the full multi-page PDF analysis report generated via Script 07.")
    
    pdf_path = get_path("../outputs/Supply_Chain_Analysis_Report.pdf")
    
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.success("Analysis Report Ready")
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name="Supply_Chain_Analysis_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            st.info("""
            **Report Contents:**
            - Executive Summary & KPIs
            - Root Cause Delay Analysis
            - Supplier Risk Matrix
            - ML Model Performance
            - 180-Day Revenue Forecast
            - Strategic Recommendations
            """)
        
        with c2:
            st.caption("Document Preview")
            import base64
            base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.warning("PDF Report not found. Please run Script 07 to generate it.")

# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    st.title("Supply Chain Intelligence")
    st.text("Strategic operations dashboard and predictive risk modeling.")
    st.markdown("---")

    dc = load_dataco()
    pr = load_products()
    tr = load_trade()

    filters = sidebar_filters(dc)
    df_filt = apply_filters(dc, filters)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"System Record Count: {len(df_filt):,}")

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Performance", 
        "Disruption", 
        "Suppliers", 
        "Risk Engine",
        "SQL Intelligence",
        "Executive Report"
    ])
    
    with t1: tab_kpis(df_filt)
    with t2: tab_delays(df_filt)
    with t3: tab_suppliers(pr)
    with t4: tab_predictor()
    with t5: tab_sql()
    with t6: tab_report()

if __name__ == "__main__":
    main()
