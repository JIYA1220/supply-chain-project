"""
╔══════════════════════════════════════════════════════════════╗
║   SUPPLY CHAIN DISRUPTION ANALYSIS                           ║
║   SCRIPT 7 OF 7 — STABILIZED MULTI-PAGE STRATEGIC AUDIT     ║
╠══════════════════════════════════════════════════════════════╣
║  FINAL STABILIZATION:                                        ║
║  • No Overlapping: Massive leading and paragraph spacing      ║
║  • Guaranteed Page 3: Fixed table sizing and margins          ║
║  • Corrected Numbering: Strict 1-per-page flow                ║
║  • Professional Spacing: Clear gaps between headers/text      ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import warnings
from datetime import datetime

import numpy  as np
import pandas as pd
from colorama import Fore, Style, init

from reportlab.lib              import colors
from reportlab.lib.pagesizes    import A4
from reportlab.lib.styles       import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units        import cm
from reportlab.lib.enums        import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus         import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image
)

warnings.filterwarnings("ignore")
init(autoreset=True)

# ════════════════════════════════════════════════════════════════
#  PATHS & BRANDING
# ════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
DATA_DIR   = os.path.join(BASE_DIR, "data", "cleaned")
CHART_DIR  = os.path.join(BASE_DIR, "outputs", "charts")
OUT_PATH   = os.path.join(BASE_DIR, "outputs", "Supply_Chain_Analysis_Report.pdf")

# Corporate Theme
INDIGO      = colors.HexColor("#4F46E5")
AMBER       = colors.HexColor("#F59E0B")
DARK_NAVY   = colors.HexColor("#0F172A")
SLATE_700   = colors.HexColor("#334155")
SLATE_400   = colors.HexColor("#94A3B8")
SLATE_100   = colors.HexColor("#F1F5F9")
BORDER_GRAY = colors.HexColor("#E2E8F0")
WHITE       = colors.white

def ok(m):   print(f"{Fore.GREEN}  ✔  {m}{Style.RESET_ALL}")
def section(t): print(f"\n{Fore.CYAN}{'═'*62}\n  {t}\n{'═'*62}{Style.RESET_ALL}")

# ════════════════════════════════════════════════════════════════
#  STYLES (High padding and leading)
# ════════════════════════════════════════════════════════════════

def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "cover_title": ParagraphStyle("ct", fontSize=44, fontName="Helvetica-Bold", textColor=WHITE, leading=52),
        "cover_sub": ParagraphStyle("cs", fontSize=20, fontName="Helvetica", textColor=SLATE_400, leading=28, spaceBefore=20),
        "h1": ParagraphStyle("h1", fontSize=24, fontName="Helvetica-Bold", textColor=DARK_NAVY, spaceBefore=10, spaceAfter=25, leading=30),
        "h2": ParagraphStyle("h2", fontSize=16, fontName="Helvetica-Bold", textColor=INDIGO, spaceBefore=30, spaceAfter=15, leading=22),
        "body": ParagraphStyle("body", fontSize=11, fontName="Helvetica", textColor=SLATE_700, leading=20, alignment=TA_JUSTIFY, spaceAfter=20),
        "kpi_val": ParagraphStyle("kv", fontSize=24, fontName="Helvetica-Bold", textColor=INDIGO, alignment=TA_CENTER, leading=28),
        "kpi_lab": ParagraphStyle("kl", fontSize=9, fontName="Helvetica-Bold", textColor=colors.grey, alignment=TA_CENTER, textTransform='uppercase', leading=12),
        "th": ParagraphStyle("th", fontSize=10, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER, leading=14),
        "tc": ParagraphStyle("tc", fontSize=10, fontName="Helvetica", textColor=SLATE_700, alignment=TA_CENTER, leading=14),
        "cap": ParagraphStyle("cap", fontSize=9, fontName="Helvetica-Oblique", textColor=colors.grey, alignment=TA_CENTER, spaceBefore=15, leading=12)
    }
    return styles

# ════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_intelligence():
    def read(f):
        p = os.path.join(DATA_DIR, f)
        return pd.read_csv(p, low_memory=False) if os.path.exists(p) else pd.DataFrame()

    dc = read("dataco_clean.csv")
    intel = {"date": datetime.now().strftime("%B %d, %Y")}
    if not dc.empty:
        intel["revenue"] = dc["sales"].sum()
        intel["orders"]  = len(dc)
        intel["late"]    = dc["is_delayed"].mean() * 100
        intel["health"]  = max(0, 100 - (intel["late"] * 0.8))
        intel["worst_m"] = dc.groupby("market")["is_delayed"].mean().idxmax()
        intel["top_cat"] = dc.groupby("category_name")["sales"].sum().idxmax()
    else:
        intel.update({"revenue": 0, "orders": 0, "late": 0, "health": 0, "worst_m": "N/A", "top_cat": "N/A"})
    return intel

# ════════════════════════════════════════════════════════════════
#  SAFE ASSET INJECTION
# ════════════════════════════════════════════════════════════════

def add_visual(story, name, s, caption):
    path = os.path.join(CHART_DIR, name)
    if os.path.exists(path):
        story.append(Image(path, width=16.5*cm, height=8.5*cm, kind="proportional"))
        story.append(Paragraph(f"<i>{caption}</i>", s["cap"]))
    else:
        story.append(Spacer(1, 2*cm))
        story.append(Paragraph(f"[Statistical Model Asset: {name} - Re-run Script 04]", s["body"]))
        story.append(Spacer(1, 2*cm))

# ════════════════════════════════════════════════════════════════
#  REPORT ASSEMBLY
# ════════════════════════════════════════════════════════════════

def build_elite_report():
    section("Engineering Stabilized Strategic Briefing")
    intel = load_intelligence()
    s = build_styles()
    
    doc = SimpleDocTemplate(OUT_PATH, pagesize=A4, topMargin=1.5*cm, bottomMargin=1.5*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    
    def page_layout(canvas, doc):
        canvas.saveState()
        if doc.page == 1:
            canvas.setFillColor(DARK_NAVY); canvas.rect(0, 0, A4[0], A4[1], fill=1)
            canvas.setFillColor(INDIGO); canvas.rect(0, 0, 3*cm, A4[1], fill=1)
        else:
            canvas.setStrokeColor(BORDER_GRAY); canvas.setLineWidth(0.5)
            canvas.line(1.5*cm, A4[1]-1.2*cm, A4[0]-1.5*cm, A4[1]-1.2*cm)
            canvas.setFont("Helvetica", 8); canvas.setFillColor(colors.grey)
            canvas.drawString(1.5*cm, 0.8*cm, f"Strategic Operations Briefing | Confidential | {intel['date']}")
            canvas.drawRightString(A4[0]-1.5*cm, 0.8*cm, f"Page {doc.page-1}")
        canvas.restoreState()

    story = []
    
    # ── PAGE 1: COVER
    story.append(Spacer(1, 8*cm))
    story.append(Paragraph("SUPPLY CHAIN", s["cover_title"]))
    story.append(Paragraph("STRATEGIC AUDIT", s["cover_title"]))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Data Engineering & Risk Prediction Suite", s["cover_sub"]))
    story.append(Spacer(1, 1.5*cm))
    story.append(HRFlowable(width="40%", thickness=5, color=AMBER, hAlign="LEFT"))
    story.append(PageBreak())
    
    # ── PAGE 2: TABLE OF CONTENTS
    story.append(Paragraph("Analytical Framework", s["h1"]))
    story.append(Spacer(1, 0.5*cm))
    toc = [
        ("I. Executive Performance Summary", "Consolidated KPIs and business health."),
        ("II. Service Reliability Audit", "Logistical friction and delay root causes."),
        ("III. Revenue Architecture", "Analysis of market value distributions."),
        ("IV. Supplier Integrity Matrix", "Upstream risk and defect variance."),
        ("V. Predictive Intelligence Drivers", "Algorithmic factors impacting delivery."),
        ("VI. Strategic Revenue Forecasting", "180-day outlook and trend patterns."),
        ("VII. Global Trade Context", "Macroeconomic trade corridor patterns."),
        ("VIII. Strategic Action Roadmap", "Prioritized initiatives for 2026.")
    ]
    for t, d in toc:
        story.append(Paragraph(f"<b>{t}</b>", s["h2"]))
        story.append(Paragraph(d, s["body"]))
    story.append(PageBreak())
    
    # ── PAGE 3: EXECUTIVE SUMMARY (TOPIC I)
    story.append(Paragraph("I. Executive Performance Summary", s["h1"]))
    story.append(Paragraph(
        f"This audit analyzes {intel['orders']:,} global operations. Current revenue velocity is recorded at "
        f"${intel['revenue']/1e6:.1f}M, with a primary volume driver identified in the {intel['top_cat']} segment. "
        f"The pipeline health score is currently rated at {intel['health']:.0f}/100.", s["body"]
    ))
    story.append(Spacer(1, 1*cm))
    k_data = [
        [Paragraph(f"${intel['revenue']/1e6:.1f}M", s["kpi_val"]), Paragraph(f"{intel['late']:.1f}%", s["kpi_val"])],
        [Paragraph("Gross Revenue", s["kpi_lab"]), Paragraph("Variance (Late)", s["kpi_lab"])],
        [Spacer(1, 1*cm), Spacer(1, 1*cm)],
        [Paragraph(f"{intel['orders']:,}", s["kpi_val"]), Paragraph(f"{intel['health']:.0f}/100", s["kpi_val"])],
        [Paragraph("Order Volume", s["kpi_lab"]), Paragraph("Health Score", s["kpi_lab"])]
    ]
    kt = Table(k_data, colWidths=[8*cm, 8*cm])
    kt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), SLATE_100),
        ('BOX', (0,0), (-1,-1), 1, BORDER_GRAY),
        ('TOPPADDING', (0,0), (-1,-1), 25),
        ('BOTTOMPADDING', (0,0), (-1,-1), 25),
    ]))
    story.append(kt)
    story.append(PageBreak())
    
    # ── PAGE 4: SERVICE RELIABILITY (TOPIC II)
    story.append(Paragraph("II. Service Reliability Audit", s["h1"]))
    story.append(Paragraph(
        "Service reliability identifies the delta between promised delivery dates and real-world fulfillment. "
        "High variance in standard shipping modes suggests a need for priority-based routing for high-value orders.", s["body"]
    ))
    add_visual(story, "03_delay_analysis.png", s, "Decomposition of delay severity and late delivery rates by mode.")
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Operational Insight:", s["h2"]))
    story.append(Paragraph(f"The {intel['worst_m']} corridor exhibits the highest statistical variance.", s["body"]))
    story.append(PageBreak())
    
    # ── PAGE 5: REVENUE ARCHITECTURE (TOPIC III)
    story.append(Paragraph("III. Revenue Architecture", s["h1"]))
    story.append(Paragraph(
        "Revenue architecture maps the density of sales against regional markets. This distribution provides "
        "insight into the pricing sensitivity and volume-velocity relationship across global segments.", s["body"]
    ))
    add_visual(story, "02_sales_distribution.png", s, "Regional sales density and statistical distribution of order valuations.")
    story.append(PageBreak())
    
    # ── PAGE 6: SUPPLIER INTEGRITY (TOPIC IV)
    story.append(Paragraph("IV. Supplier Integrity Matrix", s["h1"]))
    story.append(Paragraph(
        "The integrity matrix is a forward-looking indicator mapping manufacturing defect rates against "
        "delivery lead-times. Upper-quadrant providers represent systemic risks to fulfillments SLAs.", s["body"]
    ))
    add_visual(story, "06_supplier_risk.png", s, "Supply chain risk matrix: Defect rates vs manufacturing lead-time variance.")
    story.append(PageBreak())
    
    # ── PAGE 7: PREDICTIVE INTELLIGENCE (TOPIC V)
    story.append(Paragraph("V. Predictive Intelligence Drivers", s["h1"]))
    story.append(Paragraph(
        "Utilizing Gradient Boosting ensembles, we isolated the primary factors driving supply chain failure. "
        "Scheduling windows and order timing are identified as the most impactful operational levers.", s["body"]
    ))
    add_visual(story, "12_feature_importance.png", s, "Algorithmic Importance: Quantifying the impact of features on delay probability.")
    story.append(PageBreak())
    
    # ── PAGE 8: REVENUE FORECASTING (TOPIC VI)
    story.append(Paragraph("VI. Strategic Revenue Forecasting", s["h1"]))
    story.append(Paragraph(
        "Time-series modeling via Facebook Prophet projects momentum for the next 180 days. Management "
        "should align carrier capacity with the peak-volume windows identified in this outlook.", s["body"]
    ))
    add_visual(story, "13_sales_forecast.png", s, "Prophet Forecast: 180-day revenue outlook with confidence interval bands.")
    story.append(PageBreak())
    
    # ── PAGE 9: GLOBAL TRADE (TOPIC VII)
    story.append(Paragraph("VII. Global Trade Context", s["h1"]))
    story.append(Paragraph(
        "Macroeconomic commodity flux patterns provide the context for localized supply chain stability. "
        "Shifts in international export/import volumes signal potential upstream supply constraints.", s["body"]
    ))
    add_visual(story, "07_global_trade.png", s, "Global commodity flux: Top 15 countries and export/import trends.")
    story.append(PageBreak())
    
    # ── PAGE 10: ACTION ROADMAP (TOPIC VIII)
    story.append(Paragraph("VIII. Strategic Action Roadmap", s["h1"]))
    story.append(Paragraph("Prioritized initiatives designed to optimize operational efficiency and reduce variance.", s["body"]))
    rd_data = [
        [Paragraph("INITIATIVE", s["th"]), Paragraph("PRIORITY", s["th"]), Paragraph("IMPACT", s["th"])],
        ["High-Risk Supplier Performance Audit", "CRITICAL", "HIGH"],
        ["Predictive ERP Integration (Auto-Flagging)", "URGENT", "MEDIUM"],
        ["LATAM corridor Routing Optimization", "STRATEGIC", "HIGH"],
        ["Dynamic Shipping SLA Renegotiation", "GROWTH", "MEDIUM"]
    ]
    rt = Table(rd_data, colWidths=[9*cm, 4*cm, 5*cm])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), INDIGO), ('GRID', (0,0), (-1,-1), 0.5, BORDER_GRAY),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('PADDING', (0,0), (-1,-1), 12),
        ('BACKGROUND', (0,1), (1,1), colors.HexColor("#FEE2E2"))
    ]))
    story.append(rt)
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph("Operational Conclusion", s["h2"]))
    story.append(Paragraph("Transitioning to predictive routing will provide the moat required for 2026 growth.", s["body"]))

    # Final build
    doc.build(story, onFirstPage=page_layout, onLaterPages=page_layout)
    ok(f"Professional 10-Page Audit Finalized → {OUT_PATH}")

if __name__ == "__main__":
    build_elite_report()
