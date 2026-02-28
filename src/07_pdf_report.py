"""
╔══════════════════════════════════════════════════════════════╗
║   SUPPLY CHAIN DISRUPTION ANALYSIS                           ║
║   SCRIPT 7 OF 7 — TIDY MULTI-PAGE STRATEGIC AUDIT           ║
╠══════════════════════════════════════════════════════════════╣
║  FIXES:                                                      ║
║  • No Overlapping: Explicit leading for all headers          ║
║  • Page 3 Restored: Clean KPI Grid (no nested spacers)       ║
║  • Topic 5 Restored: Fixed numbering sequence                ║
║  • Strict 1-per-page: Force breaks after every topic         ║
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

# Visual Palette
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
#  STYLES (Fixes overlapping and spacing)
# ════════════════════════════════════════════════════════════════

def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "cover_title": ParagraphStyle("ct", fontSize=44, fontName="Helvetica-Bold", textColor=WHITE, leading=52),
        "cover_sub": ParagraphStyle("cs", fontSize=20, fontName="Helvetica", textColor=SLATE_400, spaceBefore=15, leading=24),
        "h1": ParagraphStyle("h1", fontSize=26, fontName="Helvetica-Bold", textColor=DARK_NAVY, spaceBefore=10, spaceAfter=20, leading=32),
        "h2": ParagraphStyle("h2", fontSize=16, fontName="Helvetica-Bold", textColor=INDIGO, spaceBefore=25, spaceAfter=15, leading=20),
        "body": ParagraphStyle("body", fontSize=11, fontName="Helvetica", textColor=SLATE_700, leading=18, alignment=TA_JUSTIFY, spaceAfter=15),
        "kpi_val": ParagraphStyle("kv", fontSize=28, fontName="Helvetica-Bold", textColor=INDIGO, alignment=TA_CENTER, leading=34),
        "kpi_lab": ParagraphStyle("kl", fontSize=9, fontName="Helvetica-Bold", textColor=colors.grey, alignment=TA_CENTER, textTransform='uppercase', leading=12),
        "th": ParagraphStyle("th", fontSize=10, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_CENTER, leading=12),
        "tc": ParagraphStyle("tc", fontSize=10, fontName="Helvetica", textColor=SLATE_700, alignment=TA_CENTER, leading=14),
        "cap": ParagraphStyle("cap", fontSize=9, fontName="Helvetica-Oblique", textColor=colors.grey, alignment=TA_CENTER, spaceBefore=10, leading=11)
    }
    return styles

# ════════════════════════════════════════════════════════════════
#  INTELLIGENCE
# ════════════════════════════════════════════════════════════════

def load_intelligence():
    def read(f):
        p = os.path.join(DATA_DIR, f)
        return pd.read_csv(p, low_memory=False) if os.path.exists(p) else pd.DataFrame()

    dc = read("dataco_clean.csv")
    pr = read("product_chain_clean.csv")
    
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
#  BUILDER
# ════════════════════════════════════════════════════════════════

def add_image_safe(story, path, s, caption):
    """Adds an image to the story only if it exists, otherwise adds a professional placeholder."""
    if os.path.exists(path):
        story.append(Image(path, width=17*cm, height=8*cm, kind="proportional"))
        story.append(Paragraph(f"<i>{caption}</i>", s["cap"]))
    else:
        story.append(Spacer(1, 2*cm))
        story.append(Paragraph(f"[Statistical Asset Not Available: {os.path.basename(path)}]", s["body"]))
        story.append(Paragraph("This visual is generated during the Prophet Forecasting step. Please ensure Script 04 is fully configured with the required Stan backends.", s["body"]))
        story.append(Spacer(1, 2*cm))

def build_elite_report():
    section("Generating Tidy Strategic Briefing")
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
            canvas.drawString(1.5*cm, 0.8*cm, f"Supply Chain Intelligence | Confidential | {intel['date']}")
            canvas.drawRightString(A4[0]-1.5*cm, 0.8*cm, f"Page {doc.page-1}")
        canvas.restoreState()

    story = []
    
    # ── PAGE 1: COVER
    story.append(Spacer(1, 8*cm))
    story.append(Paragraph("SUPPLY CHAIN", s["cover_title"]))
    story.append(Paragraph("STRATEGIC AUDIT", s["cover_title"]))
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("Advanced Risk Engineering Briefing", s["cover_sub"]))
    story.append(Spacer(1, 1.5*cm))
    story.append(HRFlowable(width="40%", thickness=5, color=AMBER, hAlign="LEFT"))
    story.append(PageBreak())
    
    # ── PAGE 2: TOC
    story.append(Paragraph("Diagnostic Framework", s["h1"]))
    story.append(Spacer(1, 1*cm))
    toc = [
        ("I. Executive Performance Summary", "Consolidated KPIs and pipeline health."),
        ("II. Service Reliability Audit", "Logistical friction and delay drivers."),
        ("III. Revenue Architecture", "Market velocity and value spreads."),
        ("IV. Supplier Integrity Matrix", "Upstream risk and lead-time audit."),
        ("V. Predictive Intelligence", "Algorithmic disruption drivers."),
        ("VI. Strategic Forecasting", "180-day outlook and peak volume."),
        ("VII. Global Trade Context", "Macroeconomic trade trends."),
        ("VIII. Action Roadmap", "Prioritized strategic initiatives.")
    ]
    for t, d in toc:
        story.append(Paragraph(f"<b>{t}</b>", s["h2"]))
        story.append(Paragraph(d, s["body"]))
    story.append(PageBreak())
    
    # ── PAGE 3: SUMMARY
    story.append(Paragraph("I. Executive Performance Summary", s["h1"]))
    story.append(Paragraph(f"Total revenue has reached ${intel['revenue']/1e6:.1f}M across {intel['orders']:,} global orders. Financial health remains strong, anchored by the {intel['top_cat']} category.", s["body"]))
    story.append(Spacer(1, 1*cm))
    k_data = [
        [Paragraph(f"${intel['revenue']/1e6:.1f}M", s["kpi_val"]), Paragraph(f"{intel['late']:.1f}%", s["kpi_val"])],
        [Paragraph("Revenue", s["kpi_lab"]), Paragraph("Delay Rate", s["kpi_lab"])],
        [Paragraph(f"{intel['orders']:,}", s["kpi_val"]), Paragraph(f"{intel['health']:.0f}/100", s["kpi_val"])],
        [Paragraph("Volume", s["kpi_lab"]), Paragraph("Health Score", s["kpi_lab"])]
    ]
    kt = Table(k_data, colWidths=[8*cm, 8*cm])
    kt.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), SLATE_100), ('BOX', (0,0), (-1,-1), 1, BORDER_GRAY), ('TOPPADDING', (0,0), (-1,-1), 20), ('BOTTOMPADDING', (0,0), (-1,-1), 20)]))
    story.append(kt)
    story.append(PageBreak())
    
    # ── PAGE 4: RELIABILITY
    story.append(Paragraph("II. Service Reliability Audit", s["h1"]))
    story.append(Paragraph("Analysis identifies specific delay clusters within shipping protocols.", s["body"]))
    add_image_safe(story, os.path.join(CHART_DIR, "03_delay_analysis.png"), s, "Delay severity and late delivery rates by protocol.")
    story.append(Paragraph("Management Insight:", s["h2"]))
    story.append(Paragraph(f"The {intel['worst_m']} corridor exhibits systemic delay variance.", s["body"]))
    story.append(PageBreak())
    
    # ── PAGE 5: REVENUE
    story.append(Paragraph("III. Revenue Architecture", s["h1"]))
    story.append(Paragraph("Value distribution shows a positive skew with high-value order clusters.", s["body"]))
    add_image_safe(story, os.path.join(CHART_DIR, "02_sales_distribution.png"), s, "Regional sales density and statistical distribution of valuations.")
    story.append(PageBreak())
    
    # ── PAGE 6: SUPPLIER
    story.append(Paragraph("IV. Supplier Integrity Matrix", s["h1"]))
    story.append(Paragraph("Upstream risk is mapped by lead-time variance and defect rates.", s["body"]))
    add_image_safe(story, os.path.join(CHART_DIR, "06_supplier_risk.png"), s, "Supplier risk matrix: Defect rates vs lead-times.")
    story.append(PageBreak())
    
    # ── PAGE 7: PREDICTIVE
    story.append(Paragraph("V. Predictive Intelligence Drivers", s["h1"]))
    story.append(Paragraph("Algorithmic isolating of features that trigger logistical failure.", s["body"]))
    add_image_safe(story, os.path.join(CHART_DIR, "12_feature_importance.png"), s, "Feature importance identifying drivers of delay.")
    story.append(PageBreak())
    
    # ── PAGE 8: FORECAST
    story.append(Paragraph("VI. Strategic Forecasting", s["h1"]))
    story.append(Paragraph("180-day outlook accounting for seasonality and momentum.", s["body"]))
    add_image_safe(story, os.path.join(CHART_DIR, "13_sales_forecast.png"), s, "Prophet 180-day revenue outlook.")
    story.append(PageBreak())
    
    # ── PAGE 9: TRADE
    story.append(Paragraph("VII. Global Trade Context", s["h1"]))
    story.append(Paragraph("Macroeconomic export/import patterns across key trade corridors.", s["body"]))
    add_image_safe(story, os.path.join(CHART_DIR, "07_global_trade.png"), s, "Global commodity flux and trend analysis.")
    story.append(PageBreak())
    
    # ── PAGE 10: ROADMAP
    story.append(Paragraph("VIII. Strategic Action Roadmap", s["h1"]))
    story.append(Paragraph("Prioritized initiatives for 2026 operational optimization.", s["body"]))
    rd = [
        [Paragraph("INITIATIVE", s["th"]), Paragraph("STATUS", s["th"]), Paragraph("IMPACT", s["th"])],
        ["High-Risk Supplier Audit", "CRITICAL", "HIGH"],
        ["Predictive ERP Integration", "URGENT", "MEDIUM"],
        ["Routing Optimization", "STRATEGIC", "HIGH"]
    ]
    rt = Table(rd, colWidths=[9*cm, 4*cm, 5*cm])
    rt.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), INDIGO), ('GRID', (0,0), (-1,-1), 0.5, BORDER_GRAY), ('PADDING', (0,0), (-1,-1), 12), ('BACKGROUND', (0,1), (1,1), colors.HexColor("#FEE2E2"))]))
    story.append(rt)
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Conclusion", s["h2"]))
    story.append(Paragraph("Logistical reliability is the foundation for 2026 growth.", s["body"]))

    doc.build(story, onFirstPage=page_layout, onLaterPages=page_layout)
    ok(f"Final Audit Generated → {OUT_PATH}")

if __name__ == "__main__":
    build_elite_report()
