"""
Generate final PDF report for the macro forecasting project.
Saved to research/report/week2_report.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "research" / "report" / "week2_report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

base = getSampleStyleSheet()

def style(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)

S = {
    "title":    style("title",   "Normal", fontSize=20, leading=26, spaceAfter=4,
                      textColor=colors.HexColor("#1a1a2e"), fontName="Helvetica-Bold",
                      alignment=TA_LEFT),
    "subtitle": style("subtitle","Normal", fontSize=11, leading=15, spaceAfter=2,
                      textColor=colors.HexColor("#555555"), fontName="Helvetica"),
    "meta":     style("meta",    "Normal", fontSize=9,  leading=12, spaceAfter=16,
                      textColor=colors.HexColor("#888888"), fontName="Helvetica"),
    "h1":       style("h1",      "Normal", fontSize=13, leading=18, spaceBefore=18,
                      spaceAfter=4, textColor=colors.HexColor("#1a1a2e"),
                      fontName="Helvetica-Bold"),
    "h2":       style("h2",      "Normal", fontSize=11, leading=15, spaceBefore=10,
                      spaceAfter=3, textColor=colors.HexColor("#2d4a7a"),
                      fontName="Helvetica-Bold"),
    "body":     style("body",    "Normal", fontSize=10, leading=15, spaceAfter=8,
                      textColor=colors.HexColor("#222222"), fontName="Helvetica",
                      alignment=TA_JUSTIFY),
    "eq":       style("eq",      "Normal", fontSize=10, leading=15, spaceAfter=6,
                      leftIndent=24, textColor=colors.HexColor("#1a1a2e"),
                      fontName="Courier"),
    "bullet":   style("bullet",  "Normal", fontSize=10, leading=15, spaceAfter=4,
                      leftIndent=16, textColor=colors.HexColor("#222222"),
                      fontName="Helvetica"),
    "caption":  style("caption", "Normal", fontSize=9,  leading=12, spaceAfter=6,
                      textColor=colors.HexColor("#666666"), fontName="Helvetica-Oblique"),
}

RULE = HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"),
                  spaceAfter=6, spaceBefore=2)

def h1(text):  return [Paragraph(text, S["h1"]), RULE]
def h2(text):  return [Paragraph(text, S["h2"])]
def p(text):   return [Paragraph(text, S["body"])]
def bp(text):  return [Paragraph(f"&#8226;&#160;&#160;{text}", S["bullet"])]
def eq(text):  return [Paragraph(text, S["eq"])]
def sp(n=8):   return [Spacer(1, n)]

def table(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths)
    cmds = [
        ("FONTNAME",       (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f8f8f8"), colors.white]),
        ("GRID",           (0,0), (-1,-1), 0.4, colors.HexColor("#dddddd")),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
    ]
    if header:
        cmds += [
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ]
    t.setStyle(TableStyle(cmds))
    return [t]

doc = SimpleDocTemplate(
    str(OUT), pagesize=letter,
    leftMargin=1.1*inch, rightMargin=1.1*inch,
    topMargin=1.0*inch,  bottomMargin=1.0*inch,
)

story = []

# ── title ─────────────────────────────────────────────────────────────────────
story += [
    Paragraph("Macro Forecasting with IMF Text", S["title"]),
    Paragraph("Final Report — Results and Analysis", S["subtitle"]),
    Paragraph("STAT 8240 Data Mining II · Kennesaw State University · April 2025", S["meta"]),
    RULE,
]

# ── 1. overview ───────────────────────────────────────────────────────────────
story += h1("1. Project Overview")
story += p(
    "This project applies the IMM-TSF framework (Irregular Multimodal Time-Series "
    "Forecasting) to macroeconomic forecasting for 22 countries in Central Asia and "
    "the ECA region. The core research question is whether adding IMF Article IV staff "
    "report text as a second data stream improves GDP growth and inflation forecasts "
    "over numerical-only baselines."
)
story += p(
    "Two data streams are used: (1) World Bank WDI annual macro indicators, and "
    "(2) IMF Article IV consultation reports, which contain qualitative economic "
    "assessments published with a 3–8 month lag relative to the reference year. "
    "Both streams have irregular missingness, which the model architecture handles "
    "explicitly through binary missingness masks rather than imputation."
)

# ── 2. data ───────────────────────────────────────────────────────────────────
story += h1("2. Data")

story += h2("2.1 World Bank WDI Panel")
story += p(
    "Seven macro indicators were pulled from the World Bank API for all 22 countries "
    "across 2005–2023: GDP growth, CPI inflation, current account balance, fiscal "
    "balance, remittances, unemployment rate, and government debt (all as % of GDP "
    "where applicable). Genuine gaps in the panel — particularly for government debt "
    "in Central Asian economies — are preserved as missing rather than imputed."
)

story += h2("2.2 IMF Article IV Staff Reports")
story += p(
    "240 real Article IV staff report texts were collected across 22 countries "
    "(57.4% of the 418 possible country-year pairs). Gaps fall into three categories: "
    "Turkmenistan never publishes its reports publicly; several Central Asian countries "
    "under IMF programs have biennial rather than annual consultations; and Belarus "
    "and Ukraine have suspended or unpublished consultations due to political circumstances."
)
story += p(
    "Text was extracted from pages 1–6 of each PDF (Executive Summary and Staff "
    "Appraisal sections) using PyMuPDF, streamed in memory without writing PDFs to disk."
)

# ── 3. models ─────────────────────────────────────────────────────────────────
story += h1("3. Models and Architecture")

story += h2("3.1 Pre-alignment (Appendix H, Steps 3-4)")
story += p(
    "For each (country, query year) sample, a 10-year lookback window is extracted "
    "producing three tensors:"
)
story += bp("<b>Values</b> X ∈ ℝ^(T×D) — indicator values, zero where missing (T=10, D=7)")
story += bp("<b>Mask</b> M ∈ {0,1}^(T×D) — 1 = observed, 0 = missing")
story += bp("<b>Timestamps</b> τ ∈ [0,1]^T — calendar years normalised to unit interval")
story += sp(4)
story += p("Indicator values are z-scored using training-split statistics only to prevent leakage.")
story += p(
    "<b>Step 3</b> (Appendix H): each context timestep's feature vector is expanded from "
    "D to 2D+1 by concatenating the missingness mask and the normalised timestamp: "
    "[values | mask | τ_t]. This gives the model explicit access to when each observation "
    "was made and whether it was genuinely observed."
)
story += p(
    "<b>Step 4</b> (Appendix H): a query row [0(D) | 0(D) | τ_query] is appended, "
    "making the sequence length T+1. This row has zero values and mask (no observation) "
    "but carries the normalised query year, telling the model where in time it is "
    "predicting. Final input shape: (B, T+1, 2D+1) = (B, 11, 15)."
)

story += h2("3.2 DLinear (baseline)")
story += p(
    "Applies canonical pre-alignment (Steps 3-4) to produce a (B, 11, 15) input tensor, "
    "then decomposes it into trend (moving average, kernel=3) and seasonal residual "
    "components, applying independent linear layers to each flattened component:"
)
story += eq("y_hat = W_trend · trend(X).flatten() + W_season · season(X).flatten()")
story += p(
    "Input dimension after flattening: (T+1) × (2D+1) = 11 × 15 = 165. "
    "Unimodal (numerical only). 664 parameters."
)

story += h2("3.3 NumericalGRU (ablation)")
story += p(
    "A GRU encoder over the canonically pre-aligned numerical series (same Steps 3-4 "
    "as DLinear). Input per timestep is 2D+1=15 features; the query row is appended "
    "so the GRU sees T+1=11 steps:"
)
story += eq("h_t = GRU([x_t ; m_t ; τ_t], h_{t-1}),    y_hat = W · h_{T+1}")
story += p("Unimodal ablation — tests whether temporal dynamics help over the linear baseline. 15,682 parameters.")

story += h2("3.4 RecAvg TTF — Text Context Module")
story += p(
    "The Temporally-Tagged Fusion module aggregates past Article IV embeddings using "
    "a Gaussian kernel weighted by distance from the query year:"
)
story += eq("c(t*) = ( Σ_i  w_i · e_i ) / ( Σ_i  w_i )")
story += eq("w_i = exp( -((t* - t_i) / σ)^2 ),    σ = 1.0 year")
story += p(
    "This naturally handles the IMF publication lag — a report published late "
    "receives less weight than an on-time report. When no text is available, "
    "c(t*) is set to a zero vector."
)

story += h2("3.5 GR-Add MMF — Full Multimodal Model (eq. 12-16)")
story += p(
    "The full multimodal model follows Appendix I.2 (equations 12-16). After the "
    "backbone GRU produces a numerical forecast y_ts and the RecAvg TTF produces "
    "text context e, a fusion GRU ingests z = [y_ts ; e] to produce a hidden state "
    "from which a linear correction and gate are computed:"
)
story += eq("z        = cat( y_ts, e )                              (eq. 12-13)")
story += eq("H        = FusionGRU( z )                              (hidden state)")
story += eq("ΔY       = W_Δ · H + b_Δ                              (eq. 14, linear)")
story += eq("G        = σ( W_g · z + b_g )                         (eq. 15, gate)")
story += eq("y_fused  = G ⊙ y_ts + (1−G) ⊙ (y_ts + ΔY)           (eq. 16)")
story += p(
    "The sigmoid gate G decides how much of the text-informed correction ΔY to apply. "
    "Note: ΔY is a linear (not tanh) projection per eq. 14; the gate W_g operates on "
    "z = [y_ts ; e] (not on the GRU hidden state). "
    "When text is absent (e = 0), ΔY and G become deterministic functions of y_ts "
    "alone and the gate can suppress the correction entirely. 177,866 parameters."
)

# ── 4. embedding models ───────────────────────────────────────────────────────
story += h1("4. Text Embedding Models")
story += p(
    "Two embedding approaches were evaluated, differing in their handling of long "
    "Article IV texts:"
)
story += table(
    [
        ["",                  "BERT-base-uncased",          "text-embedding-3-small (OpenAI)"],
        ["Token limit",       "512 tokens (~380 words)",    "8,191 tokens (~6,000 words)"],
        ["Coverage",          "Truncates early in document","Reads full Executive Summary + Appraisal"],
        ["Dimensions",        "768",                        "768 (via API dimension parameter)"],
        ["Cost",              "Free (local)",               "~$0.01 for all 240 texts"],
    ],
    [1.8*inch, 2.2*inch, 2.5*inch], header=True,
)
story += sp()
story += p(
    "BERT's 512-token limit discards the Staff Appraisal section — the most "
    "forward-looking and signal-rich part of the Article IV report — which typically "
    "appears after the first 500 tokens. The OpenAI model reads the full extracted "
    "text, capturing the complete qualitative assessment."
)

# ── 5. evaluation design ──────────────────────────────────────────────────────
story += h1("5. Evaluation Design")
story += table(
    [
        ["Split",      "Years",     "Samples", "Role"],
        ["Training",   "2005–2019", "102",     "Model fitting"],
        ["Validation", "2020–2021", "40",      "Early stopping (patience = 10)"],
        ["Test",       "2022–2023", "40",      "Final held-out evaluation"],
    ],
    [1.1*inch, 1.1*inch, 0.9*inch, 3.4*inch],
)
story += sp()
story += p(
    "Strict chronological split prevents any data leakage. Primary metric is MSE "
    "on the 2022–2023 test set, reported separately for GDP growth and inflation. "
    "Secondary analysis stratifies the MSE gain (GR-Add MMF vs NumericalGRU) by "
    "per-country text missingness rate to test whether countries with fewer "
    "Article IV reports benefit more from the multimodal signal."
)

# ── 6. results ────────────────────────────────────────────────────────────────
story += h1("6. Results")

story += h2("6.1 Primary Metric — Test Set MSE")
story += table(
    [
        ["Model",          "GDP MSE", "Inf MSE", "Overall MSE", "Params"],
        ["— OpenAI text-embedding-3-small (corrected architecture) —", "", "", "", ""],
        ["DLinear",        "1.344",   "4.027",   "2.685",       "664"],
        ["NumericalGRU",   "1.483",   "4.198",   "2.841",       "15,682"],
        ["GR-Add MMF",     "1.447",   "3.909",   "2.678",       "177,866"],
    ],
    [2.6*inch, 1.0*inch, 1.0*inch, 1.2*inch, 0.9*inch],
)
story += sp()
story += p(
    "GR-Add MMF achieves the best overall MSE (2.678), beating NumericalGRU (2.841) "
    "by 6.0% and narrowly beating DLinear (2.685). The main advantage is concentrated "
    "in inflation forecasting (3.909 vs 4.198, a 6.9% reduction), which aligns with "
    "the content of Article IV reports — they explicitly discuss price pressures, "
    "monetary policy, and exchange rate dynamics. GDP forecast gains are more modest. "
    "Results use the corrected architecture matching the TIME-IMM paper (Appendix H "
    "pre-alignment, eq. 12-16 fusion, corrected RecAvg kernel)."
)

story += h2("6.2 Secondary Analysis — Stratified by Text Missingness")
story += p(
    "Countries were split at the median text missingness rate (36.8%) into low- and "
    "high-missingness groups. The table below shows per-country results with OpenAI "
    "embeddings (gain = NumericalGRU MSE − GR-Add MSE; positive = text helped):"
)
story += table(
    [
        ["Country", "Text miss%", "Texts", "GRU MSE", "GR-Add MSE", "Gain%"],
        ["— LOW missingness (≤36.8%) —", "", "", "", "", ""],
        ["JPN",  "16%", "16", "0.031",  "0.004",  "+86.3%"],
        ["KAZ",  "11%", "17", "0.636",  "0.347",  "+45.5%"],
        ["MKD",  "26%", "14", "1.121",  "1.076",  "+4.0%"],
        ["BGR",  "32%", "13", "0.726",  "0.722",  "+0.6%"],
        ["HRV",  "26%", "14", "0.568",  "0.678",  "−19.5%"],
        ["USA",  "0%",  "19", "0.160",  "0.252",  "−57.6%"],
        ["— HIGH missingness (>36.8%) —", "", "", "", "", ""],
        ["UZB",  "63%", "7",  "0.273",  "0.167",  "+38.8%"],
        ["UKR",  "63%", "7",  "11.842", "11.070", "+6.5%"],
        ["SRB",  "47%", "10", "0.840",  "0.779",  "+7.3%"],
        ["KGZ",  "53%", "9",  "0.802",  "0.879",  "−9.6%"],
        ["ARM",  "53%", "9",  "1.239",  "1.307",  "−5.5%"],
        ["GEO",  "68%", "6",  "0.977",  "1.022",  "−4.7%"],
    ],
    [0.9*inch, 1.0*inch, 0.7*inch, 1.0*inch, 1.1*inch, 1.0*inch],
)
story += sp()
story += p(
    "The hypothesis — that countries with fewer Article IV texts benefit more from "
    "GR-Add — is not clearly confirmed. LOW-missingness countries show 4/6 positive "
    "gains (JPN +86.3%, KAZ +45.5%, MKD +4.0%, BGR +0.6%); HIGH-missingness countries "
    "show only 3/6 positive (UZB +38.8%, UKR +6.5%, SRB +7.3%). The biggest gains "
    "are concentrated in anchor economies (JPN, KAZ) where OpenAI embeddings capture "
    "rich policy context."
)
story += p(
    "Notable outliers: USA (0% text missingness, 19 texts) shows −57.6% — the US "
    "economy is so well-studied numerically that the IMF text likely adds noise. "
    "Ukraine's high GRU MSE (11.842) reflects the exceptional shock volatility of "
    "2022-2023; the text provides marginal improvement (+6.5%)."
)

# ── 7. conclusions ────────────────────────────────────────────────────────────
story += h1("7. Conclusions")
story += bp(
    "<b>Text improves forecasts — with the right embeddings.</b> GR-Add MMF achieves "
    "the best overall MSE (2.678) with OpenAI text-embedding-3-small, beating "
    "NumericalGRU (2.841) by 6.0% and narrowly beating DLinear (2.685). The main gain "
    "is in inflation forecasting (3.909 vs 4.198, 6.9% reduction). BERT's 512-token "
    "limit was discarding the Staff Appraisal section — the most forward-looking part "
    "of the report — making full-context embeddings a necessary prerequisite."
)
story += bp(
    "<b>All models beat the persistence baseline by ~97-98%.</b> Persistence MSE = 105.44; "
    "GR-Add MSE = 2.678 (97.5% reduction). The bulk of this gain comes from the "
    "numerical WDI panel. Text adds an incremental but meaningful improvement on top, "
    "particularly for inflation (6.9% additional reduction)."
)
story += bp(
    "<b>Text-missingness hypothesis not clearly confirmed.</b> 4/6 LOW-miss countries "
    "show positive GR-Add gains (JPN +86.3%, KAZ +45.5%) vs 3/6 HIGH-miss countries "
    "(UZB +38.8%). The biggest gains are in anchor economies with rich text coverage "
    "and high predictive value, not in data-sparse countries as originally hypothesized. "
    "A larger sample is needed to test this definitively."
)
story += bp(
    "<b>The dataset is small by ML standards.</b> With 102 training samples across "
    "22 countries and 5 query years, results should be interpreted cautiously. The "
    "per-country test set has only 2 samples per country (2022 and 2023), making "
    "individual country estimates noisy. Embedding quality is a first-order design "
    "choice — domain-adapted embeddings (fine-tuned on economic text) could further "
    "improve the text signal."
)

story += sp(12)
story += [RULE, Paragraph("— End of Report —", S["caption"])]

doc.build(story)
print(f"Saved: {OUT}")
