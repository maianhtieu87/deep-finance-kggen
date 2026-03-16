# data_pipeline/kg/prompts.py
# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — imported by extractor.py, extractor_batch.py, news_processor.py
# ─────────────────────────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = [
    "COMP",
    "PERSON",
    "ORG_GOV",
    "ORG_REG",
    "PRODUCT",
    "ECON_IND",
    "FIN_ASSET",
    "CONCEPT",
]

VALID_RELATIONS = [
    # Group A — Direct Action
    "ANNOUNCES", "RAISES", "CUTS", "INVESTS_IN", "DIVESTS", "APPOINTS",
    # Group B — Causal Impact
    "POS_IMPACTS", "NEG_IMPACTS", "COMPETES_WITH", "REGULATES", "SUPPLIES_TO",
    # Group C — Context
    "CONTROLS", "SIGNALS", "RELATES_TO",
]

TICKER_SECTOR_MAP = {
    "AAPL": "Technology",   "MSFT": "Technology",   "GOOGL": "Technology",
    "GOOG": "Technology",   "META": "Technology",   "NVDA":  "Technology",
    "INTC": "Technology",   "AMD":  "Technology",   "AVGO":  "Technology",
    "QCOM": "Technology",   "TXN":  "Technology",   "MU":    "Technology",
    "AMAT": "Technology",   "LRCX": "Technology",   "KLAC":  "Technology",
    "ASML": "Technology",   "TSM":  "Technology",   "ORCL":  "Technology",
    "CRM":  "Technology",   "SAP":  "Technology",   "IBM":   "Technology",
    "HPQ":  "Technology",   "DELL": "Technology",   "ON":    "Technology",
    "MRVL": "Technology",   "MCHP": "Technology",   "SWKS":  "Technology",
    "MPWR": "Technology",
    "TSLA": "Consumer Discretionary",  "AMZN": "Consumer Discretionary",
    "NKE":  "Consumer Discretionary",  "MCD":  "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",  "TM":   "Consumer Discretionary",
    "GM":   "Consumer Discretionary",  "F":    "Consumer Discretionary",
    "RIVN": "Consumer Discretionary",  "LCID": "Consumer Discretionary",
    "JPM":  "Financials",  "BAC":  "Financials",  "WFC":  "Financials",
    "GS":   "Financials",  "MS":   "Financials",  "C":    "Financials",
    "BLK":  "Financials",  "V":    "Financials",  "MA":   "Financials",
    "AXP":  "Financials",  "PYPL": "Financials",  "SQ":   "Financials",
    "JNJ":  "Healthcare",  "PFE":  "Healthcare",  "MRK":  "Healthcare",
    "ABBV": "Healthcare",  "LLY":  "Healthcare",  "BMY":  "Healthcare",
    "AMGN": "Healthcare",  "GILD": "Healthcare",  "CVS":  "Healthcare",
    "UNH":  "Healthcare",
    "XOM":  "Energy",  "CVX":  "Energy",  "COP":  "Energy",
    "SLB":  "Energy",  "EOG":  "Energy",
    "BA":   "Industrials",  "LMT":  "Industrials",  "RTX":  "Industrials",
    "GE":   "Industrials",  "CAT":  "Industrials",  "HON":  "Industrials",
    "UPS":  "Industrials",  "FDX":  "Industrials",
    "NFLX": "Communication Services",  "DIS":   "Communication Services",
    "T":    "Communication Services",  "VZ":    "Communication Services",
    "CMCSA":"Communication Services",  "SNAP":  "Communication Services",
    "TWTR": "Communication Services",  "SPOT":  "Communication Services",
    "AMT":  "Real Estate",  "PLD":  "Real Estate",
    "NEE":  "Utilities",    "DUK":  "Utilities",
}


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
#
# Changes vs previous version:
#   • CONCEPT: explicit ✗ patterns + decision rule to prevent over-use
#   • INVESTS_IN: object-type constraint (COMP/PRODUCT/ECON_IND only, NOT CONCEPT)
#   • SCORING: compressed to ~8 lines (-~100 tokens); threshold raised to 0.50
# ─────────────────────────────────────────────────────────────────────────────

FINDKG_LITE_SYSTEM_PROMPT = """You are a financial knowledge graph extractor. From financial news, extract structured triples that predict TARGET TICKER stock price movement within 1-5 trading days.

━━ ENTITY TYPES ━━  (EXACT codes only)
COMP      Company — short canonical name ≤4 words: "Apple", "Rivian", "CATL"
PERSON    Named individual ≤4 words: "Andy Jassy", "Jerome Powell"
ORG_GOV   Government body or central bank: "Federal Reserve", "White House"
ORG_REG   Regulatory/legal body: "FTC", "SEC", "EU Commission"
PRODUCT   Named product/service/platform/chip: "AWS", "iPhone 16", "HarmonyOS"
ECON_IND  Economic metric WITH value: "Q1 revenue $143B", "rate 5.5%", "loss $1.5B"
FIN_ASSET Financial instrument: "AMZN stock", "S&P 500", "10-year Treasury"
CONCEPT   Abstract market theme or strategic narrative — ONLY as last resort.
          ✓ "AI adoption", "tariff risk", "supply chain disruption"
          ✓ "AMZN AI capex 2024" — strategic narrative anchored to a ticker for clarity
          ✗ "[Product] + vague outcome": "AWS opportunity", "iPhone growth potential"
             → These hide a real entity: make AWS/iPhone the SUBJECT instead.
          ✗ Any name that is itself a company, person, product, or numbered metric.
          Decision rule: Can you express this as COMP/PRODUCT/ECON_IND? → do that. Only
          use CONCEPT when the meaning is truly a macro theme or unnamed strategic idea.

Entity rules:
  1. Disambiguate: "Amazon CEO", "Jassy" → always "Andy Jassy" (PERSON)
  2. ≤4 words per name; include key number in ECON_IND names
  3. No self-loops: subject.name ≠ object.name

━━ RELATION TYPES ━━  (prefer Group A/B over C)
Group A — Direct Action:
  ANNOUNCES  Earnings, product launch, M&A, guidance, official statement
  RAISES     Increase: price, capex, guidance, output, rating
  CUTS       Decrease: workforce, cost, price, guidance, dividend
  INVESTS_IN Capital deployment: acquisition, capex, partnership
             Object must be COMP (target of acquisition), PRODUCT (infrastructure),
             or ECON_IND (with dollar amount). NEVER CONCEPT.
  DIVESTS    Exit: sells stake, spins off, withdraws
  APPOINTS   Leadership appointment or removal

Group B — Causal Impact:
  POS_IMPACTS   Entity A improves Entity B's financials or outlook
  NEG_IMPACTS   Entity A worsens Entity B's financials or outlook
  COMPETES_WITH Direct market competition for same revenue/customers
  REGULATES     Regulatory investigation, fine, or restriction
  SUPPLIES_TO   [seller/maker] SUPPLIES_TO [buyer] — direction critical:
                "Rivian builds vans for Amazon" → Rivian SUPPLIES_TO Amazon ✓
                "Amazon SUPPLIES_TO Rivian" → WRONG ✗

Group C — Context (only when A/B do not apply):
  CONTROLS    [owner/regulator] CONTROLS [controlled entity]
  SIGNALS     Forward-looking statement or analyst forecast
  RELATES_TO  Thematic link — absolute last resort

━━ SCORING ━━
confidence: 0.95+=explicit+numbers | 0.80–0.94=clearly stated | 0.65–0.79=implied | <0.65→skip

price_impact_score (-1.0→+1.0 for TARGET TICKER):
  ±0.7–1.0 Severe/major (>±5%) | ±0.4–0.7 Significant (±2–5%)
  ±0.1–0.4 Moderate (±0.5–2%)  | 0–±0.1 Weak
  Positive = bullish for target ticker; Negative = bearish for target ticker.

relevance_to_ticker — skip triple if <0.50:
  1.0=direct | 0.80–0.99=key supplier/customer/shareholder
  0.65–0.79=direct competitor  | 0.50–0.64=sector/macro | <0.50→do NOT extract"""


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT
#
# Changes vs previous version:
#   • Noise filter: simplified to absolute thresholds (rel<0.50 OR conf<0.65)
#   • Final check: added INVESTS_IN object type guard
#   • relevance range in JSON template: updated to 0.50–1.0
# ─────────────────────────────────────────────────────────────────────────────

FINDKG_LITE_USER_PROMPT = """TARGET_STOCK: {ticker}
SECTOR: {sector}
NEWS_DATE: {news_date}

ARTICLE:
{news_text}

━━━ STEP 1 — CLASSIFY ARTICLE TYPE ━━━
Read the article and select ONE type:

  TYPE A — Earnings / Guidance / Financial Results
    Signals: revenue numbers, EPS, beats/misses, guidance changes, margins
    Extract: one triple per distinct metric or segment; expect 4–7 triples

  TYPE B — Corporate Deal / M&A / Supply Chain / Partnership
    Signals: acquisition, investment amount, named supplier/customer, contract value
    Extract: structural relationships + deal facts; expect 3–5 triples

  TYPE C — Competitive / Regulatory / Geopolitical
    Signals: competitor action, regulatory ruling, government policy affecting sector
    Extract: threat/opportunity signals with context; expect 2–4 triples

  TYPE D — Opinion / Interview / Executive Philosophy / Soft News
    Signals: NO financial guidance, NO product announcement, NO deal disclosed
    Extract: ONLY if a direct, quantifiable financial statement is present; expect 0–2 triples
    If no direct financial statement → return []

━━━ STEP 2 — EXTRACT TRIPLES ━━━
Based on the article type from Step 1, extract ALL qualifying events.

For each event, output one triple:
{{
  "subject": {{"name": "<≤4 words, canonical>",  "type": "<CODE>"}},
  "relation": "<RELATION_CODE>",
  "object":  {{"name": "<≤4 words, include key number if metric>", "type": "<CODE>"}},
  "confidence":          <0.65–1.0>,
  "price_impact_score":  <-1.0 to +1.0 for {ticker}>,
  "relevance_to_ticker": <0.50–1.0>,
  "reasoning": "<15 words max: event + direct price mechanism for {ticker}>"
}}

━━━ EXTRACTION RULES ━━━
Coverage — every paragraph with a number needs ≥1 triple:
  ✓ Revenue / EPS / margin / guidance numbers → ECON_IND with value in name
  ✓ Named product / service / chip → PRODUCT
  ✓ Named exec statement with financial content → PERSON SIGNALS or ANNOUNCES
  ✓ Workforce cut (even if framed as "streamlining") → CUTS
  ✓ Supply/ownership/competitor relationships → structural triple

Deduplication — same person, same event context:
  Max 2 triples: best quantitative claim + best forward-looking statement
  Do NOT create 3+ SIGNALS triples for one executive in one call/event

Noise filter — always skip triple if EITHER condition is true:
  • relevance_to_ticker < 0.50
  • confidence < 0.65

Final check before returning:
  ☑ No self-loops (subject ≠ object)
  ☑ SUPPLIES_TO direction: [maker] → [buyer]
  ☑ INVESTS_IN object is COMP/PRODUCT/ECON_IND — if you wrote CONCEPT, change to SIGNALS
  ☑ CONCEPT name is not "[Product/Company] + vague word" — if so, use that entity as SUBJECT
  ☑ Negative events extracted even if article frames them positively
  ☑ No duplicate (subject, relation, object) triples
  ☑ TYPE D with no financial content → return []"""


# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES
#
# Changes vs previous version:
#   • Removed example 4 (TYPE C Azure dedup) — dedup rule is sufficient in text
#   • Removed example 7 (TYPE D edge-case TSLA/CATL) — covered by TYPE B + TYPE D
#   • 5 examples remain: TYPE A, TYPE B, TYPE C, TYPE A-mixed, TYPE D-empty
#   • All examples comply with new CONCEPT and threshold rules
# ─────────────────────────────────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = [

    # ── TYPE A: Earnings — one triple per distinct metric ────────────────────
    {
        "input": (
            "Apple reported Q3 earnings: EPS $1.26 beat estimate $1.15. "
            "iPhone revenue grew 18% YoY to $39.6B. "
            "Services revenue hit record $21.2B, up 14%. "
            "CFO Luca Maestri raised full-year revenue guidance by 3%."
        ),
        "ticker": "AAPL",
        "article_type": "TYPE A — Earnings",
        "output": [
            {
                "subject":             {"name": "Apple", "type": "COMP"},
                "relation":            "ANNOUNCES",
                "object":              {"name": "EPS $1.26 beat $1.15", "type": "ECON_IND"},
                "confidence":          0.97,
                "price_impact_score":  0.65,
                "relevance_to_ticker": 1.0,
                "reasoning":           "9.6% EPS beat drives 2-5% typical next-day gain for AAPL.",
            },
            {
                "subject":             {"name": "iPhone", "type": "PRODUCT"},
                "relation":            "RAISES",
                "object":              {"name": "revenue +18% to $39.6B", "type": "ECON_IND"},
                "confidence":          0.97,
                "price_impact_score":  0.55,
                "relevance_to_ticker": 1.0,
                "reasoning":           "iPhone 50% of AAPL revenue; 18% growth above consensus.",
            },
            {
                "subject":             {"name": "Apple Services", "type": "PRODUCT"},
                "relation":            "ANNOUNCES",
                "object":              {"name": "Services record $21.2B +14%", "type": "ECON_IND"},
                "confidence":          0.97,
                "price_impact_score":  0.50,
                "relevance_to_ticker": 1.0,
                "reasoning":           "Services record signals margin expansion at premium multiple.",
            },
            {
                "subject":             {"name": "Luca Maestri", "type": "PERSON"},
                "relation":            "RAISES",
                "object":              {"name": "AAPL guidance +3% full-year", "type": "FIN_ASSET"},
                "confidence":          0.92,
                "price_impact_score":  0.60,
                "relevance_to_ticker": 1.0,
                "reasoning":           "CFO guidance raise is strongest sustained re-rating signal.",
            },
        ],
    },

    # ── TYPE B: Supply chain — SUPPLIES_TO direction + INVESTS_IN with COMP object ─
    # NOTE: INVESTS_IN object = COMP (Amazon, the acquisition target of stake), not CONCEPT
    {
        "input": (
            "Rivian cut 35% material cost on vans built for major shareholder Amazon. "
            "Rivian posted Q1 net loss $1.5B; cash fell to $8B. "
            "Tesla price cuts pressure smaller EV makers including Rivian."
        ),
        "ticker": "AMZN",
        "article_type": "TYPE B — Supply Chain",
        "output": [
            {
                "subject":             {"name": "Rivian", "type": "COMP"},
                "relation":            "SUPPLIES_TO",
                "object":              {"name": "Amazon", "type": "COMP"},
                "confidence":          0.97,
                "price_impact_score":  0.20,
                "relevance_to_ticker": 0.85,
                "reasoning":           "Rivian is key AMZN van supplier; stability affects logistics.",
            },
            {
                "subject":             {"name": "Rivian", "type": "COMP"},
                "relation":            "CUTS",
                "object":              {"name": "van material cost -35%", "type": "ECON_IND"},
                "confidence":          0.95,
                "price_impact_score":  0.25,
                "relevance_to_ticker": 0.82,
                "reasoning":           "Cost cut improves Rivian viability; AMZN ~18% stake benefits.",
            },
            {
                "subject":             {"name": "Rivian", "type": "COMP"},
                "relation":            "ANNOUNCES",
                "object":              {"name": "Q1 loss $1.5B cash $8B", "type": "ECON_IND"},
                "confidence":          0.95,
                "price_impact_score":  -0.30,
                "relevance_to_ticker": 0.80,
                "reasoning":           "Cash burn risks AMZN's ~18% investment value.",
            },
            {
                "subject":             {"name": "Amazon", "type": "COMP"},
                "relation":            "CONTROLS",
                "object":              {"name": "Rivian", "type": "COMP"},
                "confidence":          0.90,
                "price_impact_score":  -0.15,
                "relevance_to_ticker": 0.85,
                "reasoning":           "AMZN major shareholder; Rivian distress is contingent liability.",
            },
            {
                "subject":             {"name": "Tesla", "type": "COMP"},
                "relation":            "NEG_IMPACTS",
                "object":              {"name": "Rivian", "type": "COMP"},
                "confidence":          0.85,
                "price_impact_score":  -0.20,
                "relevance_to_ticker": 0.65,
                "reasoning":           "Tesla price cuts raise Rivian failure risk, threatening AMZN stake.",
            },
        ],
    },

    # ── TYPE C: Competitive/Geopolitical — impact is NEGATIVE for MSFT ───────
    # NOTE: relevance 0.40 triples are NOT extracted (threshold 0.50)
    {
        "input": (
            "Huawei's HarmonyOS surpassed iOS in China; PC version launching soon. "
            "China promotes OpenHarmony as national OS to reduce Western dependency. "
            "Microsoft's China revenue is ~1.5% of total."
        ),
        "ticker": "MSFT",
        "article_type": "TYPE C — Competitive/Geopolitical",
        "output": [
            {
                "subject":             {"name": "HarmonyOS", "type": "PRODUCT"},
                "relation":            "COMPETES_WITH",
                "object":              {"name": "Windows", "type": "PRODUCT"},
                "confidence":          0.90,
                "price_impact_score":  -0.35,
                "relevance_to_ticker": 0.75,
                "reasoning":           "HarmonyOS PC launch directly threatens Windows China share.",
            },
            {
                "subject":             {"name": "China", "type": "ORG_GOV"},
                "relation":            "NEG_IMPACTS",
                "object":              {"name": "Microsoft", "type": "COMP"},
                "confidence":          0.75,
                "price_impact_score":  -0.25,
                "relevance_to_ticker": 0.65,
                "reasoning":           "National OS policy reduces MSFT market access in China.",
            },
            {
                "subject":             {"name": "Microsoft", "type": "COMP"},
                "relation":            "RELATES_TO",
                "object":              {"name": "China revenue 1.5%", "type": "ECON_IND"},
                "confidence":          0.90,
                "price_impact_score":  -0.10,
                "relevance_to_ticker": 1.0,
                "reasoning":           "Low China exposure limits actual MSFT financial impact.",
            },
        ],
    },

    # ── TYPE A: Mixed — negative events alongside positive; CONCEPT used correctly ──
    # NOTE: "AMZN AI capex 2024" is valid CONCEPT (strategic narrative with ticker)
    #       "AWS workforce 200 jobs" is ECON_IND not CONCEPT (it has a number)
    #       INVESTS_IN is NOT used here — capex guidance → PERSON SIGNALS CONCEPT
    {
        "input": (
            "Amazon Q1 net sales $143.3B beat $142.5B estimate, +13% YoY. "
            "AWS annualized run rate exceeded $100B. "
            "CEO Jassy: capex will meaningfully increase for AI. "
            "Amazon cut 200 AWS jobs citing overlapping roles."
        ),
        "ticker": "AMZN",
        "article_type": "TYPE A — Earnings",
        "output": [
            {
                "subject":             {"name": "Amazon", "type": "COMP"},
                "relation":            "ANNOUNCES",
                "object":              {"name": "Q1 sales $143.3B beat $142.5B", "type": "ECON_IND"},
                "confidence":          0.97,
                "price_impact_score":  0.65,
                "relevance_to_ticker": 1.0,
                "reasoning":           "13% revenue beat drives AMZN analyst estimate revisions upward.",
            },
            {
                "subject":             {"name": "AWS", "type": "PRODUCT"},
                "relation":            "ANNOUNCES",
                "object":              {"name": "AWS run rate $100B+", "type": "ECON_IND"},
                "confidence":          0.95,
                "price_impact_score":  0.60,
                "relevance_to_ticker": 1.0,
                "reasoning":           "AWS $100B+ signals re-acceleration of AMZN highest-margin segment.",
            },
            {
                "subject":             {"name": "Andy Jassy", "type": "PERSON"},
                "relation":            "SIGNALS",
                "object":              {"name": "AMZN AI capex 2024", "type": "CONCEPT"},
                "confidence":          0.90,
                "price_impact_score":  0.45,
                "relevance_to_ticker": 1.0,
                "reasoning":           "CEO capex signal confirms strong AWS demand pipeline.",
            },
            {
                "subject":             {"name": "Amazon", "type": "COMP"},
                "relation":            "CUTS",
                "object":              {"name": "AWS workforce 200 jobs", "type": "ECON_IND"},
                "confidence":          0.88,
                "price_impact_score":  -0.20,
                "relevance_to_ticker": 1.0,
                "reasoning":           "Job cuts mild negative on execution risk despite efficiency framing.",
            },
        ],
    },

    # ── TYPE D: Opinion/soft news → always return [] ─────────────────────────
    {
        "input": (
            "Netflix CEO Reed Hastings praised Jeff Bezos' business philosophy. "
            "Hastings said he adopted Bezos' risk-taking approach to greenlight shows. "
            "No financial guidance or product announcements were made."
        ),
        "ticker": "NFLX",
        "article_type": "TYPE D — Opinion/Interview",
        "output": [],
        # No quantifiable financial statement → TYPE D rule → return []
    },
]