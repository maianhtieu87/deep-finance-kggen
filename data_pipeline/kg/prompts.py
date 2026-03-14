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
# Design: FinDKG-inspired conciseness — every rule in ≤2 lines.
# Key additions over FinDKG:
#   • price_impact_score anchored to expected % price move
#   • SUPPLIES_TO direction rule with concrete examples
#   • Entity disambiguation + <4-word cap (from FinDKG)
#   • CONCEPT gated explicitly
# ─────────────────────────────────────────────────────────────────────────────

FINDKG_LITE_SYSTEM_PROMPT = """You are a financial knowledge graph extractor. From financial news, extract structured triples that predict TARGET TICKER stock price movement within 1-5 trading days.

━━ ENTITY TYPES ━━  (EXACT codes only)
COMP      Company — short canonical name ≤4 words: "AWS", "Apple", "Rivian"
PERSON    Named individual — full name ≤4 words: "Andy Jassy", "Jerome Powell"
ORG_GOV   Government body or central bank: "Federal Reserve", "White House"
ORG_REG   Regulatory/legal body: "FTC", "SEC", "EU Commission"
PRODUCT   Named product/service/platform/chip: "AWS", "iPhone 16", "HarmonyOS"
ECON_IND  Economic metric WITH value: "Q1 revenue $143B", "rate 5.5%", "loss $1.5B"
FIN_ASSET Financial instrument: "AMZN stock", "S&P 500", "10-year Treasury"
CONCEPT   Abstract theme only — NEVER for named entities or metrics with numbers.
          Valid: "AI adoption", "tariff risk", "ESG investment"
          Invalid: "AWS growth potential", "earnings beat", "rate hike pressure"

Entity rules (apply to every triple before writing):
  1. Disambiguate: "Amazon CEO", "Jassy", "Andy Jassy" → always "Andy Jassy" (PERSON)
  2. Simplify: entity names ≤4 words; include key number in ECON_IND names
  3. No self-loops: subject.name ≠ object.name

━━ RELATION TYPES ━━  (EXACT codes — prefer Group A/B over C)
Group A — Direct Action:
  ANNOUNCES  Earnings, product launch, M&A, guidance, official statement
  RAISES     Increase: price, capex, guidance, output, rating
  CUTS       Decrease: workforce, cost, price, guidance, dividend
  INVESTS_IN Capital deployment: acquisition, capex, partnership
  DIVESTS    Exit: sells stake, spins off, withdraws
  APPOINTS   Leadership appointment or removal

Group B — Causal Impact:
  POS_IMPACTS   Entity A improves Entity B's financials or outlook
  NEG_IMPACTS   Entity A worsens Entity B's financials or outlook
  COMPETES_WITH Direct market competition for same revenue/customers
  REGULATES     Regulatory investigation, fine, or restriction
  SUPPLIES_TO   [seller/maker] SUPPLIES_TO [buyer] — direction is critical:
                "Rivian builds vans for Amazon" → Rivian SUPPLIES_TO Amazon ✓
                "Amazon SUPPLIES_TO Rivian" → WRONG ✗

Group C — Context (only when A/B do not apply):
  CONTROLS    [owner/regulator] CONTROLS [controlled entity]
  SIGNALS     Forward-looking statement or analyst forecast
  RELATES_TO  Thematic link — absolute last resort

━━ SCORING ━━
confidence (0.50–1.0):
  0.95+ explicit with numbers | 0.80–0.94 clearly stated | 0.65–0.79 implied | <0.65 skip

price_impact_score for TARGET TICKER (-1.0 to +1.0):
  +0.7→+1.0  Major beat / transformative event  (>+5% expected move)
  +0.4→+0.7  Beat / strong product / key win    (+2–5%)
  +0.1→+0.4  Moderate positive / indirect        (+0.5–2%)
   0.0→+0.1  Weakly positive
  -0.1→ 0.0  Weakly negative
  -0.4→-0.1  Moderate negative: miss/delay/cost  (-0.5–2%)
  -0.7→-0.4  Significant: major miss/regulatory  (-2–5%)
  -1.0→-0.7  Severe: fraud/shutdown/catastrophic (>-5%)

relevance_to_ticker (0.30–1.0):
  1.0   directly about target | 0.80–0.99 key supplier/customer/shareholder
  0.60–0.79 direct competitor | 0.40–0.59 same sector | 0.30–0.39 macro
  <0.30 do NOT extract"""


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPT
# Design: FinDKG two-step structure (classify → extract) eliminates
# over/under-extraction by anchoring expected triple count per article type.
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
  "relevance_to_ticker": <0.30–1.0>,
  "reasoning": "<15 words max: event + direct price mechanism for {ticker}>"
}}

━━━ EXTRACTION RULES ━━━
Coverage — every paragraph with a number needs ≥1 triple:
  ✓ Revenue / EPS / margin / guidance numbers → ECON_IND with value in name
  ✓ Named product / service / chip → PRODUCT
  ✓ Named exec statement with financial content → PERSON + SIGNALS or ANNOUNCES
  ✓ Workforce cut (even if framed as "streamlining") → CUTS
  ✓ Supply/ownership/competitor relationships → structural triple

Deduplication — same person, same event context:
  Max 2 triples: best quantitative claim + best forward-looking statement
  Do NOT create 3+ SIGNALS triples for one executive in one call/event

Noise filter — skip if ALL of the following are true:
  • Operational detail with no stated dollar/revenue impact
  • confidence < 0.75  AND  relevance_to_ticker < 0.60

Final check before returning:
  ☑ No self-loops (subject ≠ object)
  ☑ SUPPLIES_TO direction: [maker] → [buyer]
  ☑ Negative events extracted even if article frames them positively
  ☑ No duplicate (subject, relation, object) triples
  ☑ TYPE D with no financial content → return []"""


# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOT EXAMPLES
# One example per article type to anchor expected behavior.
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
                "reasoning":           "iPhone 50% of AAPL revenue; 18% growth above consensus signals strong cycle.",
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

    # ── TYPE B: Supply chain — correct SUPPLIES_TO direction ─────────────────
    # DIRECTION RULE: Rivian manufactures → Rivian SUPPLIES_TO Amazon (not reverse)
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
                "reasoning":           "Rivian is key AMZN van supplier; stability affects AMZN logistics.",
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
                "reasoning":           "AMZN major shareholder; Rivian distress is contingent AMZN liability.",
            },
            {
                "subject":             {"name": "Tesla", "type": "COMP"},
                "relation":            "NEG_IMPACTS",
                "object":              {"name": "Rivian", "type": "COMP"},
                "confidence":          0.85,
                "price_impact_score":  -0.20,
                "relevance_to_ticker": 0.65,
                "reasoning":           "Tesla price cuts raise Rivian failure risk, threatening AMZN investment.",
            },
        ],
    },

    # ── TYPE C: Competitive/Geopolitical — threat signals with context ────────
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
                "reasoning":           "HarmonyOS PC launch directly threatens Windows China market share.",
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

    # ── TYPE C: DEDUP rule — multiple CEO statements → max 2 triples ─────────
    {
        "input": (
            "Microsoft CEO Satya Nadella: Azure grew 31%, AI contributing 7pp. "
            "He expects AI revenue to accelerate through FY2025. "
            "He noted competitive dynamics with AWS remain intense."
        ),
        "ticker": "MSFT",
        "article_type": "TYPE C — Competitive (with earnings signal)",
        "output": [
            {
                "subject":             {"name": "Azure", "type": "PRODUCT"},
                "relation":            "RAISES",
                "object":              {"name": "revenue +31% AI 7pp", "type": "ECON_IND"},
                "confidence":          0.97,
                "price_impact_score":  0.65,
                "relevance_to_ticker": 1.0,
                "reasoning":           "Azure 31% growth with 7pp AI exceeds estimates; drives MSFT re-rating.",
            },
            {
                "subject":             {"name": "Satya Nadella", "type": "PERSON"},
                "relation":            "SIGNALS",
                "object":              {"name": "MSFT AI revenue FY2025", "type": "CONCEPT"},
                "confidence":          0.85,
                "price_impact_score":  0.50,
                "relevance_to_ticker": 1.0,
                "reasoning":           "CEO FY2025 AI acceleration guidance sustains MSFT growth premium.",
            },
            # Competitive comment subsumed by Azure growth number — no 3rd triple
        ],
    },

    # ── TYPE A: Mixed — negative events must be extracted alongside positive ──
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
                "object":              {"name": "AMZN AI capex increase 2024", "type": "CONCEPT"},
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

    # ── TYPE D: Opinion/Soft news → return [] ────────────────────────────────
    {
        "input": (
            "Netflix CEO Reed Hastings praised Jeff Bezos' business philosophy. "
            "Hastings said he adopted Bezos' risk-taking approach to greenlight shows. "
            "No financial guidance or product announcements were made."
        ),
        "ticker": "NFLX",
        "article_type": "TYPE D — Opinion/Interview",
        "output": [],
        # Reasoning: No direct financial statement, no quantifiable event.
        # TYPE D rule → return []
    },

    # ── TYPE D edge case: soft article WITH one financial statement ───────────
    {
        "input": (
            "The Federal Reserve raised interest rates 25bps to 5.50%. "
            "Tesla's key battery supplier CATL raised cell prices 8%."
        ),
        "ticker": "TSLA",
        "article_type": "TYPE A — Macro + Supply Chain",
        "output": [
            {
                "subject":             {"name": "Federal Reserve", "type": "ORG_GOV"},
                "relation":            "RAISES",
                "object":              {"name": "fed funds rate 5.50%", "type": "ECON_IND"},
                "confidence":          1.0,
                "price_impact_score":  -0.45,
                "relevance_to_ticker": 0.65,
                "reasoning":           "Rate hike raises EV loan costs, reducing TSLA demand.",
            },
            {
                "subject":             {"name": "CATL", "type": "COMP"},
                "relation":            "RAISES",
                "object":              {"name": "battery cell price +8%", "type": "ECON_IND"},
                "confidence":          0.90,
                "price_impact_score":  -0.55,
                "relevance_to_ticker": 0.90,
                "reasoning":           "CATL 8% price hike compresses TSLA gross margin ~150bps.",
            },
            {
                "subject":             {"name": "CATL", "type": "COMP"},
                "relation":            "SUPPLIES_TO",
                "object":              {"name": "Tesla", "type": "COMP"},
                "confidence":          0.95,
                "price_impact_score":  -0.15,
                "relevance_to_ticker": 0.90,
                "reasoning":           "CATL supply dependency means price changes directly hit TSLA P&L.",
            },
        ],
    },
]