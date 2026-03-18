# data_pipeline/kg/prompts.py
VALID_ENTITY_TYPES = [
    "COMP", "PERSON", "ORG_GOV", "ORG_REG",
    "PRODUCT", "ECON_IND", "FIN_ASSET", "CONCEPT",
]

VALID_RELATIONS = [
    "ANNOUNCES", "RAISES", "CUTS", "INVESTS_IN", "DIVESTS", "APPOINTS",
    "POS_IMPACTS", "NEG_IMPACTS", "COMPETES_WITH", "REGULATES", "SUPPLIES_TO",
    "CONTROLS", "SIGNALS", "RELATES_TO",
]

TICKER_SECTOR_MAP = {
    "AAPL": "Technology",   "MSFT": "Technology",   "GOOGL": "Technology",
    "GOOG": "Technology",   "META": "Technology",   "NVDA":  "Technology",
    "TSLA": "Consumer Discretionary",  "AMZN": "Consumer Discretionary",
    "BA":   "Industrials",  "JPM":  "Financials",   "WMT":   "Consumer Staples",
}

FINDKG_LITE_SYSTEM_PROMPT = """You are a financial knowledge graph extractor. From financial news, extract structured triples that predict TARGET TICKER stock price movement within 1-5 trading days.

\u2501\u2501 ENTITY TYPES \u2501\u2501  (EXACT codes only)
COMP      Company \u2014 short canonical name \u22644 words: "Apple", "Rivian", "CATL"
PERSON    Named individual \u22644 words: "Andy Jassy", "Jerome Powell"
ORG_GOV   Government body or central bank: "Federal Reserve", "White House"
ORG_REG   Regulatory/legal body: "FTC", "SEC", "EU Commission", "Irish Data Protection"
PRODUCT   Named product/service/platform/chip: "AWS", "iPhone 16", "HarmonyOS"
ECON_IND  Economic metric WITH value: "Q1 revenue $143B", "rate 5.5%", "stock -3.2%"
FIN_ASSET Financial instrument: "AMZN stock", "S&P 500", "10-year Treasury"
CONCEPT   Abstract market theme \u2014 ONLY as last resort.
          \u2713 "AI adoption", "tariff risk", "supply chain disruption"
          \u2717 "[Product] + vague outcome": "AWS opportunity", "iPhone growth potential"
             \u2192 Make AWS/iPhone the SUBJECT instead.
          \u2717 Any name that is itself a company, product, person, or numbered metric.
          Decision rule: Can you use COMP/PRODUCT/ECON_IND? \u2192 do that first.

Entity rules:
  1. Disambiguate: "Amazon CEO", "Jassy" \u2192 always "Andy Jassy" (PERSON)
  2. \u22644 words per name; include key number in ECON_IND names
  3. No self-loops: subject.name \u2260 object.name (check after stripping legal suffixes)

\u2501\u2501 RELATION TYPES \u2501\u2501  (prefer Group A/B over C)
Group A \u2014 Direct Action:
  ANNOUNCES  Earnings, product launch, M&A, guidance, official statement,
             or any observed stock/price movement with a specific number.
  RAISES     Increase: price, capex, guidance, output, rating
  CUTS       Decrease: workforce, cost, price, guidance, dividend
  INVESTS_IN Capital deployment: acquisition, capex, partnership
             Object must be COMP, PRODUCT, or ECON_IND (with $ amount). NEVER CONCEPT.
  DIVESTS    Exit: sells stake, spins off, withdraws
  APPOINTS   Leadership appointment or removal

Group B \u2014 Causal Impact:
  POS_IMPACTS   Entity A improves Entity B\'s financials or outlook
  NEG_IMPACTS   Entity A worsens Entity B\'s financials or outlook
  COMPETES_WITH Direct market competition for same revenue/customers
  REGULATES     [regulator] REGULATES [regulated entity] \u2014 direction is critical:
                Subject MUST be ORG_GOV or ORG_REG. Object is COMP or PRODUCT.
                "FTC fines Apple" \u2192 FTC REGULATES Apple \u2713
                "Apple REGULATES FTC" \u2192 WRONG \u2717  (flip it)
                Multiple bodies, same event \u2192 keep max 2, highest enforcement impact.
  SUPPLIES_TO   [seller/maker] SUPPLIES_TO [buyer] \u2014 direction critical:
                "Rivian builds vans for Amazon" \u2192 Rivian SUPPLIES_TO Amazon \u2713

Group C \u2014 Context (only when A/B do not apply):
  CONTROLS    [owner/regulator] CONTROLS [controlled entity]
  SIGNALS     Forward-looking statement or analyst forecast
  RELATES_TO  Thematic link \u2014 absolute last resort.
              NEVER use for: stock/price movements with numbers (\u2192 ANNOUNCES),
              CEO/exec identity facts with no price mechanism (\u2192 skip triple),
              or resolved historical context with no current financial impact.

\u2501\u2501 SCORING \u2501\u2501
confidence: 0.95+=explicit+numbers | 0.80\u20130.94=clearly stated | 0.65\u20130.79=implied | <0.65\u2192skip

price_impact_score (-1.0\u2192+1.0 for TARGET TICKER):
  \u00b10.7\u20131.0 Severe/major (>\u00b15%) | \u00b10.4\u20130.7 Significant (\u00b12\u20135%)
  \u00b10.1\u20130.4 Moderate (\u00b10.5\u20132%)  | 0\u2013\u00b10.1 Weak

relevance_to_ticker \u2014 skip triple if <0.50:
  1.0=direct | 0.80\u20130.99=key supplier/customer/shareholder
  0.65\u20130.79=direct competitor  | 0.50\u20130.64=sector/macro | <0.50\u2192do NOT extract"""


FINDKG_LITE_USER_PROMPT = """TARGET_STOCK: {ticker}
SECTOR: {sector}
NEWS_DATE: {news_date}

ARTICLE:
{news_text}

\u2501\u2501\u2501 STEP 1 \u2014 CLASSIFY ARTICLE TYPE \u2501\u2501\u2501
Read the article and select ONE type:

  TYPE A \u2014 Earnings / Guidance / Financial Results
    Signals: revenue numbers, EPS, beats/misses, guidance changes, margins
    Extract: one triple per distinct metric or segment; expect 4\u20137 triples

  TYPE B \u2014 Corporate Deal / M&A / Supply Chain / Partnership
    Signals: acquisition, investment amount, named supplier/customer, contract value
    Extract: structural relationships + deal facts; expect 3\u20135 triples

  TYPE C \u2014 Competitive / Regulatory / Geopolitical
    Signals: competitor action, regulatory ruling, government policy affecting sector
    Extract: threat/opportunity signals with context; expect 2\u20134 triples

  TYPE D \u2014 Opinion / Interview / Executive Philosophy / Soft News
    Signals: NO financial guidance, NO product announcement, NO deal disclosed
    Extract: ONLY if a direct, quantifiable financial statement is present; expect 0\u20132 triples
    If no direct financial statement \u2192 return []

\u2501\u2501\u2501 STEP 2 \u2014 EXTRACT TRIPLES \u2501\u2501\u2501
Based on the article type from Step 1, extract ALL qualifying events.

For each event, output one triple:
{{
  "subject": {{"name": "<\u22644 words, canonical>",  "type": "<CODE>"}},
  "relation": "<RELATION_CODE>",
  "object":  {{"name": "<\u22644 words, include key number if metric>", "type": "<CODE>"}},
  "confidence":          <0.65\u20131.0>,
  "price_impact_score":  <-1.0 to +1.0 for {ticker}>,
  "relevance_to_ticker": <0.50\u20131.0>,
  "reasoning": "<15 words max: event + direct price mechanism for {ticker}>"
}}

\u2501\u2501\u2501 EXTRACTION RULES \u2501\u2501\u2501
Coverage \u2014 every paragraph with a number needs \u22651 triple:
  \u2713 Revenue / EPS / margin / guidance numbers \u2192 ECON_IND with value in name
  \u2713 Named product / service / chip \u2192 PRODUCT
  \u2713 Named exec statement with financial content \u2192 PERSON SIGNALS or ANNOUNCES
  \u2713 Workforce cut (even if framed as "streamlining") \u2192 CUTS
  \u2713 Supply/ownership/competitor relationships \u2192 structural triple
  \u2713 Stock/share price movement with number \u2192 ANNOUNCES [ECON_IND], NOT RELATES_TO

Deduplication \u2014 same person or same event, same context:
  Analyst/exec: max 2 triples \u2014 best quantitative claim + best forward-looking
  REGULATES event cluster: max 2 REGULATES triples per regulatory event \u2014 keep
    the body with enforcement power (fine/restriction amount) + most impacted
    product/service; skip other bodies reporting on the same ruling.
  Do NOT create 3+ SIGNALS triples for one analyst in one report.

Noise filter \u2014 always skip triple if EITHER condition is true:
  \u2022 relevance_to_ticker < 0.50
  \u2022 confidence < 0.65

Final check before returning:
  \u2611 No self-loops (subject \u2260 object, even after stripping Inc./Corp./Ltd.)
  \u2611 SUPPLIES_TO direction: [maker] \u2192 [buyer]
  \u2611 REGULATES subject is ORG_GOV or ORG_REG \u2014 if COMP is subject, flip direction
  \u2611 Stock/price movement with a number \u2192 ANNOUNCES [ECON_IND], NOT RELATES_TO
  \u2611 INVESTS_IN object is COMP/PRODUCT/ECON_IND \u2014 if CONCEPT, change to SIGNALS
  \u2611 CONCEPT name is not "[Product/Company] + vague word" \u2014 use entity as SUBJECT
  \u2611 Negative events extracted even if article frames them positively
  \u2611 No duplicate (subject, relation, object) triples
  \u2611 TYPE D with no financial content \u2192 return []"""


FEW_SHOT_EXAMPLES = [
    {
        "input": (
            "Apple reported Q3 earnings: EPS $1.26 beat estimate $1.15. "
            "iPhone revenue grew 18% YoY to $39.6B. "
            "Services revenue hit record $21.2B, up 14%. "
            "CFO Luca Maestri raised full-year revenue guidance by 3%."
        ),
        "ticker": "AAPL",
        "article_type": "TYPE A \u2014 Earnings",
        "output": [
            {"subject": {"name": "Apple", "type": "COMP"}, "relation": "ANNOUNCES",
             "object": {"name": "EPS $1.26 beat $1.15", "type": "ECON_IND"},
             "confidence": 0.97, "price_impact_score": 0.65, "relevance_to_ticker": 1.0,
             "reasoning": "9.6% EPS beat drives 2-5% typical next-day gain for AAPL."},
            {"subject": {"name": "iPhone", "type": "PRODUCT"}, "relation": "RAISES",
             "object": {"name": "revenue +18% to $39.6B", "type": "ECON_IND"},
             "confidence": 0.97, "price_impact_score": 0.55, "relevance_to_ticker": 1.0,
             "reasoning": "iPhone 50% of AAPL revenue; 18% growth above consensus."},
            {"subject": {"name": "Apple Services", "type": "PRODUCT"}, "relation": "ANNOUNCES",
             "object": {"name": "Services record $21.2B +14%", "type": "ECON_IND"},
             "confidence": 0.97, "price_impact_score": 0.50, "relevance_to_ticker": 1.0,
             "reasoning": "Services record signals margin expansion at premium multiple."},
            {"subject": {"name": "Luca Maestri", "type": "PERSON"}, "relation": "RAISES",
             "object": {"name": "AAPL guidance +3% full-year", "type": "FIN_ASSET"},
             "confidence": 0.92, "price_impact_score": 0.60, "relevance_to_ticker": 1.0,
             "reasoning": "CFO guidance raise is strongest sustained re-rating signal."},
        ],
    },
    {
        "input": (
            "Rivian cut 35% material cost on vans built for major shareholder Amazon. "
            "Rivian posted Q1 net loss $1.5B; cash fell to $8B. "
            "Tesla price cuts pressure smaller EV makers including Rivian."
        ),
        "ticker": "AMZN",
        "article_type": "TYPE B \u2014 Supply Chain",
        "output": [
            {"subject": {"name": "Rivian", "type": "COMP"}, "relation": "SUPPLIES_TO",
             "object": {"name": "Amazon", "type": "COMP"},
             "confidence": 0.97, "price_impact_score": 0.20, "relevance_to_ticker": 0.85,
             "reasoning": "Rivian is key AMZN van supplier; stability affects logistics."},
            {"subject": {"name": "Rivian", "type": "COMP"}, "relation": "CUTS",
             "object": {"name": "van material cost -35%", "type": "ECON_IND"},
             "confidence": 0.95, "price_impact_score": 0.25, "relevance_to_ticker": 0.82,
             "reasoning": "Cost cut improves Rivian viability; AMZN ~18% stake benefits."},
            {"subject": {"name": "Rivian", "type": "COMP"}, "relation": "ANNOUNCES",
             "object": {"name": "Q1 loss $1.5B cash $8B", "type": "ECON_IND"},
             "confidence": 0.95, "price_impact_score": -0.30, "relevance_to_ticker": 0.80,
             "reasoning": "Cash burn risks AMZN's ~18% investment value."},
            {"subject": {"name": "Amazon", "type": "COMP"}, "relation": "CONTROLS",
             "object": {"name": "Rivian", "type": "COMP"},
             "confidence": 0.90, "price_impact_score": -0.15, "relevance_to_ticker": 0.85,
             "reasoning": "AMZN major shareholder; Rivian distress is contingent liability."},
            {"subject": {"name": "Tesla", "type": "COMP"}, "relation": "NEG_IMPACTS",
             "object": {"name": "Rivian", "type": "COMP"},
             "confidence": 0.85, "price_impact_score": -0.20, "relevance_to_ticker": 0.65,
             "reasoning": "Tesla price cuts raise Rivian failure risk, threatening AMZN stake."},
        ],
    },
    {
        "input": (
            "EU's Irish Data Protection Commission fined Meta \u20ac1.2B for GDPR violations. "
            "The European Data Protection Board also ordered Meta to halt EU-US data transfers. "
            "Meta's stock dropped 3.2% immediately following the announcement. "
            "Apple faces a similar ongoing EU investigation but no ruling yet."
        ),
        "ticker": "META",
        "article_type": "TYPE C \u2014 Regulatory",
        "output": [
            {
                "subject": {"name": "Irish Data Protection", "type": "ORG_REG"},
                "relation": "REGULATES",
                "object": {"name": "Meta", "type": "COMP"},
                "confidence": 0.97, "price_impact_score": -0.65, "relevance_to_ticker": 1.0,
                "reasoning": "\u20ac1.2B GDPR fine directly impacts META cash and flags EU regulatory risk.",
            },
            {
                "subject": {"name": "Meta", "type": "COMP"},
                "relation": "ANNOUNCES",
                "object": {"name": "META stock -3.2%", "type": "ECON_IND"},
                "confidence": 0.95, "price_impact_score": -0.30, "relevance_to_ticker": 1.0,
                "reasoning": "Immediate 3.2% drop reflects market pricing EU regulatory risk.",
            },
        ],
    },
    {
        "input": (
            "Amazon Q1 net sales $143.3B beat $142.5B estimate, +13% YoY. "
            "AWS annualized run rate exceeded $100B. "
            "CEO Jassy: capex will meaningfully increase for AI. "
            "Amazon cut 200 AWS jobs citing overlapping roles."
        ),
        "ticker": "AMZN",
        "article_type": "TYPE A \u2014 Earnings",
        "output": [
            {"subject": {"name": "Amazon", "type": "COMP"}, "relation": "ANNOUNCES",
             "object": {"name": "Q1 sales $143.3B beat $142.5B", "type": "ECON_IND"},
             "confidence": 0.97, "price_impact_score": 0.65, "relevance_to_ticker": 1.0,
             "reasoning": "13% revenue beat drives AMZN analyst estimate revisions upward."},
            {"subject": {"name": "AWS", "type": "PRODUCT"}, "relation": "ANNOUNCES",
             "object": {"name": "AWS run rate $100B+", "type": "ECON_IND"},
             "confidence": 0.95, "price_impact_score": 0.60, "relevance_to_ticker": 1.0,
             "reasoning": "AWS $100B+ signals re-acceleration of AMZN highest-margin segment."},
            {"subject": {"name": "Andy Jassy", "type": "PERSON"}, "relation": "SIGNALS",
             "object": {"name": "AMZN AI capex 2024", "type": "CONCEPT"},
             "confidence": 0.90, "price_impact_score": 0.45, "relevance_to_ticker": 1.0,
             "reasoning": "CEO capex signal confirms strong AWS demand pipeline."},
            {"subject": {"name": "Amazon", "type": "COMP"}, "relation": "CUTS",
             "object": {"name": "AWS workforce 200 jobs", "type": "ECON_IND"},
             "confidence": 0.88, "price_impact_score": -0.20, "relevance_to_ticker": 1.0,
             "reasoning": "Job cuts mild negative on execution risk despite efficiency framing."},
        ],
    },
    {
        "input": (
            "Netflix CEO Reed Hastings praised Jeff Bezos\' business philosophy. "
            "Hastings said he adopted Bezos\' risk-taking approach to greenlight shows. "
            "No financial guidance or product announcements were made."
        ),
        "ticker": "NFLX",
        "article_type": "TYPE D \u2014 Opinion/Interview",
        "output": [],
    },
]