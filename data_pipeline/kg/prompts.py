# data_pipeline/kg/prompts.py

# --- General KG prompts (cũ, vẫn giữ để không lỗi các chỗ khác) ---

ENTITY_PROMPT = """
Extract key entities from the source text.
Extracted entities are subjects or objects.
This is an extraction task, please be thorough and accurate to the reference text.
Return a valid Python list of strings, e.g. ["Amazon", "AWS", "global IT spend"].
"""

RELATION_PROMPT = """
Extract subject-predicate-object triples from the source text.
Subject and object must be taken from the provided entities list.
Entities provided were previously extracted from the same source text.
This is an extraction task, please be thorough, accurate, and faithful to the reference text.
Return a valid Python list of 3-element tuples, e.g.:
[
  ("Amazon", "operates", "AWS"),
  ("AWS", "serves", "enterprise customers")
]
"""

RESOLUTION_PROMPT = """
Find duplicate {item_type} for the item and an alias that best represents the duplicates.
Duplicates are those that are the same in meaning, such as with variation in tense,
plural form, stem form, case, abbreviation, shorthand.
Return an empty list if there are none.
Return a valid Python list.
"""

# --- New: Financial price-impact triple prompt ---

PRICE_IMPACT_PROMPT = """
You are a financial information extraction assistant.

Given the full news article about a listed company, extract AT MOST 5
(subject, predicate, object) triples that are most relevant to the
STOCK PRICE of the main company in the article.

Guidelines:
- Focus on concrete events, numbers, or forward-looking statements about:
  revenue, profit, guidance, demand, margins, capex, regulation,
  macro factors, competition, product launches, layoffs, or major deals.
- Subject should be a company, business segment, product, metric, or macro variable.
  Examples: "AWS", "Amazon", "global IT spend", "Q1 net sales".
- Predicate should describe a clear action or relation, e.g.:
  "achieved", "grew", "declined", "expects", "raised guidance for",
  "cut", "faces investigation from", "announced", "will invest in".
- Object should carry the key magnitude / direction / timeframe, e.g.:
  "100 billion annualized revenue run rate",
  "over 85% on-premises",
  "13% year-over-year",
  "over the next 10–20 years",
  "in 2024",
  "above Street estimate of 142.5 billion USD".

Important:
- Prefer triples that a human analyst would actually use to explain or
  justify a price move for the main stock.
- Avoid vague or generic statements without financial or strategic content.

Output format:
Return ONLY a valid Python list of 3-element tuples, for example:

[
  ("AWS", "achieved", "100 billion annualized revenue run rate"),
  ("Global IT spend", "remains", "over 85% on-premises"),
  ("Amazon", "expects", "higher AWS capital expenditures in 2024")
]

Do not add any explanation or commentary outside the Python list literal.
"""