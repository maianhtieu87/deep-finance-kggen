ENTITY_PROMPT = """
Extract key entities from the source text.
Extracted entities are subjects or objects.
This is an extraction task, please be thorough and accurate to the reference text.
"""

RELATION_PROMPT = """
Extract subject-predicate-object triples from the source text.
Subject and object must be from entities list.
Entities provided were previously extracted from the same source text.
This is an extraction task, please be thorough, accurate, and faithful to the reference text.
"""

RESOLUTION_PROMPT = """
Find duplicate {item_type} for the item and an alias that best represents the duplicates.
Duplicates are those that are the same in meaning, such as with variation in tense,
plural form, stem form, case, abbreviation, shorthand.
Return an empty list if there are none.
"""

