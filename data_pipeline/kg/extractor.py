import dspy
from .prompts import ENTITY_PROMPT, RELATION_PROMPT

class EntityExtractor(dspy.Signature):
    """{ENTITY_PROMPT}"""
    text = dspy.InputField()
    entities = dspy.OutputField(desc="list of entities")

class RelationExtractor(dspy.Signature):
    """{RELATION_PROMPT}"""
    text = dspy.InputField()
    entities = dspy.InputField()
    triples = dspy.OutputField(desc="list of (subject, predicate, object)")

class KGExtractor:
    def __init__(self, llm):
        dspy.settings.configure(lm=llm)
        self.entity_prog = dspy.ChainOfThought(EntityExtractor)
        self.relation_prog = dspy.ChainOfThought(RelationExtractor)

    def extract(self, text):
        ent = self.entity_prog(text=text).entities
        rel = self.relation_prog(text=text, entities=ent).triples
        return ent, rel
