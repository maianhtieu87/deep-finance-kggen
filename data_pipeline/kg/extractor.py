# data_pipeline/kg/extractor.py - FinDKG STYLE

import dspy
from .prompts import FINDKG_EXTRACTION_PROMPT


class FinDKGExtractor(dspy.Signature):
    """{FINDKG_EXTRACTION_PROMPT}"""
    text = dspy.InputField()
    quintuples = dspy.OutputField(desc="JSON list of (h, h_type, r, o, o_type)")


class KGExtractor:
    """
    FinDKG-style Knowledge Graph Extractor
    
    Features:
    - Quintuple format with entity types
    - 12 entity categories + 15 relation types
    - Entity disambiguation
    - Price-relevance focus
    """
    
    def __init__(self, llm):
        dspy.settings.configure(lm=llm)
        self.extractor = dspy.ChainOfThought(FinDKGExtractor)
    
    def extract(self, text: str, top_k: int = 5):
        """
        Extract top-k price-relevant quintuples.
        
        Args:
            text: News article content
            top_k: Max number of quintuples
        
        Returns:
            List of (head, h_type, relation, tail, t_type)
        """
        result = self.extractor(text=text).quintuples
        
        # Parse JSON output
        import json
        try:
            quintuples = json.loads(result)
        except:
            # Fallback: try ast.literal_eval
            import ast
            quintuples = ast.literal_eval(result)
        
        # Validate format
        valid = []
        for q in quintuples:
            if isinstance(q, (list, tuple)) and len(q) == 5:
                h, h_t, r, o, o_t = q
                # Check relation is valid
                if r in VALID_RELATIONS:
                    valid.append((str(h), str(h_t), str(r), str(o), str(o_t)))
        
        return valid[:top_k]


# Valid relations (from FinDKG)
VALID_RELATIONS = {
    "Has", "Announce", "Operate_In", "Introduce", "Produce",
    "Control", "Participates_In", "Impact", "Positive_Impact_On",
    "Negative_Impact_On", "Relate_To", "Is_Member_Of",
    "Invests_In", "Raise", "Decrease"
}