# data_pipeline/kg/resolver.py - IMPROVED VERSION
"""
Improved Entity Resolution với 2-Step Process

CHANGES:
- Step 1: Hard rules (exact match, normalization)
- Step 2: Soft clustering (KMeans on embeddings)
- Step 3 (Optional): LLM verification for ambiguous cases
"""

import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class ImprovedEntityResolver:
    """
    Enhanced Entity Resolution with multi-step approach.
    
    Process:
    1. Normalization & Hard Rules (exact matches after cleaning)
    2. Semantic Clustering (for remaining entities)
    3. Optional LLM Verification (for high-value entities)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        kmeans_k: int = 128,
        min_cluster_size: int = 2,
        use_llm_verification: bool = False
    ):
        """
        Args:
            model_name: SentenceTransformer model for embeddings
            kmeans_k: Number of clusters for semantic grouping
            min_cluster_size: Min entities to form a cluster
            use_llm_verification: Enable LLM-based verification
        """
        self.model = SentenceTransformer(model_name)
        self.kmeans_k = kmeans_k
        self.min_cluster_size = min_cluster_size
        self.use_llm_verification = use_llm_verification
        
        # Canonical name mapping (hard-coded common aliases)
        self.canonical_map = self._load_canonical_aliases()
        
        # Results
        self.entity_to_canonical = {}
        self.canonical_to_variants = defaultdict(set)
    
    def _load_canonical_aliases(self) -> Dict[str, str]:
        """
        Load hard-coded entity aliases for common companies.
        
        Returns:
            Dict mapping variants to canonical names
        """
        aliases = {
            # Tech Giants
            "apple": "Apple Inc.",
            "apple inc": "Apple Inc.",
            "aapl": "Apple Inc.",
            "microsoft": "Microsoft Corporation",
            "microsoft corp": "Microsoft Corporation",
            "msft": "Microsoft Corporation",
            "amazon": "Amazon.com Inc.",
            "amazon.com": "Amazon.com Inc.",
            "amzn": "Amazon.com Inc.",
            "google": "Alphabet Inc.",
            "alphabet": "Alphabet Inc.",
            "googl": "Alphabet Inc.",
            "goog": "Alphabet Inc.",
            "meta": "Meta Platforms Inc.",
            "facebook": "Meta Platforms Inc.",
            "fb": "Meta Platforms Inc.",
            "tesla": "Tesla Inc.",
            "tsla": "Tesla Inc.",
            "netflix": "Netflix Inc.",
            "nflx": "Netflix Inc.",
            
            # Financial
            "jpmorgan": "JPMorgan Chase & Co.",
            "jp morgan": "JPMorgan Chase & Co.",
            "j.p. morgan": "JPMorgan Chase & Co.",
            "jpm": "JPMorgan Chase & Co.",
            "goldman sachs": "Goldman Sachs Group Inc.",
            "gs": "Goldman Sachs Group Inc.",
            "morgan stanley": "Morgan Stanley",
            "ms": "Morgan Stanley",
            "bank of america": "Bank of America Corporation",
            "bofa": "Bank of America Corporation",
            "bac": "Bank of America Corporation",
            
            # Add more as needed...
        }
        return aliases
    
    def _normalize_entity(self, entity: str) -> str:
        """
        Normalize entity string for matching.
        
        Steps:
        - Lowercase
        - Remove punctuation except dots in abbreviations
        - Remove common suffixes (Inc., Corp., Ltd.)
        - Strip whitespace
        
        Args:
            entity: Raw entity string
        
        Returns:
            Normalized string
        """
        if not entity:
            return ""
        
        # Lowercase
        normalized = entity.lower().strip()
        
        # Remove common legal suffixes
        suffixes = [
            r'\binc\.?$',
            r'\bcorp\.?$',
            r'\bcorporation$',
            r'\bco\.?$',
            r'\bcompany$',
            r'\bltd\.?$',
            r'\blimited$',
            r'\bgroup$',
            r'\bthe\b',
        ]
        
        for suffix in suffixes:
            normalized = re.sub(suffix, '', normalized, flags=re.IGNORECASE)
        
        # Remove extra punctuation (keep dots for abbreviations like "J.P.")
        normalized = re.sub(r'[^\w\s\.]', ' ', normalized)
        
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _step1_hard_rules(self, entities: List[str]) -> Tuple[Dict[str, str], Set[str]]:
        """
        Step 1: Apply hard rules for exact matches.
        
        Args:
            entities: List of raw entity strings
        
        Returns:
            (resolved_map, unresolved_set): 
                - resolved_map: {original -> canonical}
                - unresolved_set: Entities that need clustering
        """
        resolved = {}
        unresolved = set()
        
        # Track which canonicals we've seen
        seen_canonicals = {}
        
        for entity in entities:
            normalized = self._normalize_entity(entity)
            
            # Check canonical alias map
            if normalized in self.canonical_map:
                canonical = self.canonical_map[normalized]
                resolved[entity] = canonical
                seen_canonicals[normalized] = canonical
                continue
            
            # Check if we've seen this exact normalized form before
            if normalized in seen_canonicals:
                resolved[entity] = seen_canonicals[normalized]
                continue
            
            # No match found - needs clustering
            unresolved.add(entity)
            seen_canonicals[normalized] = entity  # Use original as canonical for now
        
        print(f"✅ Step 1 (Hard Rules): Resolved {len(resolved)}/{len(entities)} entities")
        print(f"   Remaining for clustering: {len(unresolved)}")
        
        return resolved, unresolved
    
    def _step2_semantic_clustering(self, entities: Set[str]) -> Dict[str, str]:
        """
        Step 2: Cluster unresolved entities using embeddings.
        
        Args:
            entities: Set of unresolved entity strings
        
        Returns:
            Dict mapping entities to cluster representatives
        """
        if len(entities) < 2:
            # Not enough to cluster
            return {e: e for e in entities}
        
        entity_list = sorted(list(entities))
        
        # Encode entities
        print(f"🔮 Encoding {len(entity_list)} entities for clustering...")
        embeddings = self.model.encode(entity_list, show_progress_bar=False)
        
        # Determine optimal K
        k = min(self.kmeans_k, len(entity_list) // self.min_cluster_size)
        k = max(k, 1)
        
        print(f"🎯 Running KMeans with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        # Map entities to cluster representatives
        cluster_map = {}
        cluster_members = defaultdict(list)
        
        for entity, label in zip(entity_list, labels):
            cluster_members[label].append(entity)
        
        # Choose representative (shortest name in cluster)
        for label, members in cluster_members.items():
            # Sort by length, prefer shorter names
            representative = sorted(members, key=lambda x: len(x))[0]
            
            for entity in members:
                cluster_map[entity] = representative
        
        print(f"✅ Step 2 (Clustering): Formed {len(cluster_members)} clusters")
        
        return cluster_map
    
    def _step3_llm_verification(self, cluster_map: Dict[str, str]) -> Dict[str, str]:
        """
        Step 3 (Optional): Use LLM to verify ambiguous clusters.
        
        For high-value entities, we can ask an LLM:
        "Are 'JPM' and 'JPMorgan Chase' the same company?"
        
        Args:
            cluster_map: Initial clustering results
        
        Returns:
            Verified and potentially refined cluster map
        """
        if not self.use_llm_verification:
            return cluster_map
        
        print("🤖 Step 3 (LLM Verification): Not implemented yet")
        # TODO: Implement LLM verification
        # This would involve:
        # 1. Identify clusters with >3 members or high variance
        # 2. Query LLM for confirmation
        # 3. Split clusters if LLM says they're different
        
        return cluster_map
    
    def resolve(self, entities: List[str]) -> Dict[str, str]:
        """
        Main resolution pipeline.
        
        Args:
            entities: List of raw entity strings from KG triples
        
        Returns:
            Dict mapping each entity to its canonical form
        """
        print(f"\n🔧 Starting Entity Resolution for {len(entities)} unique entities...")
        
        # Remove empty/None entities
        entities = [e for e in entities if e and isinstance(e, str)]
        
        if not entities:
            return {}
        
        # Step 1: Hard Rules
        step1_resolved, unresolved = self._step1_hard_rules(entities)
        
        # Step 2: Semantic Clustering (for unresolved)
        step2_resolved = {}
        if unresolved:
            step2_resolved = self._step2_semantic_clustering(unresolved)
        
        # Step 3: LLM Verification (optional)
        step2_verified = self._step3_llm_verification(step2_resolved)
        
        # Merge results
        final_map = {**step1_resolved, **step2_verified}
        
        # Build reverse mapping
        for entity, canonical in final_map.items():
            self.canonical_to_variants[canonical].add(entity)
        
        self.entity_to_canonical = final_map
        
        print(f"✅ Resolution Complete!")
        print(f"   Total Canonical Entities: {len(self.canonical_to_variants)}")
        print(f"   Avg Variants per Entity: {len(entities) / max(len(self.canonical_to_variants), 1):.2f}")
        
        return final_map
    
    def get_canonical(self, entity: str) -> str:
        """Get canonical form of an entity."""
        if entity in self.entity_to_canonical:
            return self.entity_to_canonical[entity]
        
        # Try normalization + canonical map
        normalized = self._normalize_entity(entity)
        if normalized in self.canonical_map:
            return self.canonical_map[normalized]
        
        # Fallback: return original
        return entity
    
    def save_mapping(self, filepath: str):
        """Save entity mapping to JSON."""
        output = {
            "entity_to_canonical": self.entity_to_canonical,
            "canonical_to_variants": {
                k: list(v) for k, v in self.canonical_to_variants.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Entity mapping saved to {filepath}")
    
    def load_mapping(self, filepath: str):
        """Load entity mapping from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.entity_to_canonical = data["entity_to_canonical"]
        self.canonical_to_variants = {
            k: set(v) for k, v in data["canonical_to_variants"].items()
        }
        
        print(f"📥 Entity mapping loaded from {filepath}")
        print(f"   Canonical entities: {len(self.canonical_to_variants)}")


# === USAGE EXAMPLE ===
if __name__ == "__main__":
    # Example entities from KG
    test_entities = [
        "Apple Inc.",
        "apple",
        "AAPL",
        "Tesla Inc.",
        "tesla",
        "TSLA",
        "JPMorgan Chase & Co.",
        "JP Morgan",
        "J.P. Morgan",
        "JPM",
        "Microsoft Corporation",
        "Microsoft Corp.",
        "MSFT",
        "Amazon.com, Inc.",
        "Amazon",
        "AMZN",
    ]
    
    resolver = ImprovedEntityResolver(
        kmeans_k=10,
        min_cluster_size=2,
        use_llm_verification=False
    )
    
    mapping = resolver.resolve(test_entities)
    
    print("\n📋 Sample Mappings:")
    for original, canonical in list(mapping.items())[:10]:
        print(f"   {original:30} -> {canonical}")
    
    # Save mapping
    resolver.save_mapping("entity_mapping.json")