# debug_gemini_kg.py

from data_pipeline.processors.news_processor import KGGenDSPyExtractor

TEST_TEXT = """
Tesla announced that it will build a new factory in Germany.
The Federal Reserve increased interest rates by 0.25%.
"""

def main():
    print("🔍 Testing KGGenDSPyExtractor with Gemini...")
    extractor = KGGenDSPyExtractor()
    ents, triples = extractor.extract(TEST_TEXT)
    print("\nEntities:")
    for e in ents:
        print(" -", e)
    print("\nTriples:")
    for t in triples:
        print(" -", t)

if __name__ == "__main__":
    main()
