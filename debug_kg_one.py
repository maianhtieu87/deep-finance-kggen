# debug_kg_one.py

import os
import pandas as pd

from configs.config import GlobalConfig
from data_pipeline.processors.news_processor import KGGenDSPyExtractor


def main():
    TICKER = "MSFT"
    DATE_STR = "2025-03-14"  # yyyy-mm-dd

    news_path = os.path.join(GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet")
    assert os.path.exists(news_path), f"Missing news file: {news_path}"

    df = pd.read_parquet(news_path)

    # Chuẩn hoá cột
    if "equity" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "equity"})
    if "content" not in df.columns:
        if "body" in df.columns:
            df = df.rename(columns={"body": "content"})
        elif "text" in df.columns:
            df = df.rename(columns={"text": "content"})
    if "title" not in df.columns and "headline" in df.columns:
        df = df.rename(columns={"headline": "title"})

    df["date"] = pd.to_datetime(df["date"]).dt.date
    target_date = pd.to_datetime(DATE_STR).date()

    subset = df[(df["equity"] == TICKER) & (df["date"] == target_date)]
    print(f"✅ Found {len(subset)} news rows for {TICKER} on {DATE_STR}")

    if len(subset) == 0:
        return

    row = subset.iloc[0]
    text_col = "content" if "content" in subset.columns else "title"
    text = str(row[text_col])

    print("================================================================================")
    print(f"[DEBUG] Single article for {TICKER} on {DATE_STR}")
    print(f"Chars in article: {len(text)}")
    print("--------------------------------------------------------------------------------")
    print("FULL TEXT ==================================================")
    print(text)   # hoặc merged_text nếu debug dạng per-day
    print("============================================================")


    extractor = KGGenDSPyExtractor()
    triples = extractor.extract(text)

    print(f"✅ Extracted {len(triples)} triples (sorted by price impact):")
    for i, t in enumerate(triples, start=1):
        s, p, o = t
        print(f"{i}. ({s!r}, {p!r}, {o!r})")


if __name__ == "__main__":
    main()