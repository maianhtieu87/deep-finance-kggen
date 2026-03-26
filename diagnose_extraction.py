#!/usr/bin/env python3
"""
diagnose_extraction.py — Validate multi-article concat extraction results.

Checks:
  1. Cache integrity: files readable, valid JSON, correct format
  2. Article-triple attribution: article_index correctly stripped
  3. Triple quality stats: counts, relation distribution, confidence
  4. Multi-call consistency: articles in same batch have reasonable triple counts
  5. Compare with old cache (if exists): detect regressions

Usage:
    python diagnose_extraction.py                                 # scan all cache
    python diagnose_extraction.py --ticker TSLA             # filter ticker
    python diagnose_extraction.py --date 2023-07            # filter date prefix
    python diagnose_extraction.py --date 2023-07-22 --ticker TSLA --verbose
"""
from __future__ import annotations
import argparse, json, os, sys, re, textwrap
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from configs.config import GlobalConfig

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER SETUP (LƯU OUTPUT RA FILE VÀ HIỂN THỊ TERMINAL)
# ─────────────────────────────────────────────────────────────────────────────
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ─────────────────────────────────────────────────────────────────────────────
# CACHE SCANNING
# ─────────────────────────────────────────────────────────────────────────────

def scan_cache(cache_dir: str) -> List[Dict]:
    """Load all cache files and return list of {sha1, path, data, errors}."""
    results = []
    if not os.path.exists(cache_dir):
        print(f"  Cache dir not found: {cache_dir}")
        return results

    files = [f for f in os.listdir(cache_dir) if f.endswith(".json") and not f.startswith("_")]
    print(f"  Scanning {len(files)} cache files in {cache_dir}...")

    for fname in sorted(files):
        sha1 = fname.replace(".json", "")
        path = os.path.join(cache_dir, fname)
        entry = {"sha1": sha1, "path": path, "errors": []}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entry["data"] = data
            entry["triples"] = data.get("triples", [])
            entry["version"] = data.get("_v", "unknown")
            entry["meta"] = data.get("_meta", {})
        except json.JSONDecodeError as e:
            entry["errors"].append(f"Invalid JSON: {e}")
            entry["data"] = None
            entry["triples"] = []
        except Exception as e:
            entry["errors"].append(f"Read error: {e}")
            entry["data"] = None
            entry["triples"] = []

        results.append(entry)

    return results


def filter_entries(entries: List[Dict], ticker: str = None, date_prefix: str = None) -> List[Dict]:
    """Filter cache entries by ticker and/or date from _meta."""
    if not ticker and not date_prefix:
        return entries
    filtered = []
    for e in entries:
        meta = e.get("meta", {})
        if ticker:
            pt = meta.get("primary_ticker", "")
            if pt.upper() != ticker.upper():
                continue
        if date_prefix:
            d = meta.get("date", "")
            if not str(d).startswith(date_prefix):
                continue
        filtered.append(e)
    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1: Cache Integrity
# ─────────────────────────────────────────────────────────────────────────────

def check_integrity(entries: List[Dict]) -> Dict:
    """Validate cache file format and content."""
    stats = {
        "total": len(entries),
        "valid": 0,
        "invalid_json": 0,
        "empty_triples": 0,
        "has_triples": 0,
        "versions": Counter(),
        "errors": [],
    }

    for e in entries:
        if e["errors"]:
            stats["invalid_json"] += 1
            for err in e["errors"]:
                stats["errors"].append(f"  {e['sha1'][:8]}: {err}")
            continue

        stats["valid"] += 1
        stats["versions"][e["version"]] += 1

        triples = e["triples"]
        if not triples:
            stats["empty_triples"] += 1
        else:
            stats["has_triples"] += 1

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2: Article-Triple Attribution (article_index leak check)
# ─────────────────────────────────────────────────────────────────────────────

def check_article_index(entries: List[Dict]) -> Dict:
    """Check that article_index is properly stripped from cached triples."""
    stats = {
        "triples_checked": 0,
        "has_article_index": 0,     # BAD: should be stripped
        "clean": 0,                  # GOOD: no article_index
        "leaked_files": [],
    }

    for e in entries:
        for t in e.get("triples", []):
            stats["triples_checked"] += 1
            if "article_index" in t:
                stats["has_article_index"] += 1
                if e["sha1"][:8] not in [f[:8] for f in stats["leaked_files"]]:
                    stats["leaked_files"].append(e["sha1"][:8])
            else:
                stats["clean"] += 1

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3: Triple Quality Stats
# ─────────────────────────────────────────────────────────────────────────────

def check_quality(entries: List[Dict]) -> Dict:
    """Aggregate triple quality metrics."""
    all_triples = []
    for e in entries:
        all_triples.extend(e.get("triples", []))

    if not all_triples:
        return {"total_triples": 0}

    relations = Counter()
    conf_sum, rel_sum, imp_sum = 0.0, 0.0, 0.0
    has_src = 0
    has_reasoning = 0

    A = {"ANNOUNCES", "RAISES", "CUTS", "INVESTS_IN", "DIVESTS", "APPOINTS"}
    B = {"POS_IMPACTS", "NEG_IMPACTS", "COMPETES_WITH", "REGULATES", "SUPPLIES_TO"}

    for t in all_triples:
        rel = t.get("relation", "UNKNOWN")
        relations[rel] += 1
        conf_sum += float(t.get("confidence", 0))
        rel_sum  += float(t.get("relevance_to_ticker", 0))
        imp_sum  += float(t.get("price_impact_score", 0))
        if t.get("_src"):
            has_src += 1
        if t.get("reasoning"):
            has_reasoning += 1

    n = len(all_triples)
    ga = sum(v for k, v in relations.items() if k in A)
    gb = sum(v for k, v in relations.items() if k in B)

    return {
        "total_triples": n,
        "avg_per_article": n / max(1, sum(1 for e in entries if e.get("triples"))),
        "avg_confidence": conf_sum / n,
        "avg_relevance": rel_sum / n,
        "avg_impact": imp_sum / n,
        "group_a": ga,
        "group_b": gb,
        "group_c": n - ga - gb,
        "group_ab_pct": 100 * (ga + gb) / n if n else 0,
        "has_src_tag": has_src,
        "has_reasoning": has_reasoning,
        "relations": relations.most_common(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4: Multi-Call Consistency
# ─────────────────────────────────────────────────────────────────────────────

def check_multi_call(entries: List[Dict]) -> Dict:
    """Check if multi-article calls produced reasonable results."""
    stats = {
        "articles_with_0_triples": 0,
        "articles_with_many_triples": 0,  # >15
        "max_triples_per_article": 0,
        "distribution": Counter(),
    }

    for e in entries:
        n = len(e.get("triples", []))
        stats["distribution"][min(n, 20)] += 1
        if n == 0:
            stats["articles_with_0_triples"] += 1
        if n > 15:
            stats["articles_with_many_triples"] += 1
        stats["max_triples_per_article"] = max(stats["max_triples_per_article"], n)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5: Specific Date Deep Dive
# ─────────────────────────────────────────────────────────────────────────────

def deep_dive(entries: List[Dict], verbose: bool = False):
    """Print detailed info for filtered entries."""
    by_date = defaultdict(list)
    for e in entries:
        d = e.get("meta", {}).get("date", "unknown")
        by_date[d].append(e)

    for date_str in sorted(by_date.keys()):
        date_entries = by_date[date_str]
        total_triples = sum(len(e.get("triples", [])) for e in date_entries)
        print(f"\n  {date_str}: {len(date_entries)} articles → {total_triples} triples")

        for e in date_entries:
            sha1 = e["sha1"][:8]
            n_t = len(e.get("triples", []))
            ver = e.get("version", "?")
            pt = e.get("meta", {}).get("primary_ticker", "?")
            print(f"    [{sha1}] v={ver} ticker={pt} triples={n_t}")

            if verbose:
                full_text = e.get("meta", {}).get("full_text", "")
                if full_text:
                    print(f"\n      [ORIGINAL CONTENT]")
                    print(f"      {'-'*70}")
                    print(textwrap.indent(full_text.strip(), '        '))
                    print(f"      {'-'*70}")

                if e.get("triples"):
                    print(f"      [EXTRACTED TRIPLES]")
                    for i, t in enumerate(e["triples"]):
                        s = t.get("subject", {}).get("name", "?")
                        r = t.get("relation", "?")
                        o = t.get("object", {}).get("name", "?")
                        c = float(t.get("confidence", 0))
                        rsn = t.get("reasoning", "")
                        
                        print(f"      #{i+1} {s} —{r}→ {o}  (conf={c:.2f})")
                        if rsn:
                            print(textwrap.indent(f"Reasoning: {rsn}", '         '))
                else:
                    print("      [NO TRIPLES EXTRACTED]")
                print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Diagnose multi-article extraction results")
    p.add_argument("--ticker", default=None, help="Filter by primary_ticker")
    p.add_argument("--date", default=None, help="Filter by date prefix (e.g. 2023-07)")
    p.add_argument("--cache-dir", default=None, help="Override cache dir")
    p.add_argument("--verbose", "-v", action="store_true", help="Show triple details and original content")
    args = p.parse_args()

    # TẠO THƯ MỤC LƯU OUTPUT VÀ CẤU HÌNH TÊN FILE
    out_dir = os.path.join(ROOT, "data_test")
    os.makedirs(out_dir, exist_ok=True)
    
    # Đặt tên file dựa vào ticker và date truyền vào
    t_name = args.ticker.upper() if args.ticker else "ALL"
    d_name = args.date if args.date else "ALL_DATES"
    output_filename = os.path.join(out_dir, f"{t_name}_{d_name}.txt")

    # Redirect sys.stdout để in ra màn hình và ghi vào file cùng lúc
    sys.stdout = Logger(output_filename)

    cache_dir = args.cache_dir or GlobalConfig.kg_cache_dir()

    print("=" * 70)
    print("  Extraction Diagnostic — Multi-Article Concat Validation")
    print(f"  Log saved to: {output_filename}")
    print("=" * 70)
    print(f"  Cache: {cache_dir}")
    if args.ticker: print(f"  Filter ticker: {args.ticker}")
    if args.date:   print(f"  Filter date:   {args.date}")

    # ── Scan ──────────────────────────────────────────────────────────────
    all_entries = scan_cache(cache_dir)
    if not all_entries:
        print("\n  No cache files found. Run extract_corpus.py first.")
        return

    entries = filter_entries(all_entries, args.ticker, args.date)
    print(f"  Filtered: {len(entries)} / {len(all_entries)} cache files")

    # ── Check 1: Integrity ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  CHECK 1: Cache Integrity")
    print(f"{'─'*70}")
    integrity = check_integrity(entries)
    print(f"  Total files:    {integrity['total']}")
    print(f"  Valid JSON:     {integrity['valid']}")
    print(f"  Invalid JSON:   {integrity['invalid_json']}")
    print(f"  With triples:   {integrity['has_triples']}")
    print(f"  Empty (0 tri):  {integrity['empty_triples']}")
    print(f"  Versions:       {dict(integrity['versions'])}")
    if integrity["errors"]:
        print(f"  Errors ({len(integrity['errors'])}):")
        for err in integrity["errors"][:10]:
            print(f"    {err}")
    status1 = "✅ PASS" if integrity["invalid_json"] == 0 else "❌ FAIL"
    print(f"  → {status1}")

    # ── Check 2: Article Index ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  CHECK 2: article_index Stripped (no leak to cache)")
    print(f"{'─'*70}")
    idx_check = check_article_index(entries)
    print(f"  Triples checked:      {idx_check['triples_checked']}")
    print(f"  Clean (no index):     {idx_check['clean']}")
    print(f"  Leaked (has index):   {idx_check['has_article_index']}")
    if idx_check["leaked_files"]:
        print(f"  Leaked files: {idx_check['leaked_files'][:10]}")
    status2 = "✅ PASS" if idx_check["has_article_index"] == 0 else "⚠️ WARNING (article_index not stripped)"
    print(f"  → {status2}")

    # ── Check 3: Quality ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  CHECK 3: Triple Quality Stats")
    print(f"{'─'*70}")
    quality = check_quality(entries)
    if quality["total_triples"] == 0:
        print("  No triples found.")
    else:
        print(f"  Total triples:        {quality['total_triples']}")
        print(f"  Avg per article:      {quality['avg_per_article']:.1f}")
        print(f"  Avg confidence:       {quality['avg_confidence']:.2f}")
        print(f"  Avg relevance:        {quality['avg_relevance']:.2f}")
        print(f"  Avg impact:           {quality['avg_impact']:+.2f}")
        print(f"  Group A (actions):    {quality['group_a']}")
        print(f"  Group B (causal):     {quality['group_b']}")
        print(f"  Group C (context):    {quality['group_c']}")
        print(f"  Group A+B:            {quality['group_ab_pct']:.0f}%  {'✅ Good' if quality['group_ab_pct'] >= 50 else '⚠️ Low'}")
        print(f"  Has _src tag:         {quality['has_src_tag']}/{quality['total_triples']}")
        print(f"  Has reasoning:        {quality['has_reasoning']}/{quality['total_triples']}")
        print(f"\n  Relations:")
        for rel, cnt in quality["relations"]:
            bar = "█" * min(cnt, 30)
            print(f"    {rel:<22} {bar} {cnt}")

    # ── Check 4: Multi-Call Consistency ───────────────────────────────────
    print(f"\n{'─'*70}")
    print("  CHECK 4: Multi-Call Consistency")
    print(f"{'─'*70}")
    mc = check_multi_call(entries)
    print(f"  Articles with 0 triples:  {mc['articles_with_0_triples']}")
    print(f"  Articles with >15 triples: {mc['articles_with_many_triples']}")
    print(f"  Max triples/article:      {mc['max_triples_per_article']}")
    print(f"\n  Distribution (triples per article):")
    for n in sorted(mc["distribution"].keys()):
        cnt = mc["distribution"][n]
        label = f"{n:>2}" if n < 20 else "20+"
        bar = "█" * min(cnt, 40)
        print(f"    {label} triples: {bar} {cnt}")

    zero_pct = 100 * mc["articles_with_0_triples"] / max(1, len(entries))
    status4 = "✅ PASS" if zero_pct < 30 else "⚠️ WARNING (>30% empty articles)"
    print(f"  → {status4} ({zero_pct:.0f}% empty)")

    # ── Check 5: Deep Dive (if filtered) ──────────────────────────────────
    if args.ticker or args.date:
        print(f"\n{'─'*70}")
        print("  CHECK 5: Deep Dive (filtered articles)")
        print(f"{'─'*70}")
        deep_dive(entries, verbose=args.verbose)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}")
    print(f"  {status1}  Cache Integrity")
    print(f"  {status2}  article_index Stripped")
    print(f"  {'✅ PASS' if quality.get('group_ab_pct', 0) >= 50 else '⚠️ CHECK'}  Quality (A+B={quality.get('group_ab_pct', 0):.0f}%)")
    print(f"  {status4}  Multi-Call Consistency")

    all_pass = (
        integrity["invalid_json"] == 0
        and idx_check["has_article_index"] == 0
        and quality.get("group_ab_pct", 0) >= 50
        and zero_pct < 30
    )
    print(f"\n  {'✅ ALL CHECKS PASSED' if all_pass else '⚠️ SOME CHECKS NEED REVIEW'}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()