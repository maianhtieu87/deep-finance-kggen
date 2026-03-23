# configs/ticker_aliases.py
"""
Single Source of Truth cho tất cả biến thể tên gọi của mỗi ticker.

VÌ SAO CẦN FILE NÀY:
  LLM extract ra nhiều biến thể tên cho cùng 1 công ty:
    "Walmart", "Walmart Inc.", "Walmart Inc", "WMT"
  Nếu không normalize, smart_dedup_triples() coi chúng là entity khác nhau
  → triples trùng lặp không bị merge → embedding bị dominate bởi 1 sự kiện.

  Trước đây:
    - TICKER_NAME_MAP bị DUPLICATE ở extractor_batch.py và embed_news.py
    - _normalize_object_for_dedup() HARDCODE "wmt|walmart" chỉ cho 1 ticker
    - Substring matching với len>3 guard → "BA" (2 chars) bị skip

  File này giải quyết TẤT CẢ:
    1. Single source → import everywhere, không duplicate
    2. Comprehensive aliases → EXACT match only, không substring (an toàn)
    3. Derived regex pattern → stock % normalize cho TẤT CẢ 9 tickers
    4. Không cần len guard → BA, GOOGL, META đều hoạt động

CÁCH THÊM TICKER MỚI:
  Chỉ cần thêm entry vào TICKER_ALIASES. Tất cả derived maps tự cập nhật.

IMPORT:
  from configs.ticker_aliases import (
      TICKER_ALIASES, NAME_TO_TICKER, TICKER_NAME_MAP,
      ALL_TICKER_NAMES_PATTERN, normalize_entity_name,
  )
"""

import re
from typing import Dict, List, Set

# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY DEFINITION — Chỉ cần chỉnh sửa ở đây khi thêm/sửa ticker
# ─────────────────────────────────────────────────────────────────────────────

TICKER_ALIASES: Dict[str, List[str]] = {
    "TSLA":  ["Tesla", "TSLA", "Tesla Motors"],
    "AAPL":  ["Apple", "AAPL"],
    "AMZN":  ["Amazon", "AMZN", "Amazon.com"],
    "MSFT":  ["Microsoft", "MSFT"],
    "GOOGL": ["Google", "GOOGL", "Alphabet", "GOOG"],
    "META":  ["Meta", "META", "Meta Platforms", "Facebook"],
    "BA":    ["Boeing", "BA"],
    "JPM":   ["JPMorgan", "JPM", "JP Morgan", "JPMorgan Chase"],
    "WMT":   ["Walmart", "WMT"],
}

# ─────────────────────────────────────────────────────────────────────────────
# LEGAL SUFFIX REGEX
# ─────────────────────────────────────────────────────────────────────────────

_LEGAL_SUFFIX_RE = re.compile(
    r',?\s*(Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|Limited|'
    r'Group|Platforms?|Holdings?|Services?|LLC|LLP|PLC|S\.A\.|N\.V\.)\.?\s*$',
    re.IGNORECASE,
)

def _normalize_alias(name: str) -> str:
    """
    Normalize tên: strip legal suffixes LẶP LẠI (max 3 vòng) + lowercase.
    "Meta Platforms Inc." → strip "Inc." → "Meta Platforms" → strip "Platforms" → "meta"
    """
    if not name:
        return ""
    n = name
    for _ in range(3):
        stripped = _LEGAL_SUFFIX_RE.sub('', n).strip()
        if stripped == n.strip():
            break
        n = stripped
    return re.sub(r'\s+', ' ', n).strip().lower()

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED MAPS (auto-generated từ TICKER_ALIASES)
# ─────────────────────────────────────────────────────────────────────────────

def _build_name_to_ticker() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for ticker, aliases in TICKER_ALIASES.items():
        canonical = ticker.lower()
        for alias in aliases:
            normalized = _normalize_alias(alias)
            if normalized in mapping and mapping[normalized] != canonical:
                raise ValueError(f"Alias conflict: '{alias}'→'{normalized}' in {mapping[normalized]} and {canonical}")
            mapping[normalized] = canonical
        if canonical not in mapping:
            mapping[canonical] = canonical
    return mapping

def _build_ticker_name_map() -> Dict[str, List[str]]:
    return {ticker: list(aliases) for ticker, aliases in TICKER_ALIASES.items()}

def _build_all_names_pattern() -> str:
    all_names: Set[str] = set()
    for aliases in TICKER_ALIASES.values():
        for alias in aliases:
            all_names.add(alias)
    return "|".join(re.escape(n) for n in sorted(all_names, key=len, reverse=True))

NAME_TO_TICKER: Dict[str, str] = _build_name_to_ticker()
TICKER_NAME_MAP: Dict[str, List[str]] = _build_ticker_name_map()
ALL_TICKER_NAMES_PATTERN: str = _build_all_names_pattern()

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def normalize_entity_name(name: str) -> str:
    """
    Normalize tên entity → canonical ticker (lowercase) nếu match.
    EXACT MATCH only — không substring, không len guard.
    """
    normalized = _normalize_alias(name)
    return NAME_TO_TICKER.get(normalized, normalized)

def is_known_ticker_name(name: str) -> bool:
    return _normalize_alias(name) in NAME_TO_TICKER

# Validate on import
for _tk, _aliases in TICKER_ALIASES.items():
    assert _aliases, f"Ticker {_tk}: empty alias list"