"""Utilities to normalize user-provided ticker symbols."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ALIAS_FILE = Path(__file__).resolve().parent.parent / "ticker_aliases.json"


def _alias_key(raw: str) -> str:
    return "".join(raw.strip().lower().split())


def _load_aliases() -> Dict[str, str]:
    if not ALIAS_FILE.exists():
        return {}
    try:
        with ALIAS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    aliases: Dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        cleaned_key = _alias_key(key)
        cleaned_value = value.strip()
        if cleaned_key and cleaned_value:
            aliases[cleaned_key] = cleaned_value
    return aliases


ALIAS_MAP = _load_aliases()


def _get_alias_target(raw: str) -> str | None:
    global ALIAS_MAP
    key = _alias_key(raw)
    target = ALIAS_MAP.get(key)
    if target is not None:
        return target
    # Reload alias map in case文件更新
    ALIAS_MAP = _load_aliases()
    return ALIAS_MAP.get(key)


def _normalize_single(raw: str) -> str | None:
    """Return a yfinance-compatible ticker or None if unknown."""
    cleaned = raw.strip().upper()
    if not cleaned:
        return None

    alias_target = _get_alias_target(raw)
    if alias_target:
        cleaned = alias_target.strip().upper()

    # Already contains suffix
    if "." in cleaned:
        base, suffix = cleaned.split(".", 1)
        suffix = suffix.upper()
        if not base:
            return None
        if suffix in {"SH", "SS"}:
            suffix = "SS"
        elif suffix in {"SZ"}:
            suffix = "SZ"
        elif suffix in {"BJ"}:
            suffix = "BJ"
        else:
            # Assume existing suffix works
            return f"{base}.{suffix}"
        return f"{base}.{suffix}"

    # Prefix like SH600519 or SZ000001
    if len(cleaned) > 2 and cleaned[:2] in {"SH", "SZ", "SS", "BJ"} and cleaned[2:].isdigit():
        base = cleaned[2:]
        suffix = cleaned[:2]
        if suffix in {"SH", "SS"}:
            return f"{base}.SS"
        if suffix == "SZ":
            return f"{base}.SZ"
        if suffix == "BJ":
            return f"{base}.BJ"

    # Pure digits (A-shares codes)
    if cleaned.isdigit() and len(cleaned) == 6:
        first = cleaned[0]
        if cleaned.startswith(("688", "689")) or first in {"5", "6", "9"}:
            suffix = "SS"  # Shanghai, incl. STAR market
        elif first == "8":
            suffix = "BJ"  # Beijing exchange
        else:
            suffix = "SZ"  # Shenzhen, incl. GEM
        return f"{cleaned}.{suffix}"

    # Should be a US-style ticker (letters)
    if re.fullmatch(r"[A-Z\.]{1,10}", cleaned):
        return cleaned

    return None


def normalize_tickers(tickers: Iterable[str]) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Normalize incoming tickers.

    Returns:
        normalized: list of unique normalized tickers in request order
        mapping: dict normalized -> original label
        invalid: list of raw inputs that could not be normalized
    """
    mapping: "OrderedDict[str, str]" = OrderedDict()
    invalid: List[str] = []

    for raw in tickers:
        cleaned = raw.strip()
        if not cleaned:
            continue
        normalized = _normalize_single(cleaned)
        if not normalized:
            invalid.append(cleaned)
            continue
        if normalized not in mapping:
            # For display use uppercase letters, digits keep as-is
            label = cleaned.upper() if cleaned.isalpha() else cleaned
            mapping[normalized] = label

    return list(mapping.keys()), dict(mapping), invalid


CHINA_SUFFIXES = {"SS", "SZ", "BJ"}


def market_for_ticker(normalized_ticker: str) -> str:
    """Classify normalized ticker into market buckets."""
    upper = normalized_ticker.upper()
    suffix = upper.split(".")[-1] if "." in upper else ""
    if suffix in CHINA_SUFFIXES:
        return "china_a"
    return "global"
