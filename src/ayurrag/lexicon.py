"""Ayurveda terminology lexicon: entity linking and query expansion."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources

from .text import normalize


@lru_cache(maxsize=1)
def load_lexicon() -> dict[str, dict[str, list[str]]]:
    path = resources.files("ayurrag.data").joinpath("lexicon.json")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def alias_index() -> list[tuple[str, str, str]]:
    """Return ``(alias_lower, canonical, category)`` sorted by alias length."""
    rows: list[tuple[str, str, str]] = []
    for category, entries in load_lexicon().items():
        for canonical, aliases in entries.items():
            for alias in {canonical.lower(), *[a.lower() for a in aliases]}:
                rows.append((alias, canonical, category))
    rows.sort(key=lambda row: len(row[0]), reverse=True)
    return rows


@lru_cache(maxsize=1)
def _alias_lookup() -> dict[str, tuple[str, str]]:
    return {alias: (canonical, category) for alias, canonical, category in alias_index()}


@lru_cache(maxsize=1)
def _matcher() -> re.Pattern[str]:
    aliases = [re.escape(alias) for alias, _, _ in alias_index()]
    return re.compile(r"(?<![\w])(?:" + "|".join(aliases) + r")(?![\w])", re.I)


def extract_entities(text: str) -> list[str]:
    lookup = _alias_lookup()
    found: set[str] = set()
    for match in _matcher().finditer(normalize(text)):
        entry = lookup.get(match.group(0).lower())
        if entry:
            found.add(entry[0])
    return sorted(found)


def entities_with_category(text: str) -> dict[str, list[str]]:
    lookup = _alias_lookup()
    result: dict[str, list[str]] = {}
    for match in _matcher().finditer(normalize(text)):
        entry = lookup.get(match.group(0).lower())
        if entry:
            canonical, category = entry
            result.setdefault(category, [])
            if canonical not in result[category]:
                result[category].append(canonical)
    return result


def aliases_for(canonical: str) -> list[str]:
    for _category, entries in load_lexicon().items():
        if canonical in entries:
            return sorted({canonical, *entries[canonical]})
    return [canonical]


def expand_query(query: str, max_terms: int = 10) -> str:
    """Append known aliases for entities mentioned in the query.

    This is cheap, deterministic and multilingual (Sanskrit transliteration,
    botanical names and common English names). It is the first expansion
    stage; an optional LLM expansion can run on top of it.
    """
    additions: list[str] = []
    seen = {token.lower() for token in re.findall(r"\w+", query)}
    for canonical in extract_entities(query):
        for alias in aliases_for(canonical):
            if alias.lower() not in seen and len(alias) > 2:
                additions.append(alias)
                seen.add(alias.lower())
            if len(additions) >= max_terms:
                break
        if len(additions) >= max_terms:
            break
    if not additions:
        return query
    return f"{query} ({', '.join(additions)})"

