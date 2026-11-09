"""PubMed/Europe PMC literature ingestion.

Uses the public Europe PMC REST API (no key required, polite rate limits).
Only bibliographic metadata and abstracts are indexed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from ..schema import SourceDocument
from ..text import normalize
from .common import doc

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
DEFAULT_QUERY = (
    '(ayurveda OR ayurvedic OR panchakarma OR "traditional Indian medicine" '
    'OR ashwagandha OR "withania somnifera" OR triphala OR guduchi) '
    "AND HAS_ABSTRACT:Y"
)


def _authors(raw: str) -> str:
    if not raw:
        return ""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) <= 2:
        return ", ".join(parts)
    return f"{parts[0]} et al."


def fetch_pubmed(
    raw_dir: Path,
    *,
    query: str = DEFAULT_QUERY,
    limit: int = 8000,
    page_size: int = 500,
    polite_delay: float = 0.15,
) -> list[SourceDocument]:
    cache = raw_dir / "pubmed_abstracts.json"
    if cache.exists():
        records = json.loads(cache.read_text())
        if len(records) >= limit:
            records = records[:limit]
            return _to_documents(records)

    records: list[dict] = []
    cursor = "*"
    seen: set[str] = set()
    with httpx.Client(timeout=60.0) as client:
        while len(records) < limit:
            params = {
                "query": query,
                "format": "json",
                "pageSize": page_size,
                "resultType": "core",
                "cursorMark": cursor,
                "sort": "CITED desc",
            }
            payload = None
            for attempt in range(4):
                try:
                    response = client.get(BASE, params=params)
                    response.raise_for_status()
                    payload = response.json()
                    break
                except (httpx.HTTPError, ValueError):
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            results = payload.get("resultList", {}).get("result", [])
            if not results:
                break
            for item in results:
                pmid = item.get("pmid") or item.get("id")
                abstract = item.get("abstractText") or ""
                if not pmid or not abstract or pmid in seen:
                    continue
                journal = (item.get("journalInfo") or {}).get("journal") or {}
                records.append(
                    {
                        "pmid": str(pmid),
                        "doi": item.get("doi") or "",
                        "title": normalize(item.get("title") or "").strip(),
                        "abstract": normalize(abstract).strip(),
                        "authors": _authors(item.get("authorString") or ""),
                        "journal": journal.get("title") or item.get("journalTitle") or "",
                        "year": str(item.get("pubYear") or ""),
                        "pub_type": item.get("pubType") or "",
                    }
                )
                seen.add(pmid)
                if len(records) >= limit:
                    break
            cursor = payload.get("nextCursorMark") or ""
            if not cursor:
                break
            time.sleep(polite_delay)

    cache.write_text(json.dumps(records, ensure_ascii=False, indent=1))
    return _to_documents(records)


def _to_documents(records: list[dict]) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for record in records:
        title = record["title"].rstrip(".")
        citation = f"{record['authors']}. {title}. {record['journal']} ({record['year']}). PMID {record['pmid']}"
        docs.append(
            doc(
                source="pubmed",
                corpus="literature",
                title=title or f"PMID {record['pmid']}",
                text=f"{title}\n\n{record['abstract']}",
                url=f"https://europepmc.org/article/MED/{record['pmid']}",
                license="Abstract via Europe PMC / NLM",
                doc_key=record["pmid"],
                metadata={
                    "work": title,
                    "citation": citation,
                    "pmid": record["pmid"],
                    "doi": record["doi"],
                    "journal": record["journal"],
                    "year": record["year"],
                    "authors": record["authors"],
                    "pub_type": record["pub_type"],
                },
            )
        )
    return docs
