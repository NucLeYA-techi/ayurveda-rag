"""Patent ingestion from public Google Patents documents.

Google Patents blocks automated search from many data-centre IPs. We therefore
snowball from a small set of verified Ayurveda-related publication numbers:
each document page links to its citations, cited-by and similar documents.
Pages are read through the public r.jina.ai reader (returns the original HTML)
when direct access is blocked. Only public abstracts and claims are stored.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from ..schema import SourceDocument
from ..text import clean_ocr, normalize
from .common import doc

log = logging.getLogger(__name__)

JINA = "https://r.jina.ai/"

# Verified Ayurveda / herbal-medicine patent documents (ashwagandha, turmeric,
# neem, triphala-related compositions). Used only as snowball seeds; no content
# is bundled in the repository.
SEEDS = [
    "US6153198A",
    "US7108870B2",
    "US7318938B2",
    "US11638738B2",
    "US12564552B2",
    "WO2023141607A2",
    "WO2012160569A1",
    "EP3200770B1",
    "JP2004162695A",
    "JP2007261849A",
    "JPH0796241A",
    "US5401504A",
    "EP0436257B1",
    "US6713092B1",
    "US20050226950A1",
    "US20070036873A1",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

JURISDICTIONS = ("US", "WO", "EP", "GB", "AU", "CA", "IN", "JP", "KR", "CN")
RELEVANT = re.compile(
    r"ashwagandha|withania|somnifera|turmeric|curcuma|neem|azadirachta|triphala|ayurved|"
    r"herbal|botanical|plant extract|panchakarma|boswellia|bacopa|tinospora|emblica|"
    r"ocimum|shatavari|guduchi|rasayana|medicinal plant",
    re.I,
)


def _get_html(client: httpx.Client, url: str) -> str:
    """Fetch a patent page directly, falling back to the Jina reader."""
    for attempt in range(2):
        try:
            response = client.get(url, timeout=45.0)
            if response.status_code == 200 and 'itemprop="abstract"' in response.text:
                return response.text
        except httpx.HTTPError:
            pass
        try:
            response = client.get(JINA + url, headers={"X-Return-Format": "html"}, timeout=150.0)
            if response.status_code == 200:
                return response.text
        except httpx.HTTPError:
            pass
        time.sleep(2 * (attempt + 1))
    return ""


def _ids_from_page(html: str) -> list[str]:
    ids = re.findall(r"/patent/([A-Z]{2}[0-9A-Z]+)/en", html)
    seen: set[str] = set()
    ordered: list[str] = []
    for publication in ids:
        if publication in seen or not publication.startswith(JURISDICTIONS):
            continue
        seen.add(publication)
        ordered.append(publication)
    return ordered


def _fetch_detail(client: httpx.Client, publication: str) -> dict:
    url = f"https://patents.google.com/patent/{publication}/en"
    html = _get_html(client, url)
    if not html:
        return {}
    soup = BeautifulSoup(html, "lxml")
    abstract = soup.select_one('section[itemprop="abstract"]')
    claims = soup.select_one('section[itemprop="claims"]')
    description = soup.select_one('section[itemprop="description"]')
    page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
    title = page_title.split(" - ")[1].strip() if " - " in page_title else page_title
    assignee_meta = soup.find("meta", {"name": "DC.contributor"})
    return {
        "title": normalize(title).strip(),
        "abstract": abstract.get_text(" ", strip=True) if abstract else "",
        "claims": claims.get_text(" ", strip=True)[:12000] if claims else "",
        "description": description.get_text(" ", strip=True)[:3000] if description else "",
        "assignee": normalize(assignee_meta.get("content") or "").strip() if assignee_meta else "",
    }


def fetch_patents(
    raw_dir: Path,
    *,
    limit: int = 120,
    detail_limit: int = 60,
    workers: int = 4,
) -> list[SourceDocument]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "patents.json"
    if cache.exists():
        records = json.loads(cache.read_text())
        if records:
            return _to_documents(records[:limit])

    with httpx.Client(headers=HEADERS, timeout=45.0, follow_redirects=True) as client:
        candidates: list[str] = list(SEEDS)
        seen = set(candidates)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_get_html, client, f"https://patents.google.com/patent/{seed}/en"): seed
                for seed in SEEDS
            }
            for future in as_completed(futures):
                try:
                    html = future.result()
                except Exception:  # noqa: BLE001
                    continue
                for publication in _ids_from_page(html):
                    if publication not in seen:
                        seen.add(publication)
                        candidates.append(publication)

        records: list[dict] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_fetch_detail, client, publication): publication for publication in candidates[:detail_limit]}
            for future in as_completed(futures):
                publication = futures[future]
                try:
                    detail = future.result()
                except Exception as exc:  # noqa: BLE001
                    log.debug("patent detail failed for %s: %s", publication, exc)
                    continue
                if not detail or not (detail.get("abstract") or detail.get("claims")):
                    continue
                haystack = " ".join(
                    [detail.get("title") or "", detail.get("abstract") or "", detail.get("claims") or ""]
                )
                if not RELEVANT.search(haystack):
                    continue
                records.append(
                    {
                        "publication_number": publication,
                        "title": detail["title"] or publication,
                        "assignee": detail.get("assignee") or "Unknown assignee",
                        "abstract": detail.get("abstract") or "",
                        "claims": detail.get("claims") or "",
                        "description": detail.get("description") or "",
                    }
                )

    records.sort(key=lambda record: record["publication_number"])
    cache.write_text(json.dumps(records, ensure_ascii=False, indent=1))
    log.info("patents: %d documents with abstracts/claims", len(records))
    return _to_documents(records[:limit])


def _to_documents(records: list[dict]) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for record in records:
        body_parts = []
        if record.get("abstract"):
            body_parts.append(f"Abstract: {record['abstract']}")
        if record.get("claims"):
            body_parts.append(f"Claims: {record['claims']}")
        if not body_parts:
            continue
        text = clean_ocr("\n\n".join(body_parts))
        publication = record["publication_number"]
        assignee = record.get("assignee") or "Unknown assignee"
        citation = f"{publication} — {record['title']} ({assignee})"
        docs.append(
            doc(
                source="patents",
                corpus="patent",
                title=record["title"] or publication,
                text=text,
                url=f"https://patents.google.com/patent/{publication}/en",
                license="Public patent document",
                doc_key=publication,
                metadata={
                    "work": record["title"] or publication,
                    "citation": citation,
                    "publication_number": publication,
                    "assignee": assignee,
                },
            )
        )
    return docs
