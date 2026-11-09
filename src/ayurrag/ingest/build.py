"""Corpus assembly: parse sources, chunk, persist."""

from __future__ import annotations

import json
import logging

from ..chunking import chunk_corpus, corpus_stats
from ..config import Settings
from ..schema import SourceDocument
from .charaka import parse_charaka
from .common import write_jsonl
from .legal import parse_legal
from .patents import fetch_patents
from .pubmed import fetch_pubmed
from .sushruta import parse_sushruta

log = logging.getLogger(__name__)

ALL_SOURCES = ("charaka", "sushruta", "pubmed", "patents", "legal")


def collect_documents(
    settings: Settings,
    sources: tuple[str, ...] = ALL_SOURCES,
    *,
    pubmed_limit: int = 8000,
    patent_limit: int = 120,
    patent_detail_limit: int = 60,
) -> list[SourceDocument]:
    raw = settings.paths.raw_dir
    docs: list[SourceDocument] = []
    if "charaka" in sources:
        parsed = parse_charaka(raw / "charaka")
        log.info("charaka: %d chapters, %d verses", len(parsed), sum(len(d.units) for d in parsed))
        docs.extend(parsed)
    if "sushruta" in sources:
        parsed = parse_sushruta(raw)
        log.info("sushruta: %d volumes, %d passages", len(parsed), sum(len(d.units) for d in parsed))
        docs.extend(parsed)
    if "pubmed" in sources:
        parsed = fetch_pubmed(raw, limit=pubmed_limit)
        log.info("pubmed: %d abstracts", len(parsed))
        docs.extend(parsed)
    if "patents" in sources:
        parsed = fetch_patents(raw, limit=patent_limit, detail_limit=patent_detail_limit)
        log.info("patents: %d records", len(parsed))
        docs.extend(parsed)
    if "legal" in sources:
        parsed = parse_legal(raw)
        log.info("legal: %d documents", len(parsed))
        docs.extend(parsed)
    return docs


def build_corpus(
    settings: Settings,
    sources: tuple[str, ...] = ALL_SOURCES,
    *,
    pubmed_limit: int = 8000,
    patent_limit: int = 120,
    patent_detail_limit: int = 60,
) -> list[dict]:
    docs = collect_documents(
        settings,
        sources,
        pubmed_limit=pubmed_limit,
        patent_limit=patent_limit,
        patent_detail_limit=patent_detail_limit,
    )
    chunks = chunk_corpus(docs, settings.chunking)
    rows = [chunk.model_dump() for chunk in chunks]
    output = settings.paths.raw_dir / "chunks.jsonl"
    write_jsonl(output, rows)
    stats = corpus_stats(chunks)
    (settings.paths.raw_dir / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("wrote %d chunks to %s", len(chunks), output)
    return rows
