"""Structure-aware chunking with provenance-preserving headers."""

from __future__ import annotations

import re
from collections import Counter

from .config import ChunkingConfig
from .lexicon import extract_entities
from .schema import Chunk, SourceDocument, make_id
from .text import approx_tokens, content_hash, normalize, split_long_text


def _label_range(first: str, last: str) -> str:
    if not first:
        return ""
    if not last or first == last:
        return first
    first_prefix, _, first_tail = first.rpartition(" ")
    last_prefix, _, last_tail = last.rpartition(" ")
    if first_prefix and first_prefix == last_prefix:
        return f"{first_prefix} {first_tail}–{last_tail}"
    return f"{first}–{last}"


def _citation(doc: SourceDocument, label_range: str) -> str:
    work = str(doc.metadata.get("work") or doc.title)
    if label_range:
        return f"{work}, {label_range}"
    return work


def _make_chunk(doc: SourceDocument, index: int, body: str, header: str, metadata: dict) -> Chunk:
    body = normalize(body).strip()
    text = f"{header}\n{body}" if header and not body.startswith(header) else body
    return Chunk(
        chunk_id=make_id(doc.doc_id, str(index)),
        doc_id=doc.doc_id,
        source=doc.source,
        corpus=doc.corpus,
        title=doc.title,
        text=text,
        body=body,
        header=header,
        metadata=metadata,
        entities=extract_entities(text),
        token_count=approx_tokens(text),
    )


def chunk_document(doc: SourceDocument, cfg: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    base_meta = {
        **doc.metadata,
        "url": doc.url,
        "license": doc.license,
        "lang": doc.lang,
    }

    if doc.units:
        group: list[str] = []
        labels: list[str] = []
        size = 0

        def flush() -> None:
            nonlocal group, labels, size
            if not group:
                return
            label_range = _label_range(labels[0], labels[-1])
            header = f"{doc.title} | {label_range}" if label_range else doc.title
            citation = _citation(doc, label_range)
            meta = {**base_meta, "citation": citation, "label": label_range}
            chunks.append(_make_chunk(doc, len(chunks), " ".join(group), header, meta))
            group, labels, size = [], [], 0

        for unit, label in zip(doc.units, doc.unit_labels or [""] * len(doc.units), strict=False):
            unit = normalize(unit).strip()
            if len(unit) < cfg.min_chars:
                continue
            if len(unit) > cfg.max_chars:
                flush()
                for piece in split_long_text(unit, cfg.max_chars, cfg.overlap_chars):
                    header = f"{doc.title} | {label}" if label else doc.title
                    meta = {**base_meta, "citation": _citation(doc, label), "label": label}
                    chunks.append(_make_chunk(doc, len(chunks), piece, header, meta))
                continue
            if size + len(unit) + 1 > cfg.max_chars and group:
                flush()
            group.append(unit)
            labels.append(label)
            size += len(unit) + 1
        flush()
        return chunks

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    buffer: list[str] = []
    length = 0

    def flush_text() -> None:
        nonlocal buffer, length
        if buffer:
            chunks.append(_make_chunk(doc, len(chunks), " ".join(buffer), doc.title, dict(base_meta)))
            buffer, length = [], 0

    for paragraph in paragraphs:
        paragraph = normalize(paragraph)
        if len(paragraph) < cfg.min_chars:
            continue
        for piece in split_long_text(paragraph, cfg.max_chars, cfg.overlap_chars):
            if length + len(piece) + 1 > cfg.max_chars and buffer:
                flush_text()
            buffer.append(piece)
            length += len(piece) + 1
    flush_text()
    return chunks


def chunk_corpus(docs: list[SourceDocument], cfg: ChunkingConfig) -> list[Chunk]:
    seen: set[str] = set()
    chunks: list[Chunk] = []
    for document in docs:
        for chunk in chunk_document(document, cfg):
            digest = content_hash(chunk.body)
            if digest in seen:
                continue
            seen.add(digest)
            chunks.append(chunk)
    return chunks


def corpus_stats(chunks: list[Chunk]) -> dict[str, dict[str, int]]:
    stats: dict[str, Counter] = {}
    doc_ids: dict[str, set[str]] = {}
    for chunk in chunks:
        entry = stats.setdefault(chunk.source, Counter())
        entry["chunks"] += 1
        entry["tokens"] += chunk.token_count
        doc_ids.setdefault(chunk.source, set()).add(chunk.doc_id)
    for source, ids in doc_ids.items():
        stats[source]["docs"] = len(ids)
    return {source: dict(counter) for source, counter in stats.items()}
