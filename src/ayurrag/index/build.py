"""Build the SQLite store, BM25 index and dense index from chunks.jsonl."""

from __future__ import annotations

import logging
import time

from ..config import Settings
from ..schema import Chunk
from ..store import ChunkStore
from . import BM25Index, DenseIndex

log = logging.getLogger(__name__)


def build_indexes(settings: Settings) -> dict[str, int]:
    rows = Chunk.load_jsonl(settings.paths.raw_dir / "chunks.jsonl")
    if not rows:
        raise SystemExit("no chunks found; run `ayurrag ingest` first")
    ids = [chunk.chunk_id for chunk in rows]

    start = time.time()
    store = ChunkStore(settings.paths.index_dir / "corpus.sqlite")
    store.clear()
    store.add_chunks(rows)
    log.info("store: %d chunks in %.1fs", store.count(), time.time() - start)

    start = time.time()
    bm25 = BM25Index(settings.paths.index_dir / "bm25")
    bm25.build([chunk.text for chunk in rows], ids)
    log.info("bm25: indexed in %.1fs", time.time() - start)

    start = time.time()
    dense = DenseIndex(settings.paths.index_dir / "dense", settings.embedding)
    dense.build(rows, ids)
    log.info("dense: %d vectors in %.1fs", len(ids), time.time() - start)

    return {"chunks": len(rows), "vectors": len(ids)}
