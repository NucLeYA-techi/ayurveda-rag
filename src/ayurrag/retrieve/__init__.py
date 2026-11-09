"""Hybrid retrieval: BM25 + dense (RRF), optional graph expansion and reranking."""

from __future__ import annotations

import logging

from ..config import Settings
from ..index import BM25Index, DenseIndex
from ..lexicon import expand_query, extract_entities
from ..schema import Chunk, ScoredChunk
from ..store import ChunkStore

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        return [float(value) for value in self.model.predict(pairs, batch_size=16, show_progress_bar=False)]


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        index_dir = settings.paths.index_dir
        self.store = ChunkStore(index_dir / "corpus.sqlite")
        self.bm25 = BM25Index(index_dir / "bm25").load()
        self.dense = DenseIndex(index_dir / "dense", settings.embedding).load()
        self.reranker = Reranker(settings.retrieval.reranker)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source: str | None = None,
        corpus: str | None = None,
        use_expansion: bool = True,
        use_rerank: bool | None = None,
        use_graph: bool | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        bm25_weight: float | None = None,
        dense_weight: float | None = None,
    ) -> list[ScoredChunk]:
        cfg = self.settings.retrieval
        top_k = top_k or cfg.final_top_k
        use_rerank = cfg.enable_rerank if use_rerank is None else use_rerank
        use_graph = cfg.graph_expansion if use_graph is None else use_graph
        allowed = self.store.filter_ids(source=source, corpus=corpus) if (source or corpus) else None

        expanded = expand_query(query) if use_expansion else query
        queries = [query] if expanded == query else [query, expanded]

        fused: dict[str, float] = {}
        for variant in queries:
            channels: list[tuple[str, list[str]]] = []
            if use_bm25:
                channels.append(("bm25", self._bm25(variant, cfg.bm25_top_k, allowed)))
            if use_dense:
                channels.append(("dense", self._dense(variant, cfg.dense_top_k, allowed)))
            for name, ids in channels:
                weight = (bm25_weight if bm25_weight is not None else cfg.bm25_weight) if name == "bm25" else (
                    dense_weight if dense_weight is not None else cfg.dense_weight
                )
                for rank, chunk_id in enumerate(ids):
                    fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (cfg.rrf_k + rank + 1)

        if use_graph and cfg.graph_expansion:
            for rank, chunk_id in enumerate(self._graph_candidates(query, allowed)):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 0.5 / (cfg.rrf_k + rank + 1)

        ranked_ids = sorted(fused, key=lambda key: fused[key], reverse=True)
        candidates = self.store.get_many(ranked_ids[: cfg.rerank_top_n])

        if use_rerank and self.reranker:
            ordered = [candidates[chunk_id] for chunk_id in ranked_ids[: cfg.rerank_top_n] if chunk_id in candidates]
            scores = self.reranker.score(query, ordered)
            scored = [
                ScoredChunk(chunk=chunk, score=score, retriever="reranked")
                for chunk, score in zip(ordered, scores, strict=False)
            ]
        else:
            scored = [
                ScoredChunk(chunk=candidates[chunk_id], score=fused[chunk_id], retriever="rrf")
                for chunk_id in ranked_ids[: cfg.rerank_top_n]
                if chunk_id in candidates
            ]

        scored.sort(key=lambda item: item.score, reverse=True)
        for rank, item in enumerate(scored):
            item.rank = rank + 1
        return scored[:top_k]

    def _bm25(self, query: str, k: int, allowed: set[str] | None) -> list[str]:
        try:
            ids, _ = self.bm25.search(query, k=k * 3 if allowed else k)
        except Exception as exc:  # empty index / OOV query
            log.debug("bm25 search failed: %s", exc)
            return []
        return [chunk_id for chunk_id in ids if allowed is None or chunk_id in allowed][:k]

    def _dense(self, query: str, k: int, allowed: set[str] | None) -> list[str]:
        ids, _ = self.dense.search(query, k=k * 3 if allowed else k)
        return [chunk_id for chunk_id in ids if allowed is None or chunk_id in allowed][:k]

    def _graph_candidates(self, query: str, allowed: set[str] | None) -> list[str]:
        entities = extract_entities(query)
        if not entities:
            return []
        ids = self.store.chunks_for_entities(entities, limit=self.settings.retrieval.graph_top_k * 3)
        return [chunk_id for chunk_id in ids if allowed is None or chunk_id in allowed][: self.settings.retrieval.graph_top_k]
