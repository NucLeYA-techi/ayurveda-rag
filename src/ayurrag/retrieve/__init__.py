"""Multi-stage retrieval.

Pipeline: query planning (entity aliases + lightweight decomposition) ->
BM25 and dense channels -> score-normalized convex fusion (or RRF) ->
entity-graph expansion -> cross-encoder reranking blended with first-stage
scores -> optional MMR diversification.

The design is intentionally explicit: each stage can be disabled for
ablations, and no framework hides the ranking semantics.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from ..config import Settings
from ..index import BM25Index, DenseIndex
from ..lexicon import aliases_for, extract_entities
from ..schema import Chunk, ScoredChunk
from ..store import ChunkStore

log = logging.getLogger(__name__)

WORD_RE = re.compile(r"[a-z0-9]+")
SPLIT_RE = re.compile(r"[?;]")
CLAUSE_RE = re.compile(r",?\s+and\s+(?=(?:how|what|which|who|where|when|why)\b)", re.I)


@dataclass(frozen=True)
class QueryVariant:
    text: str
    weight: float
    kind: str


def _tokens(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def decompose_query(query: str, max_parts: int = 3) -> list[str]:
    """Split genuinely multi-part questions without an LLM.

    Only clauses that are large enough and materially different from the
    original query are kept; this avoids the "decompose into the same query"
    failure mode that plagues naive splitters.
    """
    original = _tokens(query)
    candidates: list[str] = []
    for part in SPLIT_RE.split(query):
        part = part.strip(" ,.")
        if len(part) > 30:
            candidates.append(part)
        for clause in CLAUSE_RE.split(part):
            clause = clause.strip(" ,.")
            if len(clause) > 30:
                candidates.append(clause)
    result: list[str] = []
    for candidate in candidates:
        tokens = _tokens(candidate)
        if not tokens:
            continue
        overlap = len(tokens & original) / len(tokens | original)
        if overlap >= 0.85:
            continue
        if candidate.lower() == query.lower():
            continue
        if candidate not in result:
            result.append(candidate)
        if len(result) >= max_parts:
            break
    return result


def plan_queries(
    query: str, cfg, use_expansion: bool = True, max_subqueries: int | None = None
) -> list[QueryVariant]:
    variants = [QueryVariant(query, 1.0, "original")]
    if not use_expansion:
        return variants

    entities = extract_entities(query)
    if entities:
        normalized_query = " ".join(query.lower().split())
        aliases: list[str] = []
        seen_aliases: set[str] = set()
        for entity in entities:
            for alias in aliases_for(entity):
                key = " ".join(alias.lower().split())
                if key in seen_aliases or key in normalized_query or len(alias) <= 2:
                    continue
                seen_aliases.add(key)
                aliases.append(alias)
        aliases = aliases[:12]
        if aliases:
            variants.append(QueryVariant(" ".join(aliases), cfg.alias_weight, "aliases"))

    subquery_limit = cfg.max_subqueries if max_subqueries is None else max_subqueries
    if subquery_limit:
        for subquery in decompose_query(query, subquery_limit):
            variants.append(QueryVariant(subquery, cfg.subquery_weight, "subquery"))

    # Deduplicate by normalized text while keeping the highest weight.
    deduped: dict[str, QueryVariant] = {}
    for variant in variants:
        key = " ".join(variant.text.lower().split())
        if key not in deduped or variant.weight > deduped[key].weight:
            deduped[key] = variant
    return list(deduped.values())


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize one channel to [0, 1] for cross-channel fusion."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high - low < 1e-9:
        return {key: 0.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


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
        if self.store.conn.execute("SELECT COUNT(*) FROM entity_edges").fetchone()[0] == 0:
            edges = self.store.rebuild_entity_graph()
            if edges:
                log.info("built %d entity-graph edges", edges)
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
        use_expansion: bool | None = None,
        use_rerank: bool | None = None,
        use_graph: bool | None = None,
        use_bm25: bool = True,
        use_dense: bool = True,
        bm25_weight: float | None = None,
        dense_weight: float | None = None,
        fusion: str | None = None,
        rerank_blend: float | None = None,
        use_mmr: bool | None = None,
        max_subqueries: int | None = None,
        _trace: dict | None = None,
    ) -> list[ScoredChunk]:
        cfg = self.settings.retrieval
        top_k = top_k or cfg.final_top_k
        use_expansion = cfg.query_expansion if use_expansion is None else use_expansion
        use_rerank = cfg.enable_rerank if use_rerank is None else use_rerank
        use_graph = cfg.graph_expansion if use_graph is None else use_graph
        use_mmr = cfg.mmr if use_mmr is None else use_mmr
        fusion = fusion or cfg.fusion
        rerank_blend = cfg.rerank_blend if rerank_blend is None else rerank_blend
        bm25_weight = cfg.bm25_weight if bm25_weight is None else bm25_weight
        dense_weight = cfg.dense_weight if dense_weight is None else dense_weight
        allowed = self.store.filter_ids(source=source, corpus=corpus) if (source or corpus) else None
        started = time.perf_counter()

        variants = plan_queries(query, cfg, use_expansion, max_subqueries)
        trace = _trace if _trace is not None else {}
        trace.update(
            {
                "query": query,
                "entities": extract_entities(query),
                "variants": [
                    {"kind": variant.kind, "text": variant.text, "weight": variant.weight}
                    for variant in variants
                ],
                "channels": [],
                "config": {
                    "fusion": fusion,
                    "bm25_weight": bm25_weight,
                    "dense_weight": dense_weight,
                    "candidate_top_k": cfg.candidate_top_k,
                    "rerank_top_n": cfg.rerank_top_n,
                    "rerank_blend": rerank_blend,
                    "graph_expansion": use_graph,
                    "query_expansion": use_expansion,
                    "mmr": use_mmr,
                },
            }
        )
        fused: dict[str, float] = {}

        for variant in variants:
            channels: list[tuple[str, list[str], list[float]]] = []
            bm25_count = 0
            dense_count = 0
            if use_bm25:
                ids, scores = self._bm25(variant.text, cfg.bm25_top_k, allowed)
                bm25_count = len(ids)
                channels.append(("bm25", ids, scores))
            if use_dense:
                ids, scores = self._dense(variant.text, cfg.dense_top_k, allowed)
                dense_count = len(ids)
                channels.append(("dense", ids, scores))
            trace["channels"].append(
                {
                    "kind": variant.kind,
                    "query": variant.text,
                    "bm25": bm25_count,
                    "dense": dense_count,
                }
            )

            if fusion == "rrf":
                for name, ids, _scores in channels:
                    weight = (bm25_weight if name == "bm25" else dense_weight) * variant.weight
                    for rank, chunk_id in enumerate(ids):
                        fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (cfg.rrf_k + rank + 1)
            else:
                normalized: dict[str, float] = {}
                for name, ids, scores in channels:
                    channel = _normalize(dict(zip(ids, scores, strict=False)))
                    weight = bm25_weight if name == "bm25" else dense_weight
                    for chunk_id, score in channel.items():
                        normalized[chunk_id] = normalized.get(chunk_id, 0.0) + weight * score
                for chunk_id, score in normalized.items():
                    fused[chunk_id] = fused.get(chunk_id, 0.0) + variant.weight * score

        graph_ids: list[str] = []
        if use_graph:
            graph_ids = self._graph_candidates(query, allowed, hops=cfg.graph_hops)
            total = max(1, len(graph_ids))
            for rank, chunk_id in enumerate(graph_ids):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + cfg.graph_weight * (1.0 - rank / total)

        ranked_ids = sorted(fused, key=lambda key: fused[key], reverse=True)[: cfg.candidate_top_k]
        candidates = self.store.get_many(ranked_ids)
        ordered_ids = [chunk_id for chunk_id in ranked_ids if chunk_id in candidates]
        ordered = [candidates[chunk_id] for chunk_id in ordered_ids]
        retrieval_done = time.perf_counter()
        trace["fusion"] = {
            "method": fusion,
            "pool_size": len(fused),
            "candidate_count": len(ordered_ids),
            "graph_candidates": len(graph_ids),
        }
        trace["timings_ms"] = {"retrieval": round((retrieval_done - started) * 1000, 1)}

        if use_rerank:
            rerank_scores = self.reranker.score(query, ordered[: cfg.rerank_top_n])
            reranked_ids = ordered_ids[: len(rerank_scores)]
            base = _normalize({chunk_id: fused[chunk_id] for chunk_id in reranked_ids})
            cross = _normalize(dict(zip(reranked_ids, rerank_scores, strict=False)))
            merged = {
                chunk_id: rerank_blend * cross.get(chunk_id, 0.0)
                + (1.0 - rerank_blend) * base.get(chunk_id, 0.0)
                for chunk_id in reranked_ids
            }
            scored = [
                ScoredChunk(chunk=candidates[chunk_id], score=merged[chunk_id], retriever="reranked")
                for chunk_id in sorted(merged, key=merged.get, reverse=True)
            ]
        else:
            normalized = _normalize({chunk_id: fused[chunk_id] for chunk_id in ordered_ids})
            scored = [
                ScoredChunk(chunk=candidates[chunk_id], score=normalized[chunk_id], retriever="fused")
                for chunk_id in ordered_ids
            ]

        if use_mmr and len(scored) > top_k:
            scored = self._mmr(query, scored, cfg.mmr_pool, cfg.mmr_lambda, top_k)
        trace["rerank"] = {
            "enabled": use_rerank,
            "model": cfg.reranker,
            "candidate_count": min(len(ordered_ids), cfg.rerank_top_n) if use_rerank else 0,
            "blend": rerank_blend,
        }
        trace["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000, 1)
        for rank, item in enumerate(scored):
            item.rank = rank + 1
        return scored[:top_k]

    def search_with_trace(self, query: str, **kwargs) -> tuple[list[ScoredChunk], dict]:
        """Run retrieval and return the stage-by-stage trace used by the UI."""
        trace: dict = {}
        hits = self.search(query, _trace=trace, **kwargs)
        return hits, trace

    def _mmr(
        self,
        query: str,
        scored: list[ScoredChunk],
        pool: int,
        lambda_: float,
        top_k: int,
    ) -> list[ScoredChunk]:
        candidates = scored[:pool]
        if len(candidates) <= top_k:
            return candidates
        vectors = self.dense.encode([item.chunk.text for item in candidates], is_query=False)
        relevance = _normalize({str(index): item.score for index, item in enumerate(candidates)})
        selected: list[int] = []
        remaining = set(range(len(candidates)))
        while remaining and len(selected) < top_k:
            best_index, best_value = -1, float("-inf")
            for index in remaining:
                sim = max(
                    (float(vectors[index] @ vectors[chosen]) for chosen in selected),
                    default=0.0,
                )
                value = lambda_ * relevance[str(index)] - (1.0 - lambda_) * sim
                if value > best_value:
                    best_index, best_value = index, value
            selected.append(best_index)
            remaining.discard(best_index)
        return [candidates[index] for index in selected]

    def _bm25(self, query: str, k: int, allowed: set[str] | None) -> tuple[list[str], list[float]]:
        try:
            ids, scores = self.bm25.search(query, k=k * 3 if allowed else k)
        except Exception as exc:  # empty index / OOV query
            log.debug("bm25 search failed: %s", exc)
            return [], []
        rows = [
            (chunk_id, float(score))
            for chunk_id, score in zip(ids, scores, strict=False)
            if allowed is None or chunk_id in allowed
        ][:k]
        return [row[0] for row in rows], [row[1] for row in rows]

    def _dense(self, query: str, k: int, allowed: set[str] | None) -> tuple[list[str], list[float]]:
        ids, scores = self.dense.search(query, k=k * 3 if allowed else k)
        rows = [
            (chunk_id, float(score))
            for chunk_id, score in zip(ids, scores, strict=False)
            if allowed is None or chunk_id in allowed
        ][:k]
        return [row[0] for row in rows], [row[1] for row in rows]

    def _graph_candidates(self, query: str, allowed: set[str] | None, hops: int = 1) -> list[str]:
        entities = extract_entities(query)
        if not entities:
            return []
        ids: list[str] = []
        direct = self.store.chunks_for_entities(entities, limit=self.settings.retrieval.graph_top_k * 3)
        ids.extend(direct)
        if hops > 0:
            related = self.store.related_entities(entities, limit=self.settings.retrieval.graph_top_k)
            if related:
                ids.extend(self.store.chunks_for_entities(related, limit=self.settings.retrieval.graph_top_k * 2))
        unique: list[str] = []
        seen: set[str] = set()
        for chunk_id in ids:
            if chunk_id in seen or (allowed is not None and chunk_id not in allowed):
                continue
            seen.add(chunk_id)
            unique.append(chunk_id)
            if len(unique) >= self.settings.retrieval.graph_top_k * 3:
                break
        return unique
