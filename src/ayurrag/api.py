"""FastAPI service and showcase UI for retrieval and grounded answers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_settings
from .generate.answer import AnswerGenerator
from .generate.verify import CitationVerifier
from .retrieve import Retriever
from .schema import ScoredChunk

STATIC_DIR = Path(__file__).parent / "static"
UNKNOWN_ANSWER = "I don't know. The indexed documents do not contain enough evidence to answer this question."
UNKNOWN_DETAIL = "The indexed documents do not contain enough evidence to answer this question."


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=8, ge=1, le=30)
    source: str | None = None
    corpus: str | None = None


class AskRequest(SearchRequest):
    verify: bool = True


@lru_cache(maxsize=1)
def components(config_path: str | None = None):
    settings = load_settings(config_path)
    return settings, Retriever(settings), AnswerGenerator(settings)


def _evidence(hits: list[ScoredChunk]) -> list[dict]:
    return [
        {
            "rank": hit.rank,
            "score": round(hit.score, 5),
            "chunk_id": hit.chunk.chunk_id,
            "citation": hit.chunk.citation,
            "source": hit.chunk.source,
            "corpus": hit.chunk.corpus,
            "retriever": hit.retriever,
            "text": hit.chunk.body,
        }
        for hit in hits
    ]


def create_app(config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="AyurRAG", version="0.1.0")
    settings, retriever, generator = components(config_path)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": retriever.store.count()}

    @app.post("/search")
    def search(request: SearchRequest) -> dict:
        hits, trace = retriever.search_with_trace(
            request.query,
            top_k=request.top_k,
            source=request.source,
            corpus=request.corpus,
        )
        return {
            "query": request.query,
            "results": _evidence(hits),
            "trace": trace,
        }

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        started = perf_counter()
        hits, trace = retriever.search_with_trace(
            request.query,
            top_k=request.top_k,
            source=request.source,
            corpus=request.corpus,
        )
        retrieval_done = perf_counter()
        search_total = trace.get("timings_ms", {}).get("total")

        answer = generator.generate(request.query, hits)
        generation_done = perf_counter()

        verifier = CitationVerifier()
        if request.verify and answer.citations and not answer.insufficient_evidence:
            answer = verifier.verify(answer)
        verification_done = perf_counter()

        # When the answer is withheld, the model's explanation is not the
        # delivered answer; do not present citations for it as if they were
        # support for the final text.
        citations = [] if answer.insufficient_evidence else answer.citations
        supported = sum(1 for citation in citations if citation.supported is True)
        unsupported = sum(1 for citation in citations if citation.supported is False)
        unverified = len(citations) - supported - unsupported

        trace.setdefault("timings_ms", {})
        trace["timings_ms"].update(
            {
                "search_total": search_total,
                "generation": round((generation_done - retrieval_done) * 1000, 1),
                "verification": round((verification_done - generation_done) * 1000, 1),
                "total": round((verification_done - started) * 1000, 1),
            }
        )
        trace["generation"] = {
            "model": answer.metadata.get("model") or settings.llm.model,
            "n_evidence": len(hits),
        }
        trace["verification"] = {
            "enabled": request.verify,
            "model": verifier.model_name if request.verify and citations else None,
            "citations": len(citations),
            "supported": supported,
            "unsupported": unsupported,
            "unverified": unverified,
        }

        model_explanation = None
        if answer.insufficient_evidence:
            model_explanation = answer.answer
            answer_text = UNKNOWN_ANSWER
        else:
            answer_text = answer.answer

        payload = answer.model_dump()
        payload["answer"] = answer_text
        payload["citations"] = [citation.model_dump() for citation in citations]
        payload["model_explanation"] = model_explanation
        payload["display_message"] = UNKNOWN_DETAIL if answer.insufficient_evidence else None
        payload["evidence"] = _evidence(hits)
        payload["trace"] = trace
        return payload

    return app


app = create_app()
