"""Core data models shared by ingestion, retrieval and the API."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, Field

CorpusName = Literal["classical", "literature", "patent", "legal", "reference"]


def make_id(*parts: str) -> str:
    digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


class SourceDocument(BaseModel):
    """A normalized document before chunking.

    ``text`` is clean body text. Parsers may additionally provide ``units``:
    pre-segmented passages (verses, sections, abstract lines). The chunker
    keeps units intact, which yields precise citations for classical texts.
    """

    doc_id: str
    source: str
    corpus: CorpusName
    title: str
    text: str = ""
    units: list[str] = Field(default_factory=list)
    unit_labels: list[str] = Field(default_factory=list)
    url: str = ""
    license: str = ""
    lang: str = "en"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    corpus: CorpusName
    title: str
    text: str
    body: str
    header: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    entities: list[str] = Field(default_factory=list)
    token_count: int = 0

    @property
    def citation(self) -> str:
        return str(self.metadata.get("citation") or f"{self.source}: {self.title}")

    @classmethod
    def load_jsonl(cls, path) -> list[Chunk]:
        import orjson

        with open(path, "rb") as handle:
            return [cls.model_validate(orjson.loads(line)) for line in handle if line.strip()]


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    retriever: str = ""
    rank: int = 0


class Citation(BaseModel):
    marker: str
    chunk_id: str
    citation: str
    snippet: str
    supported: bool | None = None
    support_score: float | None = None


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[ScoredChunk] = Field(default_factory=list)
    insufficient_evidence: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
