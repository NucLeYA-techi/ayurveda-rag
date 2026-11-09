"""Citation-constrained answer generation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..config import Settings
from ..llm import LLMClient, parse_json
from ..schema import Answer, Chunk, Citation, ScoredChunk

SYSTEM_PROMPT = """You are an evidence-grounded research assistant for Ayurveda traditional \
knowledge, scientific literature, patents and Indian/international biodiversity-IPR law.

Rules:
1. Use ONLY the numbered evidence (S1..Sn). Do not use outside knowledge.
2. Cite every factual sentence with the evidence markers that support it, e.g. [S1] or [S1][S4].
3. Separate what classical texts claim from what modern studies show, what patents disclose, and \
what the law provides. Never conflate them.
4. If the evidence does not answer the question, set insufficient_evidence to true and explain \
what is missing. Never speculate.
5. Do not give medical advice, diagnosis or treatment recommendations. Legal statements are \
"potentially relevant provisions", never conclusions on patentability or infringement.
6. Be concise (max 180 words). Quote at most one short phrase per source.

Return strict JSON:
{"answer": "text with [S#] citations", "citations": ["S1", "S2"], "insufficient_evidence": false}"""


def _context_block(chunks: list[ScoredChunk], max_chars: int = 1200) -> tuple[str, dict[str, Chunk]]:
    lines = []
    mapping: dict[str, Chunk] = {}
    for index, item in enumerate(chunks, start=1):
        marker = f"S{index}"
        mapping[marker] = item.chunk
        label = item.chunk.citation
        body = item.chunk.body[:max_chars]
        lines.append(f"[{marker}] ({item.chunk.corpus}; {label})\n{body}\n")
    return "\n".join(lines), mapping


class AnswerGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = LLMClient(settings.llm, settings.paths.artifacts_dir / "llm_cache.sqlite")

    def generate(
        self,
        question: str,
        chunks: list[ScoredChunk],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Answer:
        context, mapping = _context_block(chunks)
        user = f"Question: {question}\n\nEvidence:\n{context}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        raw = self.llm.chat(messages, model=model, temperature=temperature)
        try:
            payload = parse_json(raw)
        except ValueError:
            payload = {"answer": raw.strip(), "citations": [], "insufficient_evidence": False}

        text = str(payload.get("answer", "")).strip()
        markers = re.findall(r"\[(S\d+)\]", text)
        citations: list[Citation] = []
        seen: set[str] = set()
        for marker in markers:
            if marker in seen or marker not in mapping:
                continue
            seen.add(marker)
            chunk = mapping[marker]
            citations.append(
                Citation(
                    marker=f"[{marker}]",
                    chunk_id=chunk.chunk_id,
                    citation=chunk.citation,
                    snippet=chunk.body[:280],
                )
            )
        return Answer(
            question=question,
            answer=text,
            citations=citations,
            evidence=chunks,
            insufficient_evidence=bool(payload.get("insufficient_evidence")),
            metadata={
                "model": model or self.settings.llm.model,
                "n_evidence": len(chunks),
                "markers": sorted(seen),
            },
        )


def split_answer_sentences(answer: str) -> list[str]:
    answer = re.sub(r"\[S\d+\]", "", answer)
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    return [part.strip() for part in parts if len(part.strip()) > 25]


def citation_precision(citations: Iterable[Citation]) -> float:
    citations = list(citations)
    if not citations:
        return 0.0
    return sum(1 for citation in citations if citation.supported is True) / len(citations)
