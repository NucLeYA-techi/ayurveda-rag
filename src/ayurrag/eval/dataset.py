"""Build a domain retrieval benchmark from the indexed corpus."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from ..config import Settings
from ..llm import LLMClient, parse_json
from ..schema import Chunk

log = logging.getLogger(__name__)

QUERY_PROMPT = """You build retrieval benchmarks from Ayurveda research sources.
Given one passage, write realistic search queries whose answer is derivable from THIS passage only.

Rules:
- Produce exactly 1 query per requested style.
- Available styles: natural_question, terminology_heavy, comparative.
- natural_question: a plain-language research question.
- terminology_heavy: use the Sanskrit, botanical or technical terms that appear in the passage.
- comparative: ask how two entities/ideas in the passage relate.
- Never copy a full sentence from the passage. Max 30 words.
- Do not mention "the passage".

Return strict JSON: {"queries": [{"style": "...", "query": "..."}]}"""

STYLES = ("natural_question", "terminology_heavy")


def build_dataset(
    settings: Settings,
    output: Path,
    sample_size: int = 200,
    seed: int = 13,
    per_source: int | None = None,
) -> list[dict]:
    chunks = Chunk.load_jsonl(settings.paths.raw_dir / "chunks.jsonl")
    rng = random.Random(seed)
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if chunk.token_count >= 50 and len(chunk.body) > 200:
            by_source.setdefault(chunk.source, []).append(chunk)

    sources = sorted(by_source)
    quota = per_source or max(1, sample_size // max(1, len(sources)))
    samples: list[Chunk] = []
    for source in sources:
        pool = by_source[source]
        rng.shuffle(pool)
        samples.extend(pool[:quota])
    rng.shuffle(samples)
    samples = samples[:sample_size]

    llm = LLMClient(settings.llm, settings.paths.artifacts_dir / "llm_cache.sqlite")
    rows: list[dict] = []
    for index, chunk in enumerate(samples, start=1):
        style = STYLES[index % len(STYLES)]
        messages = [
            {"role": "system", "content": QUERY_PROMPT},
            {
                "role": "user",
                "content": f"Source: {chunk.source} | {chunk.citation}\nStyle: {style}\nPassage:\n{chunk.body[:2400]}",
            },
        ]
        try:
            payload = parse_json(llm.chat(messages, temperature=0.4, max_tokens=400))
            queries = [item.get("query", "").strip() for item in payload.get("queries", [])]
            query = next((value for value in queries if len(value) > 12), "")
        except Exception as exc:  # noqa: BLE001
            log.warning("query generation failed for %s: %s", chunk.chunk_id, exc)
            query = ""
        if not query:
            continue
        rows.append(
            {
                "query": query,
                "style": style,
                "gold_chunk_ids": [chunk.chunk_id],
                "gold_citation": chunk.citation,
                "source": chunk.source,
                "corpus": chunk.corpus,
                "passage": chunk.body[:1800],
            }
        )
        if index % 20 == 0:
            log.info("generated %d/%d queries", index, len(samples))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("wrote %d evaluation queries to %s", len(rows), output)
    return rows


UNANSWERABLE = [
    "What is the capital of France?",
    "Who won the 2022 FIFA World Cup?",
    "What is the boiling point of mercury in Celsius?",
    "Which company manufactured the first iPhone?",
    "How many moons does Jupiter have?",
    "What did Charaka write about X-ray crystallography?",
    "Which clinical trial proved that ashwagandha cures Alzheimer's disease?",
    "What is the exact royalty rate set by the 2025 Indian patent amendment?",
    "How does the Sushruta Samhita describe RNA interference?",
    "What is the GDP of India in 2024?",
    "Which Ayurvedic herb is approved by the FDA for treating diabetes?",
    "What are the side effects of the COVID-19 vaccine according to Charaka?",
    "What is the patent number for a turmeric-based cancer drug?",
    "How many verses are in the Rigveda?",
    "What did the WTO ruling say about the Neem patent in 2024?",
]
