"""Evaluation harness: retrieval ablations and end-to-end answer quality."""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

from ..config import Settings
from ..generate.answer import AnswerGenerator
from ..generate.verify import CitationVerifier, uncited_sentences
from ..retrieve import Retriever
from ..schema import Chunk, ScoredChunk
from .dataset import UNANSWERABLE, build_dataset
from .judge import LLMJudge
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k

log = logging.getLogger(__name__)

ABLATIONS: dict[str, dict] = {
    "bm25": {"use_bm25": True, "use_dense": False, "use_expansion": False, "use_rerank": False, "use_graph": False},
    "dense": {"use_bm25": False, "use_dense": True, "use_expansion": False, "use_rerank": False, "use_graph": False},
    "hybrid": {"use_bm25": True, "use_dense": True, "use_expansion": False, "use_rerank": False, "use_graph": False, "bm25_weight": 1.0, "dense_weight": 1.0},
    "hybrid+expand": {"use_bm25": True, "use_dense": True, "use_expansion": True, "use_rerank": False, "use_graph": False, "bm25_weight": 1.0, "dense_weight": 1.0},
    "bm25+expand": {"use_bm25": True, "use_dense": False, "use_expansion": True, "use_rerank": False, "use_graph": False},
    "weighted8+expand": {"use_bm25": True, "use_dense": True, "use_expansion": True, "use_rerank": False, "use_graph": False, "bm25_weight": 8.0, "dense_weight": 1.0},
    "hybrid+rerank": {"use_bm25": True, "use_dense": True, "use_expansion": False, "use_rerank": True, "use_graph": False, "bm25_weight": 1.0, "dense_weight": 1.0},
    "full": {"use_bm25": True, "use_dense": True, "use_expansion": True, "use_rerank": True, "use_graph": True},
}


def load_dataset(settings: Settings, sample_size: int) -> list[dict]:
    path = settings.paths.artifacts_dir / "retrieval_gold.jsonl"
    if not path.exists():
        log.info("building retrieval benchmark at %s", path)
        build_dataset(settings, path, sample_size=max(sample_size, 200))
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[:sample_size] if sample_size else rows


def evaluate_retrieval(settings: Settings, rows: list[dict], retriever: Retriever) -> dict[str, dict]:
    results: dict[str, dict] = {}
    ks = (5, 10, 20)
    for name, flags in ABLATIONS.items():
        per_query = []
        for row in rows:
            hits = retriever.search(row["query"], top_k=20, **flags)
            ranked = [hit.chunk.chunk_id for hit in hits]
            record = {}
            for k in ks:
                record[f"recall@{k}"] = recall_at_k(ranked, row["gold_chunk_ids"], k)
                record[f"mrr@{k}"] = mrr_at_k(ranked, row["gold_chunk_ids"], k)
                record[f"ndcg@{k}"] = ndcg_at_k(ranked, row["gold_chunk_ids"], k)
            record["source"] = row["source"]
            per_query.append(record)
        summary: dict[str, float | int] = {"n": len(per_query)}
        for k in ks:
            for metric in ("recall", "mrr", "ndcg"):
                key = f"{metric}@{k}"
                summary[key] = round(statistics.fmean(row[key] for row in per_query), 4) if per_query else 0.0
        breakdown = {}
        for source in sorted({row["source"] for row in per_query}):
            subset = [row for row in per_query if row["source"] == source]
            breakdown[source] = round(statistics.fmean(row["recall@10"] for row in subset), 4)
        summary["recall@10_by_source"] = breakdown
        results[name] = summary
        log.info("retrieval %s: recall@10=%.3f ndcg@10=%.3f", name, summary["recall@10"], summary["ndcg@10"])
    return results


def _evidence_text(chunks) -> str:
    parts = []
    for index, item in enumerate(chunks, start=1):
        parts.append(f"[S{index}] {item.chunk.citation}\n{item.chunk.body[:1200]}")
    return "\n\n".join(parts)


def evaluate_generation(settings: Settings, rows: list[dict], limit: int = 60) -> dict:
    retriever = Retriever(settings)
    generator = AnswerGenerator(settings)
    verifier = CitationVerifier()
    judge = LLMJudge(settings)
    records = []
    for row in rows[:limit]:
        chunks = retriever.search(row["query"], top_k=settings.retrieval.final_top_k)
        answer = generator.generate(row["query"], chunks)
        if answer.citations:
            answer = verifier.verify(answer)
        judged = judge.score(
            question=row["query"],
            answer=answer.answer,
            evidence=_evidence_text(chunks),
            reference=row["passage"],
        )
        sentences = [s for s in answer.answer.replace("\n", " ").split(". ") if len(s.strip()) > 30]
        cited = sum(1 for s in sentences if "[S" in s)
        records.append(
            {
                "query": row["query"],
                "source": row["source"],
                "answer": answer.answer,
                "citations": [citation.model_dump() for citation in answer.citations],
                "insufficient_evidence": answer.insufficient_evidence,
                "citation_coverage": round(cited / len(sentences), 4) if sentences else 0.0,
                "citation_precision": round(
                    statistics.fmean(
                        1.0 if c.supported else 0.0 for c in answer.citations if c.supported is not None
                    ),
                    4,
                )
                if any(c.supported is not None for c in answer.citations)
                else None,
                "uncited_sentences": len(uncited_sentences(answer)),
                "judge": judged,
            }
        )

    def mean(key: str) -> float | None:
        values = [record["judge"].get(key) for record in records if record["judge"] and record["judge"].get(key)]
        return round(statistics.fmean(values), 3) if values else None

    summary = {
        "n": len(records),
        "correctness": mean("correctness"),
        "groundedness": mean("groundedness"),
        "completeness": mean("completeness"),
        "citation_quality": mean("citation_quality"),
        "harmful_overreach_rate": round(
            statistics.fmean(1.0 if record["judge"].get("harmful_overreach") else 0.0 for record in records if record["judge"]),
            4,
        )
        if any(record["judge"] for record in records)
        else None,
        "citation_coverage": round(statistics.fmean(record["citation_coverage"] for record in records), 4),
        "citation_precision_nli": round(
            statistics.fmean(record["citation_precision"] for record in records if record["citation_precision"] is not None),
            4,
        )
        if any(record["citation_precision"] is not None for record in records)
        else None,
        "insufficient_evidence_rate": round(
            statistics.fmean(1.0 if record["insufficient_evidence"] else 0.0 for record in records), 4
        ),
        "records": records,
    }
    log.info("generation summary: %s", {k: v for k, v in summary.items() if k != "records"})
    return summary


def evaluate_refusal(settings: Settings, limit: int = 15) -> dict:
    retriever = Retriever(settings)
    generator = AnswerGenerator(settings)
    judge = LLMJudge(settings)
    rows = []
    for question in UNANSWERABLE[:limit]:
        chunks = retriever.search(question, top_k=6)
        answer = generator.generate(question, chunks)
        refused = answer.insufficient_evidence or judge.is_refusal(answer.answer)
        rows.append(
            {
                "question": question,
                "refused": refused,
                "insufficient_flag": answer.insufficient_evidence,
                "answer": answer.answer,
                "top_citation": chunks[0].chunk.citation if chunks else None,
            }
        )
    return {
        "n": len(rows),
        "refusal_rate": round(statistics.fmean(1.0 if row["refused"] else 0.0 for row in rows), 4),
        "records": rows,
    }


def evaluate_external(settings: Settings, limit: int = 48) -> dict:
    """Oracle-context PubMedQA check: gold contexts, same generator and judge."""
    path = Path(__file__).resolve().parents[3] / "benchmarks" / "pubmedqa_sample.jsonl"
    if not path.exists():
        log.warning("external benchmark missing at %s", path)
        return {"n": 0, "error": "missing benchmark"}
    generator = AnswerGenerator(settings)
    verifier = CitationVerifier()
    judge = LLMJudge(settings)
    records = []
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()][:limit]
    for row in rows:
        chunks = []
        for index, passage in enumerate(row["contexts"], start=1):
            chunk = Chunk(
                chunk_id=f"pubmedqa-{row['pubid']}-{index}",
                doc_id=f"pubmedqa-{row['pubid']}",
                source="PubMedQA",
                corpus="literature",
                title=row["question"],
                text=passage,
                body=passage,
                metadata={"citation": f"PubMedQA PMID {row['pubid']} — {row['question'][:90]}"},
                token_count=len(passage.split()),
            )
            chunks.append(ScoredChunk(chunk=chunk, score=1.0, retriever="gold_context", rank=index))
        answer = generator.generate(row["question"], chunks)
        if answer.citations:
            answer = verifier.verify(answer)
        judged = judge.score(
            question=row["question"],
            answer=answer.answer,
            evidence=_evidence_text(chunks),
            reference=f"{row['long_answer']}\n\nFinal decision: {row['final_decision']}",
        )
        lowered = answer.answer.lower()
        records.append(
            {
                "pubid": row["pubid"],
                "question": row["question"],
                "final_decision": row["final_decision"],
                "answer": answer.answer,
                "citations": [citation.model_dump() for citation in answer.citations],
                "decision_mentioned": row["final_decision"] in lowered,
                "judge": judged,
            }
        )

    def mean(key: str) -> float | None:
        values = [record["judge"].get(key) for record in records if record["judge"] and record["judge"].get(key)]
        return round(statistics.fmean(values), 3) if values else None

    summary = {
        "n": len(records),
        "correctness": mean("correctness"),
        "groundedness": mean("groundedness"),
        "completeness": mean("completeness"),
        "citation_quality": mean("citation_quality"),
        "harmful_overreach_rate": round(
            statistics.fmean(1.0 if record["judge"].get("harmful_overreach") else 0.0 for record in records if record["judge"]),
            4,
        )
        if any(record["judge"] for record in records)
        else None,
        "decision_mentioned_rate": round(
            statistics.fmean(1.0 if record["decision_mentioned"] else 0.0 for record in records), 4
        ),
        "records": records,
    }
    log.info("external PubMedQA summary: %s", {k: v for k, v in summary.items() if k != "records"})
    return summary


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    lines = ["# Evaluation report", ""]
    if "retrieval" in payload:
        lines += ["## Retrieval ablations", "", "| pipeline | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 |", "|---|---|---|---|---|---|"]
        for name, summary in payload["retrieval"].items():
            lines.append(
                f"| {name} | {summary['recall@5']:.3f} | {summary['recall@10']:.3f} | "
                f"{summary['recall@20']:.3f} | {summary['mrr@10']:.3f} | {summary['ndcg@10']:.3f} |"
            )
    if "generation" in payload:
        gen = {k: v for k, v in payload["generation"].items() if k != "records"}
        lines += ["", "## Answer quality (LLM judge, 1-5)", "", "| metric | value |", "|---|---|"]
        for key, value in gen.items():
            lines.append(f"| {key} | {value} |")
    if "external" in payload and payload["external"].get("n"):
        ext = {k: v for k, v in payload["external"].items() if k != "records"}
        lines += ["", "## External PubMedQA (oracle contexts, LLM judge, 1-5)", "", "| metric | value |", "|---|---|"]
        for key, value in ext.items():
            lines.append(f"| {key} | {value} |")
    if "refusal" in payload:
        lines += ["", f"Refusal rate on unanswerable questions: {payload['refusal']['refusal_rate']:.3f}"]
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def run_eval(settings: Settings, kind: str = "all", limit: int = 100, ablations: bool = False) -> dict:
    rows = load_dataset(settings, sample_size=limit)
    report_path = settings.paths.artifacts_dir / "eval_report.json"
    payload: dict = json.loads(report_path.read_text()) if report_path.exists() else {}
    if kind in ("retrieval", "all"):
        retriever = Retriever(settings)
        if ablations:
            payload["retrieval"] = evaluate_retrieval(settings, rows, retriever)
        else:
            payload["retrieval"] = evaluate_retrieval(settings, rows, retriever)
    if kind in ("generation", "all"):
        payload["generation"] = evaluate_generation(settings, rows, limit=min(limit, 80))
        payload["refusal"] = evaluate_refusal(settings)
    if kind in ("external", "all"):
        payload["external"] = evaluate_external(settings)
    write_report(report_path, payload)
    return payload
