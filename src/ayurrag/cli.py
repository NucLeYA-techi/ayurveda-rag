"""Command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_settings
from .index.build import build_indexes
from .ingest.build import ALL_SOURCES, build_corpus

log = logging.getLogger("ayurrag")


def _configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    sources = tuple(source.strip() for source in args.sources.split(",") if source.strip())
    unknown = set(sources) - set(ALL_SOURCES)
    if unknown:
        raise SystemExit(f"unknown sources: {sorted(unknown)}; choose from {ALL_SOURCES}")
    rows = build_corpus(
        settings,
        sources,
        pubmed_limit=args.pubmed_limit,
        patent_limit=args.patent_limit,
        patent_detail_limit=args.patent_detail_limit,
    )
    print(json.dumps({"chunks": len(rows)}, indent=2))


def cmd_index(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    stats = build_indexes(settings)
    print(json.dumps(stats, indent=2))


def cmd_search(args: argparse.Namespace) -> None:
    from .retrieve import Retriever

    settings = load_settings(args.config)
    retriever = Retriever(settings)
    results = retriever.search(
        args.query,
        top_k=args.top_k,
        source=args.source,
        corpus=args.corpus,
        use_expansion=False if args.no_expansion else None,
        use_rerank=False if args.no_rerank else None,
        use_graph=False if args.no_graph else None,
    )
    for item in results:
        chunk = item.chunk
        print(f"\n#{item.rank} [{item.score:.4f}] {chunk.citation}")
        print(f"    {chunk.body[:400].replace(chr(10), ' ')}")


def cmd_ask(args: argparse.Namespace) -> None:
    from .generate.answer import AnswerGenerator
    from .generate.verify import CitationVerifier
    from .retrieve import Retriever

    settings = load_settings(args.config)
    retriever = Retriever(settings)
    chunks = retriever.search(
        args.query,
        top_k=args.top_k,
        source=args.source,
        corpus=args.corpus,
        use_expansion=False if args.no_expansion else None,
        use_rerank=False if args.no_rerank else None,
        use_graph=False if args.no_graph else None,
    )
    generator = AnswerGenerator(settings)
    answer = generator.generate(args.query, chunks, model=args.model)
    if not args.no_verify and answer.citations:
        answer = CitationVerifier().verify(answer)
    print("\n" + answer.answer + "\n")
    if answer.insufficient_evidence:
        print("(flagged: insufficient evidence)")
    print("Citations:")
    for citation in answer.citations:
        status = "supported" if citation.supported else ("unsupported" if citation.supported is False else "unverified")
        print(f"  {citation.marker} {citation.citation} — {status}")
    if args.show_evidence:
        print("\nEvidence:")
        for item in answer.evidence:
            print(f"  [{item.rank}] {item.chunk.citation} ({item.score:.4f})")


def cmd_stats(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    stats_path = settings.paths.raw_dir / "corpus_stats.json"
    if stats_path.exists():
        print(stats_path.read_text())
    from .store import ChunkStore

    store = ChunkStore(settings.paths.index_dir / "corpus.sqlite")
    print(json.dumps({"indexed_chunks": store.count()}, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("ayurrag.api:app", host=args.host, port=args.port, reload=args.reload)


def cmd_eval(args: argparse.Namespace) -> None:
    from .eval.run import run_eval

    settings = load_settings(args.config)
    report = run_eval(settings, kind=args.kind, limit=args.limit, ablations=args.ablations)
    print(json.dumps(report, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ayurrag", description="Evidence-grounded Ayurveda RAG")
    parser.add_argument("--config", default=None, help="path to YAML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="download sources and build chunks.jsonl")
    ingest.add_argument("--sources", default=",".join(ALL_SOURCES))
    ingest.add_argument("--pubmed-limit", type=int, default=8000)
    ingest.add_argument("--patent-limit", type=int, default=120)
    ingest.add_argument("--patent-detail-limit", type=int, default=60)
    ingest.set_defaults(func=cmd_ingest)

    index = sub.add_parser("index", help="build BM25 + dense indexes")
    index.set_defaults(func=cmd_index)

    search = sub.add_parser("search", help="retrieval only")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=8)
    search.add_argument("--source", default=None)
    search.add_argument("--corpus", default=None)
    search.add_argument("--no-expansion", action="store_true")
    search.add_argument("--no-rerank", action="store_true")
    search.add_argument("--no-graph", action="store_true")
    search.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="retrieval + grounded answer")
    ask.add_argument("query")
    ask.add_argument("--top-k", type=int, default=8)
    ask.add_argument("--source", default=None)
    ask.add_argument("--corpus", default=None)
    ask.add_argument("--model", default=None)
    ask.add_argument("--no-verify", action="store_true")
    ask.add_argument("--no-expansion", action="store_true")
    ask.add_argument("--no-rerank", action="store_true")
    ask.add_argument("--no-graph", action="store_true")
    ask.add_argument("--show-evidence", action="store_true")
    ask.set_defaults(func=cmd_ask)

    stats = sub.add_parser("stats", help="corpus and index statistics")
    stats.set_defaults(func=cmd_stats)

    serve = sub.add_parser("serve", help="run the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    evaluation = sub.add_parser("eval", help="run evaluation")
    evaluation.add_argument("--kind", choices=["retrieval", "generation", "external", "all"], default="all")
    evaluation.add_argument("--limit", type=int, default=100)
    evaluation.add_argument("--ablations", action="store_true")
    evaluation.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
