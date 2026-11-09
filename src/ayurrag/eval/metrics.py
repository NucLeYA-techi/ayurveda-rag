"""Retrieval metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(ranked: list[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(gold_set & set(ranked[:k])) / len(gold_set)


def mrr_at_k(ranked: list[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    for rank, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, chunk_id in enumerate(ranked[:k], start=1) if chunk_id in gold_set)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def aggregate(rows: list[dict], k: int) -> dict[str, float]:
    if not rows:
        return {"n": 0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    return {
        "n": len(rows),
        "recall": sum(row[f"recall@{k}"] for row in rows) / len(rows),
        "mrr": sum(row[f"mrr@{k}"] for row in rows) / len(rows),
        "ndcg": sum(row[f"ndcg@{k}"] for row in rows) / len(rows),
    }
