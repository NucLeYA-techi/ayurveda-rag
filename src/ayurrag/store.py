"""SQLite-backed chunk store with entity postings."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .schema import Chunk


class ChunkStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                source TEXT,
                corpus TEXT,
                title TEXT,
                text TEXT,
                body TEXT,
                header TEXT,
                metadata TEXT,
                entities TEXT,
                token_count INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
            CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus);
            CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id TEXT,
                entity TEXT,
                PRIMARY KEY (chunk_id, entity)
            );
            CREATE INDEX IF NOT EXISTS idx_entity ON chunk_entities(entity);
            CREATE TABLE IF NOT EXISTS entity_edges (
                entity_a TEXT,
                entity_b TEXT,
                weight REAL,
                PRIMARY KEY (entity_a, entity_b)
            );
            CREATE INDEX IF NOT EXISTS idx_edge_a ON entity_edges(entity_a);
            CREATE INDEX IF NOT EXISTS idx_edge_b ON entity_edges(entity_b);
            """
        )
        self.conn.commit()

    def clear(self) -> None:
        self.conn.executescript(
            "DELETE FROM chunks; DELETE FROM chunk_entities; DELETE FROM entity_edges;"
        )
        self.conn.commit()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.source,
                chunk.corpus,
                chunk.title,
                chunk.text,
                chunk.body,
                chunk.header,
                json.dumps(chunk.metadata, ensure_ascii=False),
                json.dumps(chunk.entities, ensure_ascii=False),
                chunk.token_count,
            )
            for chunk in chunks
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        self.conn.executemany(
            "INSERT OR IGNORE INTO chunk_entities VALUES (?,?)",
            [(chunk.chunk_id, entity) for chunk in chunks for entity in chunk.entities],
        )
        self.conn.commit()

    def get(self, chunk_id: str) -> Chunk | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", chunk_ids)
        return {row[0]: self._row_to_chunk(row) for row in rows}

    def all_ids(self) -> list[str]:
        return [row[0] for row in self.conn.execute("SELECT chunk_id FROM chunks")]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def filter_ids(self, source: str | None = None, corpus: str | None = None) -> set[str]:
        clauses, params = [], []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if corpus:
            clauses.append("corpus = ?")
            params.append(corpus)
        if not clauses:
            return set(self.all_ids())
        sql = "SELECT chunk_id FROM chunks WHERE " + " AND ".join(clauses)
        return {row[0] for row in self.conn.execute(sql, params)}

    def rebuild_entity_graph(self) -> int:
        """Build co-occurrence edges between entities in the same chunk.

        This is the lightweight knowledge-graph layer: one hop from a query
        entity reaches entities that co-occur with it in classical passages,
        patents, papers or legal text. It is not an LLM-extracted semantic KG.
        """
        from collections import Counter

        self.conn.execute("DELETE FROM entity_edges")
        by_chunk: dict[str, list[str]] = {}
        for chunk_id, entity in self.conn.execute("SELECT chunk_id, entity FROM chunk_entities"):
            by_chunk.setdefault(chunk_id, []).append(entity)
        edges: Counter[tuple[str, str]] = Counter()
        for entities in by_chunk.values():
            unique = sorted(set(entities))
            for index, left in enumerate(unique):
                for right in unique[index + 1 :]:
                    edges[(left, right)] += 1
        self.conn.executemany(
            "INSERT OR REPLACE INTO entity_edges VALUES (?,?,?)",
            [(left, right, float(weight)) for (left, right), weight in edges.items()],
        )
        self.conn.commit()
        return len(edges)

    def related_entities(self, entities: list[str], limit: int = 10) -> list[str]:
        if not entities:
            return []
        from collections import Counter

        marks = ",".join("?" * len(entities))
        rows = self.conn.execute(
            f"SELECT entity_b, weight FROM entity_edges WHERE entity_a IN ({marks}) "
            f"UNION ALL "
            f"SELECT entity_a, weight FROM entity_edges WHERE entity_b IN ({marks})",
            [*entities, *entities],
        )
        scores: Counter[str] = Counter()
        source = set(entities)
        for neighbor, weight in rows:
            if neighbor not in source:
                scores[neighbor] += float(weight)
        return [entity for entity, _ in scores.most_common(limit)]

    def chunks_for_entities(self, entities: list[str], limit: int = 200) -> list[str]:
        if not entities:
            return []
        marks = ",".join("?" * len(entities))
        rows = self.conn.execute(
            f"SELECT chunk_id, COUNT(*) AS hits FROM chunk_entities "
            f"WHERE entity IN ({marks}) GROUP BY chunk_id ORDER BY hits DESC LIMIT ?",
            [*entities, limit],
        )
        return [row[0] for row in rows]

    def sample_chunks(self, source: str | None = None, limit: int = 100, seed: int = 13) -> list[Chunk]:
        sql = "SELECT * FROM chunks"
        params: list = []
        if source:
            sql += " WHERE source = ?"
            params.append(source)
        sql += " ORDER BY RANDOM() LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    @staticmethod
    def _row_to_chunk(row: tuple) -> Chunk:
        return Chunk(
            chunk_id=row[0],
            doc_id=row[1],
            source=row[2],
            corpus=row[3],
            title=row[4],
            text=row[5],
            body=row[6],
            header=row[7],
            metadata=json.loads(row[8]),
            entities=json.loads(row[9]),
            token_count=row[10],
        )

    def close(self) -> None:
        self.conn.close()
