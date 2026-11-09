"""Sparse (BM25) and dense indexes."""

from __future__ import annotations

import json
from pathlib import Path

import bm25s
import faiss
import numpy as np

from ..config import EmbeddingConfig
from ..schema import Chunk

EMBEDDING_BATCH = 64


class BM25Index:
    def __init__(self, directory: Path):
        self.directory = directory

    def build(self, texts: list[str], ids: list[str]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tokens = bm25s.tokenize(texts, stopwords=[], show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        retriever.save(str(self.directory))
        (self.directory / "bm25_ids.json").write_text(json.dumps(ids))
        self._retriever = retriever
        self._ids = ids

    def load(self) -> BM25Index:
        self._retriever = bm25s.BM25.load(str(self.directory), load_corpus=False)
        self._ids = json.loads((self.directory / "bm25_ids.json").read_text())
        return self

    def search(self, query: str, k: int) -> tuple[list[str], np.ndarray]:
        tokens = bm25s.tokenize([query], stopwords=[], show_progress=False)
        results, scores = self._retriever.retrieve(tokens, k=k, show_progress=False)
        ids = [self._ids[i] for i in results[0]]
        return ids, scores[0]


class DenseIndex:
    def __init__(self, directory: Path, cfg: EmbeddingConfig):
        self.directory = directory
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.model, device="cpu")
        return self._model

    def encode(self, texts: list[str], *, is_query: bool = False, batch_size: int | None = None):
        prefix = self.cfg.query_prefix if is_query else self.cfg.passage_prefix
        if prefix:
            texts = [prefix + text for text in texts]
        return self.model.encode(
            texts,
            batch_size=batch_size or self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
        ).astype(np.float32)

    def build(self, chunks: list[Chunk], ids: list[str]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        embeddings = []
        for start in range(0, len(chunks), EMBEDDING_BATCH):
            batch = [chunk.text for chunk in chunks[start : start + EMBEDDING_BATCH]]
            embeddings.append(self.encode(batch))
        matrix = np.vstack(embeddings)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(self.directory / "dense.faiss"))
        (self.directory / "dense_ids.json").write_text(json.dumps(ids))
        (self.directory / "dense_meta.json").write_text(
            json.dumps({"model": self.cfg.model, "dim": int(matrix.shape[1]), "count": len(ids)})
        )

    def load(self) -> DenseIndex:
        self._index = faiss.read_index(str(self.directory / "dense.faiss"))
        self._ids = json.loads((self.directory / "dense_ids.json").read_text())
        return self

    @property
    def size(self) -> int:
        return self._index.ntotal

    def search(self, query: str, k: int) -> tuple[list[str], np.ndarray]:
        vector = self.encode([query], is_query=True)
        scores, indices = self._index.search(vector, min(k, self._index.ntotal))
        ids = [self._ids[i] for i in indices[0] if i >= 0]
        return ids, scores[0][: len(ids)]
