"""Configuration loading.

Settings live in ``configs/default.yaml`` and can be overridden with the
``AYURRAG_*`` environment variables (see :class:`Settings`).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")
    artifacts_dir: Path = Path("data/artifacts")

    def resolve(self, root: Path = REPO_ROOT) -> None:
        for name in ("data_dir", "raw_dir", "index_dir", "artifacts_dir"):
            path = getattr(self, name)
            if not path.is_absolute():
                setattr(self, name, (root / path).resolve())


class LLMConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "deepseek-v4.1-flash-free"
    judge_model: str = "gemini-3.5-flash-lite"
    api_key_env: str = "AYURRAG_LLM_API_KEY"
    api_key: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1800
    timeout: float = 180.0
    max_retries: int = 4

    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get(self.api_key_env, "")


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    multilingual_model: str = "intfloat/multilingual-e5-small"
    batch_size: int = 32
    normalize: bool = True
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    passage_prefix: str = ""


class RetrievalConfig(BaseModel):
    bm25_top_k: int = 60
    dense_top_k: int = 60
    rrf_k: int = 60
    bm25_weight: float = 0.8
    dense_weight: float = 0.2
    fusion: str = "convex"  # convex score fusion | rrf rank fusion
    candidate_top_k: int = 120
    rerank_top_n: int = 80
    rerank_blend: float = 0.5
    final_top_k: int = 8
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_rerank: bool = True
    graph_expansion: bool = False
    graph_top_k: int = 20
    graph_weight: float = 0.08
    graph_hops: int = 1
    query_expansion: bool = True
    max_subqueries: int = 0
    alias_weight: float = 0.3
    subquery_weight: float = 0.2
    mmr: bool = False
    mmr_lambda: float = 0.75
    mmr_pool: int = 24


class ChunkingConfig(BaseModel):
    max_chars: int = 1400
    overlap_chars: int = 180
    min_chars: int = 40


class EvalConfig(BaseModel):
    judge_model: str = "gemini-3.5-flash-lite"
    sample_size: int = 200
    seed: int = 13


class Settings(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    config_path = Path(path) if path else DEFAULT_CONFIG
    data: dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text()) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    settings = Settings.model_validate(data)
    settings.paths.resolve()
    for directory in (settings.paths.raw_dir, settings.paths.index_dir, settings.paths.artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings(os.environ.get("AYURRAG_CONFIG"))
