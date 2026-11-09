"""OpenAI-compatible LLM client with streaming, retries and an on-disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx

from .config import LLMConfig

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class ResponseCache:
    """SQLite-backed cache so evaluations are reproducible and cheap to re-run."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses "
            "(key TEXT PRIMARY KEY, model TEXT, prompt TEXT, response TEXT, created REAL)"
        )
        self._conn.commit()

    @staticmethod
    def key(model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, "t": temperature, "m": max_tokens},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, model: str, messages: list[dict[str, str]], response: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses(key, model, prompt, response, created) VALUES (?,?,?,?,?)",
            (key, model, json.dumps(messages, ensure_ascii=False), response, time.time()),
        )
        self._conn.commit()


class LLMClient:
    def __init__(self, cfg: LLMConfig, cache_path: Path | None = None):
        self.cfg = cfg
        self.api_key = cfg.resolved_api_key()
        self.cache = ResponseCache(cache_path) if cache_path else None
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.timeout, connect=15.0),
            headers=headers,
        )

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> str:
        model = model or self.cfg.model
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.cfg.max_tokens
        key = ResponseCache.key(model, messages, temperature, max_tokens)
        if use_cache and self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                text = self._stream(payload)
                if not text.strip():
                    raise LLMError("empty completion")
                if use_cache and self.cache:
                    self.cache.put(key, model, messages, text)
                return text
            except (httpx.HTTPError, LLMError, json.JSONDecodeError) as exc:
                last_error = exc
                delay = min(45.0, (2**attempt) + random.random() * 1.5)
                log.warning("LLM call failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise LLMError(f"LLM call failed after {self.cfg.max_retries} attempts: {last_error}")

    def _stream(self, payload: dict[str, Any]) -> str:
        pieces: list[str] = []
        with self._client.stream("POST", "/chat/completions", json=payload) as response:
            if response.status_code >= 400:
                body = response.read().decode("utf-8", "replace")[:500]
                raise LLMError(f"HTTP {response.status_code}: {body}")
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        pieces.append(content)
        return "".join(pieces)

    def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        text = self.chat(messages, **kwargs)
        return parse_json(text)


def parse_json(text: str) -> Any:
    """Extract the first JSON object/array from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"no JSON found in response: {text[:200]}")
    candidate = text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(candidate):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return json.loads(candidate[: i + 1])
        raise
