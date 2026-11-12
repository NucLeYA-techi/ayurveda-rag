"""OpenAI-compatible LLM client with streaming, retries and an on-disk cache."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import sqlite3
import time
from dataclasses import dataclass
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
    def key(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        provider: str = "",
    ) -> str:
        payload = json.dumps(
            {
                "provider": provider,
                "model": model,
                "messages": messages,
                "t": temperature,
                "m": max_tokens,
            },
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


@dataclass(frozen=True)
class _Provider:
    name: str
    client: httpx.Client
    model: str
    judge_model: str


def _provider_client(
    base_url: str,
    api_key: str,
    timeout: float,
    user_agent: str | None = None,
) -> httpx.Client:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if user_agent:
        headers["User-Agent"] = user_agent
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(timeout, connect=15.0),
        headers=headers,
    )


def _non_retryable(exc: Exception) -> bool:
    """Auth/request errors will not improve by retrying the same provider."""
    text = str(exc)
    return any(f"HTTP {code}" in text for code in (400, 401, 403, 404, 422, 426))


class LLMClient:
    def __init__(self, cfg: LLMConfig, cache_path: Path | None = None):
        self.cfg = cfg
        self.api_key = cfg.resolved_api_key()
        self.cache = ResponseCache(cache_path) if cache_path else None
        self._providers = [
            _Provider(
                name="primary",
                client=_provider_client(cfg.base_url, self.api_key, cfg.timeout),
                model=cfg.model,
                judge_model=cfg.judge_model,
            )
        ]
        if cfg.fallback:
            fallback_key = cfg.fallback.resolved_api_key()
            if not fallback_key:
                log.warning(
                    "fallback provider %s has no API key (%s); disabled",
                    cfg.fallback.name,
                    cfg.fallback.api_key_env,
                )
            else:
                self._providers.append(
                    _Provider(
                        name=cfg.fallback.name,
                        client=_provider_client(
                            cfg.fallback.base_url,
                            fallback_key,
                            cfg.timeout,
                            cfg.fallback.user_agent,
                        ),
                        model=cfg.fallback.model,
                        judge_model=cfg.fallback.judge_model,
                    )
                )
        self._client = self._providers[0].client

    def close(self) -> None:
        for provider in self._providers:
            provider.client.close()

    def _model_for(self, provider: _Provider, requested: str | None) -> str:
        if requested is None or requested == self.cfg.model:
            return provider.model
        if requested == self.cfg.judge_model:
            return provider.judge_model
        return provider.model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> str:
        temperature = self.cfg.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.cfg.max_tokens
        last_error: Exception | None = None

        for index, provider in enumerate(self._providers):
            model_name = self._model_for(provider, model)
            key = ResponseCache.key(model_name, messages, temperature, max_tokens, provider.name)
            if use_cache and self.cache:
                cached = self.cache.get(key)
                if cached is not None:
                    return cached

            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            attempts = self.cfg.max_retries if provider.name == "primary" else min(2, self.cfg.max_retries)
            for attempt in range(attempts):
                try:
                    text = self._stream(provider.client, payload)
                    if not text.strip():
                        raise LLMError("empty completion")
                    if use_cache and self.cache:
                        self.cache.put(key, model_name, messages, text)
                    return text
                except (httpx.HTTPError, LLMError, json.JSONDecodeError) as exc:
                    last_error = exc
                    if _non_retryable(exc):
                        break
                    if attempt + 1 < attempts:
                        delay = min(45.0, (2**attempt) + random.random() * 1.5)
                        log.warning(
                            "LLM provider %s failed (%s); retrying in %.1fs",
                            provider.name,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
            if index + 1 < len(self._providers):
                log.warning(
                    "LLM provider %s failed (%s); trying %s",
                    provider.name,
                    last_error,
                    self._providers[index + 1].name,
                )
        raise LLMError(
            f"LLM call failed after {len(self._providers)} provider(s): {last_error}"
        )

    def _stream(self, client: httpx.Client, payload: dict[str, Any]) -> str:
        pieces: list[str] = []
        with client.stream("POST", "/chat/completions", json=payload) as response:
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
