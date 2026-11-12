import json

import httpx

from ayurrag import llm as llm_mod
from ayurrag.config import LLMConfig, LLMProviderConfig
from ayurrag.llm import LLMClient


def test_fallback_provider_and_judge_model_mapping(monkeypatch):
    seen_models: list[str] = []

    def primary_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid key"})

    def fallback_handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen_models.append(payload["model"])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"FALLBACK_OK"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    def fake_client(base_url, api_key, timeout, user_agent=None):
        assert api_key
        handler = primary_handler if "primary" in base_url else fallback_handler
        return httpx.Client(
            base_url=base_url,
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(llm_mod, "_provider_client", fake_client)

    cfg = LLMConfig(
        base_url="https://primary.test/v1",
        model="primary-model",
        judge_model="primary-judge",
        api_key="primary-key",
        max_retries=1,
        fallback=LLMProviderConfig(
            name="backup",
            base_url="https://backup.test/v1",
            model="fallback-model",
            judge_model="fallback-judge",
            api_key="backup-key",
        ),
    )
    client = LLMClient(cfg)
    try:
        reply = client.chat(
            [{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=32,
            use_cache=False,
        )
        assert reply == "FALLBACK_OK"

        client.chat(
            [{"role": "user", "content": "judge"}],
            model=cfg.judge_model,
            temperature=0,
            max_tokens=32,
            use_cache=False,
        )
    finally:
        client.close()

    assert seen_models == ["fallback-model", "fallback-judge"]
