"""FastAPI service for retrieval and grounded answers."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import load_settings
from .generate.answer import AnswerGenerator
from .generate.verify import CitationVerifier
from .retrieve import Retriever


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=8, ge=1, le=30)
    source: str | None = None
    corpus: str | None = None


class AskRequest(SearchRequest):
    verify: bool = True


@lru_cache(maxsize=1)
def components(config_path: str | None = None):
    settings = load_settings(config_path)
    return settings, Retriever(settings), AnswerGenerator(settings)


def create_app(config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="AyurRAG", version="0.1.0")
    settings, retriever, generator = components(config_path)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": retriever.store.count()}

    @app.post("/search")
    def search(request: SearchRequest) -> dict:
        hits = retriever.search(
            request.query,
            top_k=request.top_k,
            source=request.source,
            corpus=request.corpus,
        )
        return {
            "query": request.query,
            "results": [
                {
                    "rank": hit.rank,
                    "score": round(hit.score, 5),
                    "chunk_id": hit.chunk.chunk_id,
                    "citation": hit.chunk.citation,
                    "source": hit.chunk.source,
                    "corpus": hit.chunk.corpus,
                    "text": hit.chunk.body,
                }
                for hit in hits
            ],
        }

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        hits = retriever.search(
            request.query,
            top_k=request.top_k,
            source=request.source,
            corpus=request.corpus,
        )
        answer = generator.generate(request.query, hits)
        if request.verify and answer.citations:
            answer = CitationVerifier().verify(answer)
        payload = answer.model_dump()
        payload["evidence"] = [
            {"rank": hit.rank, "score": round(hit.score, 5), "citation": hit.chunk.citation, "text": hit.chunk.body}
            for hit in hits
        ]
        return payload

    return app


app = create_app()


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AyurRAG</title>
<style>
  :root { color-scheme: light dark; --fg:#1b1b1b; --muted:#666; --border:#d8d8d8; --surface:#fff; }
  @media (prefers-color-scheme: dark) { :root { --fg:#e8e8e8; --muted:#aaa; --border:#444; --surface:#1e1e1e; } }
  body { font: 15px/1.55 system-ui, sans-serif; color:var(--fg); background:var(--surface); max-width: 860px; margin: 3rem auto; padding: 0 1rem; }
  h1 { font-size: 1.4rem; margin-bottom: .25rem; }
  p.sub { color: var(--muted); margin-top: 0; }
  form { display:flex; gap:.5rem; margin:1.5rem 0; }
  input { flex:1; padding:.6rem .7rem; border:1px solid var(--border); border-radius:6px; background:transparent; color:var(--fg); }
  button { padding:.6rem 1rem; border-radius:6px; border:1px solid var(--border); background:var(--fg); color:var(--surface); cursor:pointer; }
  .answer { white-space:pre-wrap; border-left:3px solid var(--border); padding-left:1rem; }
  .cite { color:var(--muted); font-size:.85rem; }
  .err { color:#b00020; }
  details { margin-top:1rem; }
  summary { cursor:pointer; color:var(--muted); }
  pre { white-space:pre-wrap; font-size:.85rem; }
</style>
</head>
<body>
<h1>AyurRAG</h1>
<p class="sub">Evidence-grounded Ayurveda, patents, literature and IPR research. Every claim is cited; the expert decides.</p>
<form id="f">
  <input id="q" placeholder="e.g. What does Charaka say about the three pillars of health?" required>
  <button>Ask</button>
</form>
<div id="out"></div>
<script>
const f = document.getElementById('f'), out = document.getElementById('out');
f.addEventListener('submit', async (e) => {
  e.preventDefault();
  out.innerHTML = '<p class="cite">Retrieving evidence and generating a grounded answer…</p>';
  try {
    const r = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query: document.getElementById('q').value, top_k: 8})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'request failed');
    const cites = (d.citations || []).map(c => `<li><code>${c.marker}</code> ${c.citation} <span class="cite">${c.supported === false ? '— unsupported' : c.supported === true ? '— supported' : ''}</span></li>`).join('');
    out.innerHTML = `<div class="answer">${d.answer.replace(/</g,'&lt;')}</div>
      ${d.insufficient_evidence ? '<p class="cite">Flagged: insufficient evidence.</p>' : ''}
      <h3>Citations</h3><ul class="cite">${cites || '<li>none</li>'}</ul>
      <details><summary>Retrieved evidence</summary><pre>${d.evidence.map(e => `[${e.rank}] ${e.citation}\\n${e.text}`).join('\\n\\n').replace(/</g,'&lt;')}</pre></details>`;
  } catch (err) { out.innerHTML = `<p class="err">${err.message}</p>`; }
});
</script>
</body>
</html>"""
