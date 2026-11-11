(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const form = $("ask-form");
  const input = $("query");
  const source = $("source");
  const button = $("ask-button");
  const status = $("status");
  const result = $("result");
  const question = $("question");
  const answerBlock = $("answer-block");
  const verdict = $("verdict");
  const traceList = $("trace-list");
  const traceTotal = $("trace-total");
  const evidenceList = $("evidence-list");
  const evidenceCount = $("evidence-count");
  const evidenceDetails = $("evidence-details");

  const SOURCE_LABELS = {
    charaka: "Charaka",
    sushruta: "Sushruta",
    pubmed: "PubMed",
    patents: "Patents",
    legal: "Legal",
  };

  function el(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function fmtMs(value) {
    if (typeof value !== "number" || Number.isNaN(value)) return "";
    if (value >= 1000) return `${(value / 1000).toFixed(1)} s`;
    return `${Math.round(value)} ms`;
  }

  function pct(value) {
    if (typeof value !== "number") return "—";
    return `${Math.round(value * 100)}%`;
  }

  function sourceLabel(value) {
    return SOURCE_LABELS[value] || value || "—";
  }

  function addKv(container, rows) {
    const dl = el("dl", "kv");
    rows.forEach(([term, value]) => {
      dl.append(el("dt", "", term));
      const dd = el("dd");
      if (value instanceof Node) dd.append(value);
      else dd.textContent = value;
      dl.append(dd);
    });
    container.append(dl);
  }

  function addStep(number, title, timing) {
    const step = el("li", "step");
    const head = el("div", "step-head");
    head.append(el("span", "step-no", String(number).padStart(2, "0")));
    head.append(el("h3", "", title));
    head.append(el("span", "step-time", fmtMs(timing)));
    const body = el("div", "step-body");
    step.append(head, body);
    traceList.append(step);
    return body;
  }

  function setBusy(busy) {
    button.disabled = busy;
    button.textContent = busy ? "Working…" : "Ask";
  }

  function showError(message) {
    status.className = "status error";
    status.textContent = message;
    setBusy(false);
  }

  function appendAnswer(container, text, citations) {
    const byMarker = new Map((citations || []).map((citation) => [citation.marker, citation]));
    const parts = String(text || "").split(/(\[S\d+\])/g);
    parts.forEach((part) => {
      if (!part) return;
      const marker = byMarker.get(part);
      if (!marker) {
        container.append(document.createTextNode(part));
        return;
      }
      const span = el("span", "marker", part);
      const support = marker.supported === true ? "supported" : marker.supported === false ? "unsupported" : "not verified";
      span.title = `${marker.citation} — ${support}`;
      span.addEventListener("click", () => {
        const target = document.getElementById(`evidence-${part.replace(/[\[\]]/g, "")}`);
        if (target) {
          evidenceDetails.open = true;
          target.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      });
      container.append(span);
    });
  }

  function renderVerdict(data) {
    verdict.replaceChildren();
    if (data.insufficient_evidence) {
      verdict.append(el("span", "tag bad", "Insufficient evidence"));
    } else {
      verdict.append(el("span", "tag ok", "Evidence-grounded"));
    }

    const citations = data.citations || [];
    if (!citations.length) {
      verdict.append(el("span", "", "No citations emitted"));
      return;
    }
    const supported = citations.filter((citation) => citation.supported === true).length;
    const unsupported = citations.filter((citation) => citation.supported === false).length;
    const unverified = citations.length - supported - unsupported;
    verdict.append(el("span", "", `${supported}/${citations.length} citations supported`));
    if (unsupported) verdict.append(el("span", "unsupported", `${unsupported} unsupported`));
    if (unverified) verdict.append(el("span", "unverified", `${unverified} not checked`));
  }

  function renderTrace(data) {
    const trace = data.trace || {};
    const variants = trace.variants || [];
    const channels = trace.channels || [];
    const config = trace.config || {};
    const fusion = trace.fusion || {};
    const rerank = trace.rerank || {};
    const timings = trace.timings_ms || {};

    traceList.replaceChildren();
    traceTotal.textContent = timings.total ? `${fmtMs(timings.total)} end to end` : "";

    const plan = addStep(1, "Query plan");
    plan.append(el("p", "", variants.length === 1 ? "The question was retrieved as written." : `${variants.length} retrieval variants were issued.`));
    if ((trace.entities || []).length) {
      plan.append(el("p", "", `Lexicon entities: ${trace.entities.join(", ")}`));
    } else {
      plan.append(el("p", "", "No curated lexicon entity matched; retrieval relies on the question text."));
    }
    variants.forEach((variant) => {
      const row = el("div", "variant");
      row.append(el("span", "kind", variant.kind));
      row.append(document.createTextNode(" "));
      row.append(el("code", "code-inline", variant.text));
      row.append(document.createTextNode(` · weight ${Number(variant.weight).toFixed(2)}`));
      plan.append(row);
    });

    const retrieve = addStep(2, "Candidate retrieval", timings.retrieval);
    const bm25Total = channels.reduce((sum, channel) => sum + (channel.bm25 || 0), 0);
    const denseTotal = channels.reduce((sum, channel) => sum + (channel.dense || 0), 0);
    addKv(retrieve, [
      ["Channels", "BM25 lexical + dense semantic"],
      ["Candidates", `${bm25Total} BM25 · ${denseTotal} dense (before filtering)`],
      ["Fusion", `${fusion.method || config.fusion || "—"} · ${config.bm25_weight ?? "—"} lexical / ${config.dense_weight ?? "—"} semantic`],
      ["Candidate pool", `${fusion.pool_size ?? "—"} scored → ${fusion.candidate_count ?? "—"} hydrated`],
      ["Entity graph", fusion.graph_candidates ? `${fusion.graph_candidates} extra candidates` : "off"],
    ]);
    channels.forEach((channel) => {
      retrieve.append(
        el("div", "variant", `${channel.kind}: ${channel.bm25 || 0} BM25, ${channel.dense || 0} dense — ${channel.query}`),
      );
    });

    const rerankStep = addStep(
      3,
      "Reranking",
      (timings.search_total || timings.retrieval || 0) - (timings.retrieval || 0),
    );
    if (rerank.enabled) {
      addKv(rerankStep, [
        ["Model", rerank.model || "—"],
        ["Reranked", `${rerank.candidate_count || 0} candidates`],
        ["Score blend", `${pct(rerank.blend)} cross-encoder / ${pct(1 - rerank.blend)} first-stage`],
      ]);
    } else {
      rerankStep.append(el("p", "", "Cross-encoder reranking is disabled; first-stage fusion order is used."));
    }

    const evidence = data.evidence || [];
    const evidenceStep = addStep(4, "Evidence set");
    const bySource = {};
    evidence.forEach((item) => {
      bySource[item.source] = (bySource[item.source] || 0) + 1;
    });
    const sourceSummary = Object.entries(bySource)
      .map(([key, count]) => `${sourceLabel(key)} ×${count}`)
      .join(" · ") || "none";
    const topScore = evidence.length ? Number(evidence[0].score).toFixed(3) : "—";
    addKv(evidenceStep, [
      ["Passages", `${evidence.length} selected for generation`],
      ["Sources", sourceSummary],
      ["Top score", topScore],
    ]);

    const generation = trace.generation || {};
    const genStep = addStep(5, "Answer generation", timings.generation);
    addKv(genStep, [
      ["Model", generation.model || data.metadata?.model || "—"],
      ["Evidence supplied", `${generation.n_evidence ?? evidence.length} passages`],
      ["Constraint", "Only [S#] evidence; no outside knowledge"],
    ]);

    const verification = trace.verification || {};
    const verifyStep = addStep(6, "Citation verification", timings.verification);
    const citations = data.citations || [];
    const supported = citations.filter((citation) => citation.supported === true).length;
    const unsupported = citations.filter((citation) => citation.supported === false).length;
    addKv(verifyStep, [
      ["Verifier", verification.model || (verification.enabled ? "not run" : "off")],
      ["Citations", `${citations.length} emitted · ${supported} supported · ${unsupported} unsupported`],
      ["Insufficient evidence", data.insufficient_evidence ? "yes — answer withheld" : "no"],
    ]);
  }

  function renderEvidence(data) {
    const evidence = data.evidence || [];
    evidenceList.replaceChildren();
    evidenceCount.textContent = `(${evidence.length})`;

    if (!evidence.length) {
      evidenceList.append(el("p", "evidence-snippet", "No passages were retrieved."));
      return;
    }

    const citations = new Map((data.citations || []).map((citation) => [citation.marker, citation]));
    evidence.forEach((item, index) => {
      const marker = `S${index + 1}`;
      const citation = citations.get(`[${marker}]`);
      const article = el("article", "evidence");
      article.id = `evidence-${marker}`;

      const meta = el("div", "evidence-meta");
      meta.append(el("span", "evidence-rank", `#${item.rank || index + 1}`));
      meta.append(el("span", "evidence-citation", item.citation || item.chunk_id || "untitled"));
      meta.append(el("span", "evidence-score", `score ${Number(item.score || 0).toFixed(3)}`));
      meta.append(el("span", "tag", sourceLabel(item.source)));
      if (item.corpus) meta.append(el("span", "tag", item.corpus));
      if (item.retriever) meta.append(el("span", "tag", item.retriever));
      if (citation?.supported === true) meta.append(el("span", "tag ok", "supported"));
      if (citation?.supported === false) meta.append(el("span", "tag bad", "unsupported"));
      article.append(meta);

      const text = String(item.text || "");
      article.append(el("p", "evidence-snippet", text.length > 620 ? `${text.slice(0, 620)}…` : text));
      if (text.length > 620) {
        const details = el("details", "evidence-more");
        details.append(el("summary", "", "Full passage"));
        details.append(el("pre", "", text));
        article.append(details);
      }
      evidenceList.append(article);
    });
  }

  function render(data) {
    result.hidden = false;
    question.textContent = data.question || input.value;
    answerBlock.replaceChildren();

    if (data.insufficient_evidence) {
      answerBlock.append(el("p", "unknown-title", "I don’t know."));
      answerBlock.append(
        el(
          "p",
          "unknown-copy",
          data.display_message ||
            "The indexed documents do not contain enough evidence to answer this question.",
        ),
      );
      if (data.model_explanation && data.model_explanation !== data.answer) {
        const details = el("details", "evidence-more");
        details.append(el("summary", "", "What the model flagged as missing"));
        details.append(el("pre", "", data.model_explanation));
        answerBlock.append(details);
      }
    } else {
      const answerText = el("div", "answer-text");
      appendAnswer(answerText, data.answer, data.citations);
      answerBlock.append(answerText);
    }

    renderVerdict(data);
    renderTrace(data);
    renderEvidence(data);
    status.className = "status";
    status.textContent = "";
    setBusy(false);
  }

  async function ask(query) {
    setBusy(true);
    status.className = "status";
    status.textContent = "Retrieving evidence, generating a constrained answer, checking citations…";
    result.hidden = true;

    const payload = { query, top_k: 8, verify: true };
    if (source.value) payload.source = source.value;

    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || data.message || `Request failed (${response.status})`);
      }
      render(data);
    } catch (error) {
      showError(error instanceof Error ? error.message : "Request failed.");
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const query = input.value.trim();
    if (query.length < 2) {
      showError("Enter a question with at least two characters.");
      return;
    }
    ask(query);
  });

  document.querySelectorAll("[data-example]").forEach((example) => {
    example.addEventListener("click", () => {
      input.value = example.dataset.example;
      input.focus();
      ask(input.value);
    });
  });
})();
