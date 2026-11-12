# Evaluation record

This document records every benchmark run performed for AyurRAG, including the
benchmark construction, metric definitions, per-source breakdowns, failures, and
the caveats that bound the claims. It is the reference for the numbers in
`README.md` and `data/artifacts/eval_report.md`.

Last updated: **2026-11-12**  
System: **CPU-only**, no GPU, 31 GB RAM, local index; generator and judge are
remote OpenAI-compatible models.

## 1. Run ledger

| ID | Evaluation | Questions | Scoring | Status |
|---|---|---|---|---|
| E1 | Retrieval ablations (6 pipelines) | 180 | retrieval metrics | completed |
| E2 | Default retrieval after alias calibration | 180 | retrieval metrics | completed |
| E3 | End-to-end domain QA | 60 | LLM judge + NLI | completed |
| E4 | Refusal / abstention | 15 | `insufficient_evidence` or refusal cue | completed |
| E5 | External PubMedQA, oracle contexts | 48 | LLM judge | completed |
| V1 | Test verification | 13 tests | pass/fail | completed |

E1 is the full ablation sweep. E2 is a separate rerun of only the shipped
pipeline after the alias weight was calibrated from `0.15` to `0.30`; the
non-expansion E1 rows are unaffected, while `full-v2` uses expansion and is
reported from the current rerun. E3–E5 are the stored end-to-end runs. E5 is
**not** a retrieval benchmark: gold contexts are supplied to the generator, so
it measures grounded answering and citation discipline only.

## 2. Indexed corpus and models

### Corpus at evaluation time

| Source | Corpus | Chunks |
|---|---|---:|
| PubMed / Europe PMC | literature | 16,649 |
| Sushruta Samhita | classical | 1,871 |
| Charaka Samhita | classical | 1,737 |
| India Code / CBD / Nagoya | legal | 243 |
| Google Patents | patent | 68 |
| **Total** | | **20,568** |

### Pipeline settings relevant to the scores

| Component | Model / setting |
|---|---|
| Lexical retriever | BM25 (`bm25s`), top 60 |
| Dense retriever | `BAAI/bge-small-en-v1.5`, 384-d, top 60 |
| Fusion | score-normalized convex, 0.8 BM25 / 0.2 dense |
| First-stage candidates | up to 120 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 80, score blend 0.5 |
| Final context | top 8 chunks |
| Query expansion | enabled; alias weight 0.30, max 0 subqueries |
| Graph expansion | disabled by default |
| Generator | `deepseek-v4.1-flash-free`, temperature 0.1 |
| Judge | `glm-5.3-flash-free`, temperature 0.0 |
| Citation verifier | `cross-encoder/nli-deberta-v3-small` |

The generator and judge are served through the OpenAI-compatible endpoint
configured by `llm.base_url`; all prompts and scores are cached in
`data/artifacts/llm_cache.sqlite` and `data/artifacts/judge_cache.sqlite`.

The reported generation/judge scores below were collected with the earlier
proxy models `deepseek-v4.1-flash-free` and `glm-5.3-flash-free`. The current
default is `deepseek-flash` for both generation and judging; changing the model
is a new configuration and requires re-running the generation evaluation.
Using one model for both roles can inflate judge scores through self-preference,
so treat the new configuration's scores as directional.

## 3. Benchmarks

### 3.1 Domain retrieval benchmark — 180 questions

Built by `src/ayurrag/eval/dataset.py` from held-out chunks in the indexed
corpus:

- One query per sampled chunk; the chunk is the single gold passage.
- Sampling is stratified by source, seed `13`.
- Two automatically generated styles: plain-language natural question and
  terminology-heavy question. The generator is instructed not to copy a full
  sentence from the passage and to make the answer derivable from that passage.
- Queries are shuffled after generation; evaluation uses the cached
  `data/artifacts/retrieval_gold.jsonl`.

| Source | Questions | Natural | Terminology-heavy |
|---|---:|---:|---:|
| Charaka Samhita | 36 | 19 | 17 |
| Sushruta Samhita | 36 | 19 | 17 |
| PubMed | 38 | 16 | 22 |
| Patents | 34 | 17 | 17 |
| Legal | 36 | 22 | 14 |
| **Total** | **180** | **93** | **87** |

Mean query length is 130 characters (median 130). Every query has exactly one
gold chunk, so Recall@k is binary per query and nDCG has a single relevant item.

**Known bias:** because queries are generated from the gold passage, they retain
some of its vocabulary. This favours BM25 and under-represents natural paraphrase.
The benchmark is useful for source-stratified, reproducible retrieval ablations,
not as an estimate of open-web retrieval quality.

### 3.2 End-to-end domain QA — 60 questions

The first 60 cached retrieval questions, after the shuffle, are answered through
the full shipped pipeline (`ayurrag ask` equivalent): retrieval → optional
expansion → rerank → grounded generation → NLI citation verification → judge.

| Source | Questions | `insufficient_evidence` |
|---|---:|---:|
| Sushruta | 13 | 2 |
| Patents | 13 | 2 |
| Charaka | 12 | 2 |
| PubMed | 12 | 3 |
| Legal | 10 | 0 |
| **Total** | **60** | **9 (15%)** |

The judge was asked to score each answer against the gold passage and the
retrieved evidence. Five judge calls returned no parseable JSON; judge means
therefore use **55 scored answers**, while coverage/flag rates use all 60.
`insufficient_evidence` is the generator's self-flag, not proof that the raw text
is an abstention: in the evaluation the raw output is still scored, whereas the
API replaces a flagged answer with the plain “I don't know” response. Nine of the
60 raw outputs carry that flag. This is recorded rather than silently imputed, and
the split is reported in §5.2.

### 3.3 Refusal benchmark — 15 questions

Fifteen fixed unanswerable questions are run through the same retriever and
generator. They cover out-of-domain facts, anachronistic claims attributed to
classical texts, fabricated patents/legal amendments, and misleading medical
claims. The full list is in `src/ayurrag/eval/dataset.py` (`UNANSWERABLE`).

A response counts as refused if either the generator returns
`insufficient_evidence` or the answer contains an explicit refusal cue
(`cannot answer`, `no evidence`, `does not contain`, etc.).

### 3.4 External PubMedQA — 48 questions, oracle contexts

`benchmarks/pubmedqa_sample.jsonl` is a deterministic stratified sample of the
PubMedQA labelled test set:

- 48 questions: 16 `yes`, 16 `no`, 16 `maybe`.
- Sampling seed `17`; each question keeps up to six gold abstract paragraphs.
- Source: `qiaojin/PubMedQA`; license MIT.
- The gold contexts are passed directly to the generator as `[S#]` evidence.
  Retrieval is bypassed, so this measures out-of-domain grounded generation,
  citation discipline, and decision reporting — not domain retrieval.

## 4. Metrics

### Retrieval

For a query with gold set \(G\) and ranked list \(R\):

- **Recall@k** = \(|G \cap R_{1:k}| / |G|\).
- **MRR@k** = \(1 / \mathrm{rank}\) of the first gold item in the top \(k\), or 0.
- **nDCG@k** = \(DCG@k / IDCG@k\), with one unit of gain per relevant item.

The harness reports `n`, Recall/MRR/nDCG at 5, 10, and 20, plus Recall@10 by
source. Scores are rounded to four decimals in JSON and three in the report.

### Answer quality

The judge receives the question, gold passage, retrieved evidence, and answer,
and returns integer scores 1–5 plus a harmful-overreach flag:

| Field | Definition |
|---|---|
| correctness | factual agreement with the gold passage/evidence; contradictions penalised |
| groundedness | every substantive claim is supported by the supplied evidence; outside knowledge penalised |
| completeness | coverage of the facts the question asks for |
| citation_quality | cited evidence markers actually support the sentences they annotate |
| harmful_overreach | medical advice, definitive legal conclusion, or patentability/infringement verdict |

Additional automatic metrics:

- **citation_coverage** = answer sentences (>30 chars) containing at least one
  `[S#]` marker / all such sentences.
- **citation_precision_nli** = fraction of emitted citations whose cited passage
  entails the sentence under the NLI model. This is strict and sentence-level;
  it is not a human judgement.
- **insufficient_evidence_rate** = fraction of answers flagged as unanswerable.
- **harmful_overreach_rate** = fraction of judged answers flagged by the judge.

### External decision reporting

`decision_mentioned` checks whether the PubMedQA label (`yes`, `no`, `maybe`)
appears as a word in the answer. The original implementation used a substring
check; it was corrected to token-boundary matching on 2026-11-12 because `no`
was matching `not`, `north`, etc. The corrected rate is reported below. The
answers themselves and the judge scores are unchanged.

## 5. Results

### 5.1 Retrieval ablations (E1, 180 questions)

| Pipeline | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.944 | 0.967 | 0.967 | 0.860 | 0.886 |
| Dense (`bge-small-en-v1.5`) | 0.811 | 0.839 | 0.878 | 0.723 | 0.751 |
| Hybrid, equal-weight RRF | 0.872 | 0.933 | 0.961 | 0.800 | 0.832 |
| Hybrid, convex 0.8/0.2 | 0.950 | 0.961 | 0.967 | 0.874 | 0.895 |
| Convex + cross-encoder rerank | 0.961 | 0.972 | 0.978 | 0.890 | 0.910 |
| Subquery decomposition + entity graph | 0.961 | 0.972 | 0.978 | 0.888 | 0.909 |
| **Default after alias calibration (E2)** | **0.961** | **0.972** | **0.978** | **0.888** | **0.909** |

Interpretation:

- BM25 is strong because the benchmark is lexical by construction. Dense-only is
  materially worse (R@10 0.839), so the hybrid is justified even on this biased set.
- Convex fusion beats equal-weight RRF on every reported metric. The cross-encoder
  reranker adds recall@10/20 and MRR/nDCG@10 while also reaching R@5 0.961.
- Alias expansion keeps R@5/10/20 at the rerank level; it slightly lowers
  MRR/nDCG@10 versus pure rerank, consistent with adding candidates.
- Subquery decomposition + graph expansion now matches the default on R@5–R@20
  but is marginally worse on MRR/nDCG and costs extra retrieval work. It remains
  opt-in (`graph_expansion: true`, `max_subqueries: 3`) because it has no material
  end-to-end benefit.

#### Shipped pipeline after alias calibration (E2)

Full precision table for the current default configuration:

| Metric | @5 | @10 | @20 |
|---|---:|---:|---:|
| Recall | 0.9611 | 0.9722 | 0.9778 |
| MRR | 0.8869 | 0.8882 | 0.8885 |
| nDCG | 0.9056 | 0.9089 | 0.9103 |

Recall@10 by source:

| Source | Recall@10 |
|---|---:|
| Legal | 1.0000 |
| PubMed | 1.0000 |
| Charaka | 0.9722 |
| Patents | 0.9706 |
| Sushruta | 0.9167 |

Sushruta is the weakest source. Its passages have more OCR variation and
transliteration variants, which is the main motivation for the alias lexicon and
for keeping a semantic channel.

### 5.2 End-to-end domain QA (E3, 60 questions, 55 judged)

| Metric | Score |
|---|---:|
| Correctness | 4.582 / 5 |
| Groundedness | 3.673 / 5 |
| Completeness | 4.491 / 5 |
| Citation quality (judge) | 3.418 / 5 |
| Harmful overreach rate | 0.000 |
| `insufficient_evidence` rate | 0.150 |
| Citation coverage | 0.8896 |
| Citation precision (NLI, strict) | 0.4681 |
| Uncited sentences | 33 total across 60 answers |

Because the raw generation is scored even when the generator self-flags, the
all-60 mean is not the same as the quality of answers a user would see. The
retrospective split is:

| Subset | n | Judged | Correctness | Groundedness | Completeness | Citation quality | Coverage | NLI precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All raw outputs | 60 | 55 | 4.582 | 3.673 | 4.491 | 3.418 | 0.890 | 0.468 |
| Not flagged | 51 | 47 | 4.660 | 3.830 | 4.596 | 3.553 | 0.892 | 0.495 |
| Flagged | 9 | 8 | 4.125 | 2.750 | 3.875 | 2.625 | 0.874 | 0.313 |

The flagged subset is weaker but is not empty boilerplate; the API would withhold
it. The displayed-answer means are therefore the “not flagged” row, while the
headline table follows the stored all-60 report.

Judge score distributions (55 judged answers):

| Score | Correctness | Groundedness | Completeness | Citation quality |
|---:|---:|---:|---:|---:|
| 1 | 2 | 1 | 2 | 1 |
| 2 | 0 | 12 | 2 | 21 |
| 3 | 1 | 12 | 1 | 8 |
| 4 | 13 | 9 | 12 | 4 |
| 5 | 39 | 21 | 38 | 21 |

Per-source breakdown:

| Source | n | Judged | Insufficient | Correctness | Groundedness | Completeness | Citation quality | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Charaka | 12 | 9 | 2 | 4.00 | 3.22 | 3.89 | 3.11 | 0.817 |
| Sushruta | 13 | 11 | 2 | 4.45 | 3.27 | 4.55 | 3.09 | 0.892 |
| PubMed | 12 | 12 | 3 | 4.75 | 3.92 | 4.50 | 3.83 | 0.965 |
| Patents | 13 | 13 | 2 | 4.85 | 3.62 | 4.77 | 3.15 | 0.856 |
| Legal | 10 | 10 | 0 | 4.70 | 4.30 | 4.60 | 3.90 | 0.927 |

Reading the result:

- Correctness and completeness are high; the answer usually contains the
  requested facts. Groundedness is lower because the judge penalises claims that
  the retrieved evidence does not explicitly support, even when they are true.
- Citation quality is the weak point (3.42/5; 21 answers scored 2). The generator
  often attaches a topically relevant but not entailing passage. This is exactly
  why the NLI verifier exists; the API exposes `supported: false` instead of
  hiding it.
- NLI precision (0.468) is lower than judge citation quality (3.42). The NLI model
  is a small CPU cross-encoder and operates on long abstractive sentences; it is
  deliberately conservative and under-credits valid but paraphrased citations.
  Do not present 0.468 as a human-verified precision.
- Nine raw outputs were flagged `insufficient_evidence` (15%), spread across all
  sources except legal. These are not all literal abstentions: some contain an
  answer plus the flag, and the API would replace them with “I don't know”. The
  split above shows the flagged subset scoring lower on every judge dimension.
  This is a false-refusal risk, not a success metric.

### 5.3 Refusal (E4, 15 questions)

**Refusal rate: 1.000 (15/15).** All 15 were flagged by `insufficient_evidence`;
none was a bare unrelated answer.

The set covers:

1. Out-of-domain facts: capital of France, 2022 FIFA World Cup, mercury boiling
   point, first iPhone manufacturer, Jupiter's moons, India's 2024 GDP, Rigveda
   verse count.
2. Anachronistic/unsupported attributions: Charaka on X-ray crystallography,
   Sushruta on RNA interference, Charaka on COVID-19 vaccine side effects.
3. Fabricated legal/patent claims: exact 2025 Indian patent royalty amendment,
   turmeric-based cancer drug patent number, 2024 WTO Neem patent ruling.
4. Misleading medical claims: a trial proving ashwagandha cures Alzheimer's, an
   FDA-approved Ayurvedic herb for diabetes.

This is an adversarial smoke test, not a calibrated abstention benchmark. It shows
that the refusal contract works for obvious OOD queries; it does not measure the
false-refusal rate on hard but answerable questions. That false-refusal risk is
visible in E3 (9/60 abstentions), and one known case was a terminology mismatch
that was fixed by alias calibration.

### 5.4 External PubMedQA (E5, 48 questions, oracle contexts, 47 judged)

| Metric | Score |
|---|---:|
| Correctness | 4.298 / 5 |
| Groundedness | 4.957 / 5 |
| Completeness | 3.894 / 5 |
| Citation quality | 4.809 / 5 |
| Harmful overreach rate | 0.000 |
| Decision mentioned (corrected, word-boundary) | 0.3125 |

Per-label breakdown:

| Label | n | Judged | Correctness | Groundedness | Completeness | Citation quality | Decision mentioned |
|---|---:|---:|---:|---:|---:|---:|---:|
| yes | 16 | 16 | 4.188 | 5.000 | 3.938 | 4.938 | 0.188 |
| no | 16 | 15 | 4.333 | 4.933 | 3.933 | 4.800 | 0.750 |
| maybe | 16 | 16 | 4.375 | 4.938 | 3.812 | 4.688 | 0.000 |

Interpretation:

- Groundedness and citation quality are near ceiling with gold contexts; citation
  discipline generalizes to biomedical abstracts.
- Completeness is lower because gold PubMedQA long answers often include study
  design, sample sizes, and limitations that the concise generator omits.
- The answer is encouraged to explain evidence, not to emit a bare label. The low
  decision-mention rate is therefore expected, especially for `maybe`; the earlier
  reported value (0.396) was inflated by a substring bug (`no` matched inside words
  such as `not`). The corrected rate is 0.3125.
- One `no` question had no parseable judge JSON, so the judge means use 47 answers.

## 6. Verification

- `pytest`: 13 tests passed after adding the token-boundary regression test for
  `decision_mentioned`.
- `ruff check src tests`: clean.
- Browser smoke checks: grounded answer with `[S#]` citations, `I don't know`
  refusal state, and source-filtered retrieval were all exercised against the live
  API.
- The HTML/JS UI is served from `/`; `GET /health`, `GET /search`, and `POST /ask`
  are covered by the API contract tests or live checks.

These are correctness checks, not additional quality benchmarks.

## 7. Caveats and threats to validity

1. **Synthetic retrieval queries are lexical.** They are generated from the gold
   passage, so BM25 may look better than it would on real paraphrased user queries.
   The dense-only row is the main evidence that the semantic channel adds value;
   a human-written paraphrase set is still needed.
2. **One judge, one run.** `glm-5.3-flash-free` is a single LLM judge, not an
   ensemble or human panel. Scores are directional, not ground truth. Five domain
   judge calls and one external judge call failed to parse and were excluded from
   the corresponding means.
3. **E3 is a stored run from before the final UI/alias/prompt polish.** E2 is the
   current retrieval default; E3–E5 were not re-run after every presentation-only
   or prompt-clarification change. Treat E2 as the current retrieval number, and
   E3–E5 as the stored end-to-end run. Re-run `ayurrag eval --kind all --limit 60`
   before claiming a new end-to-end SOTA point.
4. **NLI precision is strict.** It under-credits correct abstractive citations;
   the judge's citation quality is a separate, more forgiving signal. Neither is a
   human audit.
5. **External PubMedQA is oracle-context.** It does not test retrieval and is not
   Ayurveda-specific. It should not be compared directly to domain retrieval scores.
6. **Refusal is tested on obvious OOD questions.** Fifteen hand-written adversarial
   questions do not estimate false-refusal rate on hard answerable queries.
7. **No multilingual benchmark.** BhashaBench-Ayur is gated and was not bundled.
   The multilingual embedding is configured but not evaluated here.
8. **No formal latency evaluation.** Timings exist in the API trace, but they are
   machine- and model-proxy-dependent and are not reported as a benchmark.
9. **Decision-mention metric was corrected after the original run.** The corrected
   value uses the same stored answers and judge scores; only the matching logic
   changed.

## 8. Reproduction

```bash
uv sync
export AYURRAG_LLM_API_KEY=...        # or put it in .env (auto-loaded)
export AYURRAG_FALLBACK_API_KEY=...   # optional secondary provider

# retrieval ablations, 180 questions
PYTHONPATH=src .venv/bin/ayurrag eval --kind retrieval --limit 180 --ablations

# current default retrieval only
PYTHONPATH=src .venv/bin/ayurrag eval --kind retrieval --limit 180

# end-to-end domain QA + refusal
PYTHONPATH=src .venv/bin/ayurrag eval --kind generation --limit 60

# external PubMedQA oracle-context check
PYTHONPATH=src .venv/bin/ayurrag eval --kind external --limit 48
```

The cached benchmark is `data/artifacts/retrieval_gold.jsonl`; delete it only if
you intend to regenerate queries (this changes the benchmark). Scores are written
to `data/artifacts/eval_report.{json,md}`. The LLM and judge caches make reruns
cheap but mean a changed prompt will miss the cache and produce new scores.

## 9. Artifact index

| Path | Contents |
|---|---|
| `data/artifacts/eval_report.json` | full summaries and per-question records |
| `data/artifacts/eval_report.md` | compact human-readable report |
| `data/artifacts/retrieval_gold.jsonl` | 180 cached retrieval questions |
| `data/artifacts/llm_cache.sqlite` | generator/query-generation cache |
| `data/artifacts/judge_cache.sqlite` | judge-answer cache |
| `benchmarks/pubmedqa_sample.jsonl` | 48-question PubMedQA fixture |
| `benchmarks/README.md` | external fixture provenance |
| `docs/evaluation.md` | this record |
