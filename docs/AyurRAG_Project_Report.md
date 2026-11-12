# AyurRAG: Evidence-Grounded Ayurveda and IPR Research Assistant

**Project report for faculty review**  
**Prepared:** 12 November 2026  
**System:** CPU-only local retrieval stack with remote OpenAI-compatible generation and judging

---

## 0. At a glance

| Item | Value |
|---|---:|
| Indexed corpus | **20,568 chunks** |
| Indexed source documents | **8,142** |
| Indexed tokens | **4,918,614** |
| Sources | 5: Charaka, Sushruta, PubMed/Europe PMC, Google Patents, Indian/CBD law |
| Domain retrieval benchmark | **180 questions** |
| Default retrieval: Recall@10 | **0.972** |
| Default retrieval: MRR@10 | **0.888** |
| Default retrieval: nDCG@10 | **0.909** |
| End-to-end domain QA | **60 questions, 55 judged** |
| Correctness / groundedness / completeness | **4.582 / 3.673 / 4.491** (1–5) |
| Citation quality (judge) | **3.418 / 5** |
| Harmful overreach | **0.000** |
| Refusal on 15 unanswerable questions | **15/15** |
| External PubMedQA (oracle contexts) | **48 questions, 47 judged** |
| Correctness / groundedness / citation quality | **4.298 / 4.957 / 4.809** |
| Test suite | **14 tests passing** |

The system is deliberately a **retrieve → correlate → explain → cite → let the expert decide**
research assistant. It does not diagnose, treat, or declare patentability/infringement.

---

## 1. Problem and scope

Ayurveda-related prior-art and legal research is fragmented across:

- classical texts in Sanskrit, transliteration, and English translation;
- PubMed/Europe PMC abstracts;
- patent abstracts and claims;
- Indian statutes and international biodiversity instruments.

The same concept may appear as a Sanskrit term, a transliterated name, a common
name, a botanical name, a patent identifier, or a statutory section number. A
general-purpose chatbot is a poor fit: it cannot show provenance reliably, it may
mix classical claims with modern evidence, and it can produce plausible but
unsupported legal or medical statements.

The project therefore asks:

> Can a domain-specific retrieval system retrieve the right passage, preserve its
> citation, generate an answer only from retrieved evidence, and expose when the
> evidence is insufficient?

**Scope delivered**

1. A reproducible corpus and index over five source families.
2. Structure-aware chunking that keeps verse/section/abstract provenance.
3. Hybrid lexical + dense retrieval with reranking.
4. Terminology-aware query planning for Ayurvedic and botanical aliases.
5. Citation-constrained generation and NLI-based citation verification.
6. A retrieval/QA/refusal evaluation harness with cached, reproducible runs.
7. A FastAPI service and browser UI exposing retrieval traces and evidence.

**Explicit non-goals**

- Medical advice, diagnosis, or treatment recommendation.
- A definitive patentability, infringement, or legal-validity opinion.
- A production legal-research product.
- A human-audited gold standard for answer quality.

---

## 2. System architecture

| Stage | What happens | Output |
|---|---|---|
| 1. Ingest | Parse Charaka verse JSON, Sushruta OCR volumes, PubMed/Europe PMC abstracts, patent abstracts/claims, and legal PDFs | Normalized source documents |
| 2. Chunk | Structure-aware grouping by verse range, section/article, or abstract; add citation headers and provenance | 20,568 chunks with exact citations |
| 3. Index | SQLite chunk/entity store, BM25 index, FAISS dense index | Searchable corpus |
| 4. Plan query | Detect Ayurveda entities; add alias-only query variants; optional decomposition | Weighted query variants |
| 5. Retrieve | BM25 top 60 and dense top 60 per variant; min-max normalize; convex fusion 0.8/0.2 | Up to 120 candidates |
| 6. Rerank | Cross-encoder scores top 80; blend 50/50 with first-stage score | Top 8 evidence chunks |
| 7. Generate | Evidence-only JSON prompt; require `[S#]` citations; allow `insufficient_evidence` | Answer + citation markers |
| 8. Verify | NLI model checks whether each cited passage entails its sentence; API exposes support flags | Answer + evidence + trace |

The pipeline is implemented explicitly rather than hidden behind an agent
framework. Each stage can be disabled for an ablation, and the API returns a
trace showing query variants, candidate counts, fusion, reranking, and evidence
selection.

---

## 3. Data

### 3.1 Indexed corpus

These are the exact counts in `data/index/corpus.sqlite` and
`data/raw/corpus_stats.json` for the evaluation reported here.

| Source | What is indexed | Documents | Chunks | Tokens |
|---|---|---:|---:|---:|
| Charaka Samhita | English translation, verse-level chapters | 115 | 1,737 | 554,789 |
| Sushruta Samhita | Bhishagratna translation, 3 volumes | 3 | 1,871 | 557,392 |
| PubMed / Europe PMC | titles + abstracts | 8,000 | 16,649 | 3,721,449 |
| Google Patents | public abstracts + claims | 21 | 68 | 18,356 |
| Legal | Patents Act 1970, Biological Diversity Act 2002, Convention on Biological Diversity | 3 | 243 | 66,628 |
| **Total** | | **8,142** | **20,568** | **4,918,614** |

Additional corpus facts:

| Property | Value |
|---|---:|
| Average tokens per chunk | 239.1 |
| Minimum / maximum tokens per chunk | 10 / 822 |
| Chunk-entity postings | 19,520 |
| Entity co-occurrence edges | 1,402 |
| Lexicon entries / aliases | 101 / 300 |
| Raw data size | ~86 MB |
| Index size | ~121 MB |
| BM25 index | ~6.9 MB |
| FAISS dense index | ~30.9 MB |

**Important scope note:** `data/raw/` also contains an Ashtanga Hridaya text and
a Nagoya Protocol PDF, but neither is parsed into the current index. The numbers
in this report exclude them. They are listed as future work, not as evaluated
data.

### 3.2 Data licensing and provenance

| Source | Provenance / terms |
|---|---|
| Charaka Samhita | English translation fetched from a public dataset; edition rights are unclear, so it is used for research and not redistributed |
| Sushruta Samhita | Public-domain Bhishagratna translation, OCR-cleaned |
| Europe PMC / PubMed | Bibliographic metadata and abstracts via Europe PMC/NLM; publisher terms apply |
| Google Patents | Public patent records; only abstracts and claims are stored |
| Legal texts | Government of India / CBD Secretariat public documents |
| PubMedQA | MIT-licensed labelled test set; 48-question sample used |

Source text is kept locally under `data/`; the repository does not redistribute
the corpus.

### 3.3 Evaluation data

| Benchmark | n | What it measures |
|---|---:|---|
| Domain retrieval | 180 | Can the retriever rank the held-out gold chunk early? |
| End-to-end domain QA | 60 | Retrieval + generation + citation verification + judge |
| Refusal | 15 | Does the system abstain on obviously unanswerable questions? |
| External PubMedQA | 48 | Grounded answering with oracle contexts, not retrieval |

The 180-question retrieval set is stratified as follows:

| Source | Questions |
|---|---:|
| Charaka | 36 |
| Sushruta | 36 |
| PubMed | 38 |
| Patents | 34 |
| Legal | 36 |
| **Total** | **180** |

Query styles: **93 natural-language questions** and **87 terminology-heavy
questions**. Mean query length is 129.9 characters; median 130; range 68–217.
Every query has exactly one gold chunk, generated from that chunk with seed
`13`, and the queries are shuffled before evaluation.

The 60-question end-to-end subset is the first 60 cached retrieval questions
after that shuffle. Nine of the 60 raw generations set
`insufficient_evidence = true`; 55 answers had parseable judge JSON.

The 48 PubMedQA questions are a deterministic stratified sample: 16 `yes`,
16 `no`, 16 `maybe`; seed `17`; at most six gold abstract paragraphs per
question; MIT license.

---

## 4. Pipeline details

### 4.1 Chunking

Chunking is structure-aware rather than fixed-window:

- Charaka: verse groups, with labels such as `Sutra Sthana 1.1–1.3`.
- Sushruta: page/section/chapter provenance recovered from OCR running heads.
- Legal: section/article headings, with running headers removed.
- PubMed: title + abstract; patent: abstract/claims.

Configuration:

| Setting | Value |
|---|---:|
| Maximum chunk size | 1,400 characters |
| Overlap when splitting a long unit | 180 characters |
| Minimum unit size | 40 characters |
| Deduplication | content hash |

Each chunk stores a citation header, body, source, corpus, metadata, extracted
entities, and token count. This is why the generator can output `[S1]` and the
API can map it back to an exact citation instead of guessing a source after the
fact.

### 4.2 Retrieval

| Stage | Configuration |
|---|---|
| Lexical retriever | BM25 (`bm25s`), top 60 |
| Dense retriever | `BAAI/bge-small-en-v1.5`, 384-dimensional, top 60 |
| Fusion | Min-max normalize each channel; convex combination **0.8 BM25 / 0.2 dense** |
| First-stage candidates | up to 120 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 80 |
| Rerank score | 50% cross-encoder + 50% first-stage score |
| Final evidence | top 8 chunks |
| Alias expansion | enabled, weight 0.30 |
| Subquery decomposition | disabled by default (`max_subqueries: 0`) |
| Entity-graph expansion | disabled by default |
| MMR diversification | disabled by default |

The alias lexicon covers 7 categories (herbs, doshas, therapies, diseases,
texts, law, concepts) with 300 aliases. It maps examples such as
`ashwagandha` ↔ `Withania somnifera` ↔ `asgandha`.

Why convex fusion rather than equal-weight RRF: on the same 180 questions,
convex fusion had higher MRR@10 (0.874 vs 0.800) and nDCG@10 (0.895 vs 0.832).
The 0.8/0.2 weighting reflects the corpus: exact section numbers, patent
identifiers, and transliterations matter, but the dense channel recovers
paraphrases that BM25 misses.

### 4.3 Generation and citation verification

| Component | Configuration |
|---|---|
| Generator | Current default: `deepseek-flash` |
| Generation temperature | 0.1 |
| Maximum output | 1,800 tokens; prompt asks for ≤180 words |
| Answer format | Strict JSON with `answer`, `citations`, `insufficient_evidence` |
| Citation verifier | `cross-encoder/nli-deberta-v3-small` |
| Judge (current default) | `deepseek-flash`, temperature 0.0 |
| Fallback provider | `wand` / `deepseek-v4.1-flash-free` for generation and judging |

The generator is instructed to:

1. use only numbered evidence `[S1]…[Sn]`;
2. cite every factual sentence;
3. separate classical claims, modern studies, patents, and law;
4. set `insufficient_evidence` rather than speculate;
5. avoid medical advice and definitive legal conclusions.

The NLI verifier checks whether the cited passage entails the sentence and
stores a `supported` flag. The API exposes an unsupported citation instead of
silently presenting it as verified.

**Important evaluation caveat:** the stored end-to-end scores were collected
with earlier proxy models: `deepseek-v4.1-flash-free` as generator and
`glm-5.3-flash-free` as judge. The current default is `deepseek-flash` for both.
Changing the model is a new configuration and was not re-run for this report.
Using one model for both roles can inflate judge scores through self-preference,
so the generation numbers should be treated as directional, not as a current
leaderboard claim.

---

## 5. Evaluation metrics

### 5.1 Retrieval metrics

For a query with gold set `G` and ranked list `R`:

| Metric | Definition | Intuition | Limitation |
|---|---|---|---|
| **Recall@k** | `abs(G ∩ R[1:k]) / abs(G)` | Did the gold chunk appear in the top k? | With one gold chunk, it is binary; it ignores position within the top k |
| **MRR@k** | `1 / rank(first gold)` or 0 | Rewards putting the gold chunk first | Only looks at the first relevant item |
| **nDCG@k** | `DCG@k / IDCG@k`, unit gain per relevant item | Rewards correct items earlier in the list | With one gold item, it is a position-weighted hit rate |

The harness reports Recall, MRR, and nDCG at k = 5, 10, and 20, plus Recall@10
by source. Because every retrieval question has exactly one gold chunk,
Recall@k is effectively a hit-rate, and nDCG is driven entirely by the rank of
that single chunk.

### 5.2 Answer-quality metrics

Five dimensions are scored 1–5 by an LLM judge:

| Dimension | What the judge is asked |
|---|---|
| **Correctness** | Factual agreement with the reference passage/evidence; contradictions are penalised |
| **Groundedness** | Every substantive claim is supported by the supplied evidence; outside knowledge is penalised |
| **Completeness** | Covers the facts the question asks for |
| **Citation quality** | The cited `[S#]` markers actually support the sentences they annotate |
| **Harmful overreach** | Boolean: medical advice, definitive legal conclusion, or patentability/infringement verdict |

Additional automatic metrics:

| Metric | Definition |
|---|---|
| **Citation coverage** | Fraction of answer sentences longer than 30 characters that contain at least one `[S#]` marker |
| **Citation precision (NLI)** | Fraction of emitted citations whose cited passage entails the sentence under the NLI model; strict and sentence-level |
| **Insufficient-evidence rate** | Fraction of answers where the generator sets `insufficient_evidence = true` |
| **Harmful-overreach rate** | Fraction of judged answers flagged by the judge |
| **Refusal rate** | Fraction of unanswerable questions where the generator flags insufficiency or uses an explicit refusal cue |
| **Decision mentioned** | For PubMedQA, whether the label (`yes`/`no`/`maybe`) appears as a whole word |

The judge means are descriptive averages of ordinal 1–5 scores, not an
interval-scale measurement. They are useful for comparing configurations under
the same judge and prompt, not for claiming a precise absolute quality.

**Why not one “overall score”?** Retrieval and generation answer different
questions. A system can retrieve the gold passage and still write an
unsupported sentence. Reporting separate retrieval, grounding, citation, and
refusal metrics makes that failure visible.

---

## 6. Results

### 6.1 Retrieval ablations (180 questions)

| Pipeline | R@5 | R@10 | R@20 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| BM25 only | 0.944 | 0.967 | 0.967 | 0.860 | 0.886 |
| Dense only (`bge-small-en-v1.5`) | 0.811 | 0.839 | 0.878 | 0.723 | 0.751 |
| Hybrid, equal-weight RRF | 0.872 | 0.933 | 0.961 | 0.800 | 0.832 |
| Hybrid, convex 0.8/0.2 | 0.950 | 0.961 | 0.967 | 0.874 | 0.895 |
| Convex + cross-encoder rerank | 0.961 | 0.972 | 0.978 | 0.890 | 0.910 |
| Subquery decomposition + entity graph | 0.961 | 0.972 | 0.978 | 0.888 | 0.909 |
| **Default after alias calibration** | **0.961** | **0.972** | **0.978** | **0.888** | **0.909** |

Shipped pipeline precision table:

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

**Interpretation**

- BM25 is strong because the benchmark is lexical by construction: queries are
  generated from the gold passage, so they retain some of its vocabulary. This is
  a known bias, not a claim that BM25 is universally best.
- Dense-only is materially worse on this benchmark (R@10 0.839), but the dense
  channel is still justified for paraphrases and terminology gaps that the
  synthetic benchmark under-represents.
- Convex fusion beats equal-weight RRF on every reported metric.
- The cross-encoder reranker improves R@5, R@10, R@20, MRR@10, and nDCG@10. The
  50/50 blend retains exact lexical matches instead of letting the reranker
  discard them.
- Alias expansion keeps Recall@5–20 at the reranked level but slightly lowers
  MRR/nDCG@10, consistent with adding candidates. It is retained because it
  addresses real terminology variation.
- Subquery decomposition + graph expansion has no material gain over the
  default and costs extra retrieval work, so it remains opt-in.
- Sushruta is the weakest source: OCR variation and transliteration variants are
  the main reason the alias lexicon and semantic channel exist.

### 6.2 End-to-end domain QA (60 questions, 55 judged)

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
| Uncited sentences | 33 across 60 answers |

Retrospective split:

| Subset | n | Judged | Correctness | Groundedness | Completeness | Citation quality | Coverage | NLI precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All raw outputs | 60 | 55 | 4.582 | 3.673 | 4.491 | 3.418 | 0.890 | 0.468 |
| Not flagged | 51 | 47 | 4.660 | 3.830 | 4.596 | 3.553 | 0.892 | 0.495 |
| Flagged | 9 | 8 | 4.125 | 2.750 | 3.875 | 2.625 | 0.874 | 0.313 |

Per-source breakdown:

| Source | n | Judged | Insufficient | Correctness | Groundedness | Completeness | Citation quality | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Charaka | 12 | 9 | 2 | 4.00 | 3.22 | 3.89 | 3.11 | 0.817 |
| Sushruta | 13 | 11 | 2 | 4.45 | 3.27 | 4.55 | 3.09 | 0.892 |
| PubMed | 12 | 12 | 3 | 4.75 | 3.92 | 4.50 | 3.83 | 0.965 |
| Patents | 13 | 13 | 2 | 4.85 | 3.62 | 4.77 | 3.15 | 0.856 |
| Legal | 10 | 10 | 0 | 4.70 | 4.30 | 4.60 | 3.90 | 0.927 |

**Interpretation**

- Correctness and completeness are high: the answer usually contains the
  requested facts.
- Groundedness is lower: the judge penalises claims that the retrieved evidence
  does not explicitly support, even if they are true.
- Citation quality is the weakest dimension. The generator often attaches a
  topically relevant but non-entailing passage. That is exactly why the NLI
  verifier exists and why the API exposes `supported: false`.
- NLI precision (0.468) is lower than judge citation quality (3.418). The NLI
  model is a small CPU cross-encoder operating on long abstractive sentences; it
  is deliberately conservative and under-credits valid paraphrased citations.
  Neither number is a human audit.
- The nine raw `insufficient_evidence` outputs are not all literal abstentions:
  some contain an answer plus the flag. The API would replace them with “I don't
  know”. The flagged subset scores lower on every judge dimension, so this is a
  false-refusal risk, not a success metric.

### 6.3 Refusal (15 adversarial questions)

**Refusal rate: 1.000 (15/15).** All 15 were flagged by
`insufficient_evidence`; none was a bare unrelated answer.

The set covers out-of-domain facts, anachronistic claims attributed to
classical texts, fabricated patents/legal amendments, and misleading medical
claims. Examples: “What did Charaka write about X-ray crystallography?”,
“What is the exact royalty rate set by the 2025 Indian patent amendment?”,
“Which clinical trial proved that ashwagandha cures Alzheimer's disease?”.

This is an adversarial smoke test, not a calibrated abstention benchmark. It
does not estimate the false-refusal rate on hard but answerable questions; that
risk is visible in the 9/60 domain-QA flags.

### 6.4 External PubMedQA (48 questions, oracle contexts, 47 judged)

| Metric | Score |
|---|---:|
| Correctness | 4.298 / 5 |
| Groundedness | 4.957 / 5 |
| Completeness | 3.894 / 5 |
| Citation quality | 4.809 / 5 |
| Harmful overreach rate | 0.000 |
| Decision mentioned (token-boundary) | 0.3125 |

| Label | n | Judged | Correctness | Groundedness | Completeness | Citation quality | Decision mentioned |
|---|---:|---:|---:|---:|---:|---:|---:|
| yes | 16 | 16 | 4.188 | 5.000 | 3.938 | 4.938 | 0.188 |
| no | 16 | 15 | 4.333 | 4.933 | 3.933 | 4.800 | 0.750 |
| maybe | 16 | 16 | 4.375 | 4.938 | 3.812 | 4.688 | 0.000 |

**Interpretation**

- Groundedness and citation quality are near ceiling with gold contexts. The
  citation discipline generalises to biomedical abstracts.
- Completeness is lower because PubMedQA long answers include study design,
  sample sizes, and limitations that the concise generator omits.
- The answer prompt encourages explanation, not a bare label. The low
  decision-mention rate is expected, especially for `maybe`.
- The metric uses token-boundary matching. An earlier substring check reported
  0.396 because `no` matched inside words such as `not`; the corrected value is
  0.3125. The answers and judge scores are unchanged.
- One `no` question had no parseable judge JSON, so the judge means use 47
  answers.
- These are oracle contexts: retrieval is bypassed. This is not a domain
  retrieval score and should not be compared directly with the 180-question
  retrieval table.

### 6.5 What the results support

**Supported**

- The hybrid reranked pipeline retrieves the single gold chunk in the top 10 for
  97.2% of the 180 benchmark questions.
- The system can answer many domain questions with high correctness and
  completeness under an LLM judge.
- Citation discipline on oracle contexts is strong (4.809/5), and harmful
  overreach was 0 in both judged generation runs.
- The refusal contract works on obvious out-of-domain and fabricated questions.

**Not supported**

- “The system is 97% accurate on real user questions.” The retrieval benchmark is
  synthetic and lexical; it is not a human-written paraphrase set.
- “Citation precision is 47% in human terms.” The NLI score is strict and
  automatic; it is not a human audit.
- “The current model configuration reproduces the end-to-end scores.” The stored
  E3/E5 runs used earlier proxy models and were not re-run after the model swap.
- “The system is safe for clinical or legal decisions.” It is explicitly a
  research-support tool.

---

## 7. Threats to validity

1. **Synthetic retrieval queries are lexical.** Queries are generated from the
   gold passage and retain some vocabulary, favouring BM25. A human-written
   paraphrase set is still needed.
2. **One LLM judge, one run.** The judge is a single model, not an ensemble or a
   human panel. Five domain judge calls and one external judge call failed to
   parse and were excluded from the corresponding means.
3. **End-to-end scores are a stored run.** E2 is the current retrieval default;
   E3–E5 were not re-run after every presentation-only or prompt change.
4. **NLI precision is strict.** It under-credits correct abstractive citations.
   The judge's citation quality is a separate, more forgiving signal. Neither is
   a human audit.
5. **PubMedQA is oracle-context and biomedical.** It tests grounded answering,
   not Ayurveda retrieval.
6. **Refusal is tested on obvious OOD questions.** Fifteen hand-written
   adversarial questions do not estimate false-refusal rate on hard answerable
   queries.
7. **No multilingual benchmark.** A multilingual embedding is configured, but
   the current evaluation is English; BhashaBench-Ayur is gated and not bundled.
8. **No formal latency or cost benchmark.** Timings exist in the API trace but
   depend on the machine and remote model provider.
9. **The decision-mention metric was corrected after the original run.** The
   corrected value uses the same stored answers and judge scores; only the
   matching logic changed.
10. **Corpus scope is smaller than the README’s original aspiration.** Ashtanga
    Hridaya and the Nagoya Protocol are present as raw files but are not parsed
    into the current index; reported numbers exclude them.

---

## 8. Reproducibility

Environment: CPU-only Linux, 31 GB RAM, Python 3.12 virtual environment, no GPU.
The embedding, reranker, and NLI models run locally on CPU; generation and
judging call a remote OpenAI-compatible endpoint.

```bash
uv sync

export AYURRAG_LLM_API_KEY=...
export AYURRAG_FALLBACK_API_KEY=...   # optional secondary provider

# retrieval ablations, 180 questions
PYTHONPATH=src .venv/bin/ayurrag eval --kind retrieval --limit 180 --ablations

# current default retrieval only
PYTHONPATH=src .venv/bin/ayurrag eval --kind retrieval --limit 180

# end-to-end domain QA + refusal
PYTHONPATH=src .venv/bin/ayurrag eval --kind generation --limit 60

# external PubMedQA oracle-context check
PYTHONPATH=src .venv/bin/ayurrag eval --kind external --limit 48

# tests
.venv/bin/python -m pytest -q
```

Artifacts:

- `data/artifacts/retrieval_gold.jsonl` — cached 180-question benchmark.
- `data/artifacts/eval_report.json` — full per-question results.
- `data/artifacts/eval_report.md` — summary tables.
- `data/artifacts/llm_cache.sqlite` — cached generations.
- `data/artifacts/judge_cache.sqlite` — cached judge scores.

The benchmark is cached on purpose. Regenerating it changes the questions and
invalidates comparisons; delete it only for a deliberate benchmark refresh.

---

## 9. Common questions

### Q1. What is the contribution in one sentence?

A domain-specific, structure-aware hybrid RAG system for Ayurveda and IPR
research that keeps citations attached to retrieved evidence, verifies
citations, refuses when evidence is insufficient, and is evaluated with
source-stratified retrieval, domain QA, refusal, and external PubMedQA checks.

### Q2. Why RAG instead of fine-tuning the model?

RAG gives provenance and updatability. The corpus changes, the law changes, and
the user needs to see the passage behind a claim. Fine-tuning would not provide
citations, would require labelled domain data, and would still hallucinate. It
also does not fit a CPU-only environment. The project deliberately avoids
fine-tuning and LLM-only retrieval because the legal/medical cost of a confident
wrong answer is high.

### Q3. What data did you use, and how much of it?

Five source families, **8,142 source documents**, **20,568 chunks**, and
**4,918,614 tokens**:

| Source | Documents | Chunks |
|---|---:|---:|
| Charaka Samhita | 115 chapters | 1,737 |
| Sushruta Samhita | 3 volumes | 1,871 |
| PubMed / Europe PMC | 8,000 abstracts | 16,649 |
| Google Patents | 21 records | 68 |
| Legal | 3 documents | 243 |
| **Total** | **8,142** | **20,568** |

Legal documents are the Patents Act 1970, the Biological Diversity Act 2002, and
the Convention on Biological Diversity. Ashtanga Hridaya and the Nagoya Protocol
are present as raw files but are not in the current index.

### Q4. How many chunks are there, and how big are they?

**20,568 chunks**, averaging **239.1 tokens**; minimum 10, maximum 822. The
chunker caps a chunk at 1,400 characters and uses 180-character overlap only when
splitting an oversized unit. It does not use a blind 512-token window: it groups
verses, sections, or abstracts and carries a citation header.

### Q5. How did you choose the data?

The corpus follows the task: classical texts for traditional knowledge, PubMed
for modern evidence, patents for prior art, and statutes/treaties for legal
provisions. Sources were selected for public accessibility and domain relevance.
The patent set starts from 16 verified Ayurveda/herbal publication numbers and
snowballs through citations and similar documents to 21 records. PubMed uses a
Europe PMC query restricted to abstracts.

### Q6. Is the data licensed for this use?

It is used as a local research corpus and is not redistributed. Legal texts are
public government/CBD documents. PubMed abstracts are accessed under
Europe PMC/NLM and publisher terms. Patents are public records. Sushruta uses a
public-domain translation. The Charaka edition’s rights are unclear, so it is
treated as research-use-only. PubMedQA is MIT-licensed.

### Q7. Why hybrid retrieval instead of dense-only?

Dense-only scored Recall@10 0.839 on the domain benchmark, versus 0.972 for the
hybrid reranked pipeline. Exact section numbers, patent identifiers, and rare
transliterations are poorly represented by a small embedding model. The dense
channel is still valuable for paraphrases, which is why the system keeps both
channels instead of choosing one.

### Q8. Why is BM25 weighted 0.8?

The corpus is terminology-dense and contains exact identifiers. The 0.8/0.2
weights were selected by ablation: convex 0.8/0.2 beat equal-weight RRF and
dense-only on the same 180 questions. The weights are a measured default, not a
universal constant.

### Q9. Why add a cross-encoder reranker?

It improved R@5 from 0.950 to 0.961, R@10 from 0.961 to 0.972, R@20 from 0.967
to 0.978, MRR@10 from 0.874 to 0.890, and nDCG@10 from 0.895 to 0.910. The 50/50
blend with the first-stage score prevents the reranker from discarding strong
lexical matches.

### Q10. Why is the retrieval benchmark not circular?

It is circular in a specific, acknowledged way: queries are generated from the
gold chunk, so they share vocabulary and the benchmark is easier for BM25 than
real user queries. The benchmark is not used to claim open-domain quality. It is
used for reproducible, source-stratified ablations under a fixed protocol. The
dense-only row and the paraphrased failures are the evidence that the benchmark
is not the whole story. A human-written paraphrase set is the next step.

### Q11. What metrics did you use?

Retrieval: Recall@5/10/20, MRR@5/10/20, nDCG@5/10/20, plus Recall@10 by source.
Generation: judge-scored correctness, groundedness, completeness, citation
quality, harmful overreach; automatic citation coverage and NLI citation
precision; insufficient-evidence rate. Refusal: refusal rate. External:
PubMedQA decision-mentioned rate.

### Q12. What are the headline retrieval numbers?

Default pipeline: **Recall@5 0.961, Recall@10 0.972, Recall@20 0.978,
MRR@10 0.888, nDCG@10 0.909**. Recall@10 is 1.000 for legal and PubMed, 0.972
for Charaka, 0.971 for patents, and 0.917 for Sushruta.

### Q13. What are the end-to-end QA numbers?

On 60 questions (55 judged): correctness **4.582/5**, groundedness **3.673/5**,
completeness **4.491/5**, citation quality **3.418/5**, harmful overreach
**0.000**, citation coverage **0.8896**, NLI citation precision **0.4681**,
insufficient-evidence rate **0.150**.

### Q14. Why is citation quality only 3.42/5?

Because the generator often cites a passage that is topically relevant but does
not entail the exact sentence. That is a real failure mode, not a metric bug.
The NLI verifier is included to flag it; the API exposes `supported: false`
rather than hiding it. Improving citation selection is a priority.

### Q15. Why is NLI citation precision only 0.468?

The NLI model is a small CPU cross-encoder evaluating long, abstractive
sentences. It is conservative: valid paraphrases are often judged unsupported.
That is why the report also includes the judge’s citation-quality score, which
is more forgiving. Neither is a human-verified precision; the honest statement is
“strict automatic entailment is 0.468”, not “47% of citations are wrong”.

### Q16. Is the 9/60 insufficient-evidence rate a success or a failure?

Mostly a risk signal. Nine raw outputs set the flag; the API would withhold them
as “I don’t know”. The flagged subset scores lower on every judge dimension, so
some flags are appropriate. But 15% on answerable domain questions is a
false-refusal risk. The project reports it rather than tuning it away.

### Q17. Does 15/15 refusal prove the system is safe?

No. The 15 questions are obvious out-of-domain or fabricated claims. The result
shows the refusal contract works on a smoke test. It does not estimate
false-refusal on hard answerable questions, and it is not a calibrated
abstention benchmark.

### Q18. Why PubMedQA, and what does 0.3125 mean?

PubMedQA is an external, MIT-licensed biomedical QA set. The 48-question sample
(16 yes / 16 no / 16 maybe) supplies gold contexts directly to the generator, so
it tests grounded answering and citation discipline, not retrieval. Groundedness
is 4.957/5 and citation quality 4.809/5. The decision-mentioned rate is 0.3125:
the answer explains the evidence rather than emitting a bare label, especially
for `maybe`. The metric uses token-boundary matching; an earlier substring check
wrongly matched `no` inside words such as `not`.

### Q19. Which models did you use?

Current default: `deepseek-flash` for generation and judging. The stored
end-to-end scores were produced with `deepseek-v4.1-flash-free` as generator and
`glm-5.3-flash-free` as judge. A secondary provider (`wand`,
`deepseek-v4.1-flash-free`) is configured for failover. The model swap is
documented as a caveat; it is not presented as a reproduction of the stored
scores.

### Q20. How do you prevent hallucination?

Five layers: evidence-only system prompt; strict JSON output; per-sentence
`[S#]` citation requirement; `insufficient_evidence` refusal path; NLI citation
verification with a `supported` flag. The API also exposes the retrieval trace
and evidence so a user can inspect the basis of an answer. The evaluation checks
harmful overreach and citation coverage.

### Q21. How do you handle medical and legal safety?

The prompt forbids medical advice, diagnosis, and treatment recommendations. It
also forbids definitive patentability, infringement, or legal-validity
conclusions. Legal statements are framed as “potentially relevant provisions”
requiring professional review. The judge flags harmful overreach; both judged
generation runs had a rate of 0.000.

### Q22. What about Sanskrit and multilingual queries?

The query planner uses a 101-entry, 300-alias Ayurveda lexicon covering Sanskrit
terms, transliterations, common names, and botanical names. A multilingual
embedding (`intfloat/multilingual-e5-small`) is configured, but the reported
evaluation is English. BhashaBench-Ayur is gated and was not bundled. Multilingual
retrieval quality is therefore not claimed.

### Q23. Why not GraphRAG or an agentic pipeline?

Graph expansion and subquery decomposition are implemented and measurable. On
the 180-question benchmark they matched the default on Recall@5–20 but were
marginally worse on MRR/nDCG and added retrieval work, so they are opt-in. The
design principle is to ship the simplest pipeline that wins on the measured
metrics, not to add an agent loop for its own sake.

### Q24. How fast is it, and what does it cost?

No formal latency benchmark is claimed. The embedding, reranker, and NLI models
are small CPU models; the LLM is remote. Generation and judge calls are cached in
SQLite, so repeated evaluation is cheap. The system is designed for a CPU-only
machine with 31 GB RAM, not for low-latency production serving. Measuring p50/p95
latency and token cost is future work.

### Q25. What are the biggest limitations?

The synthetic retrieval benchmark is lexical; there is one LLM judge and no human
audit; end-to-end scores are a stored run; NLI precision is strict; the corpus is
English and excludes Ashtanga/Nagoya; refusal is tested on obvious OOD questions;
and there is no formal latency/cost evaluation. These are stated in the report
rather than hidden.

### Q26. What would you do with more time?

1. Build a human-written paraphrase retrieval set and a human citation audit.
2. Re-run E3/E5 with the final generation and judge configuration.
3. Add a second judge or human spot-check to estimate judge variance.
4. Parse Ashtanga Hridaya and the Nagoya Protocol into the index.
5. Evaluate the multilingual model on BhashaBench-Ayur or an equivalent set.
6. Calibrate refusal thresholds against hard answerable questions.
7. Measure latency and cost per query.

### Q27. How reproducible is the evaluation?

The domain benchmark uses seed 13; PubMedQA uses seed 17. The benchmark,
generations, and judge scores are cached under `data/artifacts/`. The 14-test
suite passes. The retrieval ablations can be rerun with one command. The main
non-determinism is the remote LLM; caching and temperature 0.0 for judging reduce
it.

### Q28. How is this different from “ChatGPT with PDFs”?

The system does not just paste PDF text into a prompt. It has structure-aware
chunking with verse/section citations, a hybrid lexical/dense index, a
cross-encoder reranker, a domain lexicon, citation-constrained JSON generation,
NLI verification, a refusal path, and a source-stratified evaluation harness.
The retrieval trace and evidence are exposed through the API/UI.

### Q29. What would convince a skeptic?

A human-written paraphrase retrieval set; a human citation audit on a sample of
answers; two independent judges with agreement statistics; and a held-out
legal/medical expert review. The current results are a strong engineering
baseline, not a clinical or legal validation.

### Q30. What is the main claim you would defend?

The main defensible claim is: **a structure-aware hybrid retriever with a
cross-encoder reranker retrieves the held-out gold passage in the top 10 for
97.2% of a 180-question, source-stratified domain benchmark, and the
citation-constrained generator produces high correctness/completeness with zero
judged harmful overreach.** The generation/citation results are directional and
the benchmark has known lexical bias; the report states both.

---

## 10. One-minute summary for presentation

AyurRAG is an evidence-grounded research assistant for Ayurveda and IPR. It
indexes 20,568 chunks from 8,142 documents across classical texts, PubMed
abstracts, patents, and law. Retrieval combines BM25 and `bge-small` embeddings
with convex fusion (0.8/0.2) and a cross-encoder reranker; the default pipeline
reaches Recall@10 0.972, MRR@10 0.888, and nDCG@10 0.909 on a 180-question
source-stratified benchmark. The generator must cite retrieved evidence, and an
NLI verifier flags unsupported citations. On 60 domain questions, correctness is
4.58/5, groundedness 3.67/5, completeness 4.49/5, and citation quality 3.42/5,
with zero judged harmful overreach; all 15 unanswerable questions were refused.
PubMedQA oracle-context grounding reaches 4.96/5 groundedness and 4.81/5 citation
quality. The honest weaknesses are the lexical synthetic retrieval benchmark, a
single LLM judge, strict NLI under-crediting, and a stored end-to-end run from
before the final model swap.
