# Design notes: why this RAG stack

## Task shape

The corpus is heterogeneous and terminology-dense: Sanskrit/transliterated
classical text, PubMed abstracts, patent claims, and statutory sections. Queries
mix natural language ("what supports digestion?"), Sanskrit terms
(*agni*, *upastambha*), botanical names (*Withania somnifera*), patent numbers,
and section references. Many questions are **multi-source** (classical claim +
modern study + legal provision), and the output must be auditable.

That rules out a single dense index as the only retriever: exact section numbers,
rare transliterations and patent identifiers are poorly represented in a small
biomedical embedding model. It also rules out pure BM25: users paraphrase and
cross terminology boundaries ("three pillars of health" vs *upastambha*).

## Choices and evidence

| Choice | Why | Alternatives considered |
|---|---|---|
| BM25 + dense hybrid, score-normalized convex fusion | lexical precision on statutes/patents + semantic recall on paraphrases; measured better than equal-weight RRF on every metric (MRR 0.889 vs 0.800) | dense-only (DPR/Karpukhin et al. 2020), BM25-only, RRF (Cormack et al., SIGIR 2009) |
| `bge-small-en-v1.5` | strong small CPU embedding model (C-Pack, Xiao et al. 2023); 384-d, fast on CPU, no GPU required | `e5-small`, `all-MiniLM-L6-v2`, hosted embeddings |
| Cross-encoder rerank (`ms-marco-MiniLM-L-6-v2`) blended 50/50 with first-stage score | fixes the "pillars" → *upastambha* semantic gap without discarding exact lexical matches; standard passage-reranking recipe (Nogueira & Cho 2019) | ColBERT, LLM reranker, pure rerank score |
| Curated Ayurveda lexicon + optional co-occurrence graph | exact alias mapping across Sanskrit/botanical/common names; `entity_edges` gives one-hop related-entity expansion for recall-oriented runs | learned entity linker, GraphRAG over extracted triples (Edge et al. 2024) |
| Structure-aware chunking with verse/section citations | citation is a first-class field, not a post-hoc guess; keeps verse ranges and statutory sections intact | fixed 512-token windows |
| Citation-constrained generation + NLI verification | every sentence must carry `[S#]`; the verifier flags unsupported citations and the API returns the flag | unconstrained answer + post-hoc attribution, Self-RAG-style reflection (Asai et al. 2023) |

We avoided fine-tuning and LLM-only retrieval because the environment is CPU-only,
the domain labels are thin, and a wrong legal/medical claim is costly. A
retrieve-then-verify design is easier to audit and degrade safely.

## Evaluation design

1. **Retrieval**: queries generated from held-out chunks (one gold chunk per
   query), stratified across all sources and two query styles. This is lexical by
   construction; it favours BM25 and is reported as such. Ablations cover BM25,
   dense, equal-weight RRF, convex fusion, blended reranking, query expansion and
   graph expansion. The measured winner is convex fusion (0.8/0.2) + 50/50
   rerank blend: R@10 0.972, MRR 0.889, nDCG 0.910. Expansion/graph improved
   R@20 (0.983) but reduced early precision, so they are opt-in.
2. **End-to-end domain QA**: 60 questions over all sources, scored by an
   independent LLM judge for correctness, groundedness, completeness and citation
   quality, plus harmful-overreach and refusal checks. Judge prompts and scores
   are cached in SQLite for reproducibility.
3. **External grounding**: 48 PubMedQA labelled questions with gold contexts
   (MIT), using the same generator, verifier and judge. This checks that the
   citation discipline generalizes beyond the Ayurveda corpus; it is explicitly
   *oracle-context*, not a retrieval score.
4. **Refusal**: 15 adversarial unanswerable questions (out-of-domain facts,
   fabricated patents/trials) to test the `insufficient_evidence` path.

Known limitations: the synthetic retrieval benchmark under-represents paraphrase
queries; the NLI verifier is a small CPU model and its strict sentence-level
score under-credits abstractive but correct citations; PubMedQA is biomedical,
not Ayurvedic. BhashaBench-Ayur is gated and therefore not bundled.

## References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020.
- Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR 2009.
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering*, EMNLP 2020.
- Nogueira & Cho, *Passage Re-ranking with BERT*, arXiv:1901.04085.
- Xiao et al., *C-Pack: Packed Resources for General Chinese Embeddings*, SIGIR 2023 (BGE models).
- Jin et al., *PubMedQA: A Dataset for Biomedical Research Question Answering*, EMNLP 2019.
- Pal et al., *MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical Domain Question Answering*, CHIL 2022.
- Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, ICLR 2024.
- Edge et al., *From Local to Global: A Graph RAG Approach to Query-Focused Summarization*, arXiv:2404.16130.
- Gao et al., *Retrieval-Augmented Generation for Large Language Models: A Survey*, arXiv:2312.10997.
