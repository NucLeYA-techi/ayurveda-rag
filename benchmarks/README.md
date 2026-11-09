# External benchmark fixture

`pubmedqa_sample.jsonl` is a deterministic 48-question stratified sample (16 each of
yes/no/maybe) of the **PubMedQA** labelled test set (`pqa_labeled`, 1,000 expert-annotated
biomedical questions with PubMed abstract contexts and long answers).

- Source: https://huggingface.co/datasets/qiaojin/PubMedQA
- License: MIT
- Sampling seed: 17; contexts capped at six paragraphs per question.

It is used for an **oracle-context generation** check: the gold PubMedQA contexts are passed
to the same citation-constrained generator and LLM judge used by the domain evaluation, so it
measures grounded answering and citation discipline on out-of-domain biomedical evidence
(not retrieval, and not Ayurveda-specific).
