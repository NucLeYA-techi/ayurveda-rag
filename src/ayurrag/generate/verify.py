"""Claim-level citation verification with an NLI cross-encoder."""

from __future__ import annotations

import re

from ..schema import Answer, Chunk

MARKER_RE = re.compile(r"\[(S\d+)\]")
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "as", "is", "are",
    "was", "were", "be", "been", "by", "that", "this", "it", "its", "from", "at", "which",
}


def _tokens(text: str) -> set[str]:
    return {token for token in WORD_RE.findall(text.lower()) if token not in STOPWORDS and len(token) > 2}


def _windows(body: str, claim: str) -> list[str]:
    """Pick a few sentence windows likely to contain the claim's support.

    Whole-chunk NLI is noisy: a 1,400-character passage about several topics can
    dilute a single supported sentence into a "neutral" verdict.
    """
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", body) if part.strip()]
    if len(sentences) <= 3:
        return [body[:1600]]
    claim_tokens = _tokens(claim)
    scored = []
    for index, sentence in enumerate(sentences):
        overlap = len(claim_tokens & _tokens(sentence))
        scored.append((overlap, index))
    scored.sort(reverse=True)
    picked = {index for _, index in scored[:3]}
    windows: list[str] = []
    for index in sorted(picked):
        windows.append(sentences[index])
        if index + 1 < len(sentences):
            windows.append(" ".join(sentences[index : index + 2]))
    return windows[:6]


def _claims(sentence: str) -> list[str]:
    """Split a generated sentence into atomic claims for entailment scoring.

    Long conjunctive sentences are rarely entailed by one short evidence window
    even when each part is supported, so score the parts and aggregate.
    """
    parts = [part.strip(" .") for part in re.split(r"(?:;\s+|,\s+(?:and|but)\s+|\s+but\s+)", sentence)]
    parts = [part for part in parts if len(part) > 15]
    return parts or [sentence]


class CitationVerifier:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        self.model_name = model_name
        self._model = None
        self._entailment_index = 1

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device="cpu")
            labels = getattr(self._model.model.config, "id2label", {}) or {}
            for index, label in labels.items():
                if "entail" in str(label).lower():
                    self._entailment_index = int(index)
                    break
        return self._model

    def verify(self, answer: Answer, threshold: float = 0.55) -> Answer:
        markers = {citation.marker.strip("[]"): citation for citation in answer.citations}
        if not markers:
            return answer
        chunk_by_marker: dict[str, Chunk] = {
            f"S{index}": item.chunk for index, item in enumerate(answer.evidence, start=1)
        }
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", answer.answer) if segment.strip()]
        pairs: list[tuple[str, str]] = []
        pair_markers: list[str] = []
        pair_claims: list[int] = []
        for sentence in sentences:
            used = MARKER_RE.findall(sentence)
            clean = MARKER_RE.sub("", sentence).strip()
            if len(clean) <= 15:
                continue
            cited_chunks = [chunk_by_marker[marker] for marker in used if marker in chunk_by_marker]
            if not cited_chunks:
                continue
            claims = _claims(clean)
            for claim_index, claim in enumerate(claims):
                if len(cited_chunks) > 1:
                    # Joint evidence for multi-source sentences: a claim may be
                    # supported by the union of the cited passages.
                    joint = "\n".join(chunk.body[:700] for chunk in cited_chunks)
                    for marker in used:
                        if marker in chunk_by_marker:
                            pairs.append((joint[:1600], claim))
                            pair_markers.append(marker)
                            pair_claims.append(claim_index)
                else:
                    for window in _windows(cited_chunks[0].body, claim):
                        pairs.append((window[:1600], claim))
                        pair_markers.append(used[0])
                        pair_claims.append(claim_index)
        if not pairs:
            return answer

        scores = self.model.predict(pairs, batch_size=16, show_progress_bar=False, apply_softmax=True)
        claim_best: dict[str, dict[int, float]] = {}
        for marker, claim_index, score_row in zip(pair_markers, pair_claims, scores, strict=False):
            value = float(score_row[self._entailment_index])
            per_claim = claim_best.setdefault(marker, {})
            per_claim[claim_index] = max(per_claim.get(claim_index, 0.0), value)
        for marker, citation in markers.items():
            if marker in claim_best:
                values = list(claim_best[marker].values())
                score = sum(values) / len(values)
                citation.support_score = round(score, 4)
                citation.supported = score >= threshold
        return answer


def uncited_sentences(answer: Answer) -> list[str]:
    result = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer.answer):
        sentence = sentence.strip()
        if len(sentence) > 40 and not MARKER_RE.search(sentence):
            result.append(sentence)
    return result
