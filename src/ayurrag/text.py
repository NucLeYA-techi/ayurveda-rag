"""Text normalization, tokenization and segmentation helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata

PAGE_MARKER = re.compile(r"^\s*\(\s*\d{1,4}\s*\)\s*$")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
WORD_RE = re.compile(r"[A-Za-z\u0900-\u097F][A-Za-z\u0900-\u097F'’\-]*|\d+(?:\.\d+)?")
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+(?=[A-Z\u0900-\u097F(\"'])")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " — ").replace("\u2013", " – ")
    return re.sub(r"[ \t\u00a0]+", " ", text)


def clean_ocr(text: str) -> str:
    """Repair the most common OCR artefacts in archive.org DjVu text."""
    text = normalize(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # join words hyphenated across a line break
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        if PAGE_MARKER.match(s):
            continue
        # drop running heads/footers that consist of digits or roman numerals
        if re.fullmatch(r"[ivxlcdmIVXLCDM\d .\-–—]{1,12}", s):
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    text = normalize(text)
    parts = SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def is_devanagari(text: str) -> bool:
    return len(DEVANAGARI.findall(text)) > 0.15 * max(len(text), 1)


def approx_tokens(text: str) -> int:
    # conservative estimate: ~4 characters per token for English, ~2 for Devanagari
    return max(1, round(len(text) / (2 if is_devanagari(text) else 4)))


def content_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    boundary = max(cut.rfind(". "), cut.rfind("\n"))
    if boundary > max_chars * 0.6:
        cut = cut[: boundary + 1]
    return cut.rstrip() + "…"


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split on sentence boundaries with a small overlap."""
    if len(text) <= max_chars:
        return [text]
    sentences = split_sentences(text)
    if not sentences:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars - overlap_chars)]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if length + len(sentence) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            # carry overlap sentences
            tail: list[str] = []
            tail_len = 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks
