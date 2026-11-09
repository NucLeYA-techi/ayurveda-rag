"""Sushruta Samhita parser for the public-domain Bhishagratna translation.

The archive.org DjVu OCR is noisy, so the parser relies on running heads
(``Chap. I.] SUTRASTHANAM. 7`` and ``12 THE SUSHRUTA SAMHITA. [Chap. I.``)
to recover page, section and chapter provenance, and falls back to paragraph
units when a heading cannot be recognized.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..schema import SourceDocument
from .common import ROMAN, doc

HEAD_A = re.compile(
    r"Chap\.?\s*([IVXLnHm]+)\.?\s*\]?\s*([A-Z][A-Z'’.\- ]{2,}?)\s*STH[A-Z]*\.?\s+(\d{1,4})",
    re.I,
)
HEAD_B = re.compile(
    r"(\d{1,4})\s+THE\s+SUS[A-Z]+\s+SAM[A-Z]+\.?\s*\[?\s*Chap\.?\s*([IVXLnHm]+)",
    re.I,
)
STHANA_RE = re.compile(r"([A-Z][A-Z'’.\- ]{2,}?)\s*STH[A-Z]*\.?", re.I)
CHAPTER_RE = re.compile(r"^CHAP\w*\s+([IVXLnHm0-9]+)\s*[.,]?", re.I)
PAGE_RE = re.compile(r"^\(\s*(\d{1,4})\s*\)$")
TOC_HINT = re.compile(r"pages?\s+\d+\s*[–—-]", re.I)

STHANA_NAMES = {
    "SUTRA": "Sutra Sthana",
    "NIDANA": "Nidana Sthana",
    "NIDAN": "Nidana Sthana",
    "SARIRA": "Sharira Sthana",
    "SHARIRA": "Sharira Sthana",
    "CHIKITS": "Chikitsa Sthana",
    "CHIKITSA": "Chikitsa Sthana",
    "CHIKITSITA": "Chikitsa Sthana",
    "KALPA": "Kalpa Sthana",
    "UTTARA": "Uttara Tantra",
}

VOLUMES = {
    # file -> (volume number, default sthana)
    "sushruta_v2.txt": (1, "Sutra Sthana"),
    "sushruta_v1.txt": (2, "Nidana Sthana"),
    "sushruta_v3.txt": (3, "Uttara Tantra"),
}


def _roman(token: str) -> int | None:
    token = token.strip(" .,;:").upper()
    replacements = {"VN": "VII", "VIH": "VIII", "HI": "III", "N": "II", "M": "III"}
    token = replacements.get(token, token)
    if token.isdigit():
        return int(token)
    return ROMAN.get(token)


def _clean_line(line: str) -> str:
    line = line.replace("\u00ad", "")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _sthana_name(raw: str, fallback: str) -> str:
    key = re.sub(r"[^A-Z]", "", raw.upper())
    for prefix, name in STHANA_NAMES.items():
        if key.startswith(prefix):
            return name
    return fallback


def _is_toc(paragraph: str) -> bool:
    return bool(TOC_HINT.search(paragraph)) or paragraph.count("—") > 6


def _is_junk(paragraph: str) -> bool:
    if paragraph.count("See ") > 2 and len(paragraph) < 2500:
        return True
    if "Digitized" in paragraph or "DUE ON" in paragraph.upper():
        return True
    if paragraph.count("...") > 8 or paragraph.count(" ... ") > 4:
        return True
    words = paragraph.split()
    if words and paragraph.count(".") / len(words) > 0.45:
        return True
    if len(re.findall(r"\b[A-Z][a-z]{2,} [a-z]{3,}\b", paragraph)) >= 6:
        return True
    return False


def parse_volume(path: Path, volume: int, default_sthana: str) -> list[SourceDocument]:
    lines = path.read_text(errors="replace").splitlines()

    # Skip front matter and the table of contents: start at the first running head.
    body_start = 0
    for index, line in enumerate(lines):
        if HEAD_A.search(line) or HEAD_B.search(line):
            body_start = index
            break

    units: list[str] = []
    labels: list[str] = []
    sthana = default_sthana
    chapter = 1
    chapter_unit = 0
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, chapter_unit
        text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
        buffer = []
        if len(text) < 60 or _is_toc(text) or _is_junk(text):
            return
        chapter_unit += 1
        units.append(text)
        labels.append(f"{sthana} {chapter}.{chapter_unit}")

    for raw_line in lines[body_start:]:
        line = raw_line.strip()
        if not line:
            flush()
            continue

        head_a = HEAD_A.search(line)
        head_b = HEAD_B.search(line)
        if head_a or head_b:
            flush()
            if head_a:
                chapter = _roman(head_a.group(1)) or chapter
                sthana = _sthana_name(head_a.group(2), sthana)
            if head_b:
                chapter = _roman(head_b.group(2)) or chapter
            chapter_unit = 0
            continue

        page_match = PAGE_RE.match(line)
        if page_match:
            flush()
            continue

        chapter_match = CHAPTER_RE.match(line)
        if chapter_match and len(line) < 32:
            flush()
            chapter = _roman(chapter_match.group(1)) or chapter + 1
            chapter_unit = 0
            continue

        if line.upper() == line and len(line) < 45:
            continue
        if len(line) < 3:
            continue
        buffer.append(_clean_line(line))
    flush()

    if not units:
        return []

    title = f"Sushruta Samhita — Volume {volume}"
    return [
        doc(
            source="sushruta",
            corpus="classical",
            title=title,
            units=units,
            labels=labels,
            url=f"https://archive.org/details/{path.stem.replace('_', '')}",
            license="Public domain (Bhishagratna translation, 1907–1916)",
            lang="en",
            doc_key=f"vol{volume}",
            metadata={
                "work": "Sushruta Samhita",
                "volume": volume,
                "translator": "Kaviraj Kunja Lal Bhishagratna",
                "year": "1911",
            },
        )
    ]


def parse_sushruta(raw_dir: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for filename, (volume, sthana) in VOLUMES.items():
        path = raw_dir / filename
        if path.exists():
            docs.extend(parse_volume(path, volume, sthana))
    return docs
