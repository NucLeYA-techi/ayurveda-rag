"""Legal corpus ingestion: statutes, conventions and guidelines (PDF)."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from ..schema import SourceDocument
from ..text import clean_ocr, normalize
from .common import doc

SECTION_RE = re.compile(r"(?m)^\s*(\d{1,3})\.\s+([A-Z][^\n]{6,120})$")
BAD_HEADING = re.compile(r"\b(Subs\.|Ins\.|Omitted|Added by|w\.e\.f|ibid|Sub-section|Explanation|Rep\.)\b", re.I)
ARTICLE_RE = re.compile(r"(?m)^\s*Article\s+(\d{1,3})\s*\.?\s*([^\n]{0,120})$")

LEGAL_DOCS = [
    {
        "file": "patents_act.pdf",
        "title": "The Patents Act, 1970 (India)",
        "url": "https://ipindia.gov.in/frontend/pdf/patents/1_113_1_The_Patents_Act__1970___incorporating_all_amendments_till_1-08-2024.pdf",
        "license": "Government of India, public document",
        "kind": "section",
        "work": "Patents Act 1970",
    },
    {
        "file": "bd_act.pdf",
        "title": "The Biological Diversity Act, 2002 (India)",
        "url": "https://hpbiodiversity.gov.in/BMC/BiodiverstyAct2002english.pdf",
        "license": "Government of India, public document",
        "kind": "section",
        "work": "Biological Diversity Act 2002",
    },
    {
        "file": "cbd.pdf",
        "title": "Convention on Biological Diversity",
        "url": "https://www.cbd.int/doc/legal/cbd-en.pdf",
        "license": "CBD Secretariat, public document",
        "kind": "article",
        "work": "Convention on Biological Diversity",
    },
]


def _extract_pdf_text(path: Path) -> str:
    document = pymupdf.open(path)
    pages = []
    for page in document:
        pages.append(page.get_text("text"))
    document.close()
    text = "\n".join(pages)
    text = clean_ocr(text)
    # remove repeated running headers/footers
    lines = text.splitlines()
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if 4 < len(stripped) < 80:
            counts[stripped] = counts.get(stripped, 0) + 1
    repeated = {line for line, count in counts.items() if count > 5}
    return "\n".join(line for line in lines if line.strip() not in repeated)


def _split_sections(text: str, kind: str) -> list[tuple[str, str]]:
    pattern = ARTICLE_RE if kind == "article" else SECTION_RE
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    # Skip the table of contents: begin at the first heading that has a
    # substantive body before the next heading.
    first = 0
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if len(text[match.end():end].strip()) > 600:
            first = index
            break
    matches = matches[first:]

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        number = match.group(1)
        heading = normalize(match.group(2)).strip(" .;:")
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if BAD_HEADING.search(heading) or body.count("w.e.f") > 1:
            continue
        label = f"Article {number}" if kind == "article" else f"Section {number}"
        citation = f"{label} — {heading}" if heading else label
        content = f"{heading}. {body}" if heading else body
        if len(body) < 250 and sections:
            # running head or page fragment: keep it with the previous section
            prev_citation, prev_body = sections[-1]
            sections[-1] = (prev_citation, prev_body + " " + content)
            continue
        sections.append((citation, content))
    return sections


def parse_legal(raw_dir: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for entry in LEGAL_DOCS:
        path = raw_dir / entry["file"]
        if not path.exists():
            continue
        text = _extract_pdf_text(path)
        sections = _split_sections(text, entry["kind"])
        if not sections:
            continue
        units = [body for _, body in sections]
        labels = [citation for citation, _ in sections]
        docs.append(
            doc(
                source="legal",
                corpus="legal",
                title=entry["title"],
                units=units,
                labels=labels,
                url=entry["url"],
                license=entry["license"],
                doc_key=entry["file"],
                metadata={
                    "work": entry["work"],
                    "document": entry["title"],
                    "jurisdiction": "India" if "India" in entry["title"] else "International",
                },
            )
        )
    return docs
