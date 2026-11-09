"""Charaka Samhita parser (verse-level JSON).

Source: https://github.com/gita/Datasets/tree/main/Ayurveda/charak-samhita
The text is the English translation commonly attributed to R. K. Sharma and
Bhagwan Dash. It is fetched at build time and is not redistributed here; check
the edition's rights before deploying.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..schema import SourceDocument
from ..text import normalize
from .common import doc

SECTION_NAMES = {
    "1": ("Sutrasthana", "Sutra Sthana"),
    "2": ("Nidanasthana", "Nidana Sthana"),
    "3": ("Vimanasthana", "Vimana Sthana"),
    "4": ("Sharirasthana", "Sharira Sthana"),
    "5": ("Indriyasthana", "Indriya Sthana"),
    "6": ("Cikitsasthana", "Chikitsa Sthana"),
    "7": ("Kalpasthana", "Kalpa Sthana"),
    "8": ("Siddhisthana", "Siddhi Sthana"),
}

TITLE_RE = re.compile(r"entitled\s+[“\"](.+?)[”\"]", re.S)


def _chapter_title(verses: list[dict]) -> str:
    for verse in verses[:4]:
        match = TITLE_RE.search(verse.get("text", ""))
        if match:
            return normalize(match.group(1)).strip(" .")
    return ""


def parse_charaka(raw_dir: Path) -> list[SourceDocument]:
    docs: list[SourceDocument] = []
    for path in sorted(raw_dir.glob("*.json")):
        match = re.match(r"(\d+)\.([A-Za-z]+)_chapter(\d+)\.json", path.name)
        if not match:
            continue
        section_no, _, chapter_no = match.groups()
        raw_name, display_name = SECTION_NAMES.get(section_no, (f"Section {section_no}", f"Section {section_no}"))
        verses = json.loads(path.read_text())
        verses = [v for v in verses if str(v.get("text", "")).strip()]
        if not verses:
            continue
        chapter_title = _chapter_title(verses)
        chapter = int(chapter_no)
        units: list[str] = []
        labels: list[str] = []
        for verse in verses:
            body = normalize(verse["text"]).strip()
            if not body:
                continue
            verse_id = str(verse.get("verse_id", "")).strip()
            units.append(body)
            labels.append(f"{display_name} {chapter}.{verse_id}")
        title = f"Charaka Samhita — {display_name}, Chapter {chapter}"
        if chapter_title:
            title += f": {chapter_title}"
        docs.append(
            doc(
                source="charaka",
                corpus="classical",
                title=title,
                units=units,
                labels=labels,
                url="https://github.com/gita/Datasets/tree/main/Ayurveda/charak-samhita",
                license="Edition rights unclear; fetched for research use",
                lang="en",
                doc_key=f"{raw_name}-{chapter}",
                metadata={
                    "work": "Charaka Samhita",
                    "section": display_name,
                    "chapter": chapter,
                    "chapter_title": chapter_title,
                },
            )
        )
    return docs
