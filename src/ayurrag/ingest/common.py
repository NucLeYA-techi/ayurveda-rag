"""Helpers shared by corpus parsers."""

from __future__ import annotations

import re
from pathlib import Path

from ..schema import SourceDocument, make_id

ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10, "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20, "XXI": 21,
    "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25, "XXVI": 26, "XXVII": 27,
    "XXVIII": 28, "XXIX": 29, "XXX": 30, "XXXI": 31, "XXXII": 32, "XXXIII": 33,
    "XXXIV": 34, "XXXV": 35, "XXXVI": 36, "XXXVII": 37, "XXXVIII": 38,
    "XXXIX": 39, "XL": 40, "XLI": 41, "XLII": 42, "XLIII": 43, "XLIV": 44,
    "XLV": 45, "XLVI": 46, "XLVII": 47, "XLVIII": 48, "XLIX": 49, "L": 50,
}

ROMAN_RE = re.compile(r"\b([IVXL]+)\b")


def roman_to_int(token: str) -> int | None:
    token = token.upper().replace("I", "I")
    token = token.replace("Vn", "VII").replace("VIH", "VIII").replace("HI", "III")
    token = token.replace("n", "II").replace("m", "III")
    token = token.strip(" .,;:")
    return ROMAN.get(token)


def make_doc_id(source: str, *parts: str) -> str:
    return make_id(source, *parts)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    import orjson

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row) + b"\n")


def read_jsonl(path: Path) -> list[dict]:
    import orjson

    with path.open("rb") as handle:
        return [orjson.loads(line) for line in handle if line.strip()]


def doc(
    *,
    source: str,
    corpus: str,
    title: str,
    text: str = "",
    units: list[str] | None = None,
    labels: list[str] | None = None,
    url: str = "",
    license: str = "",
    lang: str = "en",
    metadata: dict | None = None,
    doc_key: str = "",
) -> SourceDocument:
    return SourceDocument(
        doc_id=make_doc_id(source, doc_key or title),
        source=source,
        corpus=corpus,  # type: ignore[arg-type]
        title=title,
        text=text,
        units=units or [],
        unit_labels=labels or [],
        url=url,
        license=license,
        lang=lang,
        metadata=metadata or {},
    )
