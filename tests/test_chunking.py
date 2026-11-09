from ayurrag.chunking import chunk_corpus
from ayurrag.config import ChunkingConfig
from ayurrag.schema import SourceDocument


def document() -> SourceDocument:
    units = [
        "The quest for longevity begins with the desire to live. " * 10,
        "Ayurveda is the science of life and longevity. " * 10,
        "The three pillars are food, sleep and conduct. " * 10,
    ]
    return SourceDocument(
        doc_id="test",
        source="charaka",
        corpus="classical",
        title="Charaka Samhita",
        units=units,
        unit_labels=["Sutra Sthana 1.1", "Sutra Sthana 1.2", "Sutra Sthana 1.3"],
        metadata={"work": "Charaka Samhita"},
    )


def test_labels_merge_into_compact_range():
    chunks = chunk_corpus([document()], ChunkingConfig(max_chars=4000, overlap_chars=0))
    assert chunks[0].citation == "Charaka Samhita, Sutra Sthana 1.1–1.3"
    assert chunks[0].source == "charaka"
    assert chunks[0].token_count > 0


def test_single_unit_label_is_not_a_range():
    chunks = chunk_corpus([document()], ChunkingConfig(max_chars=4000, overlap_chars=0))
    assert chunks[0].citation == "Charaka Samhita, Sutra Sthana 1.1–1.3"
