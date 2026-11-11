from ayurrag.config import RetrievalConfig
from ayurrag.retrieve import _normalize, decompose_query, plan_queries


def test_decompose_multipart_question():
    query = "What conditions allow a UK applicant to claim priority, and how are pre-registration infringements treated?"
    parts = decompose_query(query)
    assert any("pre-registration" in part for part in parts)
    assert all(part.lower() != query.lower() for part in parts)


def test_decompose_single_question_is_not_split():
    assert decompose_query("What does Charaka say about the three pillars of health?") == []


def test_plan_expands_known_entity_aliases():
    variants = plan_queries("ashwagandha stress", RetrievalConfig(max_subqueries=0))
    kinds = {variant.kind for variant in variants}
    assert "aliases" in kinds
    alias_text = next(v.text for v in variants if v.kind == "aliases")
    assert "withania" in alias_text.lower()


def test_normalize_bounds():
    assert _normalize({"a": 2.0, "b": 4.0, "c": 6.0}) == {"a": 0.0, "b": 0.5, "c": 1.0}
