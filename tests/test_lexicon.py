from ayurrag.lexicon import expand_query, extract_entities


def test_extracts_common_and_botanical_names():
    entities = extract_entities("Ashwagandha (Withania somnifera) and Turmeric were studied.")
    assert "Ashwagandha" in entities
    assert "Turmeric" in entities


def test_query_expansion_adds_aliases():
    expanded = expand_query("ashwagandha benefits")
    assert "withania somnifera" in expanded
    assert expanded != "ashwagandha benefits"


def test_concept_aliases_expand_pillars_query():
    expanded = expand_query("What are the three pillars of health?")
    assert "three subsupports" in expanded or "trayopastambha" in expanded
