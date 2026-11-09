from ayurrag.generate.verify import _windows


def test_window_selection_prefers_supporting_sentence():
    body = (
        "The sky is blue and the grass is green. "
        "First, as regards the three subsupports, they are: food, sleep and continence. "
        "The body is sustained by their right use. "
        "Purgation and emesis are described next."
    )
    claim = "Food, sleep and continence are the three subsupports."
    windows = _windows(body, claim)
    assert any("food, sleep and continence" in window.lower() for window in windows)
