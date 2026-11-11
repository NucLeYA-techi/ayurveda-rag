from ayurrag.eval.metrics import mentions_decision, mrr_at_k, ndcg_at_k, recall_at_k


def test_mentions_decision_uses_token_boundaries():
    assert mentions_decision("Yes, the trial supports it.", "yes")
    assert mentions_decision("The answer is no.", "no")
    assert not mentions_decision("There is not enough evidence.", "no")
    assert not mentions_decision("The north cohort differed.", "no")


def test_recall_mrr_ndcg():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["c"], 4) == 1.0
    assert recall_at_k(ranked, ["z"], 4) == 0.0
    assert mrr_at_k(ranked, ["c"], 4) == 1 / 3
    assert ndcg_at_k(ranked, ["c"], 4) == 0.5
