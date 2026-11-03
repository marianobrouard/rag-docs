from rag_docs.evaluate import retrieval_hit


def test_retrieval_hit_with_expected_source_retrieved_scores_true() -> None:
    result = retrieval_hit(
        {"sources": ["bonos.md", "vacaciones.md"]}, {"sources": ["bonos.md"]}
    )
    assert result["score"] is True


def test_retrieval_hit_with_one_of_two_sources_missing_scores_false() -> None:
    result = retrieval_hit(
        {"sources": ["bonos.md", "relocacion.md"]},
        {"sources": ["vacaciones.md", "bonos.md"]},
    )
    assert result["score"] is False


def test_retrieval_hit_with_unanswerable_question_scores_none() -> None:
    result = retrieval_hit({"sources": ["bonos.md"]}, {"sources": []})
    assert result["score"] is None


def test_retrieval_hit_with_failed_target_scores_none_instead_of_crashing() -> None:
    result = retrieval_hit({"error": "ServerError: 503"}, {"sources": ["bonos.md"]})
    assert result["score"] is None
    assert result["comment"] == "target failed"
