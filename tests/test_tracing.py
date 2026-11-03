from rag_docs.minimal import question_only


def test_question_only_with_heavy_inputs_keeps_just_the_question() -> None:
    inputs: dict[str, object] = {
        "question": "¿Cuántos días?",
        "store": object(),
        "settings": object(),
        "k": 3,
    }
    assert question_only(inputs) == {"question": "¿Cuántos días?"}
