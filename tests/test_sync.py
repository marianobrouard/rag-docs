from rag_docs.evaluate import items_to_upload, to_example

ITEMS = [
    {
        "id": "q01",
        "type": "simple",
        "question": "¿Cuántos días?",
        "expected_answer": "28",
        "expected_sources": ["vacaciones.md"],
    },
    {
        "id": "q02",
        "type": "simple",
        "question": "¿Cuánto?",
        "expected_answer": "1500",
        "expected_sources": ["tarjeta_corporativa.md"],
    },
]


def test_to_example_with_dataset_item_builds_inputs_outputs_and_metadata() -> None:
    example = to_example(ITEMS[0])
    assert example["inputs"] == {"question": "¿Cuántos días?"}
    assert example["outputs"] == {"answer": "28", "sources": ["vacaciones.md"]}
    assert example["metadata"] == {"id": "q01", "type": "simple"}


def test_items_to_upload_with_empty_dataset_returns_everything() -> None:
    assert len(items_to_upload(ITEMS, set())) == 2


def test_items_to_upload_with_some_existing_ids_returns_only_the_new_ones() -> None:
    new = items_to_upload(ITEMS, {"q01"})
    assert [e["metadata"]["id"] for e in new] == ["q02"]


def test_items_to_upload_with_everything_already_uploaded_returns_nothing() -> None:
    assert items_to_upload(ITEMS, {"q01", "q02"}) == []
