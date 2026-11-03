import json
from pathlib import Path

from rag_docs.minimal import DATA_DIR

DATASET_PATH = Path(__file__).resolve().parents[1] / "evals" / "dataset.json"


def test_dataset_with_all_questions_has_unique_ids() -> None:
    items = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids)) == 29


def test_dataset_expected_sources_point_to_existing_documents() -> None:
    items = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    existing = {p.name for p in DATA_DIR.glob("*.md")}
    for item in items:
        assert set(item["expected_sources"]) <= existing, item["id"]


def test_dataset_unanswerable_questions_have_no_sources() -> None:
    items = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    for item in items:
        if item["type"] in {"unanswerable", "unanswerable_tempting", "injection"}:
            assert item["expected_sources"] == []
            assert item["expected_answer"] == "NO_ANSWER"
