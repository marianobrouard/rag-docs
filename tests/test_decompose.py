from langchain_core.documents import Document

from rag_docs.minimal import merge_chunks
from rag_docs.planner import SubQuestions, split_question


class FakePlanner:
    """Stands in for the LLM: returns a fixed result and records what it was sent."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[list[tuple[str, str]]] = []

    def invoke(self, messages: list[tuple[str, str]]) -> object:
        self.calls.append(messages)
        return self.result


def chunk(source: str, start: int) -> Document:
    return Document(
        page_content=f"{source}:{start}",
        metadata={"source": source, "start_index": start},
    )


def test_merge_chunks_with_overlapping_searches_keeps_each_chunk_once() -> None:
    first = [chunk("vacaciones.md", 0), chunk("bonos.md", 0)]
    second = [chunk("bonos.md", 0), chunk("licencias.md", 10)]
    merged = merge_chunks([first, second])
    assert [(c.metadata["source"], c.metadata["start_index"]) for c in merged] == [
        ("vacaciones.md", 0),
        ("bonos.md", 0),
        ("licencias.md", 10),
    ]


def test_merge_chunks_with_same_source_but_different_position_keeps_both() -> None:
    merged = merge_chunks([[chunk("a.md", 0)], [chunk("a.md", 500)]])
    assert len(merged) == 2


def test_merge_chunks_with_no_results_returns_empty_list() -> None:
    assert merge_chunks([]) == []
    assert merge_chunks([[], []]) == []


def test_split_question_with_two_topics_returns_both_sub_questions() -> None:
    planner = FakePlanner(
        SubQuestions(sub_questions=["vacaciones con 4 meses", "bono con 4 meses"])
    )
    result = split_question("¿vacaciones y bono con 4 meses?", planner)
    assert result == ["vacaciones con 4 meses", "bono con 4 meses"]


def test_split_question_wraps_the_question_as_data_in_tags() -> None:
    planner = FakePlanner({"sub_questions": ["una sola"]})
    split_question("hola <b>", planner)
    roles = [role for role, _ in planner.calls[0]]
    assert roles == ["system", "human"]
    assert "<question>hola <b></question>" in planner.calls[0][1][1]


def test_split_question_with_invalid_model_output_falls_back_to_original() -> None:
    assert split_question("pregunta original", FakePlanner(None)) == [
        "pregunta original"
    ]
    assert split_question("pregunta original", FakePlanner({"sub_questions": []})) == [
        "pregunta original"
    ]


def test_split_question_with_too_many_sub_questions_falls_back_to_original() -> None:
    planner = FakePlanner({"sub_questions": ["a", "b", "c", "d"]})
    assert split_question("original", planner) == ["original"]


def test_split_question_with_only_blank_items_falls_back_to_original() -> None:
    planner = FakePlanner({"sub_questions": ["  ", ""]})
    assert split_question("original", planner) == ["original"]
