from pathlib import Path

from rag_docs.report import ReportRow, build_report, write_report


def make_row(**overrides: object) -> ReportRow:
    base: dict[str, object] = {
        "example_id": "q01",
        "kind": "simple",
        "question": "¿Cuántos días?",
        "expected_answer": "28 días",
        "answer": "28 días",
        "expected_sources": ["vacaciones.md"],
        "retrieved_sources": ["vacaciones.md"],
        "answer_correct": True,
        "retrieval_hit": True,
        "judge_comment": "ok",
        "error": None,
    }
    base.update(overrides)
    return ReportRow(**base)  # type: ignore[arg-type]  # test helper with dynamic fields


def test_build_report_with_all_passing_rows_says_nothing_failed() -> None:
    report = build_report("exp", [make_row(), make_row(example_id="q02")])
    assert "Ninguna pregunta puntuada falló." in report
    assert "Respuestas correctas: 2 de 2 puntuadas" in report


def test_build_report_with_wrong_answer_lists_it_first_with_judge_comment() -> None:
    bad = make_row(
        example_id="q10",
        answer_correct=False,
        answer="3000 USD",
        judge_comment="omite la condición",
    )
    report = build_report("exp", [make_row(), bad])
    failures_part = report.split("## Lo que salió bien")[0]
    assert "### q10" in failures_part
    assert "omite la condición" in failures_part
    assert "Respuestas correctas: 1 de 2 puntuadas" in report


def test_build_report_with_errored_row_counts_it_and_shows_the_error() -> None:
    errored = make_row(
        example_id="q05",
        answer=None,
        answer_correct=None,
        retrieval_hit=None,
        error="GoogleRateLimitError: 429",
    )
    report = build_report("exp", [errored])
    assert "Con error (sin puntuar): 1" in report
    assert "## Errores de infraestructura" in report
    assert "### q05" not in report
    assert "GoogleRateLimitError: 429" in report
    assert "0 de 0 puntuadas" in report


def test_build_report_with_unanswerable_row_marks_search_as_not_applicable() -> None:
    row = make_row(
        example_id="q15",
        kind="unanswerable",
        expected_sources=[],
        retrieval_hit=None,
    )
    report = build_report("exp", [row])
    assert "| q15 | unanswerable |" in report
    assert "n/a" in report


def test_write_report_with_unsafe_name_creates_sanitized_file(tmp_path: Path) -> None:
    path = write_report(tmp_path / "reports", "base/line v2", "hola")
    assert path.name == "base_line_v2.md"
    assert path.read_text(encoding="utf-8") == "hola"
