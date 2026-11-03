from rag_docs.report import ReportRow, build_report


def judge_failed_row() -> ReportRow:
    return ReportRow(
        example_id="q19",
        kind="multi_doc_3",
        question="una pregunta",
        expected_answer="algo",
        answer="una respuesta",
        expected_sources=["a.md", "b.md"],
        retrieved_sources=["a.md"],
        answer_correct=None,
        retrieval_hit=False,
        judge_comment="GoogleAPIError('503 UNAVAILABLE')",
        error=None,
    )


def test_build_report_with_judge_failure_counts_it_as_unscored_and_lists_the_cause() -> (
    None
):
    report = build_report("exp", [judge_failed_row()])
    assert "Con error (sin puntuar): 1" in report
    assert "el juez no pudo corregir: GoogleAPIError" in report


def test_build_report_with_judge_failure_does_not_show_the_error_as_judge_opinion() -> (
    None
):
    report = build_report("exp", [judge_failed_row()])
    failures_part = report.split("## Errores de infraestructura")[0]
    assert "### q19" in failures_part  # the search failure is still a quality failure
    assert "el juez no pudo corregir esta respuesta" in failures_part
    assert "503 UNAVAILABLE" not in failures_part
