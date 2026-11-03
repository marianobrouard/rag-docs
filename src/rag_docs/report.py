"""Turn evaluation rows into a plain Markdown report you can read (or paste) anywhere."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportRow:
    example_id: str
    kind: str
    question: str
    expected_answer: str
    answer: str | None
    expected_sources: list[str]
    retrieved_sources: list[str]
    answer_correct: bool | None
    retrieval_hit: bool | None
    judge_comment: str
    error: str | None


def _is_failure(row: ReportRow) -> bool:
    """A quality failure. Rows with an infrastructure error are reported separately."""
    return row.error is None and (
        row.answer_correct is False or row.retrieval_hit is False
    )


def _problem(row: ReportRow) -> str | None:
    """Why a row could not be fully scored: the RAG call failed, or the judge did."""
    if row.error is not None:
        return row.error
    if row.answer_correct is None:
        return f"el juez no pudo corregir: {row.judge_comment or 'sin detalle'}"
    return None


def _error_block(rows: Sequence[ReportRow]) -> list[str]:
    by_message: dict[str, list[str]] = {}
    for row in rows:
        by_message.setdefault(str(_problem(row))[:200], []).append(row.example_id)
    lines = [
        "## Errores de infraestructura (sin puntuar, no son fallos de calidad)",
        "",
    ]
    for message, ids in by_message.items():
        lines.append(f"- {', '.join(ids)}: {message}")
    return lines + [""]


def _mark(value: bool | None) -> str:
    return "n/a" if value is None else ("✅" if value else "❌")


def _summary(rows: Sequence[ReportRow]) -> list[str]:
    judged = [r for r in rows if r.answer_correct is not None]
    searched = [r for r in rows if r.retrieval_hit is not None]
    errored = [r for r in rows if _problem(r) is not None]
    return [
        f"- Preguntas en total: {len(rows)}",
        f"- Respuestas correctas: {sum(bool(r.answer_correct) for r in judged)} de {len(judged)} puntuadas",
        f"- Búsqueda correcta: {sum(bool(r.retrieval_hit) for r in searched)} de {len(searched)} aplicables",
        f"- Con error (sin puntuar): {len(errored)}",
    ]


def _judge_text(row: ReportRow) -> str:
    if row.answer_correct is None:
        return "(el juez no pudo corregir esta respuesta; ver errores más abajo)"
    return row.judge_comment or "(sin comentario)"


def _failure_block(row: ReportRow) -> list[str]:
    lines = [f"### {row.example_id} ({row.kind}) ❌", f"- **Pregunta:** {row.question}"]
    lines += [
        f"- **Esperado:** {row.expected_answer}",
        f"- **Respondió:** {(row.answer or '').strip()}",
        f"- **Archivos esperados:** {', '.join(row.expected_sources) or '(ninguno)'}",
        f"- **Archivos recuperados:** {', '.join(row.retrieved_sources)}",
        f"- **Búsqueda:** {_mark(row.retrieval_hit)}  |  **Respuesta:** {_mark(row.answer_correct)}",
        f"- **Comentario del juez:** {_judge_text(row)}",
        "",
    ]
    return lines


def build_report(
    experiment_name: str, rows: Sequence[ReportRow], url: str | None = None
) -> str:
    ordered = sorted(rows, key=lambda r: r.example_id)
    errors = [r for r in ordered if _problem(r) is not None]
    failures = [r for r in ordered if _is_failure(r)]
    passed = [r for r in ordered if _problem(r) is None and not _is_failure(r)]

    lines = [f"# Informe: {experiment_name}", ""]
    if url:
        lines += [f"LangSmith: {url}", ""]
    lines += [
        "## Resumen",
        *_summary(ordered),
        "",
        "## Lo que falló (mira esto primero)",
    ]
    if failures:
        lines.append("")
        for row in failures:
            lines += _failure_block(row)
    else:
        lines += ["", "Ninguna pregunta puntuada falló.", ""]
    if errors:
        lines += _error_block(errors)

    lines += [
        "## Lo que salió bien",
        "",
        "| id | tipo | pregunta | búsqueda | respuesta |",
    ]
    lines.append("|---|---|---|---|---|")
    for row in passed:
        lines.append(
            f"| {row.example_id} | {row.kind} | {row.question} | "
            f"{_mark(row.retrieval_hit)} | {_mark(row.answer_correct)} |"
        )
    return "\n".join(lines) + "\n"


def write_report(directory: Path, experiment_name: str, content: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", experiment_name)
    path = directory / f"{safe_name}.md"
    path.write_text(content, encoding="utf-8")
    return path
