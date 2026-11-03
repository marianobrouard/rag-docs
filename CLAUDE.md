# rag-docs

RAG assistant over fictional company policies, with an evaluation harness (LangSmith).
Python 3.12+, managed with `uv`. Code, comments and commits in English.

## Commands
- Install: `uv sync`
- Format and lint: `uv run ruff format . && uv run ruff check .`
- Types: `uv run mypy --strict src tests`
- Tests (offline, no API keys): `uv run pytest`
- Ask one question: `uv run python -m rag_docs.minimal "question" [--k N] [--decompose]`
- Run the exam: `uv run python -m rag_docs.evaluate --prefix NAME --pause 20 [--k N] [--decompose]`

## Rules for this project
- Never read, print or write `.env`. Keys go in the environment; `.env.example` has no values.
- Unit tests must not call any LLM or network. Only the exam (`evaluate`) uses the real APIs.
- Run exams one at a time: parallel runs trigger free-tier 429/503 errors.
- `evals/dataset.json` sync to LangSmith is add-only by question id. Editing an existing question
  does not update it there; create a new dataset name instead.
- A report row that failed with a 503 or a judge error is NOT a quality failure; keep them separate.
- Model IDs live in configuration (`.env`), never hard-coded: they change often.
