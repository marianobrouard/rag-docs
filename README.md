# rag-docs

A small, measured RAG (retrieval-augmented generation) assistant that answers questions about
company policies and cites the source file. Built to learn the full loop of an LLM application:
**build it, evaluate it, find out why it fails, and fix it with evidence.**

The documents in `data/` belong to a fictional company ("Café Andino") and were written for this
project. No real data is included.

## How it works

```
question ──► (optional) split into sub-questions ──► vector search (top-k chunks per query)
                                                              │
        answer + source files  ◄── LLM reads ONLY the chunks ◄┘   (chunks wrapped as untrusted data)
```

1. **Load and split** the Markdown files into ~500-character chunks.
2. **Embed** chunks (Gemini embeddings) into an in-memory vector store.
3. **Retrieve** the `k` closest chunks. With `--decompose`, the question is first split into up to
   3 independent sub-questions and each one gets its own search; results are merged without duplicates.
4. **Answer** with an LLM instructed to use only the retrieved text, to include conditions and
   exceptions, and to say "I don't know" otherwise.

Built with LangChain (splitting, vector store, model wrappers) and LangSmith (tracing and
experiments). Pydantic validates every structured LLM output (judge verdicts, sub-questions).

## Quickstart

```bash
uv sync
cp .env.example .env          # then fill in the keys; never commit .env
uv run python -m rag_docs.minimal "¿Cuántos días de vacaciones tengo con 12 años de antigüedad?"
uv run python -m rag_docs.minimal "..." --k 1 --decompose
```

Needs Python 3.12+ (managed by `uv`) and a Google AI Studio API key. A LangSmith key is optional
for plain questions and required for evaluations.

## Evaluation

`evals/dataset.json` holds 29 questions with the expected answer and the expected source files:
simple lookups, paraphrases, contradictions between policy versions, calculations, multi-document
questions, partial information, unanswerable questions, a prompt-injection attempt and
questions with typos.

```bash
uv run python -m rag_docs.evaluate --prefix my-run --pause 20 --k 3 [--decompose]
```

Two scores per question:

| Metric | How it is computed |
|---|---|
| `retrieval_hit` | Plain code: were all expected source files among the retrieved chunks? (n/a for unanswerable questions) |
| `answer_correct` | An LLM judge compares the answer with the expected one and explains its verdict |

Each run creates a LangSmith experiment **and** a readable Markdown report in `evals/reports/`
(git-ignored) with failures first, and infrastructure errors listed separately so that a
rate-limit error is never confused with a quality failure.

## Results so far

| Run | Dataset | `k` | Answers correct | Retrieval correct | Notes |
|---|---|---|---|---|---|
| baseline | 16 questions | 3 | 16 / 16 | 14 / 14 | |
| break test | 16 questions | 1 | 15 / 16 | 13 / 14 | only the two-topic question (q14) failed |
| current | 29 questions | 3 | 27 / 27 scored | 23 / 24 | 2 questions unscored (Google 503); typo questions all passed |

**Findings**

- Every real quality failure was a *retrieval* failure on a multi-topic question: the topics
  compete for the `k` slots, the needed document is not retrieved, and the model honestly answers
  "I don't know" instead of inventing. Reproduced in independent runs (q14 with `k=1`, q19 with `k=3`).
- Semantic search tolerated typos and WhatsApp-style abbreviations in all three typo questions.
- Operational reliability mattered more than answer quality: the free tier of the model API returned
  429 and 503 errors in most runs. The evaluation harness therefore reports them separately.

## Limitations (read before trusting the numbers)

- Small dataset, written by the same author as the documents; the corpus is fictional and tiny
  (everything would also fit in a single prompt, so RAG is not strictly necessary here).
- **The judge is the same model family as the assistant**, which tends to be lenient.
- **Question decomposition (`--decompose`) is implemented and unit-tested, but its effect has not
  been measured yet.**
- Free-tier model names and limits change; the model IDs are configuration, not code.

## Possible next steps

Verified citations (check each quoted span against the retrieved text), a persistent vector store,
PDF ingestion, a different (stronger) model as judge, and a fallback model for provider outages.

## Project layout

```
src/rag_docs/
  minimal.py    load, split, embed, retrieve, answer (and the CLI)
  planner.py    splits a question into sub-questions (validated, falls back to the original)
  evaluate.py   runs the dataset through the RAG and logs the experiment to LangSmith
  report.py     builds the readable Markdown report (pure functions)
  config.py     settings from environment variables
data/           fictional policy documents
evals/          dataset.json (questions and expected answers)
tests/          offline unit tests (no network, no API keys)
```

## Development

```bash
uv run ruff format . && uv run ruff check .
uv run mypy --strict src tests
uv run pytest
```

## Security notes

- Secrets only through environment variables; `.env` is git-ignored and `.env.example` has no values.
- Retrieved documents and user questions are passed to the model inside XML tags and labelled as data,
  never as instructions. Judge and planner treat their input the same way.
- Every network call has an explicit timeout and bounded retries. Tracing records only the question,
  not the vector store or the settings.
