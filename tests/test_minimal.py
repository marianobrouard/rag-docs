from langchain_core.documents import Document

from rag_docs.minimal import DATA_DIR, format_context, load_documents, split_documents


def test_load_documents_with_data_dir_returns_all_policies() -> None:
    docs = load_documents(DATA_DIR)
    sources = {d.metadata["source"] for d in docs}
    assert {"vacaciones.md", "reintegros.md", "trabajo_remoto.md"} <= sources
    assert len(docs) == 15


def test_split_documents_with_policies_keeps_source_and_limits_size() -> None:
    chunks = split_documents(load_documents(DATA_DIR))
    assert len(chunks) > 3
    assert all(c.metadata["source"].endswith(".md") for c in chunks)
    assert all(len(c.page_content) <= 500 for c in chunks)


def test_format_context_with_one_chunk_wraps_it_in_xml_tags() -> None:
    text = format_context([Document(page_content="hola", metadata={"source": "a.md"})])
    assert text.startswith("<documents>")
    assert '<doc source="a.md">' in text
