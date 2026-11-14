"""Real PDF extraction, embeddings and Chroma, without any generation calls."""
import chromadb
import pymupdf

import ai_core


def test_pdf_ingestion_preserves_identity_and_owner_with_duplicate_filenames(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path / "vectors"))
    store = client.create_collection(
        "pdf_ingestion", embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"},
    )
    monkeypatch.setattr(ai_core, "collection", store)
    monkeypatch.setattr(ai_core, "_corpus_version", 0)
    monkeypatch.setattr(ai_core, "MAX_PAGES_PER_FILE", 1)
    texts = ["Helios processes scientific documents using local semantic retrieval.",
             "Selene measures rainfall using sensors in coastal weather stations."]
    for index, text in enumerate(texts, 1):
        pdf = tmp_path / f"source-{index}.pdf"
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), text)
            document.save(pdf)
        assert ai_core.ingest_pdf(str(pdf), "paper.pdf", user_id=index, doc_id=index)

    data = store.get(include=["documents", "metadatas", "embeddings"])
    assert len(data["ids"]) == 2
    assert len(data["embeddings"][0]) == ai_core.EMBED_DIM
    visible = store.get(where=ai_core._build_where(user_id=1), include=[])
    assert len(visible["ids"]) == 1
    top_chunks = [(1.0, text, meta)
                  for text, meta in zip(data["documents"], data["metadatas"])]
    pages = ai_core._expand_to_pages(top_chunks, max_pages=2)
    assert {meta["doc_id"] for _text, meta in pages} == {1, 2}
    assert {text.strip() for text, _meta in pages} == set(texts)

    ai_core.delete_file_from_db("paper.pdf", user_id=1, doc_id=1)
    remaining = store.get(include=["metadatas"])
    assert len(remaining["ids"]) == 1
    assert remaining["metadatas"][0]["doc_id"] == 2
