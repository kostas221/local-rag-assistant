"""Safe migration preserves private metadata, IDs and embeddings."""
import sqlite3

import chromadb
import pytest

import transfer_vector_store
import vector_store


def test_legacy_database_is_rejected_before_chroma_can_migrate_it(tmp_path, monkeypatch):
    database = tmp_path / "chroma.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE collections (id TEXT, topic TEXT)")
    original = database.read_bytes()
    monkeypatch.setattr(chromadb, "__version__", "1.5.9")
    with pytest.raises(RuntimeError, match="Legacy Chroma"):
        vector_store.persistent_client(str(tmp_path))
    assert database.read_bytes() == original


def test_export_import_preserves_all_data_and_refuses_existing_destination(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "export.json"
    destination = tmp_path / "destination"
    client = chromadb.PersistentClient(path=str(source))
    collection = client.create_collection("ai_research_docs", embedding_function=None,
                                          metadata={"hnsw:space": "cosine"})
    collection.add(ids=["one", "two"], documents=["αρχείο", "other document"],
                   embeddings=[[0.5, 1.0], [1.0, 0.25]], metadatas=[
                       {"doc_id": 10, "user_id": 7, "is_public": False},
                       {"doc_id": 20, "user_id": 8, "is_public": True},
                   ])
    assert transfer_vector_store.export_store(source, output, "ai_research_docs") == 2
    assert transfer_vector_store.import_store(output, destination) == 2
    with pytest.raises(ValueError, match="NEW directory"):
        transfer_vector_store.import_store(output, destination)
    with pytest.raises(ValueError, match="already exists"):
        transfer_vector_store.export_store(source, output, "ai_research_docs")
