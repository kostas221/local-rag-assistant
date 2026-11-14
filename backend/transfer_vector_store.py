"""Export with the OLD Chroma runtime; import with the NEW one into a new directory.

Stop writes before export. The JSON contains private document text and embeddings.
No models or network calls are needed. Existing import targets are never reused.
"""
import argparse
import json
from pathlib import Path

import chromadb


def export_store(source: Path, output: Path, collection_name: str):
    if not (source / "chroma.sqlite3").is_file():
        raise ValueError("Source database does not exist")
    if output.exists():
        raise ValueError("Export file already exists")
    client = chromadb.PersistentClient(path=str(source))
    collection = client.get_collection(collection_name, embedding_function=None)
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    embeddings = data["embeddings"]
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    payload = {"format": 1, "chroma_version": chromadb.__version__,
               "collection": collection_name, "metadata": collection.metadata,
               "ids": data["ids"], "documents": data["documents"],
               "metadatas": data["metadatas"], "embeddings": embeddings}
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    return len(payload["ids"])


def import_store(source: Path, destination: Path):
    if destination.exists():
        raise ValueError("Destination must be a NEW directory")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("format") != 1:
        raise ValueError("Unsupported export format")
    keys = ("ids", "documents", "metadatas", "embeddings")
    count = len(payload["ids"])
    if any(len(payload[key]) != count for key in keys) or len(set(payload["ids"])) != count:
        raise ValueError("Invalid export: counts or IDs do not match")
    client = chromadb.PersistentClient(path=str(destination))
    collection = client.create_collection(payload["collection"],
                                          metadata=payload["metadata"], embedding_function=None)
    for start in range(0, count, 100):
        collection.add(**{key: payload[key][start:start + 100] for key in keys})
    actual = collection.get(include=["documents", "metadatas", "embeddings"])
    expected_by_id = {cid: i for i, cid in enumerate(payload["ids"])}
    if collection.count() != count or set(actual["ids"]) != set(expected_by_id):
        raise RuntimeError("Import verification failed: IDs differ")
    for i, cid in enumerate(actual["ids"]):
        j = expected_by_id[cid]
        for key in ("documents", "metadatas"):
            if actual[key][i] != payload[key][j]:
                raise RuntimeError(f"Import verification failed: {key} differ")
        if list(actual["embeddings"][i]) != payload["embeddings"][j]:
            raise RuntimeError("Import verification failed: embeddings differ")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--collection", default="ai_research_docs")
    restore = commands.add_parser("import")
    restore.add_argument("--source", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        count = export_store(args.source, args.output, args.collection)
    else:
        count = import_store(args.source, args.destination)
    print(f"{args.command}: {count} records verified")


if __name__ == "__main__":
    main()
