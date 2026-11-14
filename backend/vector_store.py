"""Open persistent stores without silently upgrading legacy production data."""
import sqlite3
from pathlib import Path

import chromadb


def persistent_client(path: str):
    database = Path(path).resolve() / "chroma.sqlite3"
    if database.exists() and int(chromadb.__version__.split(".")[0]) >= 1:
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(collections)")}
        if "topic" in columns:
            raise RuntimeError(
                "Legacy Chroma database detected. Export with the old runtime and import "
                "into a NEW directory using transfer_vector_store.py. "
                "See docs/STABILIZATION_2026-11-01.md."
            )
    return chromadb.PersistentClient(path=path)
