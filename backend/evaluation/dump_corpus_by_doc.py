"""Εξάγει το κείμενο του corpus ΑΝΑ ΕΓΓΡΑΦΟ σε ξεχωριστά .txt — βοήθημα authoring.

Χρησιμοποιείται για να συνταχθούν grounded multi_hop ερωτήσεις: για να ξέρω τι
πραγματικά λέει κάθε paper (όχι από μνήμη) και να διαλέξω keywords που ΥΠΑΡΧΟΥΝ
στο κείμενο και είναι localized σε ένα έγγραφο.

Read-only στη ChromaDB, μηδέν μοντέλα/API. Τρέχει σε δευτερόλεπτα.

Χρήση (μέσα στο backend container):
    python evaluation/dump_corpus_by_doc.py evaluation/corpus_dump
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import chromadb

COLLECTION_NAME = "ai_research_docs"


def _page_num(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evaluation/corpus_dump")
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []

    # ομαδοποίηση: doc_id -> file_name -> page -> [chunks]
    by_doc = defaultdict(lambda: defaultdict(list))
    file_of = {}
    for doc, meta in zip(docs, metas):
        meta = meta or {}
        did = meta.get("doc_id", "?")
        fn = str(meta.get("file_name", "?"))
        file_of[did] = fn
        by_doc[did][_page_num(meta.get("page"))].append(doc)

    index_lines = []
    for i, did in enumerate(sorted(by_doc), start=1):
        fn = file_of[did]
        pages = by_doc[did]
        safe = f"doc{i:02d}_" + Path(fn).stem[:40].replace(os.sep, "_")
        path = out_dir / f"{safe}.txt"
        with path.open("w", encoding="utf-8") as f:
            f.write(f"FILE: {fn}\nDOC_ID: {did}\nPAGES: {len(pages)}\n")
            f.write("=" * 78 + "\n")
            for pg in sorted(pages):
                f.write(f"\n----- p.{pg} -----\n")
                f.write("\n".join(pages[pg]))
                f.write("\n")
        first_page = pages[min(pages)]
        title = " ".join("".join(first_page).split())[:110]
        index_lines.append(f"{path.name}  ({len(pages)} σελ.)  {title}")

    print(f"Έγραψα {len(by_doc)} έγγραφα στο {out_dir}/\n")
    for line in index_lines:
        print(" ", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
