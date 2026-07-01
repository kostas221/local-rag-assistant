# backend/migrate_add_userid.py
"""ΜΙΑ ΦΟΡΑ: γεμίζει user_id/is_public στα ΠΑΛΙΑ chunks της ChromaDB που δεν τα είχαν,
αντιστοιχίζοντας το file_name με τον ιδιοκτήτη από τον πίνακα Document (SQL).
Τρέξε: docker exec ai_backend python migrate_add_userid.py
"""
from ai_core import collection
import database
import models


def main():
    # 1. Χάρτης file_name -> (user_id, is_public) από τη SQL
    db = database.SessionLocal()
    try:
        owner_map = {
            doc.file_name: (doc.user_id, bool(doc.is_public))
            for doc in db.query(models.Document).all()
        }
    finally:
        db.close()

    print(f"Βρέθηκαν {len(owner_map)} έγγραφα στη SQL.")

    # 2. Όλα τα chunks της Chroma
    data = collection.get()
    ids, metas = data["ids"], data["metadatas"]
    print(f"Βρέθηκαν {len(ids)} chunks στη ChromaDB.")

    upd_ids, upd_metas, skipped, already = [], [], 0, 0
    for cid, meta in zip(ids, metas):
        meta = dict(meta or {})
        if meta.get("user_id") is not None:
            already += 1
            continue  # ήδη ΟΚ
        fname = meta.get("file_name")
        if fname in owner_map:
            uid, is_pub = owner_map[fname]
            meta["user_id"] = uid
            meta["is_public"] = is_pub
            upd_ids.append(cid)
            upd_metas.append(meta)
        else:
            skipped += 1  # orphan: chunk χωρίς αντίστοιχο Document

    # 3. Ενημέρωση σε batches
    BATCH = 500
    for i in range(0, len(upd_ids), BATCH):
        collection.update(ids=upd_ids[i:i + BATCH],
                          metadatas=upd_metas[i:i + BATCH])

    print(f"✅ Ενημερώθηκαν: {len(upd_ids)} | Ήδη ΟΚ: {already} | "
          f"Orphans (skip): {skipped}")


if __name__ == "__main__":
    main()
