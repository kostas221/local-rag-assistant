"""Πόσα chunks του ευρετηρίου ΔΕΝ έχουν doc_id (legacy μονοπάτι);

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Το `_expand_to_pages` κάνει ΔΕΥΤΕΡΟ collection.get() για να φέρει ολόκληρη τη
σελίδα. Αυτό το query ΔΕΝ κληρονομεί το where φίλτρο του χρήστη — είναι νέο.
  · doc_id != -1  -> το φίλτρο είναι εγγενώς ασφαλές (μοναδικό ανά έγγραφο)
  · doc_id == -1  -> πέφτει σε {file_name, page}, και το file_name ΔΕΝ είναι
                     μοναδικό μεταξύ χρηστών: δύο άνθρωποι με ομώνυμο ιδιωτικό
                     PDF παίρνουν ο ένας τις σελίδες του άλλου.
Αυτό το script λέει αν το δεύτερο μονοπάτι είναι νεκρός κώδικας ή ενεργό ρίσκο.

ΜΗΔΕΝ μοντέλα, ΜΗΔΕΝ embeddings — μόνο metadata. Τρέχει σε ~5 s.

    docker compose exec backend python evaluation/count_legacy_chunks.py
"""
import os
import sys
from collections import Counter

import chromadb

db_path = os.getenv("VECTOR_DB_PATH", "/app/vector_db")


def main() -> int:
    client = chromadb.PersistentClient(path=db_path)
    try:
        col = client.get_collection(name="ai_research_docs")
    except Exception as e:
        print(f"Δεν βρέθηκε το collection στο {db_path}: {e}")
        return 1

    got = col.get(include=["metadatas"])
    metas = got["metadatas"] or []
    total = len(metas)
    if not total:
        print("Το ευρετήριο είναι ΑΔΕΙΟ.")
        return 1

    legacy = [m for m in metas if m.get("doc_id") in (None, -1)]
    pct = 100.0 * len(legacy) / total
    print(f"chunks σύνολο           : {total}")
    print(f"legacy (doc_id -1/None) : {len(legacy)}  ({pct:.1f}%)")

    if legacy:
        print("\nΑΡΧΕΙΑ ΣΤΟ LEGACY ΜΟΝΟΠΑΤΙ (file_name -> chunks, users):")
        per_file = Counter(m.get("file_name") for m in legacy)
        for fname, n in per_file.most_common():
            users = sorted({m.get("user_id") for m in legacy
                            if m.get("file_name") == fname},
                           key=lambda x: (x is None, x))
            flag = "  <-- ΙΔΙΟ ΟΝΟΜΑ ΣΕ >1 ΧΡΗΣΤΗ" if len(users) > 1 else ""
            short = str(fname)[:48]
            print(f"  {short:<48} {n:4d} chunks · users {users}{flag}")
        print("\nΤο legacy μονοπάτι ΕΙΝΑΙ ενεργό -> το φίλτρο χρήστη χρειάζεται.")
    else:
        print("\nΚΑΝΕΝΑ legacy chunk: το μονοπάτι είναι σήμερα ΝΕΚΡΟΣ ΚΩΔΙΚΑΣ.")
        print("Η διόρθωση παραμένει σωστή (ένα παλιό backup restore το ξυπνά),")
        print("αλλά ΔΕΝ αλλάζει ούτε ένα αποτέλεσμα σήμερα — άρα η επαλήθευση")
        print("είναι «τα πάντα ταυτόσημα», όχι «κάτι βελτιώθηκε».")
    return 0


if __name__ == "__main__":
    sys.exit(main())
