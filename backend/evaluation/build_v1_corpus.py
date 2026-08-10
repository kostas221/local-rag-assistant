"""Χτίζει ΚΑΘΑΡΗ, ΞΕΧΩΡΙΣΤΗ βάση με ΜΟΝΟ τα 2 Berkeley papers του v1.

ΓΙΑΤΙ ΞΕΧΩΡΙΣΤΗ ΒΑΣΗ ΚΑΙ ΟΧΙ EVAL_CORPUS=v1:
Το `EVAL_CORPUS=v1` περιορίζει το `where` φίλτρο στα 2 papers — αλλά το BM25
index και ο dense πίνακας χτίζονται πάνω σε ΟΛΟ το corpus (418 chunks). Το IDF
του BM25 άρα υπολογίζεται από 7 papers, όχι από 2. Τα νούμερα ΔΕΝ θα ήταν
συγκρίσιμα με ό,τι είναι ήδη γραμμένο στη διπλωματική.

ΓΙΑΤΙ ΕΠΕΙΓΕΙ (μετρημένο 10/8/2026):
Το RESULTS.md του v1 γράφει **386 chunks** για chunk_size=1500. Τα ίδια δύο
papers στο v2 δίνουν **191**. Λόγος 2.02 -> το αρχικό ingest είχε τρέξει ΔΥΟ
ΦΟΡΕΣ και άφησε ορφανά διπλότυπα. Επειδή το κλειδί σελίδας περιέχει doc_id, τα
δύο αντίγραφα μετρούσαν ως ΞΕΧΩΡΙΣΤΕΣ σελίδες -> με MAX_PAGES=8 το σύστημα
έστελνε 4 πραγματικές σελίδες σε δύο αντίτυπα. Δηλαδή ΜΙΣΟ context. Αυτό
εξηγεί και το «MRR variance 0.81-0.86» που το RESULTS.md καταγράφει χωρίς αιτία:
δύο ταυτόσημα chunks παίρνουν ίδιο σκορ και η σειρά τους είναι αυθαίρετη.

ΤΟ SCRIPT ΕΙΝΑΙ ΑΣΦΑΛΕΣ: γράφει ΜΟΝΟ στο VECTOR_DB_PATH που του δίνεις. Αν
δείχνει στην κύρια βάση, ΣΤΑΜΑΤΑΕΙ.

    docker compose exec -e VECTOR_DB_PATH=./vector_db_v1 backend \
        python evaluation/build_v1_corpus.py
"""
import collections
import glob
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

V1_PAPERS = {
    "EECS-2009-28.pdf": dict(doc_id=12, user_id=1, is_public=False),
    "1902.03383v1.pdf": dict(doc_id=13, user_id=1, is_public=False),
}
PDF_DIR = "/app/uploaded_docs"
MAIN_DB = "./vector_db"


def main() -> int:
    db = os.getenv("VECTOR_DB_PATH", MAIN_DB)
    if os.path.abspath(db) == os.path.abspath(MAIN_DB):
        print("*** ΣΤΑΜΑΤΩ: το VECTOR_DB_PATH δείχνει στην ΚΥΡΙΑ βάση.\n"
              "    Τρέξε με -e VECTOR_DB_PATH=./vector_db_v1")
        return 1
    print(f"Βάση προορισμού: {db}\n")

    import ai_core

    existing = ai_core.collection.get(include=["metadatas"])["metadatas"]
    if existing:
        cnt = collections.Counter(m["file_name"] for m in existing)
        print(f"*** ΣΤΑΜΑΤΩ: η βάση έχει ήδη {len(existing)} chunks: {dict(cnt)}\n"
              f"    Σβήσε τον φάκελο {db} και ξανατρέξε — ΜΗΝ κάνεις ingest από\n"
              f"    πάνω, αυτό ακριβώς παρήγαγε τα 386 διπλότυπα του v1.")
        return 1

    paths = {}
    for p in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
        paths[os.path.basename(p).split("_", 1)[-1]] = p
    missing = [f for f in V1_PAPERS if f not in paths]
    if missing:
        print(f"*** ΣΤΑΜΑΤΩ: δεν βρέθηκαν PDF: {missing}")
        return 1

    for fname, meta in V1_PAPERS.items():
        print(f"ingest {fname} ...", flush=True)
        ok = ai_core.ingest_pdf(paths[fname], fname, user_id=meta["user_id"],
                                is_public=meta["is_public"],
                                doc_id=meta["doc_id"])
        if not ok:
            print(f"*** ΑΠΕΤΥΧΕ: {fname}")
            return 1

    metas = ai_core.collection.get(include=["metadatas"])["metadatas"]
    ids = ai_core.collection.get(include=[])["ids"]
    per_file = collections.Counter(m["file_name"] for m in metas)
    pages = collections.defaultdict(set)
    for m in metas:
        pages[m["file_name"]].add(m["page"])

    print("\n" + "=" * 70)
    print("ΑΠΟΤΕΛΕΣΜΑ")
    print("=" * 70)
    for f in sorted(per_file):
        print(f"  {f:<24} {per_file[f]:>4} chunks · {len(pages[f]):>2} σελίδες")
    print(f"  {'ΣΥΝΟΛΟ':<24} {len(ids):>4} chunks")

    # Έλεγχος διπλοτύπων: δύο ingests δίνουν ίδιο (file,page,chunk_idx) με
    # διαφορετικό id. Ακριβώς αυτό κρύφτηκε στο v1.
    key = collections.Counter((m["file_name"], m["page"],
                               ai_core._chunk_idx_from_id(i))
                              for i, m in zip(ids, metas))
    dupes = {k: v for k, v in key.items() if v > 1}
    print(f"\n  ΔΙΠΛΟΤΥΠΑ: {len(dupes)}"
          + ("  <-- ΠΡΟΒΛΗΜΑ" if dupes else "  (κανένα — καθαρό ingest)"))

    print(f"\n  Το v1 RESULTS.md γράφει 386 chunks. Εδώ: {len(ids)}.")
    if len(ids) < 300:
        print("  -> ΕΠΙΒΕΒΑΙΩΝΕΤΑΙ ότι τα 386 ήταν διπλότυπα.")

    print("\nΕΠΟΜΕΝΟ — αντίγραψε το translation cache για να ΜΗΝ ξαναπληρώσεις")
    print("μεταφράσεις (το cache έχει key την ΕΡΩΤΗΣΗ, είναι ανεξάρτητο corpus):")
    print(f"  docker compose exec backend sh -c 'cp {MAIN_DB}/_translation_cache.json "
          f"{db}/ 2>/dev/null || true'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
