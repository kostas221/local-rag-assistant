"""Ελεγχόμενο re-ingest ΟΛΟΥ του corpus, με επαλήθευση σε κάθε βήμα.

ΓΙΑΤΙ ΞΕΧΩΡΙΣΤΟ SCRIPT ΚΑΙ ΟΧΙ ΤΟ UI: το delete+upload από το UI είναι δύο
ανεξάρτητα HTTP requests. Αν το δεύτερο ξεκινήσει πριν τελειώσει το πρώτο,
μένουν ΟΡΦΑΝΑ chunks — αόρατα στο UI αλλά ενεργά στο retrieval. Έτσι προέκυψε
η ασυμφωνία "386 vs 193 chunks" του v1. Εδώ όλα γίνονται σειριακά, σε ΜΙΑ
διεργασία, και κάθε βήμα επαληθεύεται πριν προχωρήσει το επόμενο.

ΣΗΜΑΝΤΙΚΟ: τρέχει σε ΑΛΛΗ διεργασία από το uvicorn. Το _corpus_version είναι
μεταβλητή μνήμης ανά διεργασία, οπότε ο ζωντανός server ΔΕΝ θα μάθει ποτέ ότι
άλλαξε το corpus και θα κρατήσει το παλιό BM25 cache + dense matrix.
    -> ΜΕΤΑ το script, ΥΠΟΧΡΕΩΤΙΚΑ: docker compose restart backend

Χρήση:
    python reingest_corpus.py --dry-run    # δείχνει τι θα κάνει, δεν αγγίζει τίποτα
    python reingest_corpus.py              # το κάνει
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, "/app")
import ai_core

PDF_DIR = "/app/uploaded_docs"


def snapshot():
    """Τρέχουσα κατάσταση: chunks ανά αρχείο + τα metadata τους (τα χρειαζόμαστε
    για να ξανακάνουμε ingest με ΤΟΝ ΙΔΙΟ ιδιοκτήτη/doc_id)."""
    data = ai_core.collection.get(include=["metadatas"])
    per_file = {}
    for _cid, m in zip(data["ids"], data["metadatas"]):
        f = m.get("file_name")
        e = per_file.setdefault(f, {"n": 0, "user_id": m.get("user_id"),
                                    "is_public": m.get("is_public"),
                                    "doc_id": m.get("doc_id"), "pages": set()})
        e["n"] += 1
        e["pages"].add(m.get("page"))
    return per_file, len(data["ids"])


def index_health():
    """store == index; αν διαφέρουν, λείπουν διανύσματα από το HNSW."""
    total = len(ai_core.collection.get(include=[])["ids"])
    if total == 0:
        return 0, 0
    r = ai_core.collection.query(query_texts=["cloud computing"],
                                 n_results=total)
    return total, len(r["ids"][0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    before, total_before = snapshot()
    print("=" * 74)
    print("ΠΡΙΝ")
    print("=" * 74)
    for f, e in sorted(before.items()):
        print(f"  {f:<26} {e['n']:>4} chunks | {len(e['pages']):>3} σελίδες "
              f"| user={e['user_id']} doc_id={e['doc_id']} pub={e['is_public']}")
    s, i = index_health()
    print(f"  ΣΥΝΟΛΟ {total_before} chunks | store={s} index={i} "
          f"{'OK' if s == i else '*** ΛΕΙΠΟΥΝ ' + str(s - i) + ' ***'}")

    # Αντιστοίχιση file_name -> πραγματική διαδρομή στον δίσκο
    paths = {}
    for p in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
        paths[os.path.basename(p).split("_", 1)[-1]] = p

    missing = [f for f in before if f not in paths]
    if missing:
        print(f"\n*** ΣΤΑΜΑΤΩ: δεν βρέθηκε PDF για {missing}. Re-ingest θα "
              f"έσβηνε chunks που δεν μπορούν να ξαναχτιστούν. ***")
        return 1

    if args.dry_run:
        print(f"\n[dry-run] Θα γινόταν delete + ingest σε {len(before)} αρχεία.")
        return 0

    print("\n" + "=" * 74)
    print("RE-INGEST (σειριακά, ένα αρχείο τη φορά)")
    print("=" * 74)
    for f, e in sorted(before.items()):
        ai_core.delete_file_from_db(f, user_id=e["user_id"],
                                    doc_id=e["doc_id"] if e["doc_id"] != -1 else None)
        left = len([m for m in ai_core.collection.get(include=["metadatas"])["metadatas"]
                    if m.get("file_name") == f])
        if left:
            print(f"  *** ΣΤΑΜΑΤΩ: μετά το delete έμειναν {left} chunks του {f}. "
                  f"Το delete_file_from_db καταπίνει εξαιρέσεις — δες τα logs. ***")
            return 1
        ai_core.ingest_pdf(paths[f], f, e["user_id"],
                           is_public=bool(e["is_public"]),
                           doc_id=e["doc_id"] if e["doc_id"] != -1 else None)
        print(f"  {f:<26} OK", flush=True)

    after, total_after = snapshot()
    s, i = index_health()
    print("\n" + "=" * 74)
    print("ΜΕΤΑ")
    print("=" * 74)
    print(f"{'αρχείο':<26} {'πριν':>6} {'μετά':>6} {'Δ':>7}  σελίδες")
    for f in sorted(set(before) | set(after)):
        b = before.get(f, {"n": 0, "pages": set()})
        a = after.get(f, {"n": 0, "pages": set()})
        print(f"{f:<26} {b['n']:>6} {a['n']:>6} {a['n']-b['n']:>+7}  "
              f"{len(b['pages'])} -> {len(a['pages'])}")
    print(f"\nΣΥΝΟΛΟ {total_before} -> {total_after} ({total_after-total_before:+d})")
    print(f"Υγεία ευρετηρίου: store={s} index={i} "
          f"{'OK' if s == i else '*** ΛΕΙΠΟΥΝ ' + str(s - i) + ' ***'}")
    print("\n>>> ΤΩΡΑ ΥΠΟΧΡΕΩΤΙΚΑ: docker compose restart backend")
    print(">>> (ο ζωντανός server κρατά ακόμα το ΠΑΛΙΟ BM25 cache στη μνήμη του)")
    return 0 if s == i else 1


if __name__ == "__main__":
    sys.exit(main())
