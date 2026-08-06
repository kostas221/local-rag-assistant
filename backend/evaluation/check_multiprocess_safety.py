"""Αντέχει η ChromaDB 0.4.6 δεύτερη διεργασία στο ΙΔΙΟ persistent directory;

ΓΙΑΤΙ ΠΡΩΤΑ ΑΥΤΟ: το README αναφέρει ως #1 περιορισμό ότι το σύστημα τρέχει με
ΕΝΑΝ worker, επειδή τα caches (BM25 index, dense matrix) και ο μετρητής
_corpus_version ζουν μέσα στη διεργασία. Η προφανής διόρθωση είναι να μετακομίσει
ο μετρητής σε κοινό σημείο (Postgres ή αρχείο).

ΑΛΛΑ: αν το ίδιο το store δεν αντέχει δεύτερη διεργασία, η διόρθωση του μετρητή
δεν ξεκλειδώνει τίποτα — απλώς μετακινεί το πρόβλημα ένα επίπεδο πιο κάτω, και
χειρότερα, το κρύβει. Το ChromaDB 0.4.6 PersistentClient κρατά SQLite + αρχεία
index στον δίσκο· η τεκμηρίωσή του δεν εγγυάται multi-process πρόσβαση.

ΤΙ ΔΟΚΙΜΑΖΕΤΑΙ (σε ΑΝΤΙΓΡΑΦΟ του store, ποτέ στο παραγωγικό):
  1. Δύο διεργασίες ανοίγουν τον ίδιο φάκελο ταυτόχρονα και ΔΙΑΒΑΖΟΥΝ.
  2. Η μία γράφει (add) ενώ η άλλη διαβάζει.
  3. Βλέπει η δεύτερη διεργασία ό,τι έγραψε η πρώτη, ΧΩΡΙΣ restart;

Το (3) είναι το κρίσιμο: αν η απάντηση είναι όχι, τότε με --workers>1 ένα
ingest στον worker A μένει ΑΟΡΑΤΟ στον worker B — ακριβώς το σιωπηλό σφάλμα που
περιγράφει το README, αλλά σε επίπεδο store, όπου κανένας μετρητής δεν το λύνει.

    docker compose exec backend python evaluation/check_multiprocess_safety.py
"""
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time

DIM = 8  # μικρά διανύσματα: μετράμε συμπεριφορά συγχρονισμού, όχι ποιότητα


def _worker_read(path: str, collection_name: str, queue, label: str):
    """Ανοίγει ΝΕΟ client στον ίδιο φάκελο και μετράει τα chunks."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=path)
        col = client.get_or_create_collection(name=collection_name)
        queue.put((label, "ok", len(col.get(include=[])["ids"])))
    except Exception as e:
        queue.put((label, "error", f"{type(e).__name__}: {e}"))


def _worker_write(path: str, collection_name: str, queue, n: int):
    """Γράφει n νέα chunks από ΞΕΧΩΡΙΣΤΗ διεργασία."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=path)
        col = client.get_or_create_collection(name=collection_name)
        col.add(ids=[f"w{i}" for i in range(n)],
                embeddings=[[float(i)] * DIM for i in range(n)],
                documents=[f"written by other process {i}" for i in range(n)],
                metadatas=[{"src": "writer"} for _ in range(n)])
        queue.put(("writer", "ok", len(col.get(include=[])["ids"])))
    except Exception as e:
        queue.put(("writer", "error", f"{type(e).__name__}: {e}"))


def main() -> int:
    import chromadb

    tmp = tempfile.mkdtemp(prefix="chroma_mp_")
    name = "mp_test"
    print(f"Προσωρινό store: {tmp}\n")
    try:
        # --- Αρχικό γέμισμα από ΤΗ ΔΙΚΗ ΜΑΣ διεργασία ------------------------
        client = chromadb.PersistentClient(path=tmp)
        col = client.get_or_create_collection(name=name)
        col.add(ids=[f"a{i}" for i in range(5)],
                embeddings=[[float(i)] * DIM for i in range(5)],
                documents=[f"seed {i}" for i in range(5)],
                metadatas=[{"src": "seed"} for _ in range(5)])
        print(f"1. Αρχικό γέμισμα: {len(col.get(include=[])['ids'])} chunks "
              f"(διεργασία A)")

        ctx = mp.get_context("spawn")  # νέα διεργασία, όχι fork με κληρονομιά
        queue = ctx.Queue()

        # --- ΤΕΣΤ 1: ταυτόχρονη ΑΝΑΓΝΩΣΗ από δεύτερη διεργασία ---------------
        p = ctx.Process(target=_worker_read, args=(tmp, name, queue, "reader"))
        p.start()
        p.join(timeout=120)
        _label, status, value = queue.get(timeout=10)
        print(f"2. Ανάγνωση από διεργασία B ενώ η A κρατά το store: "
              f"{status} -> {value}")
        read_ok = status == "ok" and value == 5

        # --- ΤΕΣΤ 2: ΕΓΓΡΑΦΗ από δεύτερη διεργασία --------------------------
        p = ctx.Process(target=_worker_write, args=(tmp, name, queue, 3))
        p.start()
        p.join(timeout=120)
        _label, status, value = queue.get(timeout=10)
        print(f"3. Εγγραφή 3 chunks από διεργασία B: {status} -> {value}")
        write_ok = status == "ok"

        # --- ΤΕΣΤ 3: ΤΟ ΚΡΙΣΙΜΟ. Τα βλέπει η A χωρίς restart; --------------
        time.sleep(1.0)
        seen_same_client = len(col.get(include=[])["ids"])
        print(f"4. Η διεργασία A (ΙΔΙΟ client object) βλέπει: "
              f"{seen_same_client} chunks  [αναμενόμενο 8]")

        # Και με ΝΕΟ client μέσα στην ίδια διεργασία;
        client2 = chromadb.PersistentClient(path=tmp)
        col2 = client2.get_or_create_collection(name=name)
        seen_new_client = len(col2.get(include=[])["ids"])
        print(f"5. Η διεργασία A με ΝΕΟ client: {seen_new_client} chunks "
              f"[αναμενόμενο 8]")

        # --- Πόρισμα --------------------------------------------------------
        print("\n--- ΠΟΡΙΣΜΑ ---")
        if not read_ok:
            print("• Η ΑΝΑΓΝΩΣΗ από δεύτερη διεργασία ΑΠΕΤΥΧΕ.")
        if not write_ok:
            print("• Η ΕΓΓΡΑΦΗ από δεύτερη διεργασία ΑΠΕΤΥΧΕ.")

        if write_ok and seen_same_client < 8:
            print(f"• ΤΟ ΚΡΙΣΙΜΟ ΕΥΡΗΜΑ: η εγγραφή ΠΕΤΥΧΕ αλλά η άλλη διεργασία "
                  f"ΔΕΝ τη βλέπει ({seen_same_client} αντί για 8) — ΣΙΩΠΗΛΑ, "
                  f"χωρίς σφάλμα.")
            print("  Με --workers>1 ένα ingest στον worker A θα έμενε αόρατο "
                  "στον worker B. ΚΑΝΕΝΑΣ κοινός μετρητής version δεν το λύνει: "
                  "το πρόβλημα είναι στο ίδιο το store, όχι στο cache.")
        elif write_ok and seen_same_client >= 8:
            print("• Η δεύτερη διεργασία έγραψε ΚΑΙ η πρώτη το είδε -> το store "
                  "αντέχει. Τότε ο κοινός μετρητής version ΕΧΕΙ νόημα.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\n(καθαρίστηκε το {tmp})")


if __name__ == "__main__":
    # ΠΟΤΕ στο παραγωγικό store: δουλεύουμε αποκλειστικά σε tempdir.
    os.environ.pop("VECTOR_DB_PATH", None)
    sys.exit(main())
