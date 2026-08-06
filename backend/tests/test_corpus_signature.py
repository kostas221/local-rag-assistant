"""Ακυρώνεται το cache όταν γράψει ΑΛΛΗ ΔΙΕΡΓΑΣΙΑ; (multi-worker safety)

ΤΙ ΠΡΟΣΤΑΤΕΥΕΙ: ο #1 τεκμηριωμένος περιορισμός του συστήματος ήταν ότι τρέχει
υποχρεωτικά με ΕΝΑΝ worker — τα caches (BM25 index, dense matrix) και ο μετρητής
_corpus_version ζούσαν μέσα στη διεργασία, οπότε ένα ingest στον worker A άφηνε
τον worker B να σερβίρει παλιό index ΣΙΩΠΗΛΑ, χωρίς κανένα σφάλμα.

Η υπογραφή corpus είναι πλέον (τοπικός μετρητής, mtime του store). Εδώ
δοκιμάζεται ΑΚΡΙΒΩΣ το σενάριο που έσπαγε.

ΓΙΑΤΙ ΚΑΝΕΙ import ai_core (και άρα ανήκει στο core-tests job): μια πρώτη
εκδοχή αυτού του αρχείου αντέγραφε τη λογική της υπογραφής για να τρέχει
γρήγορα χωρίς μοντέλα. Θα περνούσε ΚΑΙ με σπασμένο ai_core — δηλαδή θα έδινε
ψεύτικη ασφάλεια σε ακριβώς το σημείο που πονάει. Καλύτερα αργό και αληθινό.
"""
import os

import ai_core


def _prepare_store(tmp_path, monkeypatch, content=b"x" * 128):
    """Στήνει ψεύτικο store και το δείχνει στο ai_core (χωρίς να αγγίξει το
    παραγωγικό αρχείο του volume)."""
    store = tmp_path / "chroma.sqlite3"
    store.write_bytes(content)
    monkeypatch.setattr(ai_core, "_STORE_FILE", str(store))
    return store


def test_signature_is_stable_without_writes(tmp_path, monkeypatch):
    """Χωρίς γράψιμο η υπογραφή ΔΕΝ αλλάζει — αλλιώς τα caches θα ξαναχτίζονταν
    σε κάθε ερώτηση και το warm latency (450ms) θα γινόταν cold."""
    _prepare_store(tmp_path, monkeypatch)
    first = ai_core._corpus_signature()
    for _ in range(20):
        assert ai_core._corpus_signature() == first


def test_external_write_invalidates_cache(tmp_path, monkeypatch):
    """ΤΟ ΣΕΝΑΡΙΟ ΠΟΥ ΕΣΠΑΓΕ: άλλη διεργασία (worker) γράφει στο store. Ο
    τοπικός μετρητής ΔΕΝ αλλάζει — ζει σε άλλη μνήμη — άρα η ακύρωση πρέπει να
    έρθει αποκλειστικά από το mtime."""
    store = _prepare_store(tmp_path, monkeypatch)
    before = ai_core._corpus_signature()

    # Προσομοίωση ξένου worker. os.utime αντί για sleep: ντετερμινιστικό,
    # ακαριαίο, και δεν εξαρτάται από την ανάλυση του ρολογιού του filesystem.
    with open(store, "ab") as f:
        f.write(b"chunk written by another worker")
    st = os.stat(store)
    os.utime(store, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    after = ai_core._corpus_signature()
    assert after != before, "ξένη εγγραφή ΔΕΝ ακύρωσε το cache"
    assert after[0] == before[0], "ο τοπικός μετρητής δεν έπρεπε να αλλάξει"


def test_local_bump_still_works_without_disk_write(tmp_path, monkeypatch):
    """Ο τοπικός μετρητής παραμένει απαραίτητος: το chunk_experiment.py αλλάζει
    collection ΧΩΡΙΣ να γράψει στο ίδιο αρχείο, και πρέπει να ακυρώνει cache."""
    _prepare_store(tmp_path, monkeypatch)
    before = ai_core._corpus_signature()
    ai_core._bump_corpus_version()
    try:
        after = ai_core._corpus_signature()
        assert after != before
        assert after[1] == before[1], "το mtime δεν έπρεπε να αλλάξει"
    finally:
        # Ο μετρητής είναι module-global: χωρίς επαναφορά, τα επόμενα tests
        # (και το παραγωγικό cache της ίδιας διεργασίας) ξεκινούν μετατοπισμένα.
        ai_core._corpus_version = before[0]


def test_missing_store_does_not_crash(tmp_path, monkeypatch):
    """Πρώτο boot: το αρχείο δεν υπάρχει ακόμα. Η υπογραφή πρέπει να δίνει
    σταθερή τιμή αντί να πετάει — αλλιώς η εφαρμογή δεν σηκώνεται καθόλου."""
    monkeypatch.setattr(ai_core, "_STORE_FILE",
                        str(tmp_path / "δεν-υπάρχει.sqlite3"))
    assert ai_core._store_mtime() == 0
    sig = ai_core._corpus_signature()
    assert sig[1] == 0


def test_signature_is_hashable_tuple(tmp_path, monkeypatch):
    """Τα caches αποθηκεύουν την υπογραφή σε dict και τη συγκρίνουν με ==.
    Ένα tuple το κάνει· μια λίστα θα έσπαγε το hashing σιωπηλά."""
    _prepare_store(tmp_path, monkeypatch)
    sig = ai_core._corpus_signature()
    assert isinstance(sig, tuple)
    assert hash(sig) is not None


def test_caches_actually_rebuild_after_external_write(tmp_path, monkeypatch):
    """Ο πλήρης κύκλος, όχι μόνο η υπογραφή: γεμίζουμε το cache, προσποιούμαστε
    ξένη εγγραφή, και επαληθεύουμε ότι το _get_bm25_index ΞΑΝΑΧΤΙΖΕΙ αντί να
    επιστρέψει το παλιό. Αυτό είναι το τελικό ζητούμενο — η υπογραφή είναι απλώς
    το μέσο."""
    store = _prepare_store(tmp_path, monkeypatch)

    builds = {"n": 0}

    class _StubCollection:
        def get(self, where=None, include=None):
            builds["n"] += 1
            return {"ids": ["a", "b"], "documents": ["κείμενο ένα", "κείμενο δύο"],
                    "metadatas": [{"file_name": "f.pdf", "page": 1}] * 2}

    monkeypatch.setattr(ai_core, "collection", _StubCollection())
    monkeypatch.setattr(ai_core, "_bm25_cache", {"version": None})

    ai_core._get_bm25_index()
    assert builds["n"] == 1
    ai_core._get_bm25_index()
    assert builds["n"] == 1, "ξαναχτίστηκε χωρίς λόγο (το cache δεν κρατάει)"

    st = os.stat(store)
    os.utime(store, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    ai_core._get_bm25_index()
    assert builds["n"] == 2, "ΔΕΝ ξαναχτίστηκε μετά από ξένη εγγραφή"
