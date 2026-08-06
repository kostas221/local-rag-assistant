"""Μηχανισμοί caching του RAG pipeline — ΧΩΡΙΣ εξάρτηση από μοντέλα ή ChromaDB.

ΓΙΑΤΙ ΞΕΧΩΡΙΣΤΟ MODULE: το ai_core φορτώνει bge-m3 + cross-encoder στο import
(~40-60s). Όσο η λογική του caching ζούσε εκεί μέσα, ΔΕΝ μπορούσε να δοκιμαστεί
χωρίς να πληρωθεί αυτό το κόστος. Εδώ δεν γίνεται import κανένα μοντέλο, οπότε
τα tests του ατομικού γραψίματος και του versioning τρέχουν σε χιλιοστά.

ΣΧΕΔΙΑΣΤΙΚΗ ΑΡΧΗ: αυτό το module ΔΕΝ κρατάει state. Τα ίδια τα caches (dicts) και
τα locks ζουν στο ai_core και περνιούνται ως ΟΡΙΣΜΑΤΑ. Έτσι τα tests και το
chunk_experiment.py συνεχίζουν να κάνουν monkeypatch το `ai_core._bm25_cache`
κ.λπ. και η αντικατάσταση φτάνει πράγματι στη συνάρτηση που τρέχει — πράγμα που
ΔΕΝ θα ίσχυε αν το state μετακόμιζε εδώ.
"""
import json
import os

import numpy as np
from loguru import logger

# --- Versioned corpus caches (BM25 / dense matrix / sparse weights) ----------

def versioned(cache: dict, lock, version, build):
    """Double-checked locking πάνω σε έναν "αριθμό έκδοσης" του corpus.

    Το ΙΔΙΟ pattern ήταν γραμμένο τρεις φορές (BM25, dense, sparse). Μία
    υλοποίηση = ένα σημείο όπου μπορεί να μπει bug στο locking, αντί για τρία.

    `build()` επιστρέφει dict με τα πεδία του cache (χωρίς το "version").
    Ο έλεγχος γίνεται ΔΥΟ φορές: μία χωρίς lock (γρήγορο μονοπάτι, η συντριπτική
    πλειοψηφία των κλήσεων) και μία αφού το πάρουμε — γιατί άλλο thread μπορεί να
    το έχτισε όσο περιμέναμε, και το ξαναχτίσιμο κοστίζει δευτερόλεπτα.
    """
    if cache.get("version") == version:
        return cache
    with lock:
        if cache.get("version") == version:
            return cache
        fresh = build()
        cache.clear()
        cache.update(version=version, **fresh)
        return cache


# --- JSON cache με ατομικό γράψιμο (μεταφράσεις, sparse weights) -------------

def load_json(path: str) -> dict:
    """Ανάγνωση JSON cache· κενό dict αν λείπει ή είναι κατεστραμμένο.
    Ένα cache που δεν διαβάζεται ΔΕΝ είναι λόγος να μη σηκωθεί η εφαρμογή."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_json_atomic(path: str, data: dict, lock=None, indent: int | None = 1,
                     label: str = "cache") -> None:
    """Γράψιμο μέσω προσωρινού αρχείου + os.replace: ατομικό, ώστε ένα crash
    στη μέση να μην αφήσει κατεστραμμένο JSON (το os.replace είναι atomic σε
    POSIX και Windows). Ποτέ fatal: ένα cache είναι επιτάχυνση, όχι δεδομένα."""
    try:
        if lock is not None:
            lock.acquire()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(f"--- Δεν αποθηκεύτηκε το {label}: {e} ---")
    finally:
        if lock is not None:
            lock.release()


# --- npz cache διανυσμάτων (query embeddings) --------------------------------
# npz και όχι JSON: 1024 floats ανά εγγραφή είναι ~4KB σε binary έναντι ~20KB σε
# JSON, χωρίς rounding.

def load_npz_vectors(path: str, model_name: str, dim: int) -> dict:
    """{key: np.float32 διάνυσμα}, με ΔΙΠΛΗ επικύρωση πριν επιστραφεί.

    ΔΕΣΜΕΥΜΕΝΟ ΣΤΟ ΜΟΝΤΕΛΟ: αν αλλάξει το embedding model, τα αποθηκευμένα
    διανύσματα δείχνουν σε άλλο χώρο (ίσως άλλη διάσταση) και θα έδιναν σιωπηλά
    λάθος αποτελέσματα -> το όνομα γράφεται μέσα στο αρχείο και ελέγχεται.
    Ο έλεγχος διάστασης δεν είναι θεωρητικός: τα tests κάνουν monkeypatch τον
    embedder σε 8 διαστάσεις και είχαν μολύνει το παραγωγικό npz.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            if str(z["model"]) != model_name:
                logger.warning(
                    f"--- Query embedding cache από άλλο μοντέλο "
                    f"('{z['model']}' != '{model_name}') -> αγνοείται ---")
                return {}
            vecs = z["vectors"]
            if vecs.ndim != 2 or vecs.shape[1] != dim:
                logger.warning(
                    f"--- Query embedding cache με λάθος διάσταση "
                    f"({vecs.shape} != (n, {dim})) -> αγνοείται. "
                    f"Συνήθης αιτία: το έγραψε test με stubbed embedder. ---")
                return {}
            return dict(zip(z["queries"].tolist(), vecs))
    except (OSError, ValueError, KeyError):
        return {}


def save_npz_vectors(path: str, model_name: str, items: list, lock=None) -> None:
    """Ατομικό γράψιμο (tmp + os.replace), ίδια σύμβαση με το save_json_atomic.
    `items`: [(key, vector)]. Το savez γράφεται σε ΑΝΟΙΧΤΟ file object ώστε να
    ΜΗΝ προσθέσει μόνο του δεύτερη κατάληξη '.npz' στο tmp όνομα."""
    try:
        if lock is not None:
            lock.acquire()
        if not items:
            return
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(
                f, model=np.array(model_name),
                queries=np.array([q for q, _ in items]),
                vectors=np.stack([v for _, v in items]))
        os.replace(tmp, path)
    except (OSError, ValueError) as e:
        logger.warning(f"--- Δεν αποθηκεύτηκε το query embedding cache: {e} ---")
    finally:
        if lock is not None:
            lock.release()
