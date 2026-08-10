"""Rate limiting μοιρασμένο μεταξύ διεργασιών, με πλάτη την PostgreSQL.

ΤΙ ΗΤΑΝ ΠΡΙΝ (main.py):
    _login_attempts = defaultdict(deque)
    _chat_attempts  = defaultdict(deque)
Δύο dict στη μνήμη της διεργασίας. Τρία προβλήματα:
  1. ΧΑΝΕΤΑΙ ΣΕ RESTART — ένας επιτιθέμενος στο /login κάνει 5 προσπάθειες, ο
     server κάνει restart (deploy, crash, OOM) και το ρολόι μηδενίζει.
  2. ΔΕΝ ΜΟΙΡΑΖΕΤΑΙ — με 2 workers ο καθένας κρατά ΔΙΚΟ του μετρητή, άρα το
     πραγματικό όριο είναι 2× αυτό που δηλώνεις. Με N instances, N×.
  3. Ήταν ο τελευταίος λόγος που το σύστημα ΔΕΝ μπορούσε να τρέξει σε πάνω από
     μία διεργασία χωρίς να χαλάσει σιωπηλά. (Ο άλλος, η ακύρωση cache του
     corpus, λύθηκε με το mtime-based _corpus_signature.)

ΓΙΑΤΙ POSTGRES ΚΑΙ ΟΧΙ REDIS:
Η Postgres είναι ΗΔΗ εκεί — μηδέν νέα containers, μηδέν νέα εξάρτηση. Το
project έχει 3 containers και η προσθήκη ενός τέταρτου για δύο μετρητές θα ήταν
η ίδια ανταλλαγή που απορρίφθηκε στο Langfuse. Σε ρυθμούς χιλιάδων req/s το
Redis θα κέρδιζε· εδώ μιλάμε για 20 req/λεπτό ανά χρήστη.

Ο ΑΛΓΟΡΙΘΜΟΣ: fixed window ΕΥΘΥΓΡΑΜΜΙΣΜΕΝΟ σε πολλαπλάσια του window_sec.
Ένα atomic INSERT .. ON CONFLICT DO UPDATE αυξάνει τον μετρητή και επιστρέφει
τη νέα τιμή σε ΕΝΑ query — δεν υπάρχει race ανάμεσα σε read και write, ό,τι κι
αν τρέχει παράλληλα.

ΔΥΟ ΣΥΝΕΙΔΗΤΕΣ ΔΙΑΦΟΡΕΣ ΑΠΟ ΤΗΝ ΠΑΛΙΑ ΣΥΜΠΕΡΙΦΟΡΑ:
  α) Τα ΜΠΛΟΚΑΡΙΣΜΕΝΑ αιτήματα μετράνε κι αυτά. Πριν, ένα 429 δεν αύξανε τον
     μετρητή, οπότε ο επιτιθέμενος μπορούσε να χτυπά συνεχώς στο όριο. Τώρα το
     spam παρατείνει το μπλοκάρισμα μέχρι το τέλος του παραθύρου.
  β) Ευθυγραμμισμένο παράθυρο αντί για κυλιόμενο -> στο σύνορο δύο παραθύρων
     επιτρέπονται θεωρητικά έως 2× αιτήματα σε στιγμιαίο burst. Κλασικό
     trade-off του fixed window· για προστασία quota (όχι για DDoS) αρκεί.

Ο πίνακας δημιουργείται από το create_all() του main.py — ΚΑΜΙΑ migration.
"""
import time

from fastapi import HTTPException
from sqlalchemy import delete

import models


def _insert_for(db):
    """Το ON CONFLICT είναι dialect-specific. Η παραγωγή τρέχει PostgreSQL, τα
    tests in-memory SQLite — και οι δύο εκδοχές έχουν ΙΔΙΑ υπογραφή."""
    if db.bind.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:
        from sqlalchemy.dialects.postgresql import insert
    return insert


def check(db, bucket: str, key: str, max_hits: int, window_sec: int,
          message: str) -> None:
    """Αυξάνει τον μετρητή· πετάει 429 αν ξεπεράστηκε το όριο.

    bucket: "login" | "chat" — ώστε τα δύο όρια να μη μπερδεύονται όταν το
            κλειδί συμπίπτει (μια IP και ένα user_id μπορεί να είναι "5").
    """
    now = int(time.time())
    window_start = now - (now % window_sec)

    insert = _insert_for(db)
    stmt = (
        insert(models.RateLimit)
        .values(bucket=bucket, key=key, window_start=window_start, hits=1)
        .on_conflict_do_update(
            index_elements=["bucket", "key", "window_start"],
            # ΠΡΟΣΟΧΗ: models.RateLimit.hits + 1 μεταφράζεται σε
            # "hits = rate_limits.hits + 1" μέσα στη ΒΑΣΗ. Αν το κάναμε σε
            # Python (διάβασε -> +1 -> γράψε) θα επέστρεφε το race που όλο
            # αυτό το αρχείο υπάρχει για να εξαλείψει.
            set_={"hits": models.RateLimit.hits + 1})
        .returning(models.RateLimit.hits)
    )
    hits = db.execute(stmt).scalar_one()
    db.commit()

    # Καθαρισμός παλιών παραθύρων. Γίνεται ΜΟΝΟ όταν hits == 1, δηλαδή μία φορά
    # ανά κλειδί ανά παράθυρο: αυτορυθμίζεται με την κίνηση, δεν χρειάζεται cron
    # και δεν επιβαρύνει το hot path.
    if hits == 1:
        db.execute(delete(models.RateLimit).where(
            models.RateLimit.bucket == bucket,
            models.RateLimit.window_start < window_start - window_sec))
        db.commit()

    if hits > max_hits:
        raise HTTPException(status_code=429, detail=message)
