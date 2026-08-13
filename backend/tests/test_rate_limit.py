"""Tests για το rate_limit.py — ΜΗΔΕΝ μοντέλα, ΜΗΔΕΝ Postgres, ΜΗΔΕΝ δίκτυο.

ΓΙΑΤΙ ΞΕΧΩΡΙΣΤΟ ΑΡΧΕΙΟ ΚΑΙ ΟΧΙ ΕΜΜΕΣΑ ΜΕΣΩ ΤΟΥ /login:
Τα test_api_endpoints χτυπούν το /login και άρα εκτελούν το rate_limit.check —
αλλά ΠΟΤΕ αρκετές φορές ώστε να ενεργοποιηθεί το όριο. Δηλαδή επιβεβαιώνουν ότι
«δεν σκάει», όχι ότι «κόβει». Αν αύριο σπάσει το ON CONFLICT και ο μετρητής
μείνει κολλημένος στο 1, και τα 72 tests θα περνούσαν.

ΤΟ ΚΡΙΣΙΜΟ ΤΕΣΤ ΕΙΝΑΙ ΤΟ test_counter_increments_atomically_in_sql: επιβεβαιώνει
ότι η αύξηση γίνεται ΜΕΣΑ στη βάση (hits = hits + 1) και όχι σε Python. Αν κάποιος
το ξαναγράψει ως read-modify-write, επανέρχεται σιωπηλά το race που όλο το module
υπάρχει για να εξαλείψει — και ΔΕΝ θα φαινόταν με έναν worker.

SQLite in-memory: το ON CONFLICT DO UPDATE .. RETURNING υποστηρίζεται και στα δύο
dialects, οπότε το ίδιο μονοπάτι κώδικα δοκιμάζεται εδώ και τρέχει σε Postgres.
"""
import sys
import time

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, "/app")


@pytest.fixture()
def db():
    """Καθαρή in-memory βάση ανά test."""
    import database

    saved_engine, saved_session = database.engine, database.SessionLocal
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    database.engine = engine
    database.SessionLocal = sessionmaker(autocommit=False, autoflush=False,
                                         bind=engine)
    import models

    models.Base.metadata.create_all(bind=engine)
    session = database.SessionLocal()
    yield session
    session.close()
    database.engine, database.SessionLocal = saved_engine, saved_session


def _check(db, key="1.2.3.4", max_hits=3, window=60, bucket="login"):
    import rate_limit

    rate_limit.check(db, bucket, key, max_hits, window, "too many")


def test_allows_up_to_the_limit(db):
    for _ in range(3):
        _check(db)          # 3 αιτήματα με max_hits=3 -> κανένα 429


def test_blocks_beyond_the_limit(db):
    for _ in range(3):
        _check(db)
    with pytest.raises(HTTPException) as exc:
        _check(db)
    assert exc.value.status_code == 429


def test_blocked_requests_also_count(db):
    """ΣΥΝΕΙΔΗΤΗ αλλαγή συμπεριφοράς έναντι της in-memory εκδοχής: εκεί ένα 429
    ΔΕΝ αύξανε τον μετρητή, οπότε ο επιτιθέμενος μπορούσε να χτυπά συνεχώς στο
    όριο. Εδώ το spam παρατείνει το μπλοκάρισμα ως το τέλος του παραθύρου."""
    import models

    for _ in range(3):
        _check(db)
    for _ in range(2):
        with pytest.raises(HTTPException):
            _check(db)
    row = db.query(models.RateLimit).one()
    assert row.hits == 5


def test_counter_increments_atomically_in_sql(db):
    """Η αύξηση πρέπει να γίνεται ΜΕΣΑ στη βάση. Το ελέγχουμε με δύο ΞΕΧΩΡΙΣΤΑ
    sessions πάνω στην ίδια σύνδεση: αν η λογική ήταν read-modify-write σε
    Python, το δεύτερο session θα ξεκινούσε από στιγμιότυπο και θα έγραφε 1."""
    import database
    import models

    _check(db, max_hits=99)
    other = database.SessionLocal()
    try:
        _check(other, max_hits=99)
    finally:
        other.close()
    assert db.query(models.RateLimit).one().hits == 2


def test_separate_keys_do_not_interfere(db):
    for _ in range(3):
        _check(db, key="1.1.1.1")
    _check(db, key="2.2.2.2")      # άλλη IP -> δικός της μετρητής


def test_separate_buckets_do_not_interfere(db):
    """Μια IP «5» και ένα user_id «5» δίνουν το ΙΔΙΟ key. Χωρίς το bucket, το
    όριο του login θα κατανάλωνε το όριο του chat."""
    for _ in range(3):
        _check(db, key="5", bucket="login")
    _check(db, key="5", bucket="chat")


def test_new_window_resets_the_counter(db):
    """window_sec=1 -> το επόμενο δευτερόλεπτο είναι ΑΛΛΟ ευθυγραμμισμένο
    παράθυρο, άρα άλλη γραμμή και μετρητής από το μηδέν."""
    for _ in range(3):
        _check(db, window=1)
    with pytest.raises(HTTPException):
        _check(db, window=1)
    time.sleep(1.05)
    _check(db, window=1)           # νέο παράθυρο -> περνάει


def test_old_windows_are_cleaned_up(db):
    """Ο καθαρισμός τρέχει όταν hits == 1, δηλαδή μία φορά ανά κλειδί ανά
    παράθυρο. Χωρίς αυτόν ο πίνακας μεγαλώνει επ' άπειρον."""
    import models

    _check(db, window=1)
    time.sleep(1.05)
    _check(db, window=1)           # νέο παράθυρο -> σβήνει τα παλιά
    time.sleep(1.05)
    _check(db, window=1)
    assert db.query(models.RateLimit).count() <= 2


def test_429_carries_retry_after(db):
    """Ο πελάτης πρέπει να ΞΕΡΕΙ πόσο να περιμένει: τα μπλοκαρισμένα αιτήματα
    μετράνε, άρα το τυφλό retry αυτοτιμωρείται."""
    for _ in range(3):
        _check(db)
    with pytest.raises(HTTPException) as exc:
        _check(db)
    assert 1 <= int(exc.value.headers["Retry-After"]) <= 60
