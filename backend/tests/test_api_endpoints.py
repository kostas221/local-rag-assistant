"""Integration tests των HTTP endpoints (main.py) — ΧΩΡΙΣ Postgres, ΧΩΡΙΣ μοντέλα.

ΓΙΑΤΙ ΥΠΑΡΧΟΥΝ: το main.py ήταν το μεγαλύτερο αδοκίμαστο κομμάτι του project.
Εκεί ζει το AUTHORIZATION — ποιος βλέπει ποιο έγγραφο, ποιος σβήνει τι — και μια
σιωπηλή αστοχία εκεί δεν είναι bug απόδοσης, είναι διαρροή δεδομένων μεταξύ
χρηστών. Το retrieval eval δεν θα την έπιανε ΠΟΤΕ: τα σκορ θα ήταν τέλεια.

ΠΩΣ ΤΡΕΧΟΥΝ ΓΡΗΓΟΡΑ (δύο αντικαταστάσεις, καμία αλλαγή σε παραγωγικό κώδικα):
  1. ai_core -> stub στο sys.modules ΠΡΙΝ το import του main. Χωρίς αυτό κάθε
     τρέξιμο πλήρωνε 40-60s και 2.4GB για bge-m3 + cross-encoder, που δεν
     χρειάζονται πουθενά εδώ.
  2. database.engine / SessionLocal -> SQLite in-memory. Το main.py τα διαβάζει
     ως module attributes τη στιγμή της κλήσης, οπότε αρκεί να τα αλλάξουμε πριν
     γίνει το import — δεν χρειάζεται dependency_overrides ούτε άλλαγή κώδικα.

Το stub ΕΠΑΝΑΦΕΡΕΤΑΙ στο τέλος του module: αλλιώς το test_rag_core.py, που
θέλει το ΠΡΑΓΜΑΤΙΚΟ ai_core, θα έπαιρνε το ψεύτικο και θα έσπαγε ανάλογα με τη
σειρά εκτέλεσης — ακριβώς το είδος του bug που κρύβεται για εβδομάδες.
"""
import asyncio
import os
import sys
import types

import httpx
import pytest
from loguru import logger

# --- Στήσιμο περιβάλλοντος ΠΡΙΝ από οποιοδήποτε import του main ---------------
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-used-in-production")
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("GEMINI_API_KEY", "test-key")


def _make_ai_core_stub():
    """Ελάχιστο ai_core: μόνο όσα αγγίζει το main.py.
    Καταγράφει τις κλήσεις ώστε τα tests να επαληθεύουν ΟΤΙ κλήθηκε το σωστό —
    π.χ. ότι η διαγραφή εγγράφου φτάνει πράγματι στη vector DB."""
    stub = types.ModuleType("ai_core")
    stub.calls = []

    def ingest_pdf(file_path, filename, user_id, is_public=False, doc_id=None):
        stub.calls.append(("ingest_pdf", filename, user_id, doc_id))
        return True

    def delete_file_from_db(filename, user_id=None, doc_id=None):
        stub.calls.append(("delete_file_from_db", filename, user_id, doc_id))

    async def ask_ai(question, target_filenames, history=None, user_id=None,
                     persona="Researcher", precomputed=None):
        stub.calls.append(("ask_ai", question, user_id))
        yield {"type": "sources", "data": []}
        yield {"type": "text", "data": "stub answer"}

    stub.ingest_pdf = ingest_pdf
    stub.delete_file_from_db = delete_file_from_db
    stub.ask_ai = ask_ai
    return stub


class _Client:
    """Σύγχρονος client πάνω από ASGITransport.

    ΓΙΑΤΙ ΟΧΙ TestClient: το starlette που έρχεται με FastAPI 0.99.1 (pinned)
    καλεί `httpx.Client(app=...)`, παράμετρο που ΑΦΑΙΡΕΘΗΚΕ στο httpx 0.28 ->
    TypeError πριν καν τρέξει test. Υποβάθμιση του httpx δεν γίνεται: το
    gemini_rest.py στηρίζεται στο 0.28 για το SSE streaming. Το ASGITransport
    χτυπάει την εφαρμογή κατευθείαν, χωρίς δίκτυο και χωρίς αυτή τη γέφυρα.
    (Το main.py δεν έχει startup/lifespan events, οπότε δεν χάνεται τίποτα.)
    """

    def __init__(self, app):
        self._app = app

    def _request(self, method: str, url: str, **kwargs):
        async def go():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(transport=transport,
                                         base_url="http://testserver") as c:
                return await c.request(method, url, **kwargs)
        return asyncio.run(go())

    def get(self, url, **kw):
        return self._request("GET", url, **kw)

    def post(self, url, **kw):
        return self._request("POST", url, **kw)

    def delete(self, url, **kw):
        return self._request("DELETE", url, **kw)


@pytest.fixture(scope="module")
def app_env():
    """(client, ai_core_stub) με καθαρή in-memory βάση ανά module."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    saved_ai_core = sys.modules.get("ai_core")
    saved_main = sys.modules.pop("main", None)
    stub = _make_ai_core_stub()
    sys.modules["ai_core"] = stub

    import database

    saved_engine, saved_session = database.engine, database.SessionLocal
    # StaticPool + check_same_thread=False: το TestClient εκτελεί τα sync
    # endpoints σε ΑΛΛΟ thread από το test. Χωρίς αυτά, η in-memory SQLite είτε
    # σκάει ("same thread") είτε δίνει σε κάθε σύνδεση ΑΛΛΗ κενή βάση.
    database.engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False},
        poolclass=StaticPool)
    database.SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=database.engine)

    import main  # κάνει create_all πάνω στο ΝΕΟ engine

    yield _Client(main.app), stub

    # Το main.py κάνει logging.basicConfig(force=True) και δένει το loguru στο
    # sys.stderr. Στο teardown το pytest κλείνει εκείνο το stream -> "I/O
    # operation on closed file" σε κάθε log, που πνίγει το summary του CI.
    logger.remove()

    # Επαναφορά: το ai_core stub ΔΕΝ πρέπει να διαρρεύσει σε άλλα test modules.
    database.engine, database.SessionLocal = saved_engine, saved_session
    sys.modules.pop("main", None)
    if saved_main is not None:
        sys.modules["main"] = saved_main
    if saved_ai_core is not None:
        sys.modules["ai_core"] = saved_ai_core
    else:
        sys.modules.pop("ai_core", None)


def _register_and_login(client, username: str) -> dict:
    """Δημιουργεί χρήστη και επιστρέφει έτοιμο Authorization header."""
    client.post("/register", json={"username": username,
                                   "email": f"{username}@test.local",
                                   "password": "TestPass123!"})
    res = client.post("/login", data={"username": username,
                                      "password": "TestPass123!"})
    assert res.status_code == 200, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


@pytest.fixture(scope="module")
def alice(app_env):
    client, _ = app_env
    return _register_and_login(client, "alice")


@pytest.fixture(scope="module")
def bob(app_env):
    client, _ = app_env
    return _register_and_login(client, "bob")


# --- Health ------------------------------------------------------------------

def test_health_needs_no_auth(app_env):
    client, _ = app_env
    res = client.get("/health")
    assert res.status_code == 200


# --- Authentication ----------------------------------------------------------

@pytest.mark.parametrize("method,path", [
    ("get", "/documents"),
    ("get", "/conversations"),
    ("post", "/conversations"),
    ("delete", "/documents/1"),
])
def test_protected_endpoints_reject_anonymous(app_env, method, path):
    """Κάθε endpoint που αγγίζει δεδομένα χρήστη ΠΡΕΠΕΙ να θέλει token."""
    client, _ = app_env
    res = getattr(client, method)(path, json={} if method == "post" else None)
    assert res.status_code == 401, f"{method} {path} -> {res.status_code}"


def test_garbage_token_is_rejected(app_env):
    client, _ = app_env
    res = client.get("/documents",
                     headers={"Authorization": "Bearer not.a.real.jwt"})
    assert res.status_code == 401


def test_duplicate_registration_is_refused(app_env, alice):
    client, _ = app_env
    res = client.post("/register", json={"username": "alice",
                                         "email": "other@test.local",
                                         "password": "TestPass123!"})
    assert res.status_code == 400


def test_wrong_password_is_refused(app_env, alice):
    client, _ = app_env
    res = client.post("/login", data={"username": "alice",
                                      "password": "WrongPass123!"})
    assert res.status_code == 401


# --- Απομόνωση χρηστών (το σοβαρό κομμάτι) -----------------------------------

def test_conversations_are_isolated_between_users(app_env, alice, bob):
    """REGRESSION ΑΣΦΑΛΕΙΑΣ: η συνομιλία της Alice δεν πρέπει να εμφανίζεται
    ούτε να διαβάζεται από τον Bob. Καμία μετρική RAG δεν θα το έπιανε αυτό."""
    client, _ = app_env
    created = client.post("/conversations", json={"title": "ιδιωτικό"},
                          headers=alice)
    assert created.status_code in (200, 201), created.text
    conv_id = created.json()["id"]

    mine = client.get("/conversations", headers=alice).json()
    assert any(c["id"] == conv_id for c in mine)

    theirs = client.get("/conversations", headers=bob).json()
    assert all(c["id"] != conv_id for c in theirs), "διαρροή συνομιλίας!"

    # Ούτε με απευθείας πρόσβαση στο id
    res = client.get(f"/conversations/{conv_id}/messages", headers=bob)
    assert res.status_code in (403, 404), (
        f"ο Bob διάβασε τα μηνύματα της Alice ({res.status_code})")


def test_cannot_delete_another_users_conversation(app_env, alice, bob):
    client, _ = app_env
    conv_id = client.post("/conversations", json={"title": "δικό μου"},
                          headers=alice).json()["id"]
    res = client.delete(f"/conversations/{conv_id}", headers=bob)
    assert res.status_code in (403, 404)
    # Και πρέπει να ΥΠΑΡΧΕΙ ακόμα για την Alice
    still = client.get("/conversations", headers=alice).json()
    assert any(c["id"] == conv_id for c in still)


def test_cannot_delete_nonexistent_document(app_env, alice):
    client, _ = app_env
    res = client.delete("/documents/999999", headers=alice)
    assert res.status_code == 404


# --- Upload validation -------------------------------------------------------

def test_upload_rejects_non_pdf(app_env, alice):
    """Το ingest υποθέτει PDF. Ένα .exe με αλλαγμένη κατάληξη δεν πρέπει καν
    να φτάσει στο background task."""
    client, _ = app_env
    res = client.post("/upload", headers=alice,
                      files={"file": ("malware.exe", b"MZ\x90\x00",
                                      "application/octet-stream")})
    assert res.status_code in (400, 415, 422), res.text


# --- Chat --------------------------------------------------------------------

def test_chat_requires_authentication(app_env):
    """ΣΗΜ: το σώμα εδώ είναι ΣΚΟΠΙΜΑ ελλιπές. Το 401 πρέπει να έρχεται ΠΡΙΝ
    το validation — αλλιώς ένας ανώνυμος μαθαίνει τη δομή του API από τα 422."""
    client, _ = app_env
    res = client.post("/chat", json={"question": "τι είναι το cloud;"})
    assert res.status_code == 401


def test_chat_streams_and_passes_user_id(app_env, alice):
    """Το /chat πρέπει να περνά το user_id στο ai_core — είναι ΑΥΤΟ που κάνει
    το authorization στο retrieval (single-source στο where της Chroma)."""
    client, stub = app_env
    conv_id = client.post("/conversations", json={"title": "chat test"},
                          headers=alice).json()["id"]
    stub.calls.clear()
    res = client.post("/chat", headers=alice,
                      json={"conversation_id": conv_id,
                            "question": "τι είναι το cloud;", "history": [],
                            "target_filenames": [], "persona": "Concise"})
    assert res.status_code == 200, res.text
    assert "stub answer" in res.text

    ask_calls = [c for c in stub.calls if c[0] == "ask_ai"]
    assert len(ask_calls) == 1
    assert ask_calls[0][2] is not None, "το user_id ΔΕΝ έφτασε στο ai_core"


# --- Rate limiting ------------------------------------------------------------

def _clear_register_limit():
    """Το παράθυρο του /register είναι ΜΙΑ ΩΡΑ: χωρίς καθάρισμα, το test θα
    κατανάλωνε το όριο και θα έσπαγε όποιο fixture δημιουργεί χρήστη μετά."""
    import database
    import models

    session = database.SessionLocal()
    try:
        session.query(models.RateLimit).filter(
            models.RateLimit.bucket == "register").delete()
        session.commit()
    finally:
        session.close()


def test_register_is_rate_limited(app_env, alice, bob, monkeypatch):
    """Κάθε λογαριασμός κουβαλά ΔΙΚΟ του quota στο /chat -> χωρίς όριο εδώ, το
    όριο του /chat παρακάμπτεται με ένα loop που φτιάχνει χρήστες."""
    import main

    client, _ = app_env
    monkeypatch.setattr(main, "REGISTER_MAX_ACCOUNTS", 2)
    _clear_register_limit()
    try:
        codes = [client.post("/register", json={
            "username": f"spam{i}", "email": f"spam{i}@test.local",
            "password": "TestPass123!"}).status_code for i in range(4)]
    finally:
        _clear_register_limit()
    assert 429 in codes, codes


def test_forwarded_ip_is_ignored_without_trust_proxy(monkeypatch):
    """ΤΟ ΚΡΙΣΙΜΟ: το X-Forwarded-For το γράφει ο ΠΕΛΑΤΗΣ όταν δεν υπάρχει
    proxy. Αν το εμπιστευόμασταν πάντα, ο επιτιθέμενος θα άλλαζε "IP" σε κάθε
    αίτημα και κάθε per-IP όριο θα ήταν διακοσμητικό."""
    from starlette.requests import Request as StarletteRequest

    import main

    req = StarletteRequest({"type": "http",
                            "headers": [(b"x-forwarded-for", b"9.9.9.9")],
                            "client": ("127.0.0.1", 1234)})
    monkeypatch.setattr(main, "TRUST_PROXY", False)
    assert main._client_ip(req) == "127.0.0.1"
    monkeypatch.setattr(main, "TRUST_PROXY", True)
    assert main._client_ip(req) == "9.9.9.9"
    # --- Ingest status ------------------------------------------------------------

@pytest.mark.parametrize("ingest_ok,expected", [(True, "ready"), (False, "empty")])
def test_scanned_pdf_is_not_marked_ready(app_env, alice, monkeypatch,
                                         ingest_ok, expected):
    """Ένα PDF χωρίς εξαγώγιμο κείμενο ΔΕΝ πρέπει να δείχνει "ready": ο χρήστης
    θα το τσέκαρε στη βιβλιοθήκη και θα ρωτούσε ένα αρχείο που δεν υπάρχει καν
    στο ευρετήριο — και η αποτυχία θα φαινόταν σαν αποτυχία του RAG."""
    _client, stub = app_env
    import database
    import main
    import models

    db = database.SessionLocal()
    uid = db.query(models.User).first().id
        # Η διαδρομή δεν ανοίγεται ΠΟΤΕ — το ingest_pdf είναι stubbed. Δεν είναι
    # /tmp/... ώστε να μη χτυπά το S108 για αρχείο που δεν υπάρχει καν.
    fake_path = "uploaded_docs/scanned.pdf"
    doc = models.Document(file_name="scanned.pdf", file_path=fake_path,
                          user_id=uid, status="processing")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    doc_id = doc.id
    db.close()

    monkeypatch.setattr(stub, "ingest_pdf", lambda *a, **kw: ingest_ok)
    main.process_document(doc_id, fake_path, "scanned.pdf", uid)

    db = database.SessionLocal()
    got = db.query(models.Document).filter(models.Document.id == doc_id).first()
    assert got.status == expected
    db.close()