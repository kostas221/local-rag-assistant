"""Ο δρόμος «όλο το κείμενο» μέσα στον ΠΡΑΓΜΑΤΙΚΟ ai_core (κύκλος 2, 24/11/2026).

Χρειάζεται `import ai_core` (μοντέλα) -> τρέχει στο core-tests. Μηδέν δίκτυο: η γέννηση είναι stubbed και
η βάση είναι μια μικρή προσωρινή Chroma με γνωστά chunks. Προστατεύει:
  1. AUTHZ: έγγραφο άλλου χρήστη δεν διαβάζεται ποτέ, ούτε όταν το ζητά η επιλογή.
  2. Τον κανόνα: 1 έγγραφο ή πάνω από το όριο -> RAG· ο διακόπτης ENABLE_FULL_TEXT κλείνει τον δρόμο.
  3. Τη σύνδεση στο ask_ai: με όλο το κείμενο η αναζήτηση ΔΕΝ τρέχει, και οι πηγές το λένε στο UI.
"""
import asyncio

import chromadb

import ai_core
import gemini_rest

ROWS = [  # (id, κείμενο, doc_id, ιδιοκτήτης, σελίδα)
    ("a.pdf_p0_c0_x1", "A page one.", 1, 7, 1),
    ("a.pdf_p1_c0_x2", "A page two.", 1, 7, 2),
    ("b.pdf_p0_c0_x3", "B page one.", 2, 7, 1),
    ("c.pdf_p0_c0_x4", "C secret of another user.", 3, 8, 1),
]


def _collection(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "store"))
    col = client.create_collection("full_text_test", embedding_function=None,
                                   metadata={"hnsw:space": "cosine"})
    col.add(ids=[r[0] for r in ROWS], documents=[r[1] for r in ROWS],
            embeddings=[[1.0, 0.0]] * len(ROWS),
            metadatas=[{"doc_id": d, "user_id": u, "is_public": False, "page": p,
                        "file_name": cid.split("_p")[0]} for cid, _t, d, u, p in ROWS])
    return col


def test_two_own_documents_give_all_pages_in_selection_order(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_core, "collection", _collection(tmp_path))
    pages = ai_core._full_text_pages(None, 7, [2, 1])
    assert [(m["file_name"], m["page"]) for _t, m in pages] == [("b.pdf", 1), ("a.pdf", 1), ("a.pdf", 2)]
    assert all(m["full_text"] for _t, m in pages)


def test_one_document_stays_rag(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_core, "collection", _collection(tmp_path))
    assert ai_core._full_text_pages(None, 7, [1]) is None
    assert ai_core._full_text_pages(None, 7, [1, 1]) is None       # διπλό id δεν μετράει για δύο


def test_another_users_document_is_never_read(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_core, "collection", _collection(tmp_path))
    # Το 3 ανήκει στον χρήστη 8: μένει μόνο το 1 -> < 2 έγγραφα -> RAG
    assert ai_core._full_text_pages(None, 7, [1, 3]) is None
    pages = ai_core._full_text_pages(None, 7, [1, 2, 3])
    assert pages and all("secret" not in t for t, _m in pages)
    assert {m["doc_id"] for _t, m in pages} == {1, 2}


def test_over_the_limit_or_disabled_stays_rag(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_core, "collection", _collection(tmp_path))
    monkeypatch.setattr(ai_core, "FULL_TEXT_MAX_TOKENS", 10)
    assert ai_core._full_text_pages(None, 7, [1, 2]) is None
    monkeypatch.setattr(ai_core, "FULL_TEXT_MAX_TOKENS", 70_000)
    monkeypatch.setattr(ai_core, "ENABLE_FULL_TEXT", False)
    assert ai_core._full_text_pages(None, 7, [1, 2]) is None


def _ask(monkeypatch, doc_ids):
    seen = {}

    async def fake_stream(prompt, **kwargs):
        seen["prompt"] = prompt
        yield "text", "stub answer [S1]"

    monkeypatch.setattr(gemini_rest, "stream_generate", fake_stream)
    monkeypatch.setattr(ai_core, "USE_REST_GENERATION", True)

    async def run():
        return [ev async for ev in ai_core.ask_ai("what is in the papers?", None, user_id=7,
                                                  target_doc_ids=doc_ids)]

    events = asyncio.run(run())
    return seen.get("prompt", ""), next(e["data"] for e in events if e["type"] == "sources")


def test_ask_ai_reads_full_text_without_searching(monkeypatch):
    pages = [("A page one.", {"file_name": "a.pdf", "page": 1, "doc_id": 1, "full_text": True}),
             ("B page one.", {"file_name": "b.pdf", "page": 1, "doc_id": 2, "full_text": True})]
    monkeypatch.setattr(ai_core, "_full_text_pages", lambda *_a: pages)

    async def no_search(*_a, **_k):
        raise AssertionError("η αναζήτηση δεν πρέπει να τρέξει")

    monkeypatch.setattr(ai_core, "search_documents", no_search)
    prompt, sources = _ask(monkeypatch, [1, 2])
    assert "A page one." in prompt and "B page one." in prompt
    assert [s["full_text"] for s in sources] == [True, True]


def test_ask_ai_falls_back_to_search_when_the_router_says_no(monkeypatch):
    monkeypatch.setattr(ai_core, "_full_text_pages", lambda *_a: None)
    called = {}

    async def fake_search(query, *_a, **_k):
        called["query"] = query
        return [("searched page", {"file_name": "a.pdf", "page": 1, "doc_id": 1})]

    monkeypatch.setattr(ai_core, "search_documents", fake_search)
    prompt, sources = _ask(monkeypatch, [1, 2])
    assert called and "searched page" in prompt
    assert [s["full_text"] for s in sources] == [False]
