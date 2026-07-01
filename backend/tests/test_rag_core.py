"""Unit tests για τον RAG πυρήνα (ai_core): καθαρές συναρτήσεις + where-building.
ΣΗΜ: το import του ai_core φορτώνει τα μοντέλα μία φορά (~1 λεπτό, cached) επειδή
γίνονται load σε επίπεδο module — τα ίδια τα tests μετά είναι ακαριαία."""
import asyncio

import ai_core
from ai_core import el_tokenize, _has_greek, _build_where


# --- el_tokenize: Greek-aware tokenization για το BM25 ---

def test_tokenize_strips_accents_and_lowercases():
    assert el_tokenize("Ελληνικά Κείμενα") == ["ελληνικα", "κειμενα"]


def test_tokenize_accented_and_unaccented_match():
    """Οι χρήστες γράφουν συχνά άτονα -> πρέπει να ταυτίζονται στο BM25."""
    assert el_tokenize("ελαστικότητα") == el_tokenize("ελαστικοτητα")


def test_tokenize_splits_on_punctuation():
    assert el_tokenize("cloud, computing!") == ["cloud", "computing"]


# --- _has_greek: ανίχνευση γλώσσας για το translate-then-retrieve ---

def test_has_greek_detection():
    assert _has_greek("Τι είναι το cloud;") is True
    assert _has_greek("What is cloud computing?") is False


# --- _build_where: φίλτρα πρόσβασης & επιλογής αρχείων της ChromaDB ---

def test_where_empty_when_no_filters():
    assert _build_where(None, None) is None


def test_where_single_file():
    assert _build_where(["a.pdf"], None) == {"file_name": "a.pdf"}


def test_where_user_and_files_combined_with_and():
    w = _build_where(["a.pdf", "b.pdf"], user_id=7)
    access, files = w["$and"]
    assert access == {"$or": [{"user_id": 7}, {"is_public": True}]}
    assert files == {"$or": [{"file_name": "a.pdf"}, {"file_name": "b.pdf"}]}


# --- delete_file_from_db: REGRESSION TEST για το duplicate-delete bug ---

class _FakeCollection:
    """Ψεύτικη collection: καταγράφει με τι where κλήθηκε το delete.
    ΔΕΝ πατχάρουμε το πραγματικό αντικείμενο — είναι pydantic model του chromadb
    και μπλοκάρει το setattr. Αντικαθιστούμε το module-level ai_core.collection."""

    def __init__(self):
        self.captured = {}

    def delete(self, where):
        self.captured["where"] = where


def test_delete_prefers_doc_id_scope(monkeypatch):
    """Bug 2026-06-11: διαγραφή με file_name+user_id έσβηνε ΚΑΙ τα chunks
    ομώνυμου record (διπλο-ανέβασμα). Fix: με doc_id σβήνουμε ΜΟΝΟ αυτό."""
    fake = _FakeCollection()
    monkeypatch.setattr(ai_core, "collection", fake)
    ai_core.delete_file_from_db("a.pdf", user_id=1, doc_id=42)
    assert fake.captured["where"] == {"doc_id": 42}


def test_delete_falls_back_to_filename_and_user(monkeypatch):
    fake = _FakeCollection()
    monkeypatch.setattr(ai_core, "collection", fake)
    ai_core.delete_file_from_db("a.pdf", user_id=1)
    assert fake.captured["where"] == {"$and": [{"file_name": "a.pdf"}, {"user_id": 1}]}


# --- _rrf_fuse: Reciprocal Rank Fusion (καθαρή λογική, χωρίς DB/μοντέλα) ---

def test_rrf_doc_in_both_lists_ranks_first():
    """Doc που εμφανίζεται ΚΑΙ στο dense ΚΑΙ στο sparse top πρέπει να νικά
    ένα doc που είναι μόνο σε μία λίστα (η ουσία του RRF)."""
    all_ids = ["a", "b", "c"]
    all_texts = ["text-a", "text-b", "text-c"]
    all_metas = [{"file_name": "f.pdf"}] * 3
    dense_ids = ["a", "b"]   # 'a' #1 στο dense
    sparse_ids = ["a", "c"]  # 'a' #1 και στο sparse -> πρέπει να βγει πρώτο
    fused = ai_core._rrf_fuse(dense_ids, sparse_ids, all_ids, all_texts, all_metas)
    assert fused[0][1] == "text-a"


def test_rrf_score_formula_known_value():
    """Doc #1 (rank 0 -> +1) και στις δύο λίστες με k=60: 1/61 + 1/61 = 2/61."""
    fused = ai_core._rrf_fuse(["x"], ["x"], ["x"], ["tx"], [{"file_name": "f"}], k=60)
    assert abs(fused[0][0] - (2.0 / 61.0)) < 1e-9


def test_rrf_respects_top_n():
    ids = [f"id{i}" for i in range(10)]
    texts = [f"t{i}" for i in range(10)]
    metas = [{"file_name": "f"}] * 10
    fused = ai_core._rrf_fuse(ids, list(reversed(ids)), ids, texts, metas, top_n=3)
    assert len(fused) == 3


def test_rrf_is_deterministic():
    """Ίδια είσοδος -> ίδια σειρά. Το list(set(...)) στο RRF δεν πρέπει να
    εισάγει τυχαιότητα, γιατί στο τέλος ταξινομούμε με βάση το score."""
    ids, texts = ["a", "b", "c", "d"], ["ta", "tb", "tc", "td"]
    metas = [{"file_name": "f"}] * 4
    run1 = ai_core._rrf_fuse(["a", "b", "c"], ["d", "a"], ids, texts, metas)
    run2 = ai_core._rrf_fuse(["a", "b", "c"], ["d", "a"], ids, texts, metas)
    assert [t for _, t, _ in run1] == [t for _, t, _ in run2]


# --- Relevance gate: anti-hallucination (stub reranker/collection, χωρίς API) ---

class _StubCollection:
    """Ελάχιστο stub της ChromaDB collection ώστε να τρέξει το search_documents
    χωρίς πραγματική DB/embeddings: get() γυρνά όλο το corpus, query() γυρνά τα
    ids ως πλασματικό dense ranking."""

    def __init__(self, ids, texts, metas):
        self._ids, self._texts, self._metas = ids, texts, metas

    def get(self, where=None, include=None):
        return {"ids": self._ids, "documents": self._texts,
                "metadatas": self._metas}

    def query(self, query_texts=None, n_results=10, where=None):
        return {"ids": [self._ids[:n_results]]}


class _StubReranker:
    """Γυρνά σταθερό score για κάθε (query, chunk) ζεύγος -> ελέγχουμε
    ντετερμινιστικά αν το gate κόβει ή περνά."""

    def __init__(self, score):
        self.score = score

    def predict(self, pairs):
        return [self.score] * len(pairs)


def _patch_search(monkeypatch, rerank_score):
    """Στήνει corpus + stubs ώστε το search_documents να τρέξει offline."""
    ids = ["id0", "id1", "id2"]
    texts = ["chunk zero", "chunk one", "chunk two"]
    metas = [{"file_name": "a.pdf", "page": i + 1} for i in range(3)]
    monkeypatch.setattr(ai_core, "collection", _StubCollection(ids, texts, metas))
    monkeypatch.setattr(ai_core, "reranker", _StubReranker(rerank_score))
    # Μηδένισε το cache ώστε το BM25 index να ξαναχτιστεί από το stub corpus
    # (αλλιώς ένα stale index από προηγούμενο test/run θα χρησιμοποιούνταν).
    monkeypatch.setattr(ai_core, "_bm25_cache", {"version": None})

    async def _no_translate(q):  # καμία κλήση Gemini μέσα στα tests
        return q
    monkeypatch.setattr(ai_core, "optimize_query", _no_translate)


def test_relevance_gate_blocks_below_threshold(monkeypatch):
    """Όλα τα chunks κάτω από το MIN_RERANK_SCORE -> επιστρέφει [] (το gate κόβει)."""
    _patch_search(monkeypatch, rerank_score=ai_core.MIN_RERANK_SCORE - 0.01)
    result = asyncio.run(ai_core.search_documents("what is cloud computing"))
    assert result == []


def test_relevance_gate_passes_above_threshold(monkeypatch):
    """Chunks πάνω από το κατώφλι -> επιστρέφει (text, metadata) ζεύγη."""
    _patch_search(monkeypatch, rerank_score=ai_core.MIN_RERANK_SCORE + 0.50)
    result = asyncio.run(ai_core.search_documents("what is cloud computing"))
    assert len(result) > 0
    text, meta = result[0]
    assert isinstance(text, str) and meta["file_name"] == "a.pdf"