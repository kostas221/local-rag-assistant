"""Corrective retrieval: 2ο pass όταν το relevance gate κόβει.

ΤΙ ΠΡΟΣΤΑΤΕΥΕΙ: ο agent είναι το ΜΟΝΟ σημείο όπου το σύστημα απαντά αφού ο
reranker έχει ήδη πει «τίποτα δεν ταιριάζει». Δύο πράγματα μπορούν να πάνε
στραβά και κανένα δεν πιάνεται από μετρική RAG:
  1. να τρέξει στο happy path -> +1.2s σε ΚΑΘΕ ερώτηση (μετρήθηκε ότι δεν
     πρέπει: 61/61 ερωτήσεις του golden set δεν τον αγγίζουν ποτέ)
  2. να χαλαρώσει την άμυνα -> απάντηση χωρίς σωστό υλικό. Μετρήθηκαν 2 τέτοιες
     περιπτώσεις στο probe· γι' αυτό υπάρχει το ΑΥΣΤΗΡΟΤΕΡΟ CORRECTIVE_MIN_SCORE.

Καμία κλήση δικτύου: το rewrite είναι stubbed. Ο μετρητής κλήσεων είναι το
ουσιώδες — «δεν καλέστηκε» είναι ισχυρότερος ισχυρισμός από «γύρισε []».
"""
import asyncio

import ai_core
import gemini_rest

# Το rewrite περιέχει αυτό το σημάδι -> ο stub reranker ξεχωρίζει 1ο από 2ο pass.
MARK = "REWRITTEN"


class _StubCollection:
    """Ίδιο σκεπτικό με το test_rag_core: get() γυρνά όλο το corpus, τα
    embeddings είναι ορθοκανονική βάση ώστε η κατάταξη να είναι προβλέψιμη."""

    def __init__(self, ids, texts, metas):
        self._ids, self._texts, self._metas = ids, texts, metas

    def get(self, where=None, include=None):
        out = {"ids": self._ids, "documents": self._texts,
               "metadatas": self._metas}
        if include and "embeddings" in include:
            out["embeddings"] = [[1.0 if j == i else 0.0 for j in range(8)]
                                 for i in range(len(self._ids))]
        return out

    def query(self, query_texts=None, n_results=10, where=None):
        return {"ids": [self._ids[:n_results]]}


class _QueryAwareReranker:
    """Διαφορετικό score ανά pass. Ο reranker βλέπει pairs=[[query, text], ...],
    οπότε το ίδιο το ερώτημα δείχνει σε ποιο pass βρισκόμαστε — χωρίς αυτό δεν
    μπορεί να δοκιμαστεί ΚΑΘΟΛΟΥ το δεύτερο κατώφλι."""

    def __init__(self, first, rewritten):
        self.first, self.rewritten = first, rewritten
        self.calls = 0

    def predict(self, pairs, batch_size=None):
        self.calls += 1
        q = pairs[0][0] if pairs else ""
        score = self.rewritten if MARK in q else self.first
        return [score] * len(pairs)


def _patch(monkeypatch, first_score, rewritten_score=None, rewrite=None,
           enabled=True):
    """Στήνει corpus + stubs. Επιστρέφει (reranker, counter) ώστε τα tests να
    ελέγχουν ΠΟΣΕΣ φορές έτρεξε καθένα — όχι μόνο το τελικό αποτέλεσμα."""
    ids = ["id0", "id1", "id2"]
    texts = ["chunk zero", "chunk one", "chunk two"]
    metas = [{"file_name": "a.pdf", "page": i + 1} for i in range(3)]
    rr = _QueryAwareReranker(first_score,
                             first_score if rewritten_score is None
                             else rewritten_score)
    monkeypatch.setattr(ai_core, "collection", _StubCollection(ids, texts, metas))
    monkeypatch.setattr(ai_core, "reranker", rr)
    monkeypatch.setattr(ai_core, "_bm25_cache", {"version": None})
    monkeypatch.setattr(ai_core, "_dense_cache", {"version": None})
    # Απομόνωση από το παραγωγικό npz cache — τα stub διανύσματα είναι 8-διάστατα
    # και θα έσπαγαν μια πραγματική ερώτηση με το ίδιο κείμενο.
    monkeypatch.setattr(ai_core, "_query_emb_cache", {})
    monkeypatch.setattr(ai_core, "_save_query_emb_cache", lambda: None)
    monkeypatch.setattr(ai_core, "sentence_transformer_ef",
                        lambda texts: [[1.0] + [0.0] * 7 for _ in texts])
    monkeypatch.setattr(ai_core, "USE_BGE_SPARSE", False)
    monkeypatch.setattr(ai_core, "ENABLE_CORRECTIVE", enabled)

    async def _no_translate(q):
        return q
    monkeypatch.setattr(ai_core, "optimize_query", _no_translate)

    calls = {"n": 0}

    async def _fake_rewrite(prompt, *, model=None, api_key=None, **kw):
        calls["n"] += 1
        if callable(rewrite):
            return rewrite(prompt)
        return rewrite if rewrite is not None else f"{MARK} storage terminology"
    monkeypatch.setattr(gemini_rest, "generate_once", _fake_rewrite)
    return rr, calls


def _search():
    return asyncio.run(ai_core.search_documents("what is cloud computing"))


def test_no_retry_when_gate_passes(monkeypatch):
    """ΤΟ ΠΙΟ ΣΗΜΑΝΤΙΚΟ: στο happy path ο agent δεν πρέπει να αγγιχτεί καν.
    Αλλιώς κάθε ερώτηση πληρώνει +1.2s για δουλειά που δεν χρειάζεται."""
    rr, calls = _patch(monkeypatch, first_score=ai_core.MIN_RERANK_SCORE + 5.0)
    result = _search()
    assert len(result) > 0
    assert calls["n"] == 0, "το rewrite κλήθηκε ενώ το gate είχε περάσει"
    assert rr.calls == 1, "έγινε δεύτερο rerank χωρίς λόγο"


def test_retry_saves_question_when_rewrite_scores_high(monkeypatch):
    """Το σενάριο h005: 1ο pass κόβει, το rewrite φέρνει σκορ πάνω από το
    αυστηρότερο κατώφλι -> το σύστημα απαντά αντί να σιωπήσει."""
    rr, calls = _patch(monkeypatch,
                       first_score=ai_core.MIN_RERANK_SCORE - 1.0,
                       rewritten_score=ai_core.CORRECTIVE_MIN_SCORE + 1.0)
    result = _search()
    assert len(result) > 0
    text, meta = result[0]
    assert isinstance(text, str) and meta["file_name"] == "a.pdf"
    assert calls["n"] == 1
    assert rr.calls == 2, "το 2ο pass πρέπει να ξανακάνει rerank"


def test_retry_respects_stricter_threshold(monkeypatch):
    """ΤΟ ΣΕΝΑΡΙΟ ΤΩΝ ΨΕΥΔΑΙΣΘΗΣΕΩΝ: το rewrite περνά το ΑΡΧΙΚΟ κατώφλι αλλά
    ΟΧΙ το αυστηρότερο του retry -> πρέπει να μείνει κομμένο. Χωρίς αυτόν τον
    έλεγχο, το keyword-stuffing του rewrite περνά το gate με λάθος υλικό
    (μετρήθηκε 2 φορές: h015 στο 0.95, h016 στο -1.76)."""
    between = (ai_core.MIN_RERANK_SCORE + ai_core.CORRECTIVE_MIN_SCORE) / 2
    assert ai_core.MIN_RERANK_SCORE < between < ai_core.CORRECTIVE_MIN_SCORE
    _, calls = _patch(monkeypatch,
                      first_score=ai_core.MIN_RERANK_SCORE - 1.0,
                      rewritten_score=between)
    assert _search() == []
    assert calls["n"] == 1, "το rewrite έπρεπε να δοκιμαστεί μία φορά"


def test_disabled_agent_falls_back_to_old_behaviour(monkeypatch):
    """ENABLE_CORRECTIVE=0 -> ακριβώς η συμπεριφορά πριν τον agent. Ο διακόπτης
    πρέπει να δουλεύει: κάθε μέτρηση συγκρίνει on/off, και αν το off δεν είναι
    πραγματικά off, η σύγκριση έχει δύο μεταβλητές."""
    _, calls = _patch(monkeypatch,
                      first_score=ai_core.MIN_RERANK_SCORE - 1.0,
                      rewritten_score=ai_core.CORRECTIVE_MIN_SCORE + 5.0,
                      enabled=False)
    assert _search() == []
    assert calls["n"] == 0, "ο σβηστός agent δεν πρέπει να καλεί το API"


def test_rewrite_failure_degrades_to_silence(monkeypatch):
    """Σφάλμα/rate-limit στο Gemini -> [] (η παλιά συμπεριφορά), ΠΟΤΕ εξαίρεση
    προς τον χρήστη. Το quota τελείωσε ήδη μία φορά μέσα σε μέτρηση."""
    def _boom(_prompt):
        raise RuntimeError("429 quota exhausted")
    _patch(monkeypatch, first_score=ai_core.MIN_RERANK_SCORE - 1.0,
           rewritten_score=ai_core.CORRECTIVE_MIN_SCORE + 5.0, rewrite=_boom)
    assert _search() == []


def test_identical_rewrite_skips_second_retrieval(monkeypatch):
    """Αν το rewrite γυρίσει το ΙΔΙΟ ερώτημα (συμβαίνει σκόπιμα στα εκτός
    θέματος — q020/q049 μετρήθηκαν έτσι), το 2ο pass θα έδινε ταυτόσημο
    αποτέλεσμα. Μη σπαταλάς 450ms CPU για να το επιβεβαιώσεις."""
    rr, calls = _patch(monkeypatch, first_score=ai_core.MIN_RERANK_SCORE - 1.0,
                       rewrite="what is cloud computing")
    assert _search() == []
    assert calls["n"] == 1
    assert rr.calls == 1, "δεν πρέπει να γίνει δεύτερο rerank για ίδιο ερώτημα"


def test_empty_rewrite_is_treated_as_no_change(monkeypatch):
    """Κενή απάντηση από το μοντέλο -> κομμένο, χωρίς δεύτερη ανάκτηση."""
    rr, _ = _patch(monkeypatch, first_score=ai_core.MIN_RERANK_SCORE - 1.0,
                   rewrite="")
    assert _search() == []
    assert rr.calls == 1


def test_thresholds_are_ordered(monkeypatch):
    """Ο agent στηρίζεται στο ότι το retry είναι ΑΥΣΤΗΡΟΤΕΡΟ. Αν κάποιος γυρίσει
    το CORRECTIVE_MIN_SCORE κάτω από το MIN_RERANK_SCORE μέσω env, ο agent
    γίνεται χαλαρωτής της άμυνας αντί για βοηθός — σιωπηλά."""
    assert ai_core.CORRECTIVE_MIN_SCORE > ai_core.MIN_RERANK_SCORE
