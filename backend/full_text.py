"""Όλο το κείμενο των επιλεγμένων εγγράφων αντί για αναζήτηση, όταν χωράει (κύκλος 2, 24/11/2026).

ΑΠΟΦΑΣΗ, ΜΕΤΡΗΜΕΝΗ (docs/LONGCONTEXT_VS_RAG_PREREG_2026-11-23.md, GPT-4.1 κριτής, gemini-3.8-flash):
    1 επιλεγμένο έγγραφο                         -> RAG: ίδια ποιότητα (51 έναντι 52/52), ~3× φθηνότερο
    ≥ 2 επιλεγμένα, εκτίμηση ≤ 70.000 tokens       -> όλο το κείμενο: 2 άρθρα 28 έναντι 25/29, 3 άρθρα
                                                     11 έναντι 6/12· ασφάλεια 47/47 και στα δύο· όταν η
                                                     απάντηση είναι σε ΕΝΑ από τα επιλεγμένα 14/14·
                                                     1η λέξη σε ~3 s αντί ~27 s (αναζήτηση στη CPU)
    αλλιώς (πολλά ή μεγάλα έγγραφα)               -> RAG
    Τα 3 άρθρα ως τεκμηριωμένη εξαίρεση: κόστος 3,2× (το όριο ήταν 3×).

ΓΙΑΤΙ ΕΔΩ ΚΑΙ ΟΧΙ ΣΤΗ ΒΑΣΗ: το μέγεθος μετριέται τη στιγμή της ερώτησης, πάνω στις ΙΔΙΕΣ σελίδες που θα
σταλούν. Δεν χρειάζεται νέα στήλη ούτε migration για τα παλιά έγγραφα, και δεν υπάρχει αριθμός που
μπορεί να μείνει πίσω από το περιεχόμενο.

Leaf module (μόνο stdlib): οι κανόνες δοκιμάζονται χωρίς μοντέλα (tests/test_full_text.py).
"""
import re

MIN_DOCS = 2
MAX_TOKENS = 70_000          # η μεγαλύτερη ερώτηση 3 άρθρων ήταν 71.431 tokens -> κάτω στα 5k

# ΕΚΤΙΜΗΣΗ TOKENS ΑΠΟ ΧΑΡΑΚΤΗΡΕΣ — βαθμονομημένη στις 43 επιλογές εγγράφων του πειράματος (πραγματικά
# tokens εισόδου του Gemini έναντι χαρακτήρων των ίδιων σελίδων): chars/3,8 + 800 ΔΕΝ υποεκτίμησε ποτέ
# (+0,2% έως +16%). Υπερεκτίμηση = ασφαλής πλευρά: μια οριακή επιλογή πάει στο RAG, ποτέ το αντίθετο.
# Οι μη-λατινικοί χαρακτήρες (π.χ. ελληνικά άρθρα) ΔΕΝ μετρήθηκαν· /2 είναι συντηρητική υπόθεση.
CHARS_PER_TOKEN_LATIN = 3.8
CHARS_PER_TOKEN_OTHER = 2.0
PROMPT_OVERHEAD_TOKENS = 800

_CHUNK_IDX = re.compile(r"_c(\d+)_")


def chunk_idx(chunk_id: str) -> int:
    """Ίδιο με το ai_core._chunk_idx_from_id: id της μορφής '..._p{page}_c{idx}_{uuid}'."""
    m = _CHUNK_IDX.search(chunk_id)
    return int(m.group(1)) if m else 0


def assemble_pages(ids, documents, metadatas, doc_order) -> list[tuple[str, dict]]:
    """Όλες οι σελίδες, με τη σειρά των εγγράφων στην επιλογή και μετά με αριθμό σελίδας.

    Κάθε σελίδα = τα chunks της με τη σειρά τους, ενωμένα με '\\n' — ΙΔΙΟ κείμενο με το
    ai_core._expand_to_pages, άρα με αυτό που μετρήθηκε στο πείραμα."""
    by_page: dict = {}
    for cid, text, meta in zip(ids, documents, metadatas):
        key = (meta.get("doc_id"), int(meta.get("page")))
        entry = by_page.setdefault(key, {"file_name": meta.get("file_name"), "chunks": []})
        entry["chunks"].append((chunk_idx(cid), text))
    rank = {d: i for i, d in enumerate(doc_order)}
    pages = []
    for doc_id, page in sorted(by_page, key=lambda k: (rank.get(k[0], len(rank)), k[1])):
        entry = by_page[(doc_id, page)]
        text = "\n".join(t for _i, t in sorted(entry["chunks"], key=lambda c: c[0]))
        pages.append((text, {"file_name": entry["file_name"], "page": page, "doc_id": doc_id,
                             "full_text": True}))
    return pages


def estimate_tokens(texts) -> int:
    latin = other = 0
    for t in texts:
        a = len(t.encode("ascii", "ignore"))
        latin += a
        other += len(t) - a
    return int(latin / CHARS_PER_TOKEN_LATIN + other / CHARS_PER_TOKEN_OTHER) + PROMPT_OVERHEAD_TOKENS


def use_full_text(n_docs: int, est_tokens: int, *, min_docs: int = MIN_DOCS,
                  max_tokens: int = MAX_TOKENS) -> bool:
    return n_docs >= min_docs and est_tokens <= max_tokens
