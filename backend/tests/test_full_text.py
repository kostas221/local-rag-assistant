"""Ο router «όλο το κείμενο ή RAG» (κύκλος 2, 24/11/2026). Μόνο stdlib -> γρήγορα tests του CI.

Προστατεύει: (1) τον μετρημένο κανόνα (1 έγγραφο -> RAG· ≥2 και ≤ 70k -> όλο το κείμενο)· (2) ότι η
εκτίμηση tokens ΔΕΝ υποεκτιμά στα σημεία βαθμονόμησης του πειράματος — αν κάποιος «βελτιώσει» τις
σταθερές, το test δείχνει ότι ο router θα έστελνε περισσότερα απ' όσα μετρήθηκαν· (3) τη σειρά των
σελίδων (επιλογή -> σελίδα -> chunk), ίδια με το _expand_to_pages."""
import full_text as F

# Χαρακτήρες σελίδων ανά άρθρο (ίδια εξαγωγή + split με την εισαγωγή) και τα ΠΡΑΓΜΑΤΙΚΑ tokens εισόδου
# του Gemini στο πείραμα (μεγαλύτερο ανά επιλογή), runs/longctx/answers_*.json, 24/11/2026.
CHARS = {"1702": 44066, "1706": 52946, "1812": 62444, "1902": 117807, "EECS": 128171,
         "excamera": 73530, "mapreduce": 64397}
REAL = [(("1702",), 12367), (("1702", "1706"), 24569), (("1702", "1812"), 27140),
        (("1702", "1706", "1812"), 39212), (("1702", "1812", "1902"), 55332),
        (("EECS", "1706", "1902"), 71431), (("1902", "EECS", "excamera"), 78046),
        (("EECS", "1902"), 59351)]


def test_rule_one_document_stays_rag():
    assert not F.use_full_text(1, 5_000)


def test_rule_two_or_three_small_documents_go_full_text():
    assert F.use_full_text(2, 40_000)
    assert F.use_full_text(3, F.MAX_TOKENS)


def test_rule_too_large_stays_rag():
    assert not F.use_full_text(3, F.MAX_TOKENS + 1)
    assert not F.use_full_text(10, 200_000)


def test_estimate_never_underestimates_the_measured_prompts():
    for docs, real in REAL:
        est = int(sum(CHARS[d] for d in docs) / F.CHARS_PER_TOKEN_LATIN) + F.PROMPT_OVERHEAD_TOKENS
        assert est >= real, (docs, est, real)
        assert est <= real * 1.2, (docs, est, real)      # όχι υπερβολικά συντηρητική


def test_selection_above_the_limit_in_our_corpus_goes_to_rag():
    # Berkeley 2019 + EECS + ExCamera ≈ 78k tokens στο πείραμα -> πρέπει RAG
    est = int(sum(CHARS[d] for d in ("1902", "EECS", "excamera")) / F.CHARS_PER_TOKEN_LATIN) \
        + F.PROMPT_OVERHEAD_TOKENS
    assert not F.use_full_text(3, est)


def test_non_latin_text_is_counted_conservatively():
    assert F.estimate_tokens(["α" * 3800]) > F.estimate_tokens(["a" * 3800])


def test_pages_follow_selection_order_then_page_then_chunk():
    ids = ["b.pdf_p0_c1_x", "a.pdf_p1_c0_x", "b.pdf_p0_c0_x", "a.pdf_p0_c0_x", "a.pdf_p1_c1_x"]
    docs = ["B1-second", "A2-first", "B1-first", "A1", "A2-second"]
    metas = [{"doc_id": 7, "page": 1, "file_name": "b.pdf"}, {"doc_id": 3, "page": 2, "file_name": "a.pdf"},
             {"doc_id": 7, "page": 1, "file_name": "b.pdf"}, {"doc_id": 3, "page": 1, "file_name": "a.pdf"},
             {"doc_id": 3, "page": 2, "file_name": "a.pdf"}]
    pages = F.assemble_pages(ids, docs, metas, doc_order=[7, 3])     # ο χρήστης επέλεξε πρώτα το b
    assert [(m["file_name"], m["page"]) for _t, m in pages] == [("b.pdf", 1), ("a.pdf", 1), ("a.pdf", 2)]
    assert pages[0][0] == "B1-first\nB1-second"
    assert pages[2][0] == "A2-first\nA2-second"
    assert all(m["full_text"] for _t, m in pages)
