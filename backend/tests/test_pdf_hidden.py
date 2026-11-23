"""Κρυμμένο κείμενο σε PDF (pdf_hidden.py, βήμα 5α). Χρειάζεται pymupdf -> core-tests.

Κλειδώνουμε: (1) λευκό σε λευκή σελίδα, μικροσκοπικό, εκτός σελίδας = κρυμμένο· (2) λευκό ΠΑΝΩ σε
σκούρο σχήμα = ορατό (ετικέτες διαγραμμάτων· η πρώτη εκδοχή τα έπιανε λάθος)· (3) σελίδα χωρίς κρυμμένο
κείμενο -> ακριβώς το ίδιο get_text() με πριν."""
import pymupdf

import pdf_hidden as H


def _page(build):
    doc = pymupdf.open()
    page = doc.new_page()
    build(page)
    return doc, page


def test_white_tiny_and_offpage_text_are_hidden():
    def build(p):
        p.insert_text((72, 72), "Visible finding: 87.3% retention.")
        p.insert_text((72, 100), "white instruction", color=(1, 1, 1))
        p.insert_text((72, 120), "tiny instruction", fontsize=0.5)
        p.insert_text((900, 900), "offpage instruction")
    doc, page = _page(build)
    reasons = {s["text"]: s["reason"] for s in H.hidden_spans(page)}
    assert reasons == {"white instruction": "white", "tiny instruction": "tiny"}
    # Κείμενο έξω από τη σελίδα: το PyMuPDF δεν το εξάγει καν, άρα δεν φτάνει ποτέ στο μοντέλο.
    assert "offpage" not in page.get_text()
    text, spans = H.visible_text(page)
    assert "87.3%" in text and "instruction" not in text and len(spans) == 2
    doc.close()


def test_white_label_on_dark_bar_is_visible():
    def build(p):
        p.draw_rect(pymupdf.Rect(60, 60, 300, 90), color=None, fill=(0.1, 0.1, 0.3))
        p.insert_text((72, 80), "Input (MB/s)", color=(1, 1, 1))
    doc, page = _page(build)
    assert H.hidden_spans(page) == []
    doc.close()


def test_page_without_hidden_text_is_returned_unchanged():
    def build(p):
        p.insert_text((72, 72), "Plain page with normal text.")
    doc, page = _page(build)
    text, spans = H.visible_text(page)
    assert spans == [] and text == page.get_text()
    doc.close()
