"""Από πόσες σελίδες ενός PDF εξήχθη κείμενο (κύκλος 2, βήμα 1, 23/11/2026).

ΓΙΑΤΙ: ένα PDF με κάποιες σαρωμένες σελίδες (εικόνα χωρίς στρώμα κειμένου) γινόταν «έτοιμο» αρκεί να
έδινε κείμενο έστω μία σελίδα. Αν η απάντηση ήταν σε σαρωμένη σελίδα, ο βοηθός έλεγε «τα έγγραφα δεν
το αναφέρουν», και ο χρήστης δεν μπορούσε να ξεχωρίσει το «δεν διαβάστηκε» από το «το άρθρο δεν το
λέει». Τώρα κάθε έγγραφο κρατά: συνολικές σελίδες, σελίδες με κείμενο, και ποιες ήταν χωρίς κείμενο.

ΤΙ ΔΕΝ ΛΕΕΙ: «εξήχθη κείμενο» δεν σημαίνει «διαβάστηκε σωστά». Πίνακες και διαγράμματα μπορεί να
έδωσαν κείμενο σε λάθος σειρά ή καθόλου. Γι' αυτό η διατύπωση στο UI είναι «Text extracted from X/Y
pages», όχι «διαβάστηκαν».

Η εισαγωγή (ai_core.ingest_pdf) καλεί το `summarize` πάνω στο ΙΔΙΟ κείμενο που ευρετηριάζει. Το
`from_pdf` υπάρχει για έγγραφα που ανέβηκαν πριν από τη στήλη (migrate_add_extraction_report.py).
Leaf module: μόνο stdlib (+ pymupdf μόνο μέσα στο from_pdf).
"""

from __future__ import annotations


def summarize(page_texts: list[str], image_pages=()) -> dict:
    """page_texts: το κείμενο κάθε σελίδας με τη σειρά (1η = σελίδα 1). image_pages: αριθμοί σελίδων
    που περιέχουν εικόνα. Σελίδα χωρίς κείμενο ΚΑΙ με εικόνα = πιθανώς σαρωμένη (θέλει OCR)."""
    empty = [i for i, text in enumerate(page_texts, 1) if not (text or "").strip()]
    images = set(image_pages)
    return {
        "pages_total": len(page_texts),
        "pages_text": len(page_texts) - len(empty),
        "pages_empty": empty,
        "pages_scanned": [p for p in empty if p in images],
    }


def from_pdf(path: str) -> dict:
    import pymupdf

    with pymupdf.open(path) as doc:
        texts = [page.get_text() or "" for page in doc]
        images = [i for i, page in enumerate(doc, 1) if page.get_images()]
    return summarize(texts, images)


def pages_to_str(pages: list[int]) -> str:
    """[4, 7] -> "4,7" για αποθήκευση σε μία στήλη της βάσης."""
    return ",".join(str(p) for p in pages)


def pages_from_str(value: str | None) -> list[int]:
    return [int(p) for p in (value or "").split(",") if p.strip().isdigit()]


def describe(pages_total: int | None, pages_text: int | None, pages_empty: list[int] | None = None,
             max_listed: int = 8) -> str:
    """Η γραμμή του UI. Κενό string όταν δεν υπάρχει αναφορά (έγγραφα πριν από τη στήλη)."""
    if not pages_total:
        return ""
    line = f"Text extracted from {pages_text}/{pages_total} pages"
    empty = pages_empty or []
    if empty:
        shown = ", ".join(str(p) for p in empty[:max_listed])
        more = f" and {len(empty) - max_listed} more" if len(empty) > max_listed else ""
        line += f" · no text on p. {shown}{more}"
    return line
