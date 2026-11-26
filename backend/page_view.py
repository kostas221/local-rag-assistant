"""Εικόνα σελίδας PDF με το χωρίο υπογραμμισμένο (προβολή, 26/11/2026).

Ο χρήστης πατά «Σελίδα 5» δίπλα σε μια πηγή ή σε ένα χωρίο και βλέπει την ΠΡΑΓΜΑΤΙΚΗ σελίδα του άρθρου:
οι πίνακες όπως είναι τυπωμένοι, όχι ως σκέτο κείμενο. Είναι προσθήκη προβολής μετά την παγωμένη έκδοση
(`c15a395`): δεν αγγίζει αναζήτηση ούτε απαντήσεις, και η μέτρηση του κλειστού σετ μένει έγκυρη.

ΚΙΤΡΙΝΟ ΜΟΝΟ ΟΤΑΝ ΤΟ ΣΗΜΕΙΟ ΕΙΝΑΙ ΜΟΝΑΔΙΚΟ. Λάθος υπογράμμιση είναι χειρότερη από καμία: δείχνει στον
χρήστη σίγουρα το λάθος σημείο. Το χωρίο εντοπίζεται με την ΙΔΙΑ κανονικοποίηση και την ίδια λογική
κομματιών με τον έλεγχο χωρίων (claim_support.quote_in_source), λέξη-λέξη πάνω στη σελίδα, ώστε αλλαγές
γραμμής και λέξεις κομμένες με παύλα να μη χαλούν τον εντοπισμό. Στους πίνακες το χωρίο μπορεί να είναι
κελιά χωρισμένα με «...» ή αλλαγή γραμμής: κίτρινο μόνο αν ΚΑΘΕ κομμάτι υπάρχει ΜΙΑ φορά στη σελίδα.

Καταστάσεις (`highlight`):
  highlighted  κάθε κομμάτι βρέθηκε ακριβώς μία φορά -> κίτρινο εκεί·
  ambiguous    κάποιο κομμάτι υπάρχει σε περισσότερα σημεία -> καθόλου κίτρινο·
  not_found    το χωρίο δεν εντοπίστηκε στις λέξεις της σελίδας -> καθόλου κίτρινο·
  none         δεν ζητήθηκε χωρίο (απλό άνοιγμα σελίδας).
Το pymupdf φορτώνεται μόνο όταν χρειάζεται: το main.py εισάγει αυτό το module, και τα tests του API
τρέχουν χωρίς pymupdf.
"""
from __future__ import annotations

import re

import claim_support as CS

ZOOM = 2.0                 # ~144 dpi: διαβάζονται οι πίνακες, και η μεγέθυνση του UI έχει περιθώριο
MAX_QUOTE_CHARS = 4000
MIN_FRAGMENT = 4           # όπως το κατώτατο του apply_check (max(4, ...))


class PageNotFound(Exception):
    """Η σελίδα δεν υπάρχει στο PDF."""


def canonical_words(words: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
    """Η σελίδα στην κανονική μορφή του ελέγχου χωρίων (claim_support._canon), φτιαγμένη λέξη-λέξη ώστε
    κάθε χαρακτήρας να ξέρει από ποια λέξη ήρθε. Επιστρέφει (κείμενο, [(αρχή, τέλος, δείκτης λέξης)]).

    Οι κανόνες του _canon που περνούν ΑΝΑΜΕΣΑ σε λέξεις εφαρμόζονται στα όρια: παύλα ανάμεσα σε γράμματα
    φεύγει («distribu-» + «ted» = «distributed»), και κενό ανάμεσα σε δύο αριθμούς γίνεται «|»."""
    out = ""
    spans: list[tuple[int, int, int]] = []
    for idx, word in enumerate(words):
        c = CS._canon(word)
        if not c:
            continue
        if out:
            if out[-1] == "-" and len(out) >= 2 and out[-2].isalpha() and c[0].isalpha():
                out = out[:-1]
                s, e, i = spans[-1]
                spans[-1] = (s, e - 1, i)
            elif out[-1].isdigit() and c[0].isdigit():
                out += "|"
        start = len(out)
        out += c
        spans.append((start, len(out), idx))
    return out, spans


def _occurrences(frag: str, page: str) -> list[tuple[int, int]]:
    """Όλα τα σημεία όπου το κομμάτι υπάρχει ΟΛΟΚΛΗΡΟ (ίδιος κανόνας με το claim_support._found_whole:
    ένας αριθμός δεν επιτρέπεται να συνεχίζεται στη σελίδα)."""
    hits, start = [], page.find(frag)
    while start != -1:
        end = start + len(frag)
        left_ok = not frag[0].isdigit() or start == 0 or page[start - 1] not in CS._NUM_LEFT
        right_ok = not frag[-1].isdigit() or end == len(page) or page[end] not in CS._NUM_RIGHT
        if left_ok and right_ok:
            hits.append((start, end))
        start = page.find(frag, start + 1)
    return hits


def locate(quote: str, words: list[str]) -> tuple[str, list[int]]:
    """(κατάσταση, δείκτες λέξεων για κίτρινο). Τα κομμάτια χωρίζονται όπως στο quote_in_source: πρώτα
    «...» ή κενή γραμμή, και μόνο αν αυτό δεν βρεθεί, γραμμή-γραμμή (κελιά πίνακα)."""
    page, spans = canonical_words(words)
    quote = (quote or "")[:MAX_QUOTE_CHARS]
    for split in CS._FRAGMENT_SPLITS:
        frags = [f for f in (CS._canon(x) for x in split.split(quote)) if f]
        frags = [re.sub(r"(?<=[^\W\d_])-$", "", f) for f in frags]
        if not frags or max(map(len, frags)) < MIN_FRAGMENT:
            continue
        occ = [_occurrences(f, page) for f in frags]
        if not all(occ):
            continue
        if any(len(o) > 1 for o in occ):
            return "ambiguous", []
        idxs = sorted({i for (s, e) in (o[0] for o in occ) for (ws, we, i) in spans if ws < e and we > s})
        return "highlighted", idxs
    return "not_found", []


def render(pdf_path: str, page_no: int, quote: str | None = None, zoom: float = ZOOM) -> tuple[bytes, str]:
    """PNG της σελίδας `page_no` (από 1, όπως στις πηγές) και η κατάσταση της υπογράμμισης. Το PDF ανοίγει
    μόνο για ανάγνωση: η υπογράμμιση μπαίνει στη μνήμη, το αρχείο δεν αλλάζει."""
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        if not 1 <= page_no <= doc.page_count:
            raise PageNotFound(page_no)
        page = doc[page_no - 1]
        status = "none"
        if quote and quote.strip():
            words = page.get_text("words", sort=False)
            status, idxs = locate(quote, [w[4] for w in words])
            if idxs:
                annot = page.add_highlight_annot([pymupdf.Rect(words[i][:4]) for i in idxs])
                annot.update()
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), annots=True)
        return pix.tobytes("png"), status
