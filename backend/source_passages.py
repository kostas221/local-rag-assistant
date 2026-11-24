"""Απόσπασμα πηγής ανά πρόταση για το UI (κύκλος 2, βήμα 6, docs/SOURCE_PASSAGES_PREREG_2026-11-24.md).

Χτίζει ΑΥΤΟΥΣΙΟ τον έλεγχο του βήματος 5 (claim_support): ίδιο prompt, ίδια επαλήθευση αυτούσιου χωρίου.
Από εκείνον κρατά ΜΟΝΟ το χωρίο. Η κρίση full/partial/none απορρίφθηκε στο βήμα 5 και δεν φτάνει ποτέ
στον χρήστη, άρα εδώ δεν υπάρχει «επαληθευμένο»: χωρίο που βρέθηκε δεν σημαίνει ότι στηρίζει όλη την πρόταση.

Μία διαφορά από το βήμα 5: ο έλεγχος παίρνει μόνο τις σελίδες που ΠΑΡΑΘΕΤΕΙ η απάντηση. Στο «όλο το κείμενο»
δόθηκαν 30–50 σελίδες, και δεν θα πληρώναμε για όλες.

Leaf module (stdlib + claim_support): δοκιμάζεται χωρίς μοντέλα (tests/test_source_passages.py).
"""
from __future__ import annotations

import re

import claim_support as CS

# Ίδιες ρυθμίσεις με το probe_claim_support (βήμα 5), όπου μετρήθηκε ο έλεγχος.
CHECK_KW = {"temperature": 0.0, "max_output_tokens": 8192, "thinking_budget": 1024}

# Προτάσεις άρνησης/απουσίας («τα έγγραφα δεν αναφέρουν…»): ΑΝΤΙΓΡΑΦΟ του eval_near_ooc.REFUSAL, με
# το οποίο μετρήθηκε το βήμα 5. Το tests/test_source_passages.py ελέγχει ότι παραμένουν ίδια.
_W = r"(?:\w+\s+){0,3}"
REFUSAL = re.compile(
    r"δεν\s+" + _W + r"(?:βρ[ίι]σκ|βρ[έε]θ|περι[έε]χ|αναφ[έε]ρ|υπ[άα]ρχ|"
    r"παρ[έε]χ|μπορ|δ[ίι]ν|συζητ|εξετ|σχολι|προσδιορ|τεκμηρι|ανακτ)"
    r"|ουδεμ[ίι]α\s+αναφορ|καμ[ίι]α\s+" + _W + r"(?:αναφορ|πληροφορ|μνε[ίι]α)"
    r"|there\s+(?:is|are|was|were)\s+no\s"
    r"|(?:do|does|did)\s+not\s+" + _W + r"(?:contain|mention|provide|report|"
    r"discuss|specify|include|address|state|say|detail|present)"
    r"|(?:is|are|was|were)\s+not\s+" + _W + r"(?:contained|mentioned|provided|"
    r"reported|discussed|specified|included|addressed|present|available|found)"
    r"|\bno\s+" + _W + r"(?:information|answer|mention|reference|results?|data|"
    r"details?|evidence|benchmark|figures?)\b"
    r"|not\s+(?:explicitly\s+|specifically\s+|directly\s+)?(?:mentioned|discussed|"
    r"stated|specified|provided|reported|addressed|contained|available|present)"
    r"|cannot\s+" + _W + r"(?:find|determine|answer|provide|locate)"
    r"|unable\s+to\s+" + _W + r"(?:find|determine|answer|provide|locate)",
    re.I)

# Οι καταστάσεις που βλέπει ο χρήστης (το UI έχει ένα κείμενο για την καθεμία).
FOUND = "found"                 # χωρίο αυτούσιο σε σελίδα που παραθέτει η πρόταση
FOUND_ELSEWHERE = "found_elsewhere"  # αυτούσιο, αλλά σε άλλη από τις παρατεθειμένες σελίδες
NOT_FOUND = "not_found"         # το μοντέλο δεν έδωσε χωρίο ή δεν βρέθηκε αυτούσιο
NO_CITATION = "no_citation"     # η πρόταση δεν παραθέτει πηγή
ABSENCE = "absence"             # λέει τι ΛΕΙΠΕΙ από τις πηγές: δεν υπάρχει χωρίο να δειχτεί
STATUSES = (FOUND, FOUND_ELSEWHERE, NOT_FOUND, NO_CITATION, ABSENCE)


def cited_labels(answer: str) -> list[str]:
    """Οι ετικέτες που παραθέτει πράγματι η απάντηση, με τη σειρά πρώτης εμφάνισης."""
    labels = [lab for c in CS.split_claims(answer, REFUSAL) for lab in c.labels]
    return list(dict.fromkeys(labels))


def prepare(question: str, answer: str, texts: dict[str, str]):
    """(προτάσεις, prompt ή None). texts: ετικέτα -> κείμενο σελίδας, ΜΟΝΟ των παρατεθειμένων."""
    claims = CS.split_claims(answer, REFUSAL)
    # Προτάσεις χωρίς παραπομπή: δεν ελέγχονται (ο έλεγχος παίρνει μόνο παρατεθειμένες σελίδες).
    for c in claims:
        if c.kind == "claim" and not c.labels:
            c.kind = "uncited"
    return claims, CS.build_check_prompt(question, claims, texts)


def to_items(claims, sources_by_label: dict[str, dict]) -> list[dict]:
    """Μία γραμμή ανά πρόταση για το UI, χωρίς τις εισαγωγές λιστών («Τα εμπόδια είναι:»)."""
    items = []
    for c in claims:
        if c.kind == "lead_in":
            continue
        if c.absence:
            status = ABSENCE
        elif c.kind in ("uncited", "refusal") or not c.labels:
            status = NO_CITATION
        elif c.quote_found and c.quote_label in c.labels:
            status = FOUND
        elif c.quote_found:
            status = FOUND_ELSEWHERE
        else:
            status = NOT_FOUND
        src = sources_by_label.get(c.quote_label or "") if status in (FOUND, FOUND_ELSEWHERE) else None
        items.append({
            "i": c.i, "sentence": c.text, "labels": c.labels, "status": status,
            "quote": c.quote if src else None, "label": c.quote_label if src else None,
            "file": src.get("file") if src else None, "page": src.get("page") if src else None,
        })
    return items
