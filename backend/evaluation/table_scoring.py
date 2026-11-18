"""Αυστηρός έλεγχος απαντήσεων πινάκων: τιμή + μονάδα + γραμμή/στήλη, όχι απλή εμφάνιση κειμένου.

ΓΙΑΤΙ (18/11/2026)
------------------
Ο παλιός έλεγχος (`eval_tables.check`) κοιτά μόνο αν το κείμενο της τιμής εμφανίζεται ΚΑΠΟΥ στην
απάντηση. Για την t001 (900 s, 0.5 GiB) δίνει 2/2 και στο «9000 seconds and 10.5 GB» (το «900» μέσα
στο «9000») και στο «0.5 seconds and 900 GB» (αντιμετάθεση). Οι 16 αποθηκευμένες απαντήσεις της
baseline ελέγχθηκαν με το χέρι απέναντι στα PDF και ήταν σωστές· αδύναμο ήταν το ΤΕΣΤ.

ΠΩΣ ΚΡΙΝΕΙ ΜΙΑ ΤΙΜΗ (`facts` στο golden_tables.jsonl)
----------------------------------------------------
  · αριθμός ως ΟΛΟΚΛΗΡΟ token, συγκρίνεται αριθμητικά με πρόσημο: το 900 δεν ταιριάζει στο 9000,
    το 0.5 δεν ταιριάζει στο 10.5, το 22 ταιριάζει στο 22.0. Δεκαδικό κόμμα («2,20») δεκτό· κόμμα με
    ακριβώς τρία ψηφία μετά («29,423») = διαχωριστικό χιλιάδων.
  · `unit`: μία από τις αποδεκτές μονάδες ακριβώς μετά τον αριθμό. Οι επιτρεπόμενες παραλλαγές
    (π.χ. GiB/GB) γράφονται ΡΗΤΑ στο σετ.
  · `anchor`: όταν δύο τιμές έχουν ίδια μονάδα (π.χ. μέση τιμή / τυπική απόκλιση), η πλησιέστερη
    ετικέτα της ΙΔΙΑΣ ερώτησης πρέπει να είναι η δική της. Πλησιέστερη: μέσα στο ίδιο τμήμα της
    πρότασης (χωρίζεται σε κόμματα εκτός παρενθέσεων, «;», whereas/while/compared to) σε οποιαδήποτε
    κατεύθυνση· αν το τμήμα δεν έχει ετικέτα, η πλησιέστερη ΠΡΟΗΓΟΥΜΕΝΗ στην πρόταση, μετά στη
    γραμμή, μετά η επόμενη στη γραμμή.
  · `row`: το ίδιο για τη γραμμή του πίνακα, με εμβέλεια τη γραμμή κειμένου (στοιχείο λίστας ή
    παράγραφο): πλησιέστερη προηγούμενη, αλλιώς επόμενη.
  · `text`: λεκτική τιμή (όνομα γραμμής, υπηρεσία) με όρια λέξης.

Ντετερμινιστικό, χωρίς μοντέλο και χωρίς βαριά imports: τρέχει και στο fast CI.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

_NUM = re.compile(r"(?<![\w.])([-+]?)(\d+(?:\.\d+)?)")
_AND = re.compile(r"\band\b")
_SPLIT_WORDS = re.compile(r"\b(?:whereas|while|compared (?:to|with)|versus|vs\.)\s*")


def normalize(text: str) -> str:
    """Πεζά, NFKC, χωρίς markdown έμφαση, ενιαίοι αριθμοί. Κρατά τις αλλαγές γραμμής."""
    s = unicodedata.normalize("NFKC", text or "").lower()
    s = s.replace("−", "-").replace("–", "-")
    s = re.sub(r"\*\*|__|`", "", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)          # 29,423 -> 29423
    s = re.sub(r"(?<![\w.])(\d{1,3}) (?=\d{3}(?!\d))", r"\1", s)  # 29 423 -> 29423
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)                  # 2,20 -> 2.20
    s = re.sub(r"[ \t]+", " ", s)
    return s


def _split_outside_parens(sentence: str) -> list[tuple[int, int]]:
    """(αρχή, τέλος) των τμημάτων μιας πρότασης: κόμμα ή «;» εκτός παρενθέσεων και λέξεις αντίθεσης."""
    cuts, depth = [], 0
    for i, ch in enumerate(sentence):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch in ",;" and depth == 0:
            cuts.append((i, i + 1))
    cuts += [(m.start(), m.end()) for m in _SPLIT_WORDS.finditer(sentence)]
    spans, start = [], 0
    for a, b in sorted(cuts):
        if a >= start:
            spans.append((start, a))
            start = b
    spans.append((start, len(sentence)))
    return spans


def _spans(text: str, pattern: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(pattern, text)]


def _nearest(text: str, lo: int, hi: int, pos: int, end: int, patterns: list[str],
             either: bool) -> str | None:
    """Η ετικέτα (pattern) πιο κοντά στον αριθμό μέσα στο [lo, hi).

    either=True: πλησιέστερη σε οποιαδήποτε κατεύθυνση. Αλλιώς: πλησιέστερη προηγούμενη, και μόνο
    αν δεν υπάρχει, πλησιέστερη επόμενη."""
    before, after = [], []  # (απόσταση, pattern, κείμενο ανάμεσα σε ετικέτα και αριθμό)
    for p in patterns:
        for a, b in _spans(text[lo:hi], p):
            a, b = a + lo, b + lo
            if b <= pos:
                before.append((pos - b, p, text[b:pos]))
            elif a >= end:
                after.append((a - end, p, text[end:a]))
    if either:
        # Ετικέτα που χωρίζεται από τον αριθμό με «and» ανήκει μάλλον στην ΑΛΛΗ τιμή: στο
        # «web and API serving accounts for 32% and data processing … 21%» το 32 είναι του web.
        both = [c for c in before + after if not _AND.search(c[2])] or before + after
        return min(both)[1] if both else None
    if before:
        return min(before)[1]
    return min(after)[1] if after else None


def _structure(text: str):
    """Γραμμές -> προτάσεις -> τμήματα, ως απόλυτες θέσεις στο κείμενο."""
    lines, sentences, segments = [], [], []
    for m in re.finditer(r"[^\n]+", text):
        lines.append((m.start(), m.end()))
        line = m.group(0)
        s0 = 0
        for sm in re.finditer(r"(?<=[.!?])\s+", line):
            sentences.append((m.start() + s0, m.start() + sm.start()))
            s0 = sm.end()
        sentences.append((m.start() + s0, m.end()))
    for a, b in sentences:
        for x, y in _split_outside_parens(text[a:b]):
            segments.append((a + x, a + y))
    return lines, sentences, segments


def _containing(spans: list[tuple[int, int]], pos: int) -> tuple[int, int]:
    for a, b in spans:
        if a <= pos < b:
            return a, b
    return 0, 0


def _anchor_ok(text, structure, pos, end, own, patterns) -> bool:
    lines, sentences, segments = structure
    seg = _containing(segments, pos)
    hit = _nearest(text, *seg, pos, end, patterns, either=True)
    if hit is None:
        sen = _containing(sentences, pos)
        hit = _nearest(text, sen[0], pos, pos, end, patterns, either=False)
    if hit is None:
        line = _containing(lines, pos)
        hit = _nearest(text, *line, pos, end, patterns, either=False)
    return hit == own


def _row_ok(text, structure, pos, end, own, patterns) -> bool:
    lines = structure[0]
    line = _containing(lines, pos)
    hit = _nearest(text, *line, pos, end, patterns, either=False)
    if hit is None:  # η ετικέτα γραμμής μπορεί να είναι σε προηγούμενη γραμμή (π.χ. επικεφαλίδα)
        hit = _nearest(text, 0, pos, pos, end, patterns, either=False)
    return hit == own


def check_fact(answer_norm: str, fact: dict[str, Any], facts: list[dict[str, Any]],
               structure=None) -> bool:
    structure = structure or _structure(answer_norm)
    if "text" in fact:
        return re.search(fact["text"], answer_norm) is not None
    want = float(fact["value"])
    tol = float(fact.get("tol", 1e-9))
    anchors = sorted({f["anchor"] for f in facts if "anchor" in f})
    rows = sorted({f["row"] for f in facts if "row" in f})
    for m in _NUM.finditer(answer_norm):
        sign = -1.0 if m.group(1) == "-" else 1.0
        if abs(sign * float(m.group(2)) - want) > tol:
            continue
        pos, end = m.start(), m.end()
        if "unit" in fact:
            rest = answer_norm[end:].lstrip()
            if not any(re.match(u, rest) for u in fact["unit"]):
                continue
        if "anchor" in fact and not _anchor_ok(answer_norm, structure, pos, end, fact["anchor"], anchors):
            continue
        if "row" in fact and not _row_ok(answer_norm, structure, pos, end, fact["row"], rows):
            continue
        return True
    return False


def check_strict(answer: str, facts: list[dict[str, Any]]) -> list[bool]:
    """Για κάθε fact: βρέθηκε με σωστή τιμή, μονάδα και θέση στον πίνακα;"""
    a = normalize(answer)
    structure = _structure(a)
    return [check_fact(a, f, facts, structure) for f in facts]
