"""Κριτής επάρκειας για τη δεύτερη αναζήτηση (κύκλος 2, βήμα 2β, 23/11/2026).

ΤΙ ΚΑΝΕΙ: όταν η απάντηση πάει να βγει από τη ΔΕΥΤΕΡΗ αναζήτηση (αναδιατυπωμένη ερώτηση), ένα μοντέλο
κρίνει αν οι σελίδες απαντούν την ΑΡΧΙΚΗ ερώτηση. ΟΧΙ -> σιωπή.

ΓΙΑΤΙ: ο reranker κρίνει με λέξεις. Η αναδιατύπωση προσθέτει λέξεις του πεδίου («GDPR compliance» ->
«... cloud service providers») και μια άσχετη ερώτηση φαίνεται σχετική. Ένα μοντέλο κρίνει με νόημα.
Ίδια ιδέα με τον «retrieval evaluator» του CRAG και τον «autorater» του Sufficient Context· έρευνα και
επιλογές στο docs/CYCLE2_PLAN_2026-11-23.md, κανόνας στο docs/SUFFICIENCY_VETO_PREREG_2026-11-23.md.

ΤΟ PROMPT είναι αυτό του Αυγούστου (evaluation/probe_corrective_verdict.py), αυτούσιο: τότε είπε ΟΧΙ και
στις 3 άσχετες που έφτασαν ως εκεί (2.5). Το ίδιο κείμενο μετριέται και, αν περάσει, μπαίνει στην εφαρμογή.

Leaf module: μόνο stdlib (δοκιμάζεται χωρίς Docker).
"""

from __future__ import annotations

import re

VERDICT_PROMPT = """You are verifying whether retrieved material is sufficient to answer a question.

Read the SOURCE TEXT and the QUESTION. Decide whether the SOURCE TEXT actually
contains the information needed to answer the QUESTION, even partially.

Answer with EXACTLY ONE WORD:
YES - the SOURCE TEXT contains information that answers the question
NO  - the SOURCE TEXT does not contain it; answering would need outside knowledge

Do not explain. Output only YES or NO.

--- SOURCE TEXT ---
{context}

QUESTION: {question}

VERDICT:"""


def build_prompt(question: str, pages: list[tuple[str, dict]]) -> str:
    """Οι σελίδες με την ίδια επικεφαλίδα που βλέπει και η απάντηση ([S1: αρχείο, Page: N])."""
    context = "\n\n".join(
        f"[S{i}: {meta.get('file_name', '?')}, Page: {meta.get('page', '?')}]\n{text}"
        for i, (text, meta) in enumerate(pages, 1)
    )
    return VERDICT_PROMPT.format(context=context, question=question)


def parse_verdict(raw: str | None) -> str | None:
    """'YES' / 'NO' / None (ακατανόητο). Δέχεται «Yes.», «**NO**», «VERDICT: NO»· ΟΧΙ το «NOT SURE»."""
    t = re.sub(r"[*_`\"'.:!]", " ", (raw or "")).strip().upper()
    t = re.sub(r"^VERDICT\s+", "", t)
    first = t.split()[0] if t.split() else ""
    return first if first in ("YES", "NO") else None


def vetoes(verdict: str | None) -> bool:
    """Κόβει ΜΟΝΟ με ρητό ΟΧΙ. Αποτυχία κλήσης ή ακατανόητη απάντηση -> όχι βέτο, δηλαδή η συμπεριφορά
    πριν από τον κριτή: ο κριτής δεν μπορεί ποτέ να κάνει το σύστημα χειρότερο από σήμερα λόγω σφάλματος."""
    return verdict == "NO"
