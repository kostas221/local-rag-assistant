"""Σταματά μια μέτρηση όταν ένα όνομα μοντέλου έχει γίνει ΣΙΩΠΗΛΗ ανακατεύθυνση σε άλλο.

ΓΙΑΤΙ (18/11/2026): η Google ανακοίνωσε ότι από 1/12/2026 οι κλήσεις στο `gemini-2.5-flash`
πηγαίνουν αυτόματα στο `gemini-3.5-flash-lite` (και το `gemini-2.5-pro` στο `gemini-3.8-flash`),
χωρίς σφάλμα. Δεν υπάρχει «καρφωμένη» έκδοση (π.χ. -001). Ο κριτής όλων των μετρήσεων είναι το
`gemini-2.5-flash`: μετά την 1/12 ένα αρχείο αποτελεσμάτων θα έγραφε «κριτής gemini-2.5-flash» ενώ
θα βαθμολογούσε άλλο μοντέλο, και οι βαθμοί δεν θα συγκρίνονταν με τους παλιούς.
https://aistudio.google.com/docs/deprecations

Για σκόπιμη χρήση του νέου μοντέλου: γράψε ΡΗΤΑ το όνομα-στόχο (π.χ. JUDGE_MODEL=gemini-3.5-flash-lite),
ή ALLOW_REDIRECTED_MODEL=1 αν θέλεις πράγματι το παλιό όνομα.
"""
from __future__ import annotations

import os
from datetime import date

REDIRECTS = {
    "gemini-2.5-flash": (date(2026, 12, 1), "gemini-3.5-flash-lite"),
    "gemini-2.5-pro": (date(2026, 12, 1), "gemini-3.8-flash"),
}


def ensure_not_redirected(model: str, role: str, today: date | None = None) -> None:
    redirect = REDIRECTS.get(model)
    if not redirect or os.getenv("ALLOW_REDIRECTED_MODEL") == "1":
        return
    since, target = redirect
    if (today or date.today()) >= since:
        raise RuntimeError(
            f"{role}: το «{model}» από {since.isoformat()} απαντά ως «{target}». Τα αποτελέσματα θα "
            f"έγραφαν λάθος μοντέλο. Όρισε ρητά το μοντέλο (π.χ. {target}) ή ALLOW_REDIRECTED_MODEL=1."
        )
