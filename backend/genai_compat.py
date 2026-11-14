"""Ένα σημείο εισαγωγής για το google.generativeai, με καταστολή του FutureWarning.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ
-----------------------------
Το `google.generativeai` είναι deprecated υπέρ του `google.genai`. Η αναβάθμιση
σε Pydantic 2 έχει πλέον ελεγχθεί (docs/STABILIZATION_2026-11-01.md), οπότε ο
παλιός περιορισμός συμβατότητας δεν ισχύει. Η αλλαγή SDK παραμένει ξεχωριστή
εργασία που χρειάζεται επανέλεγχο των eval wrappers. Η παραγωγή χρησιμοποιεί REST.

Η προειδοποίηση είναι γνωστό, τεκμηριωμένο χρέος. Αυτό το module περιορίζει το FutureWarning,
τυπώνεται ~15 γραμμές σε κάθε import και πνίγει την έξοδο κάθε eval run.

Η καταστολή είναι ΣΤΟΧΕΥΜΕΝΗ: `catch_warnings` σβήνει το φίλτρο μόλις τελειώσει
το import, ώστε καμία άλλη προειδοποίηση (δική μας ή τρίτων) να μη χαθεί. Ένα
`filterwarnings(module=...)` ΔΕΝ θα δούλευε — το warning αποδίδεται στη γραμμή
που κάνει το import, όχι στο google module.

ΧΡΗΣΗ: `from genai_compat import genai` αντί για `import google.generativeai as genai`.
"""
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

__all__ = ["genai"]
