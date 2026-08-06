"""Ένα σημείο εισαγωγής για το google.generativeai, με καταστολή του FutureWarning.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ
-----------------------------
Το `google.generativeai` είναι deprecated υπέρ του `google.genai`. Η μετανάστευση
ΔΕΝ είναι εφικτή εδώ: το `google-genai` (2.15.0) απαιτεί `pydantic>=2.12.5`, ενώ
το stack είναι καρφωμένο σε `pydantic 1.10.26 / FastAPI 0.99.1` για bit-exact
αναπαραγωγιμότητα των μετρήσεων της διπλωματικής (βλ. ARCHITECTURE.md). Και δεν
υπάρχει διαφυγή σε offline container όπως έγινε με άλλα εργαλεία — το SDK πρέπει
να τρέχει ΜΕΣΑ στο serving path.

Άρα η προειδοποίηση είναι γνωστό, τεκμηριωμένο χρέος και ΟΧΙ κάτι που μπορεί να
διορθωθεί. Το πρόβλημα που λύνει αυτό το module είναι πρακτικό: το FutureWarning
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
