"""Ερωτήσεις ΔΥΟ ΕΓΓΡΑΦΩΝ: δρομολόγηση με ονόματα — leaf module, ΚΑΝΕΝΑ μοντέλο (fast CI).

ΤΙ ΛΥΝΕΙ (Φάση 2, 29-30/9/2026): σε ερώτηση για δύο papers το σύστημα ψάχνει ΜΙΑ φορά, το ένα θέμα
πιάνει τις 8 θέσεις και η σελίδα με το στοιχείο του άλλου δεν φτάνει στο Gemini (29 ερωτήσεις
δύο εγγράφων: σελίδες-τεκμήρια 30/58· όταν λείπει η σελίδα, η απάντηση είναι σωστή 6/28).
Αν η ερώτηση ΟΝΟΜΑΖΕΙ ≥ 2 έγγραφα, η ΙΔΙΑ ερώτηση ψάχνεται και ΜΕΣΑ σε κάθε έγγραφο χωριστά (εκεί
ανταγωνίζονται μόνο σελίδες του ίδιου paper) και προστίθενται έως 4 σελίδες:
    evaluation/probe_route_perdoc.py   σελίδες-τεκμήρια 30 -> 42/58, «και τα δύο papers» 7 -> 18/29
ΓΙΑΤΙ ΟΝΟΜΑΤΑ ΚΑΙ ΟΧΙ LLM:
    evaluation/probe_decomp_nonmh.py   το LLM (διάσπαση του Αυγούστου) σπάει 35/50 του κύριου σετ και
                                       36/42 των κοντινών ooc — με παράφραση, όχι με θέματα
    evaluation/probe_route_names.py    ο κανόνας ονομάτων: 0/170 ψευδώς θετικά σε 8 σετ, 0 κλήσεις
ΟΡΙΟ (γνωστό, τεκμηριωμένο): τα ονόματα ζουν στο corpus_names.json, γραμμένα με το χέρι για τα 7
papers του σώματος. Έγγραφο χωρίς ονόματα -> η λειτουργία απλώς δεν ενεργοποιείται, δηλαδή η σημερινή
συμπεριφορά. Αυτόματη εξαγωγή ονομάτων στο ingest (τίτλος, συγγραφέας, όνομα συστήματος) = ανοιχτό.
Περιφράσεις («the video processing system») και αντωνυμίες ΔΕΝ πιάνονται — επίτηδες: εκεί το LLM
routing έσπαγε και ερωτήσεις ενός θέματος.
"""
import itertools
import json


def load_names(path: str) -> dict[str, list[str]]:
    """{αρχείο: [ονόματα σε πεζά]} από το κλειδί "names"· {} αν το αρχείο λείπει ή είναι χαλασμένο
    (-> η λειτουργία απλώς δεν ενεργοποιείται, ποτέ σφάλμα στο import του ai_core)."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f).get("names") or {}
        return {doc: [a.lower() for a in aliases if a] for doc, aliases in raw.items() if aliases}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def named_docs(text: str, names: dict, in_scope) -> list[str]:
    """Τα έγγραφα ΤΟΥ SCOPE που ονομάζει το κείμενο, με τη σειρά της ΠΡΩΤΗΣ αναφοράς τους.
    Υποσυμβολοσειρά σε πεζά: τα ονόματα των papers γράφονται με λατινικούς χαρακτήρες και μέσα
    σε ελληνικές ερωτήσεις («Πώς συγκρίνει το PyWren…»)."""
    t = text.lower()
    first = {}
    for doc, aliases in names.items():
        if doc not in in_scope:
            continue
        hits = [t.find(a) for a in aliases if a in t]
        if hits:
            first[doc] = min(hits)
    return sorted(first, key=lambda d: (first[d], d))


def _key(meta: dict) -> str:
    return f"{meta.get('file_name')}:{meta.get('page')}"


def extra_pages(base: list, legs: list, limit: int) -> list:
    """Σελίδες (text, meta) εναλλάξ από κάθε έγγραφο, χωρίς όσες έχει ήδη η βάση και χωρίς διπλές,
    έως `limit`. Εναλλάξ και όχι «όλο το 1ο έγγραφο πρώτα»: το 2ο έγγραφο είναι ακριβώς αυτό που
    χάνει σήμερα τις θέσεις."""
    if limit <= 0:
        return []
    seen = {_key(m) for _t, m in base}
    out = []
    for tier in itertools.zip_longest(*legs):
        for pg in tier:
            if pg is None or _key(pg[1]) in seen:
                continue
            seen.add(_key(pg[1]))
            out.append(pg)
            if len(out) >= limit:
                return out
    return out
