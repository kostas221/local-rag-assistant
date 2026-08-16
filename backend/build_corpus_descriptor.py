"""Παράγει τον domain descriptor ΑΠΟ το corpus — ΜΙΑ φορά, μετά το ingestion.

ΓΙΑΤΙ: το optimize_query χρειάζεται να ξέρει το ΠΕΔΙΟ του corpus για να μεταφράσει
τις ελληνικές ερωτήσεις στη σωστή ορολογία (π.χ. «καθυστέρηση» -> "latency", όχι
"delay"). Ο descriptor ΗΤΑΝ ΚΑΡΦΩΜΕΝΟΣ («cloud/serverless») μέσα στο prompt -> το
σύστημα δούλευε μόνο για ένα πεδίο. Εδώ τον παράγουμε από τα ΙΔΙΑ τα έγγραφα,
οπότε το ίδιο prompt δουλεύει για ΟΠΟΙΟΔΗΠΟΤΕ corpus (ιατρικό, νομικό, ...).

Γράφει το backend/corpus_descriptor.json. Το ai_core το φορτώνει στο import (με
fallback στο ιστορικό cloud/serverless αν λείπει). ΤΡΕΞΕ ΤΟ μετά από κάθε αλλαγή
corpus, και ΜΕΤΑ `docker compose restart backend`.

ΜΕΤΡΗΘΗΚΕ ΠΡΙΝ ΜΠΕΙ (probe_domain_translation.py, 22 ελληνικές ερωτήσεις, 3 μορφές):
ο παραγόμενος descriptor ≈ ο καρφωμένος (in-corpus μέσος best-logit 4.89 vs 4.75,
ΚΑΝΕΝΑ gate crossing), out_of_corpus 0 διαρροές. Αποδεδειγμένη μη-παλινδρόμηση στο
cloud corpus· το cross-domain όφελος είναι ΕΝΔΕΙΞΗ (δεν υπάρχει 2ο corpus να μετρηθεί).

    docker compose exec backend python build_corpus_descriptor.py
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, "/app")
import ai_core
import gemini_rest

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus_descriptor.json")

# Εφάπαξ δουλειά ανά corpus -> βάζουμε το ΙΣΧΥΡΟΤΕΡΟ μοντέλο by default, όχι το
# Flash της παραγωγής. Ο descriptor είναι high-leverage (τροφοδοτεί ΚΑΘΕ μετάφραση)
# και το κόστος είναι μία κλήση για πάντα. Το επιχείρημα κόστους που κρατά το Flash
# στη μετάφραση ΔΕΝ ισχύει εδώ.
# ΣΗΜΕΙΩΣΗ: το gemini-2.5-pro είναι gated (404 για νέους λογαριασμούς)· το
# gemini-pro-latest είναι το ισχυρότερο ΓΕΝΙΚΑ ΔΙΑΘΕΣΙΜΟ (όχι preview) — επαληθεύτηκε
# με ListModels + δοκιμαστική κλήση 15/8/2026.
DEFAULT_MODEL = "gemini-pro-latest"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"μοντέλο για τη γέννηση του descriptor (default: {DEFAULT_MODEL})")
    args = ap.parse_args()

    docs = ai_core.collection.get(include=["documents"])["documents"]
    if not docs:
        print("Άδειο corpus — τίποτα να περιγράψω.")
        return 1

    # Ομοιόμορφο δείγμα από ΟΛΟ το corpus, κομμένο ώστε να χωράει σε ένα prompt.
    step = max(1, len(docs) // 30)
    sample = "\n---\n".join(d[:500] for d in docs[::step][:30])

    prompt = (
        "You are analyzing a document corpus to build a translation glossary for "
        "an English-only search engine. Below are excerpts from the corpus.\n"
        "1) In ONE short line, name the domain/field of these documents.\n"
        "2) List up to 20 STANDARD ENGLISH TECHNICAL TERMS that a search over this "
        "corpus would rely on — the exact vocabulary the documents use.\n"
        "Output EXACTLY this format, nothing else:\n"
        "DOMAIN: <one line>\n"
        "TERMS: term1, term2, term3, ...\n\n"
        f"--- CORPUS EXCERPTS ---\n{sample}"
    )
    # thinking_budget>0: τα Pro 3.x απαιτούν thinking mode (400 αλλιώς). Το
    # max_output_tokens πρέπει να χωρά thinking + ορατή απάντηση μαζί — με 512 τα
    # thinking tokens έτρωγαν τη λίστα όρων στη μέση, γι' αυτό 2048.
    raw = (await gemini_rest.generate_once(
        prompt, model=args.model, api_key=ai_core.GEMINI_API_KEY,
        thinking_budget=1024, max_output_tokens=2048)).strip()

    domain, terms = "", ""
    for line in raw.splitlines():
        if line.upper().startswith("DOMAIN:"):
            domain = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TERMS:"):
            terms = line.split(":", 1)[1].strip()

    if not domain:
        print(f"Ο descriptor ΔΕΝ αναλύθηκε σωστά — δεν γράφτηκε τίποτα:\n{raw}")
        return 1

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"domain": domain, "terms": terms}, f, ensure_ascii=False, indent=2)
    print(f"Γράφτηκε: {OUT}  (από {len(docs)} chunks, μοντέλο {args.model})")
    print(f"  DOMAIN: {domain}")
    print(f"  TERMS:  {terms}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
