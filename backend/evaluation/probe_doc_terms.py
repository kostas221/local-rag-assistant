"""Όροι ΑΝΑ ΕΓΓΡΑΦΟ — ΕΙΔΙΚΟΙ, όχι γενικές ετικέτες πεδίου.

ΤΟ ΜΕΤΡΗΜΕΝΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΛΥΝΕΙ:
Ο `build_corpus_descriptor.py` ζητάει «STANDARD ENGLISH TECHNICAL TERMS» και το
μοντέλο απαντάει με ΓΕΝΙΚΕΣ ΕΤΙΚΕΤΕΣ ΠΕΔΙΟΥ: serverless computing, cloud
computing, virtual machine, elasticity... Και τα 20.

Οι μετρήσεις της 17/8 (probe_glossary_main.py, 6 συνθήκες) δείχνουν ότι ΑΥΤΟ
είναι το πρόβλημα, ΟΧΙ η διατύπωση της οδηγίας:

    C_no_glossary  ΚΑΝΕΝΑΣ όρος                  98.52%
    A / B / F      3 ΔΙΑΦΟΡΕΤΙΚΕΣ οδηγίες        97.04%  <- ΤΑΥΤΟΣΗΜΑ
    E              ίδιοι όροι, άλλο domain       97.04%

Η ΜΟΝΗ μεταβλητή που κούνησε το νούμερο είναι η ΠΑΡΟΥΣΙΑ των όρων. Και το είδος
των όρων εξηγεί ΚΑΘΕ περίπτωση που έχουμε δει:

    ΚΕΡΔΟΣ    injection -> effusion   ειδικός όρος — το μοντέλο ΔΕΝ τον ήξερε
              pipeline  -> conduit    ειδικός όρος
    ΑΠΩΛΕΙΑ   layer -> «serverless computing»   γενική ετικέτα — την ήξερε ήδη

Ο ρόλος του glossary είναι ΟΡΘΟΓΡΑΦΙΚΟΣ: να δώσει τη λέξη που το μοντέλο δεν θα
μάντευε. Μια γενική ετικέτα δεν προσθέτει πληροφορία — μόνο ΕΛΚΕΙ τη μετάφραση
προς το κέντρο του πεδίου, και εκεί χάνεται το «layer».

ΤΙ ΚΑΝΕΙ: για κάθε αρχείο του corpus παίρνει τα ΔΙΚΑ ΤΟΥ chunks, φτιάχνει
ομοιόμορφο δείγμα και ζητάει όρους με prompt που ΑΠΑΓΟΡΕΥΕΙ ρητά τις γενικές
ετικέτες. Γράφει JSON {filename: terms} για να το καταναλώσει το
probe_glossary_main.py --terms-file.

ΓΙΑΤΙ ΑΝΑ ΕΓΓΡΑΦΟ ΚΑΙ ΟΧΙ ΣΥΝΟΛΙΚΑ: αυτό είναι το σχήμα που θα γράφει το ingest
στα metadata κάθε αρχείου. Ένα συνολικό prompt θα έδινε άλλο αποτέλεσμα από N
ξεχωριστά (το μοντέλο γενικεύει όταν βλέπει ανάμεικτο δείγμα) — άρα το probe
πρέπει να μετρήσει ΑΚΡΙΒΩΣ τον μηχανισμό που θα μπει στην παραγωγή.

ΚΟΣΤΟΣ: μία κλήση ΑΝΑ ΑΡΧΕΙΟ, εφάπαξ. ΜΗΔΕΝ γέννηση απαντήσεων.

    docker compose exec backend python evaluation/probe_doc_terms.py \
        --out evaluation/runs/doc_terms.json
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai_core
import gemini_rest

# Ίδιο μοντέλο με τον build_corpus_descriptor: εφάπαξ δουλειά ανά αρχείο, το
# επιχείρημα κόστους που κρατά το Flash στη μετάφραση ΔΕΝ ισχύει εδώ.
DEFAULT_MODEL = "gemini-pro-latest"

PROMPT = (
    "You are building a translation glossary for a search engine over ONE "
    "document. Below are excerpts from it.\n\n"
    "First, in ONE short line, name the domain/field of this document.\n\n"
    "Then list up to {n} terms that a translator would need in order to phrase a "
    "search query that MATCHES THIS DOCUMENT'S OWN WORDING.\n\n"
    "INCLUDE:\n"
    "- proper names of systems, tools, datasets, instruments, drugs, trials, "
    "algorithms, file formats (exactly as spelled in the text)\n"
    "- acronyms and their expansions\n"
    "- words whose everyday English translation would be WRONG here, i.e. this "
    "field uses a different word for the same everyday concept\n\n"
    "EXCLUDE (these are useless and actively harmful):\n"
    "- broad field labels naming the discipline itself (e.g. 'cloud computing', "
    "'machine learning', 'cardiology', 'volcanology')\n"
    "- words any competent translator already produces without a glossary\n"
    "- generic academic vocabulary (e.g. 'experiment', 'evaluation', 'dataset')\n\n"
    "Output EXACTLY these two lines, nothing else:\n"
    "DOMAIN: <one line>\n"
    "TERMS: term1, term2, term3, ...\n\n"
    "--- DOCUMENT EXCERPTS ---\n{sample}"
)


async def gloss_for(filename: str, docs: list, n: int, model: str) -> dict:
    """Μία κλήση ανά αρχείο, πάνω σε ομοιόμορφο δείγμα ΤΩΝ ΔΙΚΩΝ ΤΟΥ chunks.

    ΓΙΑΤΙ ΚΑΙ DOMAIN ΑΝΑ ΑΡΧΕΙΟ: μετρήθηκε (probe_glossary_main.py, συνθήκες G/H,
    δύο τρεξίματα το καθένα, ταυτόσημα): ΙΔΙΟΙ ειδικοί όροι με cloud-specific
    domain δίνουν 98.52%, με καρφωμένο γενικό domain 97.78% — το q047 χάνει τη
    λέξη «layer». Το domain string ΔΕΝ είναι ουδέτερο, άρα δεν μπορεί να καρφωθεί.
    """
    step = max(1, len(docs) // 30)
    sample = "\n---\n".join(d[:500] for d in docs[::step][:30])
    raw = (await gemini_rest.generate_once(
        PROMPT.format(n=n, sample=sample),
        model=model, api_key=ai_core.GEMINI_API_KEY,
        thinking_budget=1024, max_output_tokens=2048)).strip()

    out = {"domain": "", "terms": ""}
    for line in raw.splitlines():
        if line.upper().startswith("DOMAIN:"):
            out["domain"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("TERMS:"):
            out["terms"] = line.split(":", 1)[1].strip()
    # Χωρίς τις ετικέτες δεν μαντεύουμε — καλύτερα κενό παρά σκουπίδια στο prompt.
    if not out["terms"]:
        print(f"  !! {filename}: δεν αναλύθηκε ->\n{raw[:300]}")
    return out


async def terms_for(filename: str, docs: list, n: int, model: str) -> str:
    """Συμβατότητα με το probe_domain_glossary.py (συνθήκη CG): μόνο οι όροι."""
    return (await gloss_for(filename, docs, n, model))["terms"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/runs/doc_terms.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=15, help="όροι ανά αρχείο")
    ap.add_argument("--only", default="", help="αρχεία με κόμμα· κενό = όλα")
    args = ap.parse_args()

    got = ai_core.collection.get(include=["documents", "metadatas"])
    by_file: dict = {}
    for doc, meta in zip(got["documents"], got["metadatas"]):
        by_file.setdefault(meta.get("file_name", "?"), []).append(doc)

    keep = {s.strip() for s in args.only.split(",") if s.strip()}
    files = sorted(f for f in by_file if not keep or f in keep)
    print(f"{len(files)} αρχεία, {sum(len(by_file[f]) for f in files)} chunks\n")

    out = {}
    for fn in files:
        g = await gloss_for(fn, by_file[fn], args.n, args.model)
        out[fn] = g
        print(f"{fn}  ({len(by_file[fn])} chunks)\n  DOMAIN: {g['domain']}\n"
              f"  TERMS : {g['terms']}\n", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
