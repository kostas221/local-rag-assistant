"""ΤΥΦΛΕΣ περιγραφές chunk — η γέφυρα ορολογίας μπαίνει στο ΕΓΓΡΑΦΟ.

ΤΟ ΕΥΡΗΜΑ ΠΟΥ ΤΟ ΓΕΝΝΗΣΕ (12/8/2026, `probe_reference_contamination.py`)
------------------------------------------------------------------------
Η σελίδα-στόχος του `h002` δεν περιέχει **πουθενά** τη λέξη που ρωτάει η
ερώτηση. Το κείμενο λέει `vpxenc`, `command-line arguments`, `$QUALITY`· η
ερώτηση λέει `encoder settings`. Η μόνη εμφάνιση του `encoding` στη σελίδα
ήταν μέσα σε **βιβλιογραφική αναφορά άσχετου paper** — και όταν αφαιρέθηκε, το
dense rank ΧΕΙΡΟΤΕΡΕΨΕ (15 -> 21). Δηλαδή η σελίδα δεν βρισκόταν καν· τύχαινε.

Με χειροκίνητο πρόθεμα «Encoder settings and configuration.» ο cross-encoder
πήγε **−8.95 -> +1.34** (+10.29 logits) και το dense **15 -> 2**.

ΓΙΑΤΙ ΑΥΤΟ ΤΟ SCRIPT ΥΠΑΡΧΕΙ ΞΕΧΩΡΙΣΤΑ
--------------------------------------
Το χειροκίνητο πρόθεμα το έγραψε κάποιος που **ΞΕΡΕΙ ΤΗΝ ΕΡΩΤΗΣΗ**. Είναι
ακριβώς το `field-name stuffing` που ακύρωσε το `corrective_v3` — μετράει το
ΤΑΒΑΝΙ, όχι το εφικτό. Εδώ το prompt βλέπει **ΜΟΝΟ το κείμενο του chunk** και
ποτέ καμία ερώτηση, κανένα golden set. Αν η τυφλή περιγραφή δεν παράγει τη
γέφυρα, το +10.29 ήταν αυτοεκπληρούμενο και η ιδέα πεθαίνει.

ΓΛΩΣΣΑ: το prompt και η έξοδος είναι **ΑΓΓΛΙΚΑ**, στη γλώσσα του ΣΩΜΑΤΟΣ.
Μετρημένο εύρημα της 11/8: ελληνικές ερωτήσεις σε αγγλικό κείμενο έδωσαν
6 ΝΑΙ / 14 ΟΧΙ, αγγλικές 34/2 — κάθε έλεγχος με LLM τρέχει στη γλώσσα του corpus.

ΚΟΣΤΟΣ: ~330 tokens in / ~30 out ανά chunk. Για 418 chunks ≈ 0.07 $, ΜΙΑ φορά.

    # ένα chunk — ο φθηνός αποφασιστικός έλεγχος (1 κλήση)
    docker compose exec backend python evaluation/build_chunk_prefixes.py \\
        --doc excamera-nsdi17.pdf --page 15

    # ολόκληρο το corpus
    docker compose exec backend python evaluation/build_chunk_prefixes.py --all
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core
from gemini_rest import generate_once

OUT = "/app/evaluation/runs/chunk_prefixes.json"

# ΤΟ PROMPT ΔΕΝ ΒΛΕΠΕΙ ΠΟΤΕ ΕΡΩΤΗΣΗ. Οι κανόνες είναι σκόπιμα ουδέτεροι:
#   · «terminology a reader would use to search» = η γέφυρα ορολογίας, ΧΩΡΙΣ
#     να ξέρουμε ποια ερώτηση θα έρθει
#   · «only terms warranted by the text» = φράγμα κατά της παραγωγής όρων που
#     δεν στηρίζονται στο κείμενο (θα ήταν ψευδαίσθηση στο ίδιο το index)
#   · «one sentence, max 25 words» = μηχανικό όριο, όπως στο enrich_v2 που
#     ήταν η ΜΟΝΗ εκδοχή του enrichment χωρίς έκρηξη όρων
PROMPT = """You are indexing fragments of scientific papers for a search engine.

Below is one fragment. Write ONE sentence, at most 25 words, describing what \
this fragment contains. Use the standard technical terminology a reader would \
type when searching for this material, including the general category of any \
tool, artifact, table or measurement it shows.

Rules:
- Only use terms warranted by the text. Invent nothing.
- No opinions, no summary of findings, no "this fragment".
- Write it like a descriptive title. Output the sentence and nothing else.

FRAGMENT:
{text}"""

# ΕΚΔΟΧΗ 2 — ΚΑΘΗΜΕΡΙΝΗ ΔΙΑΤΥΠΩΣΗ ΜΑΖΙ ΜΕ ΤΗΝ ΟΡΟΛΟΓΙΑ.
# ΔΕΝ είναι fit σε μία ερώτηση: ολόκληρη κατηγορία του hard set (`nojargon`)
# είναι «ο χρήστης δεν ξέρει τους όρους του paper». Η v1 έγραψε σωστά
# «video encoding parameters» και έδωσε μόλις +1.45, γιατί ο cross-encoder
# θέλει τη ΛΕΞΗ. Στοχεύει την ΚΛΑΣΗ διατύπωσης, όχι το h002.
PROMPT_PLAIN = """You are indexing fragments of scientific papers for a search \
engine used by readers who are not experts in the field.

Below is one fragment. Write ONE sentence, at most 35 words, describing what \
this fragment contains. State it twice over in the same sentence: first with \
the paper's own technical terminology, then with the plain everyday words a \
non-expert would type when looking for exactly this material.

Rules:
- Only describe what is actually in the text. Invent nothing.
- Name the general kind of thing any tool or identifier is (for example, what \
a named program does), because a reader may not recognise the name.
- No opinions, no findings, no "this fragment".
- Output the sentence and nothing else.

FRAGMENT:
{text}"""

# ΕΚΔΟΧΗ 3 — doc2query. ΤΟ ΕΥΡΗΜΑ ΠΟΥ ΤΗΝ ΕΠΙΒΑΛΛΕΙ: οι δύο περιγραφικές
# εκδοχές πήγαν το dense 15 -> 9 και 15 -> 4, αλλά τον cross-encoder μόλις
# +1.45 / +1.27 ενώ χρειάζονται +6.35. Ο reranker είναι εκπαιδευμένος σε
# ζεύγη (ΕΡΩΤΗΣΗ, passage) του MS MARCO — βαθμολογεί ψηλά ό,τι ΜΟΙΑΖΕΙ με
# απάντηση σε ερώτηση. Άρα το chunk πρέπει να περιέχει ΕΡΩΤΗΣΕΙΣ, όχι τίτλο.
# Παραμένει ΤΥΦΛΟ: το prompt δεν βλέπει κανένα golden set.
PROMPT_D2Q = """Below is a fragment from a scientific paper that will be \
indexed for search.

Write 5 different questions that a reader could ask and that this fragment \
actually answers. Vary the wording deliberately:
- two using the paper's own technical terminology
- two in plain everyday words, as someone unfamiliar with the field would ask
- one short and vague, the way people type into a search box

Rules:
- Every question must be answerable from this fragment alone. Invent nothing.
- Name things explicitly. Do not write "this paper" or "the system".
- Output the 5 questions, one per line, nothing else.

FRAGMENT:
{text}"""

PROMPTS = {"v1": PROMPT, "plain": PROMPT_PLAIN, "d2q": PROMPT_D2Q}


async def describe(text: str, style: str = "v1") -> str:
    out = await generate_once(
        PROMPTS[style].format(text=text[:4000]),
        model=ai_core.GEMINI_MODEL, api_key=ai_core.GEMINI_API_KEY,
        temperature=0.1, max_output_tokens=320, thinking_budget=0)
    if style == "d2q":
        # Πολλαπλές γραμμές: κράτα τις όπως είναι, μία ερώτηση ανά γραμμή.
        lines = [ln.strip(" -•\t") for ln in out.strip().splitlines()]
        return "\n".join(ln for ln in lines if ln)
    return " ".join(out.strip().strip('"\'').split())


async def main(args) -> int:
    data = ai_core.collection.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]

    if args.all:
        targets = list(zip(ids, docs))
    else:
        targets = [(c, t) for c, t, m in zip(ids, docs, metas)
                   if m.get("file_name") == args.doc
                   and str(m.get("page")) == str(args.page)]
        if not targets:
            print(f"ΔΕΝ ΒΡΕΘΗΚΕ chunk για {args.doc}:{args.page}")
            return 1

    existing = {}
    if os.path.exists(args.out) and not args.overwrite:
        with open(args.out, encoding="utf-8") as fh:
            existing = json.load(fh)
        targets = [(c, t) for c, t in targets if c not in existing]
        print(f"{len(existing)} υπάρχουν ήδη -> μένουν {len(targets)}")

    print(f"{len(targets)} κλήσεις Gemini (~{0.00017 * len(targets):.3f} $)\n")
    for n, (cid, txt) in enumerate(targets, 1):
        try:
            pfx = await describe(txt, args.style)
        except Exception as exc:
            print(f"[{n}/{len(targets)}] {cid}: ΣΦΑΛΜΑ {exc}")
            continue
        existing[cid] = pfx
        print(f"[{n}/{len(targets)}] {cid}\n    {pfx}")
        if n % 25 == 0:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, ensure_ascii=False, indent=1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, ensure_ascii=False, indent=1)
    print(f"\n→ γράφτηκε {args.out}  ({len(existing)} προθέματα)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="τυφλές περιγραφές chunk (το prompt ΔΕΝ βλέπει ερωτήσεις)")
    ap.add_argument("--doc", default="excamera-nsdi17.pdf")
    ap.add_argument("--page", default="15")
    ap.add_argument("--all", action="store_true", help="ΟΛΟ το corpus")
    ap.add_argument("--style", choices=list(PROMPTS), default="v1",
                    help="v1 = ορολογία μόνο · plain = + καθημερινή διατύπωση")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", default=OUT)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
