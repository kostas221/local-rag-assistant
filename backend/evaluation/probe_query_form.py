"""Η ΜΟΡΦΗ του ερωτήματος μετακινεί ολόκληρη την κλίμακα του reranker;

ΤΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΤΟ ΓΕΝΝΗΣΕ
--------------------------
Στο 2ο πέρασμα του corrective agent τα `h008`/`h009` έχουν κάλυψη keywords
**100%**, ο στόχος φτάνει στο prompt, και η ΚΑΤΑΤΑΞΗ είναι σωστή (στο h008 οι
θέσεις 1-5 είναι όλες από το σωστό paper). Και όμως κόβονται, γιατί το top-1
είναι −3.16 / −4.45 — κάτω από το gate των −2.6.

Δηλαδή δεν κατατάσσει λάθος: **ΟΛΗ η κατανομή είναι μετατοπισμένη κάτω**. Στο
h009 και τα 15 logits είναι στο [−11.18, −4.45].

Η ΥΠΟΨΙΑ
--------
Το `ms-marco-MiniLM` είναι εκπαιδευμένο σε **ερωτήσεις** αναζήτησης. Ο
corrective agent παράγει **ονοματικές φράσεις** («Author contributions system
paper»). Αν η μορφή μόνη της ρίχνει την κλίμακα, τότε το κατώφλι −2.6 —
βαθμονομημένο σε μεταφρασμένες ΕΡΩΤΗΣΕΙΣ του 1ου περάσματος — συγκρίνει δύο
διαφορετικές κλίμακες, και το `CORRECTIVE_MIN_SCORE` κληρονομεί το σφάλμα.

ΓΙΑΤΙ ΤΟ ΠΡΟΧΕΙΡΟ ΤΕΣΤ ΔΕΝ ΑΡΚΕΙ
--------------------------------
Δοκιμή με χειρόγραφη ερώτηση έδωσε h008 −3.16 → **−0.42** (περνάει!) αλλά h009
−4.45 → **−8.94** (χειρότερα). Αιτία: αλλάζοντας τη διατύπωση αλλάζει ΚΑΙ το
τι ανακτούν dense/BM25 — στο h009 εξαφανίστηκαν οι σελίδες του ExCamera. Δύο
μεταβλητές μαζί.

Ο ΕΛΕΓΧΟΣ ΕΔΩ
-------------
**Οι υποψήφιοι μένουν ΚΑΡΦΩΜΕΝΟΙ.** Ανακτώνται μία φορά με το ερώτημα βάσης
και μετά βαθμολογούνται ξανά με κάθε διατύπωση. Οι λέξεις περιεχομένου είναι
ΤΑΥΤΟΣΗΜΕΣ σε όλες τις παραλλαγές — αλλάζει μόνο το ερωτηματικό πλαίσιο. Ό,τι
διαφορά μείνει, είναι της μορφής και μόνο.

ΜΗΔΕΝ κλήσεις Gemini: τα rewrites διαβάζονται από artifact, τα πλαίσια είναι
ντετερμινιστικά templates.

    docker compose exec backend python evaluation/probe_query_form.py \\
        --csv evaluation/runs/query_form.csv
"""
import argparse
import asyncio
import csv
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core

SOURCE = "/app/evaluation/runs/corrective_verdict.csv"

# Ντετερμινιστικά πλαίσια. Το `{q}` μπαίνει ΑΥΤΟΥΣΙΟ, ώστε οι λέξεις
# περιεχομένου να είναι ίδιες παντού και να απομονώνεται η μορφή.
FORMS = [
    ("ωμό",        "{q}"),
    ("what-is",    "What is {q}?"),
    ("what-about", "What does the text say about {q}?"),
    ("how-why",    "How and why does {q} happen?"),
    ("explain",    "Explain {q}"),
]


ASK = """Rewrite the following search phrase as ONE grammatical English
question that a person would actually type. Keep every content word that is
already there. Do not add new topic words. Do not explain. Output only the
question.

PHRASE: {q}"""

SETS = ["golden_set_50.jsonl", "golden_multihop_new.jsonl",
        "golden_hard_paraphrase.jsonl"]


async def as_question(phrase: str) -> str:
    """Γραμματική ερώτηση με ΤΙΣ ΙΔΙΕΣ λέξεις περιεχομένου.

    ΓΙΑΤΙ ΔΕΝ ΑΡΚΟΥΝ ΤΑ TEMPLATES: το «How and why does Cloud resource pricing
    variability across regions happen?» δεν είναι ερώτηση, είναι σπασμένα
    αγγλικά. Μετρώντας με templates μετράς τη σύνταξη, όχι τη μορφή.
    """
    from gemini_rest import generate_once
    return (await generate_once(ASK.format(q=phrase),
                                model=ai_core.GEMINI_MODEL,
                                api_key=ai_core.GEMINI_API_KEY,
                                thinking_budget=0,
                                max_output_tokens=64)).strip().split("\n")[0]


def candidates_for(query: str, n: int):
    """Τα κείμενα που θα έβλεπε ο reranker — ίδια διαδρομή με το trace_query."""
    allowed = ai_core.collection.get(
        where=ai_core._build_where(None, None), include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()
    dense_ids = ai_core._dense_exact_ids(
        dm, query, allowed, min(ai_core.DENSE_CANDIDATES, len(allowed)))
    sparse_ids = ai_core._bm25_sparse_ids(
        idx, query, allowed, ai_core.DENSE_CANDIDATES)
    rrf = ai_core._rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                            idx["metas"], k=60, top_n=n, pos=idx["pos"])
    return [text for _s, text, _m in rrf]


def score(query: str, texts):
    return list(ai_core.reranker.predict(
        [[query, t] for t in texts], batch_size=ai_core.RERANK_BATCH_SIZE))


async def main(args) -> int:
    with open(args.source, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("rewrite")]
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids)]
    print(f"{len(rows)} αναδιατυπώσεις · {len(FORMS)} μορφές · "
          f"υποψήφιοι ΚΑΡΦΩΜΕΝΟΙ ανά ερώτηση\n")

    out = []
    for r in rows:
        base = r["rewrite"]
        docs = candidates_for(base, ai_core.RERANK_CANDIDATES)
        forms = list(FORMS)
        if args.gemini_question:
            forms.append(("ΕΡΩΤΗΣΗ-LLM", await as_question(base)))
        print("=" * 78)
        print(f"{r['id']}  κάλυψη {float(r['coverage']):.0f}%  «{base}»")
        print(f"{'μορφή':<12}{'best':>8}{'μέσο':>8}{'χειρ.':>8}   gate")
        print("-" * 78)
        for name, tpl in forms:
            q = tpl.format(q=base) if "{q}" in tpl else tpl
            s = score(q, docs)
            best, mean, worst = max(s), sum(s) / len(s), min(s)
            passes = best >= ai_core.MIN_RERANK_SCORE
            corr = best >= ai_core.CORRECTIVE_MIN_SCORE
            mark = ("ΠΕΡΝΑΕΙ" if passes else "κόβει")
            mark += " +corrective" if corr else ""
            print(f"{name:<12}{best:>8.2f}{mean:>8.2f}{worst:>8.2f}   {mark}")
            out.append({"id": r["id"], "coverage": r["coverage"],
                        "form": name, "query": q, "best": round(best, 3),
                        "mean": round(mean, 3), "worst": round(worst, 3),
                        "passes_gate": passes, "passes_corrective": corr})
            if name == "ΕΡΩΤΗΣΗ-LLM":
                print(f"{'':<12}«{q}»")
                # ΚΑΙ με τη ΔΙΚΗ ΤΗΣ ανάκτηση: αυτό είναι το πραγματικό
                # αποτέλεσμα παραγωγής, ενώ το παραπάνω απομονώνει τη μορφή.
                own = candidates_for(q, ai_core.RERANK_CANDIDATES)
                so = score(q, own)
                b2 = max(so)
                print(f"{'  ^ δική της ανάκτηση':<12}{b2:>8.2f}"
                      f"{sum(so)/len(so):>8.2f}{min(so):>8.2f}   "
                      + ("ΠΕΡΝΑΕΙ" if b2 >= ai_core.MIN_RERANK_SCORE
                         else "κόβει"))
                out.append({"id": r["id"], "coverage": r["coverage"],
                            "form": "ΕΡΩΤΗΣΗ-LLM+ανάκτηση", "query": q,
                            "best": round(b2, 3),
                            "mean": round(sum(so) / len(so), 3),
                            "worst": round(min(so), 3),
                            "passes_gate": b2 >= ai_core.MIN_RERANK_SCORE,
                            "passes_corrective":
                                b2 >= ai_core.CORRECTIVE_MIN_SCORE})
        print()

    print("=" * 78)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΑ — μέση μετατόπιση ως προς το ωμό ερώτημα")
    print(f"{'μορφή':<12}{'Δbest':>9}{'Δμέσο':>9}{'περνούν gate':>15}")
    print("-" * 78)
    raw = {o["id"]: o for o in out if o["form"] == "ωμό"}
    names = [n for n, _ in FORMS]
    if args.gemini_question:
        names += ["ΕΡΩΤΗΣΗ-LLM", "ΕΡΩΤΗΣΗ-LLM+ανάκτηση"]
    for name in names:
        grp = [o for o in out if o["form"] == name]
        db = sum(o["best"] - raw[o["id"]]["best"] for o in grp) / len(grp)
        dm = sum(o["mean"] - raw[o["id"]]["mean"] for o in grp) / len(grp)
        n_ok = sum(1 for o in grp if o["passes_gate"])
        print(f"{name:<12}{db:>+9.2f}{dm:>+9.2f}{n_ok:>10}/{len(grp)}")
    print("-" * 78)
    print("ΤΟ ΚΡΙΣΙΜΟ: αν το Δμέσο κινείται μαζί με το Δbest, η μορφή")
    print("μετακινεί ΟΛΗ την κλίμακα και το απόλυτο κατώφλι είναι άκυρο.")
    print("Αν κινείται μόνο το Δbest, η μορφή βελτιώνει την ΚΑΤΑΤΑΞΗ.")
    print("=" * 78)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        print(f"→ γράφτηκε {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="μετακινεί η μορφή του ερωτήματος την κλίμακα;")
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--gemini-question", action="store_true",
                    help="ΓΡΑΜΜΑΤΙΚΗ ερώτηση από το Gemini (1 κλήση/ερώτηση)")
    ap.add_argument("--csv", default=None)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
