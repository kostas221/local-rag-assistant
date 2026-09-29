"""Χάνει η ΜΕΤΑΦΡΑΣΗ το ζητούμενο στις ελληνικές multi_hop; — ίδιες ερωτήσεις, στα αγγλικά.

ΑΦΟΡΜΗ (scoreboard, 28/9/2026): στις 29 νέες multi_hop σελίδα και από τα δύο papers έρχεται
54/58, αλλά η σελίδα με το ΣΤΟΙΧΕΙΟ 30/58 — ελληνικές 19/44, αγγλικές 11/14· «και τα δύο
papers» el 3/22 έναντι en 4/7. Υπόθεση: η μετάφραση («concise English search query with the
key terms») συμπιέζει τις μακριές ερωτήσεις και πετάει το ζητούμενο — m047, 31 λέξεις ->
«AWS Lambda functions PyWren ExCamera workers» (χάθηκε το «όριο χρόνου εκτέλεσης»).

ΤΙ ΚΑΝΕΙ: οι 22 ελληνικές του golden_multihop_v2 ξανατρέχουν με το question_en τους (γράφτηκε
ΜΑΖΙ με την ελληνική, στην ίδια γέννηση) — ΧΩΡΙΣ μετάφραση. Όλα τα άλλα ίδια με το scoreboard
(απομονωμένο store, corrective ON, Gemini παγωμένο στο ίδιο αρχείο). Η ελληνική πλευρά
διαβάζεται από τη βάση σύγκρισης (runs/scoreboard/baseline.json) — είναι ντετερμινιστική,
μετρημένο 218/219 ταυτόσημα σε δύο τρεξίματα.

ΤΙ ΔΕΝ ΑΠΟΜΟΝΩΝΕΙ: το question_en δεν είναι λέξη προς λέξη η ελληνική· είναι παράλληλη
διατύπωση. Αν τα αγγλικά βγουν καλύτερα, λέει «μια ολόκληρη αγγλική ερώτηση φέρνει το στοιχείο,
η μετάφραση της ελληνικής όχι» — όχι ποια λέξη της μετάφρασης φταίει.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (28/9/2026):
    σελίδες-τεκμήρια   el 19/44  ->  en 28-34/44
    και τα δύο papers  el 3/22   ->  en 8-13/22
    Αν en ≤ 23/44: η μετάφραση ΔΕΝ είναι η κύρια αιτία — το στοιχείο χάνεται στην ανάκτηση
    (reranker / επέκταση σε σελίδες) και η διόρθωση ανήκει στη Φάση 1-2, όχι στη μετάφραση.

ΑΠΟΤΕΛΕΣΜΑ (28/9/2026, runs/multihop_lang.csv):
    σελίδες-τεκμήρια   el 19/44  ->  en 24/44   ✗ (πρόβλεψη 28-34)
    και τα δύο papers  el 3/22   ->  en 6/22    ✗ (πρόβλεψη 8-13)
    καλύτερα στα αγγλικά 5 (m047 m048 m055 m070 m072) · χειρότερα 0 · ίδια 17.
    Η ΜΕΤΑΦΡΑΣΗ ΕΞΗΓΕΙ ΜΙΚΡΟ ΜΕΡΟΣ — ΚΑΙ Η ΣΥΜΠΙΕΣΗ ΣΧΕΔΟΝ ΚΑΘΟΛΟΥ: από τις 5 που κέρδισαν ΜΟΝΟ
    η m047 είχε συμπιεσμένη μετάφραση (31 -> 6 λέξεις)· οι m049 (33 -> 5), m061 (36 -> 6), m078,
    m079 δεν κέρδισαν τίποτα με ολόκληρη αγγλική ερώτηση. Η υπόθεση ήταν δική μου και διαψεύστηκε.
    Ο ΜΗΧΑΝΙΣΜΟΣ (ανάλυση σειράς αναφοράς, 0 κόστος): η σελίδα-τεκμήριο του paper που αναφέρεται
    1ο στην ερώτηση έρχεται 19/29, του 2ου 11/29 (αγγλικές μορφές: 16/22 έναντι 8/22). Ο reranker
    δίνει ΕΝΑΝ βαθμό για όλη την ερώτηση -> οι σελίδες του 1ου μισού παίρνουν τις θέσεις· από το
    2ο paper έρχεται «κάποια» σελίδα (paper παρόν 54/58), όχι αυτή με το στοιχείο. Πρόβλημα
    ΑΝΑΚΤΗΣΗΣ δύο-μερών ερωτήσεων, όχι μετάφρασης -> Φάση 2 (η διάσπαση ερώτησης είχε απορριφθεί
    6/8 στο ΠΑΛΙΟ σετ, όπου δεν φαινόταν πρόβλημα — η απόρριψη δεν ισχύει αυτόματα εδώ).

    docker compose exec backend python evaluation/probe_multihop_lang.py
"""
import asyncio
import csv
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import scoreboard as S

BASE = os.path.join(S.OUT_DIR, "baseline.json")
MH = os.path.join(E.HERE, "golden_multihop_v2.jsonl")
OUT = os.path.join(S.RUNS, "multihop_lang.csv")


def evidence_got(t: dict, pages: str) -> int:
    got = set(pages.split(";")) if pages else set()
    return sum(ep in got for ep in t["evidence_pages"])


async def run() -> int:
    with open(BASE, encoding="utf-8") as f:
        base = {r["id"]: r for r in json.load(f)["rows"] if r["set"] == "mh_new"}
    tests = [t for t in E.load_jsonl(MH) if t["lang"] == "el"]
    missing = [t["id"] for t in tests if t["id"] not in base]
    if missing:
        print(f"!! Λείπουν από τη βάση σύγκρισης: {missing} — τρέξε πρώτα το scoreboard")
        return 1

    ai_core, _near = E.open_store()
    from eval_engine import calculate_mrr, calculate_ndcg  # ΜΕΤΑ το open_store: ίδιο ai_core

    if ai_core.collection.count() != S.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {S.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()
    ai_core.ENABLE_CORRECTIVE = True
    frozen = S.Frozen(S.FROZEN_PATH, fresh=False)
    frozen.install()

    rows = []
    print(f"\n{len(tests)} ελληνικές multi_hop -> η αγγλική τους μορφή, χωρίς μετάφραση\n")
    for t in tests:
        q = t["question_en"]
        pages, b1, b2, rw, outcome = await E.retrieve(ai_core, q)
        r = S.make_row("mh_new_en", dict(t, lang="en"), q, (pages, b1, b2, rw, outcome, 0.0),
                       ai_core, calculate_mrr, calculate_ndcg)
        el = base[t["id"]]
        rows.append({"id": t["id"], "el_state": el["state"], "en_state": r["state"],
                     "el_ev": evidence_got(t, el["pages"]), "en_ev": evidence_got(t, r["pages"]),
                     "el_both": el["both"], "en_both": r["both"],
                     "el_best1": el["best1"], "en_best1": r["best1"],
                     "el_translation": el["translation"], "question_en": q,
                     "n_words_el": len(t["question"].split()),
                     "n_words_translation": len(el["translation"].split()),
                     "el_pages": el["pages"], "en_pages": r["pages"]})
        x = rows[-1]
        print(f"  {t['id']}  el {x['el_state']:<24} ev {x['el_ev']}/2   ->   en {x['en_state']:<24}"
              f" ev {x['en_ev']}/2", flush=True)
    frozen.save()

    n = len(rows)
    el_ev, en_ev = sum(r["el_ev"] for r in rows), sum(r["en_ev"] for r in rows)
    el_b, en_b = sum(r["el_both"] for r in rows), sum(r["en_both"] for r in rows)
    better = [r["id"] for r in rows if r["en_ev"] > r["el_ev"]]
    worse = [r["id"] for r in rows if r["en_ev"] < r["el_ev"]]
    print("\n" + "#" * 90)
    print(f"σελίδες-τεκμήρια   el {el_ev}/{2 * n}   ->   en {en_ev}/{2 * n}      (πρόβλεψη en 28-34)")
    print(f"και τα δύο papers  el {el_b}/{n}    ->   en {en_b}/{n}       (πρόβλεψη en 8-13)")
    print(f"ανά ερώτηση: καλύτερα στα αγγλικά {len(better)} · χειρότερα {len(worse)} · "
          f"ίδια {n - len(better) - len(worse)}")
    print(f"  καλύτερα: {' '.join(better) or '—'}\n  χειρότερα: {' '.join(worse) or '—'}")
    short = [r for r in rows if r["n_words_translation"] <= 0.3 * r["n_words_el"]]
    if short:
        g = sum(r["en_ev"] - r["el_ev"] for r in short)
        print(f"μεταφράσεις ≤30% του μήκους της ερώτησης: {len(short)}/{n} · κέρδος σελίδων στα "
              f"αγγλικά μέσα σε αυτές: {g:+d}   (υπόλοιπες: "
              f"{sum(r['en_ev'] - r['el_ev'] for r in rows if r not in short):+d})")
    print("#" * 90)
    print(f"Gemini: {frozen.misses} νέες κλήσεις · {frozen.hits} από το πάγωμα")
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Σώθηκε: {OUT}")
    return 1 if frozen.errors else 0


def main() -> int:
    if not E.API_KEY:
        print("!! Λείπει GEMINI_API_KEY (χρειάζεται μόνο αν κοπεί κάποια και τρέξει ο corrective)")
        return 1
    try:
        fd = os.open(S.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard ή probe στο ίδιο store (κλειδαριά {S.LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run())
    finally:
        os.close(fd)
        os.remove(S.LOCK)


if __name__ == "__main__":
    sys.exit(main())
