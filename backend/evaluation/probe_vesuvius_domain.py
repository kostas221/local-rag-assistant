"""Φταίει το ΠΕΔΙΟ «cloud» που περνάει η o4 (Βεζούβιος) τον φύλακα σε PDF άλλου θέματος;

ΑΦΟΡΜΗ (scoreboard, 4ο τρέξιμο, 28/9/2026): στο store άλλων πεδίων (Στρόμπολι + καρδιολογία) η
«Πότε εξερράγη για τελευταία φορά ο Βεζούβιος;» ΠΕΡΝΑΕΙ τον φύλακα:
    σήμερα  πεδίο «Cloud Computing and Distributed Systems»  «Vesuvius eruption last time»  −1.76
    25/9    πεδίο «scientific research papers» (CD)           «Vesuvius eruptions»           −3.46 κομμένη
Το πεδίο cloud έρχεται από το fallback: το ingest γράφει domain="" -> optimize_query ->
corpus_descriptor.json. Ίδια οδηγία (η περιοριστική, αμετάβλητη από το ac2e15b) και ίδιο γλωσσάρι·
μόνη διαφορά το πεδίο. ΑΛΛΑ n=1 και η o4 είναι γνωστή ευαίσθητη στη διατύπωση.

ΤΙ ΚΑΝΕΙ: N μεταφράσεις της o4 με ΚΑΘΕ πεδίο, με τον ΠΡΑΓΜΑΤΙΚΟ optimize_query (ίδιο prompt,
temperature 0.1, όροι του scope όπως στην παραγωγή) -> κάθε μετάφραση περνάει το 1ο πέρασμα του
ΠΡΑΓΜΑΤΙΚΟΥ search_documents (χωρίς corrective) -> best1 έναντι του φύλακα −2.6. Στο πεδίο
«scientific research papers» = το _FALLBACK_DOMAIN του ίδιου του ai_core (όταν λείπει ο descriptor).
+ ΜΙΑ απάντηση (ask_ai) στην πρώτη μετάφραση που περνάει: τι θα έβλεπε ο χρήστης.
CONTROL (0 κλήσεις): οι δύο γνωστές μεταφράσεις πρέπει να ξαναδώσουν −3.46 και −1.76 (±0.02).

ΚΟΣΤΟΣ: 2N μεταφράσεις + ≤1 απάντηση = 11 κλήσεις για N=5 (~1 σεντ), ~2 λεπτά με τη φόρτωση.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (28/9/2026):
    control: −3.46 / −1.76 ✓ (αλλιώς το store δεν είναι το ίδιο και τίποτα δεν διαβάζεται)
    με temperature 0.1 κάθε πεδίο δίνει 1-2 διαφορετικές μεταφράσεις στις 5
    cloud: περνάει 4-5/5 · γενικό: περνάει 0-2/5
    απάντηση: ΑΡΝΗΣΗ («δεν αναφέρεται ο Βεζούβιος») — το 2ο στρώμα κρατάει, όπως στις 38/40 κοντινές
ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ (γραμμένος πριν):
    διαφορά ≥ 3/5 ανάμεσα στα πεδία -> ΦΤΑΙΕΙ ΤΟ ΠΕΔΙΟ: το fallback στο cloud είναι σφάλμα παραγωγής
        για PDF άλλου θέματος -> υποψήφια διόρθωση, μετριέται με τον πίνακα
    διαφορά ≤ 1/5 -> ΔΕΝ φταίει το πεδίο: το 25/9 ήταν τυχαία διατύπωση· το αδύνατο σημείο είναι ο
        reranker (δίνει −1.76 σε ερώτηση για ΑΛΛΟ ηφαίστειο) -> ξαναμετριέται στη Φάση 1.1
    2/5 -> αδιευκρίνιστο σε αυτό το n· καταγράφεται, δεν διορθώνεται τίποτα

ΑΠΟΤΕΛΕΣΜΑ (28/9/2026, runs/vesuvius_domain.csv): control ✓ (αλλιώς δεν θα γινόταν καμία κλήση)
    cloud    5/5 ΠΕΡΝΑΕΙ — 5/5 λέξη προς λέξη «Vesuvius eruption last time» −1.76
    generic  1/5 περνάει — 4/5 «Vesuvius eruptions» −3.46 κομμένη · 1/5 «Vesuvius eruptions RATE» +0.86
    διαφορά 4/5 -> ΦΤΑΙΕΙ ΤΟ ΠΕΔΙΟ (κανόνας γραμμένος πριν). Προβλέψεις: cloud ✓ · generic ✓ ·
    1-2 μεταφράσεις ανά πεδίο ✓ (1 και 2).
    ΔΥΟ ΛΕΠΤΟΜΕΡΕΙΕΣ ΠΟΥ ΑΛΛΑΖΟΥΝ ΤΗΝ ΑΝΑΓΝΩΣΗ:
    (1) Η μετάφραση με cloud είναι η ΠΙΣΤΟΤΕΡΗ («last time» = «για τελευταία φορά»). Με γενικό
        πεδίο κόβεται επειδή ο μεταφραστής ΠΕΤΑΕΙ κομμάτι της ερώτησης. Το αδύνατο σημείο που μένει
        είναι ο reranker: −1.76 σε ερώτηση για ΑΛΛΟ ηφαίστειο πάνω σε σελίδες για το Στρόμπολι.
    (2) Η διαρροή του Αυγούστου («… RATE», όρος του γλωσσαρίου που δεν ρωτήθηκε) ΞΑΝΑΕΜΦΑΝΙΣΤΗΚΕ
        1/5 παρά την περιοριστική οδηγία. Η οδηγία ΜΕΙΩΣΕ τη διαρροή (τότε ήταν η συνηθισμένη
        μετάφραση), δεν την ΕΞΑΛΕΙΨΕ. Και περνάει με πολύ μεγαλύτερο περιθώριο (+0.86 vs −1.76).
    Άρα το γενικό πεδίο ΔΕΝ είναι καθαρή λύση για την o4 (5/5 -> 1/5, όχι 0/5). Το επιχείρημα για
    διόρθωση είναι ότι το prompt λέει «το σώμα είναι για cloud» για έγγραφα καρδιολογίας — λάθος
    πληροφορία — και η μέτρηση δείχνει ότι έχει αποτέλεσμα. n=1 ερώτηση.
    ΑΠΑΝΤΗΣΗ (--answer-for, runs/vesuvius_domain_answer.txt): ΑΡΝΗΣΗ ✓ — «Δεν μπορώ να βρω πληροφορίες
    σχετικά με το πότε εξερράγη για τελευταία φορά ο Βεζούβιος … Το κείμενο αναφέρεται κυρίως σε
    εκρήξεις του ηφαιστείου Στρόμπολι». Το 2ο στρώμα κρατάει· η ζημιά είναι ότι το 1ο (φύλακας) δεν
    κάνει τη δουλειά του σε PDF άλλου θέματος. (Η απάντηση του 1ου τρεξίματος χάθηκε στο τερματικό·
    αυτή είναι νέα κλήρωση πάνω στις ΙΔΙΕΣ σελίδες.)

    docker compose exec backend python evaluation/probe_vesuvius_domain.py
    # μόνο η απάντηση (1 κλήση) — η απάντηση του 1ου τρεξίματος τυπώθηκε μόνο στο τερματικό:
    docker compose exec backend python evaluation/probe_vesuvius_domain.py --answer-for "Vesuvius eruption last time"
"""
import argparse
import asyncio
import csv
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import scoreboard as S

import corpus_glossary

QUESTION = "Πότε εξερράγη για τελευταία φορά ο Βεζούβιος;"
GENERIC = "scientific research papers"            # GENERIC_DOMAIN του 25/9 = _FALLBACK_DOMAIN
CONTROLS = [("Vesuvius eruptions", -3.46), ("Vesuvius eruption last time", -1.76)]
OUT = os.path.join(S.RUNS, "vesuvius_domain.csv")
ANSWER_OUT = os.path.join(S.RUNS, "vesuvius_domain_answer.txt")   # 28/9: η 1η απάντηση χάθηκε στο τερματικό


def save_answer(label: str, tr: str, ans: str) -> None:
    refusal = "ΝΑΙ" if E.REFUSAL.search(ans) else "ΟΧΙ — διάβασέ την"
    text = f"ΑΠΑΝΤΗΣΗ ({label}, «{tr}») — μοτίβο άρνησης: {refusal}\n{ans}\n"
    print("\n" + text)
    with open(ANSWER_OUT, "a", encoding="utf-8") as f:
        f.write(text + "-" * 60 + "\n")
    print(f"Σώθηκε η απάντηση: {ANSWER_OUT}")


async def first_pass(ai_core, english: str):
    """best1 + απόφαση φύλακα για ΑΓΓΛΙΚΟ ερώτημα (δεν μεταφράζεται ξανά), χωρίς corrective."""
    pages, b1, _b2, _rw, outcome = await E.retrieve(ai_core, english)
    return pages, b1, outcome


async def run(args) -> int:
    ai_core, _near = E.open_store()
    n = S.open_domains_store(ai_core)
    if n != S.DOMAINS_CHUNKS:
        print(f"!! Το store άλλων πεδίων έχει {n} chunks, περίμενα {S.DOMAINS_CHUNKS} — σταματάω")
        return 1
    ai_core.ENABLE_CORRECTIVE = False
    if ai_core._FALLBACK_DOMAIN != GENERIC:
        print(f"!! Το _FALLBACK_DOMAIN άλλαξε ({ai_core._FALLBACK_DOMAIN!r}) — σταματάω")
        return 1

    ids = ai_core.collection.get(where=ai_core._build_where(None, E.TEST_USER), include=[])["ids"]
    scope_domain, scope_terms = corpus_glossary.assemble(ids, ai_core._get_bm25_index())
    print(f"\nScope: {len(ids)} chunks · domain των αρχείων {scope_domain!r} "
          f"(κενό -> παραγωγή: {ai_core._CORPUS_DOMAIN!r}) · όροι: {scope_terms[:120]}...")

    print("\n===== CONTROL (0 κλήσεις) =====")
    ok = True
    for text, want in CONTROLS:
        _p, b1, outcome = await first_pass(ai_core, text)
        good = b1 is not None and abs(b1 - want) <= 0.02
        ok &= good
        print(f"  «{text}»  {E.fmt(b1)} (αναμενόταν {want:+.2f})  {outcome}  {'OK' if good else '!! ΑΠΕΤΥΧΕ'}")
    if not ok:
        print("!! Το control δεν αναπαράχθηκε — τα αποτελέσματα ΔΕΝ διαβάζονται. Σταματάω πριν από κλήσεις.")
        return 1

    if args.answer_for:                 # ΜΟΝΟ απάντηση: σελίδες χωρίς κόστος, 1 κλήση γέννησης
        pages, b1, outcome = await first_pass(ai_core, args.answer_for)
        print(f"\n«{args.answer_for}» {E.fmt(b1)} {outcome} · σελίδες {len(pages)}")
        if not pages:
            print("Κόβεται από τον φύλακα — ο χρήστης παίρνει τη σταθερή άρνηση, δεν υπάρχει απάντηση.")
            return 0
        save_answer("μόνο απάντηση", args.answer_for, await E.answer_of(ai_core, QUESTION, pages))
        return 0

    rows, scored, answered = [], {}, None
    for cond, domain_arg in (("cloud", scope_domain), ("generic", GENERIC)):
        print(f"\n===== {cond}: πεδίο {domain_arg or ai_core._CORPUS_DOMAIN!r} × {args.n} =====")
        for i in range(args.n):
            ai_core._translation_cache.pop(QUESTION, None)
            tr = await ai_core.optimize_query(QUESTION, domain=domain_arg, terms=scope_terms)
            if tr == QUESTION:          # το optimize_query καταπίνει σφάλματα -> αμετάφραστη
                print("!! Η μετάφραση απέτυχε (γύρισε η ελληνική ερώτηση) — σταματάω")
                return 1
            if tr not in scored:
                scored[tr] = await first_pass(ai_core, tr)
            pages, b1, outcome = scored[tr]
            passed = outcome != "cut"
            rows.append({"cond": cond, "draw": i + 1, "translation": tr, "best1": E.fmt(b1),
                         "passed": int(passed), "n_pages": len(pages)})
            print(f"  {i + 1}  «{tr}»  {E.fmt(b1)}  {'ΠΕΡΝΑΕΙ' if passed else 'κόβεται'}", flush=True)
            if passed and answered is None:
                answered = (cond, tr, await E.answer_of(ai_core, QUESTION, pages))

    print("\n" + "#" * 90)
    for cond in ("cloud", "generic"):
        rs = [r for r in rows if r["cond"] == cond]
        kinds = sorted({r["translation"] for r in rs})
        print(f"  {cond:<8} περνάει {sum(r['passed'] for r in rs)}/{len(rs)} · "
              f"διαφορετικές μεταφράσεις {len(kinds)}")
    diff = abs(sum(r["passed"] for r in rows if r["cond"] == "cloud")
               - sum(r["passed"] for r in rows if r["cond"] == "generic"))
    verdict = ("ΦΤΑΙΕΙ ΤΟ ΠΕΔΙΟ" if diff >= 3 else "ΔΕΝ ΦΤΑΙΕΙ ΤΟ ΠΕΔΙΟ (αδύνατο σημείο: reranker)"
               if diff <= 1 else "ΑΔΙΕΥΚΡΙΝΙΣΤΟ σε αυτό το n")
    print(f"  διαφορά {diff}/{args.n} -> {verdict}   (κανόνας γραμμένος πριν, βλ. docstring)")
    print("#" * 90)
    if answered:
        save_answer(*answered)
    else:
        print("\nΚαμία μετάφραση δεν πέρασε — καμία απάντηση.")

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nΣώθηκε: {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=5, help="μεταφράσεις ανά πεδίο")
    ap.add_argument("--answer-for", default="",
                    help="ΜΟΝΟ απάντηση για αυτή την αγγλική μετάφραση (0 μεταφράσεις, 1 γέννηση)")
    args = ap.parse_args()
    if not E.API_KEY:
        print("!! Λείπει GEMINI_API_KEY")
        return 1
    try:
        fd = os.open(S.LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard ή probe στο ίδιο store (κλειδαριά {S.LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run(args))
    finally:
        os.close(fd)
        os.remove(S.LOCK)


if __name__ == "__main__":
    sys.exit(main())
