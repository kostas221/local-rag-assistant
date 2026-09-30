"""Τι κάνει η διάσπαση ΕΚΤΟΣ multi_hop; — Φάση 2, βήμα 4: κύριο σετ + κοντινές ooc.

ΑΦΟΡΜΗ (probe_decomp_answers.py, 29/9/2026): στις 29 multi_hop η B (βάση + έως 4 σελίδες από τα
σκέλη) πάει τα σωστά μισά 33 -> 43/58 και περνάει τον κανόνα -> ΠΡΟΧΩΡΑΕΙ. Δύο ανοιχτά:
    (α) ROUTING: η διάσπαση τρέχει σε ΚΑΘΕ ερώτηση. Πόσο συχνά σπάει ερώτηση ΕΝΟΣ θέματος — και όταν
        σπάει, χαλάει η απάντηση; Δεν μετρήθηκε ποτέ (ούτε τον Αύγουστο).
    (β) ΑΣΦΑΛΕΙΑ: στη B 4 μισά πέρασαν από «λείπει» σε «χωρίς στήριξη» (m047, m062, m072, m078).
        Με περισσότερες σελίδες το μοντέλο λέει λιγότερο «δεν υπάρχει». Οι κοντινές ooc είναι ακριβώς
        το σετ όπου η σωστή απάντηση είναι «δεν υπάρχει» -> εδώ φαίνεται αν η B ανοίγει διαρροές.

ΤΙ ΚΑΝΕΙ: ίδια ανάκτηση με το probe_decomp_mh40 (απομονωμένο store, μεταφράσεις ΠΑΓΩΜΕΝΕΣ, ο φύλακας
κρίνει ΜΟΝΟ το αρχικό ερώτημα -> ό,τι κόβεται σήμερα μένει κομμένο). Διάσπαση με ΤΟ ΙΔΙΟ prompt και
μοντέλο (gemini-2.5-flash), πάγωμα στο ίδιο runs/decomp_mh40_gemini.json. Απαντήσεις ΜΟΝΟ για τη B
(η D έμεινε εκτός κανόνα), με τον ΠΡΑΓΜΑΤΙΚΟ ask_ai και τους κριτές του scoreboard_answers ΑΥΤΟΥΣΙΟΥΣ:
κύριο -> ο παλιός του eval_engine (4 βαθμοί) · κοντινές -> της Φάσης 0.1 (σωστό / ήπια διαρροή (β) /
διαρροή (γ)). Η βάση βγαίνει ΟΛΗ από το πάγωμα της 28/9· νέες κλήσεις ΜΟΝΟ όπου άλλαξαν σελίδες.
ΕΛΕΓΧΟΙ: σελίδες βάσης = baseline.json · απαντήσεις βάσης: 0 νέες κλήσεις και βαθμοί/ετικέτες ΙΔΙΟΙ με
answers_baseline_main / _near_ooc (αλλιώς σταματάει ΠΡΙΝ ξοδέψει για τη B).

ΚΟΣΤΟΣ: 92 διασπάσεις (~0.05 $) + γέννηση & κριτής όπου αλλάζουν σελίδες (~30 × 2) ≈ 150 κλήσεις,
~0.4 $. Χρόνος σε ΑΥΤΟ το μηχάνημα ~45-60 λεπτά (η ανάκτηση κάνει ~12 s)· ο χρόνος ΔΕΝ μετριέται.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (29/9/2026) — βάση (28/9): κύριο τέλειες 47/50, faithfulness < 5
μόνο q046, μέσοι (με απάντηση) 4.90/4.80/4.92/4.96 · κοντινές σωστό 38 · (β) 1 · (γ) 1 · κομμένες 2.
    routing: κύριο multi_hop σπάει 3-4/4 · υπόλοιπες με απάντηση 6-18/41 · ooc κύριου 1-3/5 ·
             κοντινές 10-25/42 (τα combo 4/4)
    η B αλλάζει σύνολο σελίδων: κύριο 8-20 · κοντινές 8-22
    κύριο: τέλειες 44-48 · faithfulness < 5 σε 1-3 · κάλυψη λέξεων ίδια ή +1-3
    κοντινές: σωστό 34-39 · (β) 1-4 · (γ) 1-3 — ΑΝΑΜΕΝΩ επιδείνωση (ο μηχανισμός των m047/m072/m078)
ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ (γραμμένος πριν):
    (1) ΑΣΦΑΛΕΙΑ: κοντινές (β)+(γ) ≤ 3 ΚΑΙ (γ) ≤ 2   (βάση 2 και 1· +1 για τον θόρυβο της κλήρωσης)
    (2) ΚΥΡΙΟ: τέλειες ≥ 45/50 · faithfulness < 5 σε ≤ 3 με απάντηση · κανένας μέσος (με απάντηση)
        κάτω από βάση − 0.10
    Περνάει και τα δύο -> η B μένει υποψήφια: επόμενο χρόνος σε κανονικό μηχάνημα, διάσπαση μέσα
        στην κλήση μετάφρασης, υλοποίηση πίσω από διακόπτη.
    Αποτυγχάνει το (1) -> το κέρδος των multi_hop πληρώνεται με διαρροές: η B ΔΕΝ πάει ως έχει στην
        παραγωγή (επόμενο: οι επιπλέον σελίδες μόνο όταν το σκέλος περνάει ΚΑΙ δικό του φύλακα, ή η D).
    Αποτυγχάνει ΜΟΝΟ το (2) -> η διάσπαση πρέπει να σπάει λιγότερο (routing) πριν ξαναμετρηθεί.

ΑΠΟΤΕΛΕΣΜΑ (29/9/2026, runs/decomp_nonmh.csv + runs/scoreboard/answers_decomp_{main,near_ooc}.json):
    έλεγχοι: σελίδες βάσης = baseline 92/92 ✓ · απαντήσεις βάσης από το πάγωμα, ίδιες με 28/9 ✓
    ROUTING — ΤΟ PROMPT ΤΟΥ ΑΥΓΟΥΣΤΟΥ ΔΕΝ ΞΕΧΩΡΙΖΕΙ: σπάει 35/50 στο κύριο (direct_fact 12/15 ·
        enumeration 9/14 · reasoning 10/12 · multi_hop 4/4 · ooc 0/5) και 36/42 στις κοντινές. Σπάει με
        ΠΑΡΑΦΡΑΣΗ, όχι με θέματα (q002 «hardware provisioning» + «resource allocation»· n003 δύο σχεδόν
        ίδιες διατυπώσεις). Συνέπεια: η B αλλάζει σελίδες σε 34 + 34 — σελίδες μ.ό. κύριο 6.9 -> 9.4,
        κοντινές 7.6 -> 10.7, δηλαδή +~3.000-3.700 tokens στις ΠΕΡΙΣΣΟΤΕΡΕΣ ερωτήσεις, όχι μόνο στις multi_hop.
    ΚΥΡΙΟ: σχεδόν ΑΝΕΠΗΡΕΑΣΤΟ. Από τις 30 ερωτήσεις ενός θέματος με άλλες σελίδες, βαθμοί άλλαξαν ΜΟΝΟ
        στην q029 (5555 -> 5455). Τέλειες 47 -> 45 (q029, q045) · οι παλιές multi_hop q046 4353 -> 5455 και
        q047 1115 -> 5125 καλύτερα. Μέσοι (με απάντηση) 4.89/4.78/4.91/4.96 -> 5.00/4.76/4.91/5.00 ·
        faith < 5: 1 -> 0 · κάλυψη λέξεων 129 -> 130/138 · ooc 5/5.
    ΚΟΝΤΙΝΕΣ ooc: κριτής 38/1/1 -> 40/0/0 — ΑΛΛΑ ΜΕ ΤΟ ΜΑΤΙ η n024 της B ανοίγει με ΤΟ ΙΔΙΟ λάθος
        συμπέρασμα («η γλώσσα δεν επηρεάζει το autoscaling») που στη βάση κρίθηκε (γ)· ο κριτής στάθηκε σε
        άλλη πρόταση -> θόρυβος κριτή, ΟΧΙ βελτίωση. Η n038 (β) -> σωστό είναι πραγματική αλλαγή της
        απάντησης, αλλά η (β) είναι γνωστά κλήρωση (28/9). Ειλικρινής ανάγνωση: ΙΔΙΟ με τη βάση.
        Ο φόβος των multi_hop (m047/m062/m072/m078: «λείπει» -> «χωρίς στήριξη») ΔΕΝ εμφανίστηκε εδώ.
        Υπόθεση (δεν αποδείχθηκε): στις multi_hop το paper του μισού που λείπει ΕΙΝΑΙ στο σώμα και οι
        γειτονικές του σελίδες δίνουν υλικό για υπερ-επέκταση· στις κοντινές ooc το ζητούμενο δεν
        υπάρχει πουθενά.
    ΚΑΝΟΝΑΣ: (1) (β)+(γ) 0 ≤ 3 ✓ · (γ) 0 ≤ 2 ✓ (και με τη n024 ως (γ): 1) · (2) τέλειες 45 ≥ 45 ✓ (ΑΚΡΙΒΩΣ
        στο όριο) · faith < 5: 0 ≤ 3 ✓ · μέσοι ✓ -> Η B ΜΕΝΕΙ ΥΠΟΨΗΦΙΑ.
    ΠΡΟΒΛΕΨΗ: multi_hop κύριου 3-4/4 ✓ · υπόλοιπες 6-18/41 ✗ (31) · ooc κύριου 1-3/5 ✗ (0) · κοντινές
        10-25/42 ✗ (36) · combo 4/4 ✓ · B αλλάζει κύριο 8-20 ✗ (34) · κοντινές 8-22 ✗ (34) · τέλειες 44-48 ✓
        · faith < 5: 1-3 ✗ (0) · κάλυψη ίδια/+1-3 ✓ · κοντινές σωστό 34-39 ✗ · (β) 1-4 ✗ · (γ) 1-3 ✗ ·
        «επιδείνωση» ✗. Τέσσερις στις δεκατρείς — ΥΠΟΤΙΜΗΣΑ ΤΟ ΣΠΑΣΙΜΟ, ΥΠΕΡΕΚΤΙΜΗΣΑ ΤΟΝ ΚΙΝΔΥΝΟ.
    ΤΙ ΣΗΜΑΙΝΕΙ: η B δεν βλάπτει ούτε τις απλές ούτε τις ooc, αλλά με αυτό το routing πληρώνει +1 κλήση
        ΚΑΙ ~+3 σελίδες σχεδόν σε ΚΑΘΕ ερώτηση για κέρδος που υπάρχει ΜΟΝΟ στις multi_hop. Ο μοχλός πλέον
        είναι το ROUTING: οι 29+11 multi_hop ονομάζουν ΡΗΤΑ δύο papers — ένας ντετερμινιστικός έλεγχος
        («αναφέρει ≥ 2 έγγραφα του σώματος;») θα έκοβε και την κλήση και τις σελίδες στις υπόλοιπες.

    docker compose exec backend python evaluation/probe_decomp_nonmh.py --limit 3   # δοκιμή, 3 ανά σετ
    docker compose exec backend python evaluation/probe_decomp_nonmh.py
"""
import argparse
import asyncio
import csv
import hashlib
import inspect
import json
import os
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import probe_decomp_mh40 as P
import scoreboard as S
import scoreboard_answers as SA

import gemini_rest

SETS = {"main": "golden_set_50.jsonl", "near_ooc": "golden_near_ooc.jsonl"}
BASE = os.path.join(S.OUT_DIR, "baseline.json")
OUT_CSV = os.path.join(S.RUNS, "decomp_nonmh.csv")
OUT_ANS = os.path.join(S.OUT_DIR, "answers_decomp_{}")
MAX_LEAKS, MAX_HARD_LEAKS = 3, 2                                  # κανόνας (1)
MIN_PERFECT, MAX_UNFAITHFUL, MAX_MEAN_DROP = 45, 3, 0.10          # κανόνας (2)


def keys_of(pages: list) -> str:
    return ";".join(P.key(m) for _t, m in pages)


def b_and_d(pipe, subs: list, base: list) -> tuple[list, list, int]:
    """B και D ΑΚΡΙΒΩΣ όπως στο probe_decomp_mh40.strategies (χωρίς τη C, που απορρίφθηκε)."""
    mp = pipe.ai.MAX_PAGES
    sfs = [pipe.rerank(s, pipe.candidates(s)) for s in subs]
    legs = [pipe.pages(sf, mp) for sf in sfs if pipe.passes(sf)]
    if not legs:
        return base, base, 0
    reserved = P.round_robin([lp[:2] for lp in legs], mp // 2)
    taken = {P.key(m) for _t, m in reserved}
    d = reserved + [pg for pg in base if P.key(pg[1]) not in taken][:mp - len(reserved)]
    b = base + P.round_robin(legs, 4, skip={P.key(m) for _t, m in base})
    return b, d, len(legs)


def group_of(set_key: str, t: dict) -> str:
    return t.get("category", "") if set_key == "main" else t.get("near_type", "")


async def run(args) -> int:
    real_once = gemini_rest.generate_once       # ΠΡΙΝ από κατασκόπους/πάγωμα
    real_stream = gemini_rest.stream_generate
    with open(BASE, encoding="utf-8") as f:
        base_rows = {(r["set"], r["id"]): r for r in json.load(f)["rows"]}
    base_ans = {}
    for sk in SETS:
        with open(os.path.join(S.OUT_DIR, f"answers_baseline_{sk}.json"), encoding="utf-8") as f:
            base_ans[sk] = {r["id"]: r for r in json.load(f)["rows"]}
    tests = {sk: E.load_jsonl(os.path.join(E.HERE, fn)) for sk, fn in SETS.items()}
    if args.limit:
        tests = {sk: ts[:args.limit] for sk, ts in tests.items()}
    dec = {}
    if os.path.exists(P.DEC_PATH):
        with open(P.DEC_PATH, encoding="utf-8") as f:
            dec = json.load(f)

    ai_core, _near = E.open_store()
    if ai_core.collection.count() != S.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {S.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()
    ai_core.ENABLE_CORRECTIVE = True
    ai_core._save_query_emb_cache = lambda: None

    async def no_gemini(prompt, **kw):
        raise RuntimeError("νέα κλήση Gemini της παραγωγής — δεν επιτρέπεται σε αυτό το script")

    gemini_rest.generate_once = no_gemini
    frozen = S.Frozen(S.FROZEN_PATH, fresh=False)
    frozen.install()
    # ΜΕΤΑ το open_store: ίδιο ai_core. Ο κριτής του κύριου ΑΥΤΟΥΣΙΟΣ, όπως στο scoreboard_answers.
    import eval_engine
    from probe_decomposition import DEFAULT_PROMPT, covered
    missing = SA.check_main_judge(inspect.getsource(eval_engine.evaluate_answer))
    if missing:
        print(f"!! Ο κριτής του eval_engine ΑΛΛΑΞΕ — δεν βρέθηκαν: {missing}")
        return 1

    async def sdk_call(prompt):
        resp = await eval_engine.judge_model.generate_content_async(
            prompt, generation_config=eval_engine.genai.GenerationConfig(
                response_mime_type="application/json", temperature=0.0))
        return resp.text

    # ---------------------------------------------------------------- 1. ανάκτηση + διάσπαση
    pipe = P.Pipe(ai_core)
    model = ai_core.GEMINI_MODEL
    rows, pages_of, mismatch, new_dec = [], {}, [], 0
    for sk, ts in tests.items():
        print(f"\n{sk}: {len(ts)} ερωτήσεις · διάσπαση με {model}\n", flush=True)
        for t in ts:
            base, _b1, _b2, _rw, outcome = await E.retrieve(ai_core, t["question"])
            if keys_of(base) != base_rows[(sk, t["id"])]["pages"]:
                mismatch.append(f"{sk}:{t['id']}")
            q = ai_core._translation_cache.get(t["question"], t["question"])
            prompt = DEFAULT_PROMPT.format(query=q)
            h = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
            if h not in dec:
                t1 = time.perf_counter()
                try:
                    raw = await gemini_rest_once(real_once, prompt, model, ai_core.GEMINI_API_KEY)
                except gemini_rest._RateLimited as e:
                    print(f"  {t['id']} ΟΡΙΟ GEMINI — σταματάω ({str(e)[:120]})", flush=True)
                    return 1
                except Exception as e:
                    print(f"  {t['id']} διάσπαση ΑΠΕΤΥΧΕ: {type(e).__name__}: {str(e)[:200]} — σταματάω")
                    return 1
                dec[h] = {"id": t["id"], "model": model, "query": q, "out": raw,
                          "ms": round(1000 * (time.perf_counter() - t1))}
                new_dec += 1
                P.save_dec(dec)
            subs = P.parse_subs(dec[h]["out"], q)
            b, d, legs_passed = base, base, 0
            if outcome == "passed_gate" and len(subs) > 1:
                b, d, legs_passed = b_and_d(pipe, subs, base)
            pages_of[(sk, t["id"])] = {"base": base, "B": b}
            kws = t.get("keywords") or []
            row = {"set": sk, "id": t["id"], "group": group_of(sk, t), "lang": S.lang_of(t, t["question"]),
                   "outcome": outcome, "n_sub": len(subs), "legs_passed": legs_passed,
                   "base_n": len(base), "B_n": len(b), "D_n": len(d),
                   "B_changed": int(set(keys_of(b).split(";")) != set(keys_of(base).split(";"))),
                   "n_kw": len(kws), "base_cov": covered(base, kws), "B_cov": covered(b, kws),
                   "D_cov": covered(d, kws), "gemini_ms": dec[h]["ms"], "query": q,
                   "subqueries": " || ".join(subs), "base_pages": keys_of(base),
                   "B_pages": keys_of(b), "D_pages": keys_of(d)}
            rows.append(row)
            print(f"  {t['id']:<5} {row['group']:<13} {row['lang']} {S.SHORT[outcome]:<10} σκέλη {len(subs)} "
                  f"| B {'ΑΛΛΑΞΕ' if row['B_changed'] else 'ίδια':<6} σ{len(base)}->{len(b)} "
                  f"κ{row['base_cov']}->{row['B_cov']}/{len(kws)}", flush=True)
    if frozen.errors or frozen.misses:
        print(f"!! {len(frozen.errors) + frozen.misses} κλήσεις έξω από το πάγωμα της παραγωγής — ΔΕΝ ΜΕΤΡΑΕΙ")
        return 1
    print(f"\nΕΛΕΓΧΟΣ 1: σελίδες βάσης = baseline.json σε {len(rows) - len(mismatch)}/{len(rows)}"
          + (f"  !! διαφέρουν: {mismatch}" if mismatch else "  ✓")
          + f" · νέες κλήσεις διάσπασης {new_dec}")
    if mismatch:
        return 1
    if not args.limit:
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ---------------------------------------------------------------- 2. απαντήσεις (βάση ΠΡΩΤΑ)
    cache = SA.AnswerCache(SA.ANS_PATH)
    cache.install_stream(real_stream)
    n_entries0 = len(cache.data)                # πραγματικές κλήσεις = νέες εγγραφές (βλ. probe_decomp_answers)
    ans = {sk: {"base": [], "B": []} for sk in SETS}
    for strat in ("base", "B"):
        for sk, ts in tests.items():
            for t in ts:
                r0 = next(r for r in rows if r["set"] == sk and r["id"] == t["id"])
                if strat == "B" and not r0["B_changed"]:
                    ans[sk]["B"].append(dict(next(a for a in ans[sk]["base"] if a["id"] == t["id"]),
                                             strategy="B"))
                    continue
                row = await answer_row(sk, t, r0, pages_of[(sk, t["id"])][strat], ai_core, cache,
                                       real_once, sdk_call)
                row["strategy"] = strat
                ans[sk][strat].append(row)
                cache.save()
                print(f"  {strat:<4} {sk:<8} {t['id']:<5} {r0['group']:<13} "
                      + (f"ΣΦΑΛΜΑ {row['error']}" if row["error"] else verdict_of(sk, row)), flush=True)
                if row["error"]:
                    print("!! σφάλμα κλήσης — σταματάω (ό,τι πληρώθηκε είναι σωσμένο· ξανατρέξε)")
                    return 1
        if strat == "base":
            flips = [f"{sk}:{a['id']}" for sk in SETS for a in ans[sk]["base"]
                     if verdict_of(sk, a) != verdict_of(sk, base_ans[sk][a["id"]])]
            new = len(cache.data) - n_entries0
            n = sum(len(ts) for ts in tests.values())
            print(f"\nΕΛΕΓΧΟΣ 2 (απαντήσεις βάσης): νέες κλήσεις {new} · ίδιες με 28/9 σε {n - len(flips)}/{n}"
                  + ("  ✓" if not new and not flips else f"  !! {flips}"))
            if new or flips:
                print("!! Η βάση ΔΕΝ βγήκε από το πάγωμα — σταματάω πριν ξοδέψω για τη B")
                return 1
            print()

    report(rows, ans)
    print(f"Gemini: διασπάσεις {new_dec} νέες · απαντήσεις+κριτές {len(cache.data) - n_entries0} νέες · "
          f"της παραγωγής 0 ({frozen.hits} από το πάγωμα)")
    if not args.limit:
        for sk in SETS:
            summ = {s: (SA.summarize_main if sk == "main" else SA.summarize_near)(ans[sk][s])
                    for s in ("base", "B")}
            with open(OUT_ANS.format(sk) + ".json", "w", encoding="utf-8") as f:
                json.dump({"when": time.strftime("%Y-%m-%d %H:%M"), "summary": summ,
                           "rows": ans[sk]["base"] + ans[sk]["B"]}, f, ensure_ascii=False, indent=1)
        print(f"Σώθηκε: {OUT_CSV} · {OUT_ANS.format('main')}.json · {OUT_ANS.format('near_ooc')}.json")
    return 0


async def gemini_rest_once(real_once, prompt, model, api_key):
    return await real_once(prompt, model=model, api_key=api_key, max_output_tokens=P.MAX_OUT)


async def answer_row(sk, t, r0, pages, ai_core, cache, real_once, sdk_call) -> dict:
    row = {"id": t["id"], "lang": r0["lang"], "category": t.get("category", ""), "question": t["question"],
           "outcome": r0["outcome"], "n_pages": len(pages), "pages": keys_of(pages), "error": ""}
    if sk == "near_ooc":
        row["near_type"] = t.get("near_type", "")
    n_err = len(cache.errors)
    answer = await E.answer_of(ai_core, t["question"], pages)
    row["answer"] = answer
    row["refusal"] = int(not pages or bool(E.REFUSAL.search(answer)))
    row["prompt_tokens"], row["out_tokens"], row["thinking_tokens"] = (
        cache.tokens() if pages else (None, None, None))
    if len(cache.errors) > n_err:
        row["error"] = cache.errors[-1]
    elif sk == "near_ooc" and not pages:
        row |= {"label": "cut", "evidence": "", "evfound": 1}
    else:
        try:
            row |= (await SA.judge_main(cache, t, pages, answer, sdk_call) if sk == "main"
                    else await SA.judge_near(cache, t, answer, real_once))
        except Exception as e:
            row["error"] = f"κριτής: {type(e).__name__}: {e}"[:200]
            cache.errors.append(row["error"])
    for s in SA.SCORES:
        row.setdefault(s, None)
    row.setdefault("label", "")
    row.setdefault("feedback", "")
    return row


def verdict_of(sk: str, r: dict) -> str:
    return r.get("label", "") if sk == "near_ooc" else SA._tuple(r)


def report(rows: list[dict], ans: dict) -> None:
    print("\n" + "#" * 100)
    print("ROUTING — πόσες ερωτήσεις ΣΠΑΝΕ (η κλήση γίνεται ούτως ή άλλως σε ΚΑΘΕ ερώτηση):")
    for sk in SETS:
        rs = [r for r in rows if r["set"] == sk]
        by = Counter(r["group"] for r in rs)
        split = Counter(r["group"] for r in rs if r["n_sub"] > 1)
        print(f"  {sk:<9} έσπασε {sum(split.values())}/{len(rs)}   "
              + " · ".join(f"{g} {split[g]}/{n}" for g, n in by.items()))
        ch = [r for r in rs if r["B_changed"]]
        print(f"  {'':<9} η B άλλαξε σελίδες σε {len(ch)} · σελίδες μ.ό. {statistics.mean(r['base_n'] for r in rs):.1f}"
              f" -> {statistics.mean(r['B_n'] for r in rs):.1f}"
              + (f" · κάλυψη λέξεων {sum(r['base_cov'] for r in rs)} -> {sum(r['B_cov'] for r in rs)}"
                 f"/{sum(r['n_kw'] for r in rs)}" if sk == "main" else ""))
    fp = [r for r in rows if r["set"] == "main" and r["n_sub"] > 1 and r["group"] != "multi_hop"]
    if fp:
        print("\n  κύριο, έσπασαν ΧΩΡΙΣ να είναι multi_hop (για έλεγχο με το μάτι):")
        for r in fp:
            print(f"    {r['id']} {r['group']:<13} «{r['query'][:80]}»\n        -> {r['subqueries'][:160]}")

    m = {s: SA.summarize_main(ans["main"][s]) for s in ("base", "B")}
    print("\nΚΥΡΙΟ ΣΕΤ (με απάντηση)   accuracy · completeness · relevance · faithfulness   τέλειες/50  faith<5")
    unfaith = {}
    for s in ("base", "B"):
        ic = m[s]["in_corpus"]
        unfaith[s] = sum(1 for r in ans["main"][s] if r["category"] != "out_of_corpus"
                         and r["faithfulness"] is not None and r["faithfulness"] < 5)
        print(f"  {s:<24} " + " · ".join(f"{ic[x]:.2f}" for x in SA.SCORES)
              + f"   {m[s]['all']['perfect']:>8}/{m[s]['all']['n']}  {unfaith[s]:>6}")
    flips = [(a["id"], SA._tuple(a), SA._tuple(b)) for a, b in zip(ans["main"]["base"], ans["main"]["B"])
             if SA._tuple(a) != SA._tuple(b)]
    print(f"  άλλαξαν βαθμούς: {len(flips)}  " + " · ".join(f"{i} {x}->{y}" for i, x, y in flips))

    n = {s: SA.summarize_near(ans["near_ooc"][s]) for s in ("base", "B")}
    print("\nΚΟΝΤΙΝΕΣ ooc (όσες περνούν τον φύλακα)   σωστό · ήπια διαρροή (β) · διαρροή (γ)   κομμένες")
    for s in ("base", "B"):
        lb = n[s]["labels"]
        print(f"  {s:<38} {lb['correct']:>5} · {lb['soft_leak']:>17} · {lb['leak']:>12}   {n[s]['outcome']['cut']:>8}")
    nflips = [(a["id"], a["near_type"], a["label"], b["label"], b.get("evidence", ""))
              for a, b in zip(ans["near_ooc"]["base"], ans["near_ooc"]["B"]) if a["label"] != b["label"]]
    print(f"  άλλαξε ετικέτα: {len(nflips)}")
    for i, typ, x, y, ev in nflips:
        print(f"    {i} {typ:<7} {x} -> {y}   «{ev[:160]}»")

    lb = n["B"]["labels"]
    ok1 = (lb["soft_leak"] + lb["leak"] <= MAX_LEAKS, lb["leak"] <= MAX_HARD_LEAKS)
    ic_b, ic_0 = m["B"]["in_corpus"], m["base"]["in_corpus"]
    ok2 = (m["B"]["all"]["perfect"] >= MIN_PERFECT, unfaith["B"] <= MAX_UNFAITHFUL,
           all(ic_b[x] >= ic_0[x] - MAX_MEAN_DROP for x in SA.SCORES))
    print(f"\nΚΑΝΟΝΑΣ (1) ασφάλεια: (β)+(γ) ≤ {MAX_LEAKS} {'✓' if ok1[0] else '✗'} · (γ) ≤ {MAX_HARD_LEAKS} "
          f"{'✓' if ok1[1] else '✗'}   (2) κύριο: τέλειες ≥ {MIN_PERFECT} {'✓' if ok2[0] else '✗'} · "
          f"faith<5 ≤ {MAX_UNFAITHFUL} {'✓' if ok2[1] else '✗'} · μέσοι ≥ βάση−{MAX_MEAN_DROP} {'✓' if ok2[2] else '✗'}")
    print("  -> " + ("Η B ΜΕΝΕΙ ΥΠΟΨΗΦΙΑ" if all(ok1) and all(ok2) else
                     "ΑΠΟΤΥΓΧΑΝΕΙ Η ΑΣΦΑΛΕΙΑ: η B δεν πάει ως έχει στην παραγωγή" if not all(ok1) else
                     "ΑΠΟΤΥΓΧΑΝΕΙ ΜΟΝΟ ΤΟ ΚΥΡΙΟ: λιγότερο σπάσιμο πριν ξαναμετρηθεί"))
    print("#" * 100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="N ανά σετ (δοκιμή· δεν σώζει αρχεία αποτελεσμάτων)")
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
