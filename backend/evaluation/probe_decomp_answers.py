"""Γίνονται ΣΩΣΤΕΣ ΑΠΑΝΤΗΣΕΙΣ οι σελίδες που φέρνει η διάσπαση; — Φάση 2, βήμα 3.

ΑΦΟΡΜΗ (probe_decomp_mh40.py, 29/9/2026): στις 29 νέες multi_hop οι σελίδες-τεκμήρια πάνε 30 -> 37/58
με τη D (2 εγγυημένες θέσεις ανά σκέλος, 8 σελίδες) και 30 -> 40/58 με τη B (βάση + έως 4, ~11
σελίδες). Κατά τον κανόνα εκείνου του βήματος μόνο η B περνάει -> ΣΥΜΒΙΒΑΣΜΟΣ που θέλει απόφαση·
η απόφαση θέλει ΑΠΑΝΤΗΣΕΙΣ, όχι σελίδες. Το trace έδειξε ότι η σελίδα κάνει τη διαφορά (ήρθε -> σωστό
27/30 · δεν ήρθε -> 6/28), αλλά αυτό ήταν συσχέτιση πάνω σε ΑΛΛΕΣ ερωτήσεις — εδώ μετριέται η ίδια
ερώτηση με και χωρίς τη σελίδα.

ΤΙ ΚΑΝΕΙ: οι σελίδες ΚΑΘΕ στρατηγικής διαβάζονται από το runs/decomp_mh40.csv (καμία ανάκτηση εδώ) και
τα κείμενά τους από το απομονωμένο store, με ΤΟΝ ΙΔΙΟ τρόπο που τα συναρμολογεί το _expand_to_pages.
Γέννηση με τον ΠΡΑΓΜΑΤΙΚΟ ask_ai και κριτής ΑΝΑ PAPER του scoreboard_answers ΑΥΤΟΥΣΙΟΙ (ίδιο πάγωμα
runs/scoreboard_answers.json, ίδιο πρότυπο, ίδιες ρυθμίσεις) -> η βάση βγαίνει ΟΛΗ από το πάγωμα της
28/9 (answers_baseline_mh_new) και νέες κλήσεις γίνονται ΜΟΝΟ όπου άλλαξαν οι σελίδες.
ΣΕΙΡΑ ΣΕΛΙΔΩΝ: η σειρά αλλάζει το prompt. Η D εδώ κρατά τις σελίδες της βάσης ΜΕ ΤΗ ΣΕΙΡΑ ΤΟΥΣ και βάζει
τις νέες ΣΤΟ ΤΕΛΟΣ (στη θέση όσων έφυγαν) — αλλιώς θα μετρούσαμε και τη μετάθεση: στο probe οι
εγγυημένες μπήκαν πρώτες και το «άλλαξαν σελίδες» έβγαινε 27/29 ενώ το ΣΥΝΟΛΟ άλλαξε σε 23. Η B είναι
ήδη βάση-πρώτα. Όπου το σύνολο είναι ίδιο με τη βάση, το prompt είναι ίδιο -> ίδια απάντηση, 0 κλήσεις.
ΕΛΕΓΧΟΙ: (1) βάση: 0 νέες κλήσεις και ετικέτες ΤΑΥΤΟΣΗΜΕΣ με answers_baseline_mh_new σε 29/29 — αλλιώς
τα κείμενα σελίδων διαφέρουν από το store της 28/9 (τα PDF κατέβηκαν ξανά στις 29/9) και το script
σταματάει ΠΡΙΝ ξοδέψει για D/B· (2) σελίδες-τεκμήρια ανά στρατηγική = ό,τι έγραψε το probe (30/37/40).

ΚΟΣΤΟΣ: D 23 + B 28 = 51 γεννήσεις + 51 κριτές ≈ 102 κλήσεις gemini-2.5-flash, ~0.6 $, ~15 λεπτά.
ΘΟΡΥΒΟΣ: κάθε νέα απάντηση είναι ΜΙΑ κλήρωση (temperature 0.1, thinking 512) και ο κριτής γυρίζει
~2/9 κατά μία μονάδα (11/8). Μία ετικέτα που αλλάζει σε ΜΙΑ ερώτηση δεν είναι εύρημα — γι' αυτό
ζευγαρωτό bootstrap πάνω στις ερωτήσεις.

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (29/9/2026) — βάση (28/9): σωστά μισά 33/58 (ήρθε 27/30 · δεν ήρθε
6/28) · και τα δύο σωστά 7/29 · χωρίς στήριξη 4 · faithfulness 4.52 · τέλειες 7/29.
    D  σωστά μισά 36-40 · και τα δύο 9-13 · χωρίς στήριξη ≤ 4 · faithfulness ≥ 4.4
       (37 σελίδες × ~0.9 + 21 × ~0.2 ≈ 38)
    B  σωστά μισά 37-42 · και τα δύο 10-14 · χωρίς στήριξη ≤ 4 · B − D = 0 έως +3
    ήρθε η σελίδα -> σωστό ≥ 85% και στις δύο (το 27/30 κρατάει και με περισσότερες σελίδες)
ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ (γραμμένος πριν):
    η D ΠΡΟΧΩΡΑΕΙ αν: σωστά μισά ≥ 37 (+4) · κάτω όριο του ζευγαρωτού CI της διαφοράς > 0 ·
        χωρίς στήριξη ≤ 5 · faithfulness ≥ 4.4. Επόμενο βήμα τότε: routing σε ΟΛΑ τα σετ (ψευδώς
        θετικά), χρόνος σε κανονικό μηχάνημα, διάσπαση μέσα στην κλήση μετάφρασης.
    η B αξίζει τα +3.7 σελίδες ΜΟΝΟ αν περνάει τα ίδια ΚΑΙ ξεπερνά τη D κατά ≥ 3 σωστά μισά.
    Αν καμία δεν περνάει: οι σελίδες ΔΕΝ γίνονται απαντήσεις -> η διάσπαση απορρίπτεται στο επίπεδο
        της απάντησης· επόμενο HippoRAG 2.

ΑΠΟΤΕΛΕΣΜΑ (29/9/2026, runs/scoreboard/answers_decomp_mh_new.json · 96 νέες κλήσεις, gemini-2.5-flash):
    έλεγχοι: βάση 0 νέες κλήσεις, ετικέτες ίδιες με 28/9 σε 29/29 ✓ (τα PDF που κατέβηκαν ξανά δίνουν
    ΙΔΙΑ κείμενα σελίδων byte προς byte) · σελίδες-τεκμήρια 30/37/40 = probe ✓
    στρατηγική  σελ.  τεκμ./58  σωστά μισά/58  δύο σωστά/29  «λείπει»  χωρίς στήρ.  faith  τέλειες
    βάση         7.8     30          33             7            9          4        4.52     7
    D            8.0     37          39            14            4          4        4.52    14
    B           11.6     40          43            17            2          5        4.52    17
    Δ σωστά μισά (ζευγαρωτό CI 95%): D +6 [+0, +12] · B +10 [+4, +16]
    ήρθε η σελίδα -> σωστό: βάση 27/30 · D 35/37 · B 37/40 — ΤΟ 27/30 ΚΡΑΤΑΕΙ και με ~12 σελίδες.
    ΚΑΝΟΝΑΣ: D ✓ ✗ ✓ ✓ (το κάτω όριο είναι ΑΚΡΙΒΩΣ 0, όχι > 0) · B ✓ ✓ ✓ ✓ και B − D = +4 (≥ +3)
    -> Η B ΠΡΟΧΩΡΑΕΙ στο επόμενο βήμα. Τα «δύο σωστά» διπλασιάζονται και κάτι (7 -> 17).
    Η B μετρήθηκε ΣΥΝΤΗΡΗΤΙΚΑ: m010 ο κριτής έγραψε σκέτο «?» για το 2ο μισό, ενώ το feedback του
    δείχνει σωστό (-> ίσως 44).
    ⚠️ ΤΟ ΕΥΡΗΜΑ ΑΣΦΑΛΕΙΑΣ: το σύνολο «χωρίς στήριξη» κρύβει ανακατάταξη. Στη B 4 μισά πέρασαν από
    «λείπει» (σωστή άρνηση) σε «χωρίς στήριξη» — m047, m062, m072, m078, ΟΛΑ χωρίς τη σελίδα-τεκμήριο —
    και 4 άλλα διορθώθηκαν (m039, m064, m071, m079). Στη D 2 (m047, m078). Με περισσότερες σελίδες το
    μοντέλο σταματά να λέει «λείπει» και συμπληρώνει από γειτονικό υλικό: το «λείπει» πέφτει 9 -> 2, όχι
    μόνο προς το «σωστό». Μία κλήρωση ανά απάντηση -> ΕΝΔΕΙΞΗ, αλλά είναι ακριβώς ο κίνδυνος που
    πρέπει να μετρηθεί ΠΡΙΝ την παραγωγή: κοντινές ooc (42, το σωστό είναι «δεν υπάρχει») με τη B.
    ΠΡΟΒΛΕΨΗ: D σωστά ✓ (39) · D δύο ✗ (14, όχι 9-13) · D χωρίς στήριξη ✓ · D faith ✓ · B σωστά ✗ (43,
    όχι 37-42) · B δύο ✗ (17) · B χωρίς στήριξη ✗ (5) · B − D ✗ (+4) · ήρθε -> ≥ 85% ✓. Τέσσερις στις
    εννέα — ΥΠΟΤΙΜΗΣΑ τη B, ανάποδα από το βήμα των σελίδων (όπου υπερεκτίμησα τη διάσπαση).

    docker compose exec backend python evaluation/probe_decomp_answers.py --limit 2   # δοκιμή
    docker compose exec backend python evaluation/probe_decomp_answers.py
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import bootstrap_ci
import eval_near_ooc as E
import scoreboard as S
import scoreboard_answers as SA

import gemini_rest

DEC_CSV = os.path.join(S.RUNS, "decomp_mh40.csv")
BASE_ANS = os.path.join(S.OUT_DIR, "answers_baseline_mh_new.json")
OUT = os.path.join(S.OUT_DIR, "answers_decomp_mh_new")
STRATS = ("base", "D", "B")
MIN_CORRECT, MAX_UNSUP, MIN_FAITH, B_OVER_D = 37, 5, 4.4, 3      # κανόνας (docstring)


def page_keys(strat: str, row: dict) -> list[str]:
    """Σελίδες της στρατηγικής· στη D οι σελίδες της βάσης ΜΕ ΤΗ ΣΕΙΡΑ ΤΟΥΣ, οι νέες στο τέλος."""
    got = [k for k in row[f"{strat}_pages"].split(";") if k]
    if strat != "D":
        return got
    base = [k for k in row["base_pages"].split(";") if k]
    return [k for k in base if k in got] + [k for k in got if k not in base]


def page_of(ai_core, meta: dict) -> tuple[str, dict]:
    """Ό,τι κάνει το _expand_to_pages για ΜΙΑ σελίδα (κλειδί doc_id + page), χωρίς το όριο ανά αρχείο."""
    pg = ai_core.collection.get(where={"$and": [{"doc_id": meta["doc_id"]}, {"page": meta["page"]}]})
    ordered = sorted(zip(pg["ids"], pg["documents"]), key=lambda p: ai_core._chunk_idx_from_id(p[0]))
    return "\n".join(d for _i, d in ordered), {"file_name": meta["file_name"], "page": meta["page"]}


def correct_halves(r: dict) -> int:
    return (r["label_1"] == "correct") + (r["label_2"] == "correct")


async def run(args) -> int:
    real_once = gemini_rest.generate_once       # ΠΡΙΝ από κατασκόπους/πάγωμα της ανάκτησης
    real_stream = gemini_rest.stream_generate
    with open(DEC_CSV, encoding="utf-8-sig") as f:
        dec = {r["id"]: r for r in csv.DictReader(f) if r["set"] == "mh_new"}
    with open(BASE_ANS, encoding="utf-8") as f:
        base_ans = {r["id"]: r for r in json.load(f)["rows"]}
    tests = E.load_jsonl(SA.MH_PATH)
    if args.limit:
        tests = tests[:args.limit]
    orders = {t["id"]: SA.mention_order(t) for t in tests}     # σκάει ΠΡΙΝ τη φόρτωση

    ai_core, _near = E.open_store()
    if ai_core.collection.count() != S.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {S.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()
    frozen = S.Frozen(S.FROZEN_PATH, fresh=False)   # το ask_ai δεν ψάχνει εδώ· δίχτυ ασφαλείας
    frozen.install()
    cache = SA.AnswerCache(SA.ANS_PATH)
    cache.install_stream(real_stream)
    # ΠΡΑΓΜΑΤΙΚΕΣ κλήσεις = νέες εγγραφές. Το cache.misses μετράει τον κριτή ΔΙΠΛΑ: το generate_once
    # τρέχει πάνω στο stream_generate, που είναι ήδη τυλιγμένο -> ίδιο κλειδί, μία κλήση, δύο misses.
    n_entries0 = len(cache.data)
    meta_of = {f"{m['file_name']}:{m['page']}": m for m in ai_core._get_bm25_index()["metas"]}

    rows: list[dict] = []
    print(f"\n{len(tests)} ερωτήσεις × {len(STRATS)} στρατηγικές · νέες κλήσεις ΜΟΝΟ όπου άλλαξαν σελίδες\n")
    for strat in STRATS:                        # η βάση ΠΡΩΤΑ: αν δεν βγει από το πάγωμα, σταματάμε
        misses0 = cache.misses
        for t in tests:
            t0 = time.perf_counter()
            r = dec[t["id"]]
            pages = [page_of(ai_core, meta_of[k]) for k in page_keys(strat, r)]
            row = {"id": t["id"], "strategy": strat, "lang": S.lang_of(t, t["question"]),
                   "n_pages": len(pages), "pages": ";".join(page_keys(strat, r)),
                   "same_set_as_base": int(set(r[f"{strat}_pages"].split(";"))
                                           == set(r["base_pages"].split(";"))),
                   "both": S.both_docs(t, pages), "error": ""}
            for h, ep in enumerate(orders[t["id"]], 1):
                sig = SA.half_signals(t, ep, pages)
                row |= {f"doc_{h}": sig["doc"], f"ev_{h}": sig["ev"], f"paper_{h}": sig["paper"],
                        f"kw_{h}": sig["kw"]}
            n_err = len(cache.errors)
            answer = await E.answer_of(ai_core, t["question"], pages)
            row["answer"] = answer
            row["prompt_tokens"], row["out_tokens"], row["thinking_tokens"] = cache.tokens()
            if len(cache.errors) > n_err:
                row["error"] = cache.errors[-1]
            else:
                try:
                    row |= await SA.judge_mh(cache, t, row, pages, answer, ai_core, real_once)
                except Exception as e:
                    row["error"] = f"κριτής: {type(e).__name__}: {e}"[:200]
                    cache.errors.append(row["error"])
            for h in (1, 2):
                row.setdefault(f"label_{h}", "")
            for s in SA.SCORES:
                row.setdefault(s, None)
            rows.append(row)
            cache.save()                        # μετά από ΚΑΘΕ ερώτηση: ένα 429 δεν χάνει τα πληρωμένα
            print(f"  {strat:<4}" + SA.line_mh(t, row)
                  + (f"ΣΦΑΛΜΑ {row['error']}" if row["error"] else
                     "/".join(str(row[s]) for s in SA.SCORES))
                  + f"  {time.perf_counter() - t0:.1f}s", flush=True)
            if row["error"]:
                print("!! σφάλμα κλήσης — σταματάω (ό,τι πληρώθηκε είναι σωσμένο· ξανατρέξε)")
                return 1
        if strat == "base":
            flips = [r["id"] for r in rows if r["strategy"] == "base" and base_ans.get(r["id"])
                     and (r["label_1"], r["label_2"]) != (base_ans[r["id"]]["label_1"],
                                                         base_ans[r["id"]]["label_2"])]
            new = cache.misses - misses0
            print(f"\nΕΛΕΓΧΟΣ 1 (βάση): νέες κλήσεις {new} · ετικέτες ίδιες με 28/9 σε "
                  f"{len(tests) - len(flips)}/{len(tests)}" + ("  ✓" if not new and not flips else ""))
            if new or flips:
                print("!! Η βάση ΔΕΝ βγήκε από το πάγωμα: τα κείμενα σελίδων διαφέρουν από το store της "
                      "28/9 — σταματάω πριν ξοδέψω για D/B")
                return 1
            print()

    report(rows, dec, len(tests))
    print(f"Gemini: {len(cache.data) - n_entries0} νέες κλήσεις (γεννήσεις + κριτές)")
    if not args.limit:
        with open(OUT + ".json", "w", encoding="utf-8") as f:
            json.dump({"when": time.strftime("%Y-%m-%d %H:%M"), "judge": {"model": E.MODEL, **SA.JUDGE_KW},
                       "summary": {s: SA.summarize([r for r in rows if r["strategy"] == s])
                                   for s in STRATS}, "rows": rows}, f, ensure_ascii=False, indent=1)
        with open(OUT + ".csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
        print(f"Σώθηκε: {OUT}.json / .csv")
    return 0


def report(rows: list[dict], dec: dict, n: int) -> None:
    by = {s: {r["id"]: r for r in rows if r["strategy"] == s} for s in STRATS}
    ids = list(by["base"])
    print("\n" + "#" * 104)
    ev_ok = all(sum(by[s][i]["ev_1"] + by[s][i]["ev_2"] for i in ids)
                == sum(int(dec[i][f"{s}_ev"]) for i in ids) for s in STRATS)
    print("ΕΛΕΓΧΟΣ 2: σελίδες-τεκμήρια ανά στρατηγική = probe_decomp_mh40" + ("  ✓" if ev_ok else "  !! ΔΙΑΦΕΡΟΥΝ"))
    print(f"\n{'στρατηγική':<12}{'σελ.':>6}{'τεκμ./' + str(2 * n):>9}{'σωστά μισά':>12}{'δύο σωστά':>11}"
          f"{'«λείπει»':>10}{'χωρίς στήρ.':>12}{'faith':>7}{'τέλειες':>9}   Δ σωστά μισά (CI 95%)")
    verdict = {}
    for s in STRATS:
        rs = [by[s][i] for i in ids]
        sm = SA.summarize(rs)
        ch = sum(correct_halves(r) for r in rs)
        unsup = sm["ev1"]["unsupported"] + sm["ev0"]["unsupported"]
        missing = sm["ev1"]["declared_missing"] + sm["ev0"]["declared_missing"]
        line = (f"{s:<12}{statistics.mean(r['n_pages'] for r in rs):>6.1f}"
                f"{sum(r['ev_1'] + r['ev_2'] for r in rs):>9}{ch:>12}{sm['both_correct']:>11}"
                f"{missing:>10}{unsup:>12}{sm['faithfulness']:>7}{sm['perfect']:>9}")
        if s != "base":
            m, lo, hi = bootstrap_ci.paired_ci(
                [(correct_halves(by["base"][i]), correct_halves(by[s][i])) for i in ids], 10000, 42)
            line += f"   {m * n:+.1f} [{lo * n:+.1f}, {hi * n:+.1f}]"
            verdict[s] = {"ch": ch, "ok": (ch >= MIN_CORRECT, lo > 0, unsup <= MAX_UNSUP,
                                           (sm["faithfulness"] or 0) >= MIN_FAITH)}
        print(line)
        e1, e0 = sm["ev1"], sm["ev0"]
        print(f"{'':<12}ήρθε η σελίδα: σωστό {e1['correct']}/{e1['n']} · δεν ήρθε: σωστό {e0['correct']}/"
              f"{e0['n']}, «λείπει» {e0['declared_missing']}, χωρίς στήριξη {e0['unsupported']}")
    print(f"(Δ: σωστά μισά σε σύνολο {n} ερωτήσεων, ζευγαρωτό bootstrap 10.000, seed 42)")

    print(f"\nΚΑΝΟΝΑΣ (σωστά ≥ {MIN_CORRECT} · CI > 0 · χωρίς στήριξη ≤ {MAX_UNSUP} · faith ≥ {MIN_FAITH}):")
    for s, v in verdict.items():
        print(f"  {s:<4}" + " ".join("✓" if x else "✗" for x in v["ok"]) + ("   ΠΕΡΝΑΕΙ" if all(v["ok"]) else ""))
    if "D" in verdict and "B" in verdict:
        gap = verdict["B"]["ch"] - verdict["D"]["ch"]
        print(f"  B − D = {gap:+d} σωστά μισά (η B αξίζει τις +σελίδες μόνο με ≥ +{B_OVER_D})")

    print("\nΑΛΛΑΓΕΣ ΑΝΑ ΕΡΩΤΗΣΗ (ετικέτα 1ου/2ου μισού· * = ήρθε η σελίδα-τεκμήριο):")
    for i in ids:
        cells = []
        for s in STRATS:
            r = by[s][i]
            cells.append(f"{s} " + "/".join(f"{r[f'label_{h}'] or '-'}{'*' if r[f'ev_{h}'] else ''}"
                                             for h in (1, 2)))
        if len({c.split(' ', 1)[1] for c in cells}) > 1:
            print(f"  {i} {by['base'][i]['lang']}  " + "  |  ".join(cells))
    print("#" * 104)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="μόνο οι πρώτες N (δοκιμή· δεν σώζει αρχεία)")
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
