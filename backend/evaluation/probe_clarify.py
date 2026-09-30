"""Φάση 5: αξίζει ο agent «σε ποιο έγγραφο αναφέρεσαι;» με προτάσεις; — ένα probe, σχεδόν μηδέν κόστος.

ΣΗΜΕΡΑ: αν ο φύλακας κόψει (και ο corrective δεν σώσει) ΚΑΙ η ερώτηση δείχνει σε κάτι που δεν ονομάζει
(ai_core._has_dangling_referent), η απάντηση είναι η γενική «Σε ποιο σύστημα ή paper αναφέρεσαι;» —
ο χρήστης ξαναγράφει την ερώτηση από την αρχή. Στα σετ μας φτάνουν εκεί μόνο 2/235 (h009, h012).
Η ΙΔΕΑ: στο ΙΔΙΟ σημείο, 3 κουμπιά με τα έγγραφα που είναι πιο κοντά στην ερώτηση· κλικ = η ΙΔΙΑ ερώτηση
ΜΟΝΟ σε εκείνο το έγγραφο (η επιλογή εγγράφων υπάρχει ήδη στο UI), ξανά από τον φύλακα. Το σύστημα
ΡΩΤΑΕΙ, δεν μαντεύει: η αποσαφήνιση της 13/8 διάλεγε μόνη της εκδοχή και άνοιξε 5 διαρροές.

ΤΟ ΣΕΤ (golden_dangling.jsonl, 14): ΚΑΘΕ ερώτηση των σετ που ονομάζει ΕΝΑ έγγραφο, απαντιέται σήμερα και
δεν είναι multi_hop (10 en + 4 el), με το όνομα αντικατεστημένο με το χέρι από «that system / that paper
/ εκείνο το σύστημα». Λέξεις-κλειδιά και έγγραφο-στόχος από τη γονική. Μικρό δείγμα: οι περισσότερες
ερωτήσεις ρωτάνε για ΠΕΡΙΕΧΟΜΕΝΟ χωρίς να ονομάζουν paper — εύρημα από μόνο του.

ΤΙ ΜΕΤΡΑΕΙ (ο ΠΡΑΓΜΑΤΙΚΟΣ search_documents, απομονωμένο store):
    1. πού καταλήγει κάθε παραλλαγή: φύλακας / corrective / σιωπή — και αν ενεργοποιείται η διευκρίνιση
       (σιωπή ΚΑΙ ανιχνευτής, όπως στο ask_ai)
    2. όσες ΠΕΡΝΑΝΕ: από το σωστό έγγραφο; κάλυψη λέξεων έναντι της γονικής (το σιωπηλό λάθος — απάντηση
       από ΑΛΛΟ paper — η διευκρίνιση δεν το πιάνει)
    3. προτάσεις: είναι το σωστό έγγραφο στις 3 πρώτες; δύο τρόποι — έγγραφα των κορυφαίων κομματιών του
       ΚΡΙΤΗ (rerank) · του DENSE μόνο (φθηνότερο, χωρίς rerank). Τύχη: 3/7 = 43%.
    4. μετά το κλικ (search_documents με target_filenames=[σωστό έγγραφο]): περνάει ο φύλακας; κάλυψη;
ΚΟΣΤΟΣ: μεταφράσεις των 4 ελληνικών + αναδιατυπώσεις του corrective όπου κόβει — ≤ ~30 σύντομες κλήσεις,
κλάσματα του σεντ. Παγώνουν στο runs/clarify_gemini.json (όχι στο πάγωμα του scoreboard).

ΠΡΟΒΛΕΨΗ, ΓΡΑΜΜΕΝΗ ΠΡΙΝ ΤΟ ΤΡΕΞΙΜΟ (30/9/2026):
    περνάνε τον φύλακα ούτως ή άλλως 8-11/14 (το περιεχόμενο δείχνει το paper: vpxenc, fluid code, counters)
    ενεργοποιείται η διευκρίνιση 2-5/14 · όσες περνάνε: σωστό έγγραφο στις σελίδες ≥ 90%
    προτάσεις σωστό στις 3: κριτής 13-14/14 · dense 12-14/14 · στην 1η θέση (κριτής) 10-13
    μετά το κλικ: περνάει ο φύλακας 12-14/14 · κάλυψη ίση με της γονικής σε ≥ 11/14
ΚΑΝΟΝΑΣ ΑΠΟΦΑΣΗΣ (γραμμένος πριν):
    ΥΛΟΠΟΙΕΙΤΑΙ (στο σημείο της σιωπής, με τον καλύτερο από τους δύο τρόπους προτάσεων) αν:
        σωστό έγγραφο στις 3 προτάσεις ≥ 12/14 ΚΑΙ μετά το κλικ κάλυψη ≥ 1 λέξη σε ≥ 12/14.
    αλλιώς μένει η σημερινή γενική ερώτηση και η Φάση 5 καταγράφεται ως απορριφθείσα.
    Αν ενεργοποιείται σε ≤ 1/14 ακόμα και σε ερωτήσεις ΦΤΙΑΓΜΕΝΕΣ για αυτό, καταγράφεται ρητά: η αξία είναι
    στο demo, όχι στα νούμερα.

ΑΠΟΤΕΛΕΣΜΑ (30/9/2026, runs/clarify.csv · 6 κλήσεις Gemini):
    ΠΕΡΝΑΝΕ ΚΑΝΟΝΙΚΑ 12/14 — με ΙΔΙΑ κάλυψη με τις γονικές (35/35) και σελίδες του σωστού εγγράφου σε
        ΟΛΕΣ. Το όνομα σχεδόν δεν χρειάζεται: το ΠΕΡΙΕΧΟΜΕΝΟ της ερώτησης δείχνει το paper.
    Η διευκρίνιση ενεργοποιείται σε 2/14 (d040, d156 — ExCamera, «τέσσερις μετρικές/παράγοντες»).
    Προτάσεις: σωστό έγγραφο στις 3 — κριτής 14/14 (1η θέση 11) · dense 14/14 (1η 10). Ακριβείς.
    ΜΕΤΑ ΤΟ ΚΛΙΚ: περνάει ο φύλακας 12/14 · κάλυψη ≥ 1 σε 11/14 — και οι 2 που ΕΝΕΡΓΟΠΟΙΟΥΝ τη
        διευκρίνιση ΚΟΒΟΝΤΑΙ ΞΑΝΑ (0/3). Ακριβώς εκεί που θα έβγαιναν τα κουμπιά, το κλικ οδηγεί σε «δεν βρέθηκε».
    ΓΙΑΤΙ ΕΙΝΑΙ ΔΟΜΙΚΟ, ΟΧΙ ΑΤΥΧΙΑ: ο φύλακας κρίνει τον ΚΑΛΥΤΕΡΟ βαθμό (ερώτηση, κομμάτι). Ο περιορισμός σε
        ένα έγγραφο ΑΦΑΙΡΕΙ ανταγωνιστές — δεν μπορεί να ΑΝΕΒΑΣΕΙ τον βαθμό ενός κομματιού που ήδη κρίθηκε.
        Όταν κόβει, φταίει η ΑΣΘΕΝΕΣΤΕΡΗ ερώτηση («that system» αντί για «ExCamera»), όχι το ποιο έγγραφο.
    ΚΑΝΟΝΑΣ: σωστό στις 3 14 ≥ 12 ✓ · μετά το κλικ 11 < 12 ✗ -> ΜΕΝΕΙ Η ΓΕΝΙΚΗ ΕΡΩΤΗΣΗ. Η Φάση 5, όπως
        σχεδιάστηκε (κλικ = περιορισμός σε έγγραφο), ΑΠΟΡΡΙΠΤΕΤΑΙ.
    ΠΡΟΒΛΕΨΗ: περνάνε 8-11 ✗ (12) · διευκρίνιση 2-5 ✓ (2) · σωστό έγγραφο όσες περνάνε ✓ · προτάσεις κριτής
        13-14 ✓ · dense 12-14 ✓ · 1η θέση 10-13 ✓ · μετά το κλικ φύλακας 12-14 ✓ · κάλυψη = γονικής ≥ 11 ✓ (11).
        Έξι στις επτά — σωστά νούμερα, αλλά ΔΕΝ είχα προβλέψει ότι οι 2 που μετράνε θα ήταν ακριβώς οι 2 που αποτυγχάνουν.
    ΙΔΕΑ ΠΟΥ ΜΕΝΕΙ (ΔΕΝ μετρήθηκε, και ΔΕΝ μετριέται με αυτό το σετ): κλικ = αντικατάσταση του «that system» με το
        ΟΝΟΜΑ που διάλεξε ο χρήστης. Εδώ θα έβγαινε τέλειο ΕΞ ΚΑΤΑΣΚΕΥΗΣ (οι παραλλαγές φτιάχτηκαν σβήνοντας ακριβώς
        αυτό το όνομα) — κυκλικό τεστ. Θέλει ερωτήσεις γραμμένες ΑΠΟ ΤΗΝ ΑΡΧΗ χωρίς όνομα.

    docker compose run --rm --no-deps -v eval_near_ooc_store:/tmp/eval_near_ooc_chroma backend \
        python -u evaluation/probe_clarify.py
"""
import asyncio
import csv
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
import probe_decomp_mh40 as P
import scoreboard as S

VARIANTS = os.path.join(E.HERE, "golden_dangling.jsonl")
PARENTS = {"main": "golden_set_50.jsonl", "hard_new": "golden_hard_new.jsonl",
           "tables": "golden_tables.jsonl"}
FROZEN = os.path.join(S.RUNS, "clarify_gemini.json")
OUT = os.path.join(S.RUNS, "clarify.csv")
MIN_HIT3, MIN_AFTER = 12, 12                                     # κανόνας (docstring)


def top_docs(metas: list, k: int = 3) -> list[str]:
    out = []
    for m in metas:
        f = m.get("file_name")
        if f not in out:
            out.append(f)
        if len(out) == k:
            break
    return out


def rank_of(doc: str, docs: list) -> int | None:
    return docs.index(doc) + 1 if doc in docs else None


async def search(ai_core, question: str, target: list | None):
    """(σελίδες, outcome) — ό,τι κάνει το E.retrieve, αλλά με επιλογή εγγράφων."""
    E._calls.clear()
    E._prompts.clear()
    pages = await ai_core.search_documents(question, target, user_id=E.TEST_USER)
    best1 = E._calls[0] if E._calls else None
    if not pages:
        return pages, "cut"
    return pages, "passed_gate" if best1 is not None and best1 >= ai_core.MIN_RERANK_SCORE else "passed_corrective"


async def run() -> int:
    variants = E.load_jsonl(VARIANTS)
    parents = {(sk, t["id"]): t for sk, fn in PARENTS.items() for t in E.load_jsonl(os.path.join(E.HERE, fn))}
    with open(os.path.join(S.OUT_DIR, "baseline.json"), encoding="utf-8") as f:
        base = {(r["set"], r["id"]): r for r in json.load(f)["rows"]}

    ai_core, _near = E.open_store()
    if ai_core.collection.count() != S.EXPECTED_CHUNKS:
        print(f"!! Το store δεν έχει {S.EXPECTED_CHUNKS} chunks — σταματάω")
        return 1
    ai_core._translation_cache.clear()
    ai_core.ENABLE_CORRECTIVE = True
    ai_core._save_query_emb_cache = lambda: None
    frozen = S.Frozen(FROZEN, fresh=False)        # ΝΕΕΣ κλήσεις επιτρέπονται (λίγες) — παγώνουν εδώ
    frozen.install()
    from probe_decomposition import covered  # ΜΕΤΑ το open_store: ίδιο ai_core

    pipe = P.Pipe(ai_core)
    idx = pipe.idx
    rows = []
    print(f"\n{len(variants)} παραλλαγές με «εκείνο το σύστημα»\n", flush=True)
    for v in variants:
        par = parents[(v["parent_set"], v["parent_id"])]
        kws = par["keywords"]
        doc = v["doc"]
        pages, outcome = await search(ai_core, v["question"], None)
        q = ai_core._translation_cache.get(v["question"], v["question"])
        flagged = ai_core._has_dangling_referent(v["question"])      # όπως το ask_ai: στην ερώτηση του χρήστη
        rr = pipe.rerank(q, pipe.candidates(q))
        by_rerank = top_docs([m for _s, _t, m in rr])
        dense = ai_core._dense_exact_ids(pipe.dm, q, pipe.allowed, ai_core.DENSE_CANDIDATES)
        by_dense = top_docs([idx["metas"][idx["pos"][i]] for i in dense])
        after, outcome2 = await search(ai_core, v["question"], [doc])
        docs_in = [m.get("file_name") for _x, m in pages]
        row = {"id": v["id"], "parent": f"{v['parent_set']}:{v['parent_id']}", "lang": v["lang"], "doc": doc,
               "outcome": outcome, "flagged": int(flagged), "clarify": int(outcome == "cut" and flagged),
               "cov_parent": base[(v["parent_set"], v["parent_id"])]["cov"], "n_kw": len(kws),
               "cov": covered(pages, kws), "right_doc_pages": docs_in.count(doc), "n_pages": len(pages),
               "top_page_doc": docs_in[0] if docs_in else "",
               "rerank_top3": ";".join(by_rerank), "rerank_rank": rank_of(doc, by_rerank),
               "dense_top3": ";".join(by_dense), "dense_rank": rank_of(doc, by_dense),
               "after_outcome": outcome2, "after_cov": covered(after, kws), "translation": q,
               "question": v["question"]}
        rows.append(row)
        print(f"  {v['id']:<6} {v['lang']} {S.SHORT[outcome]:<10} σήμα {row['flagged']} διευκρίνιση {row['clarify']}"
              f" | κάλυψη {row['cov']}/{len(kws)} (γονική {row['cov_parent']}) σωστό έγγρ. {row['right_doc_pages']}/{len(pages)}"
              f" | προτάσεις κριτής #{row['rerank_rank'] or '-'} dense #{row['dense_rank'] or '-'}"
              f" | κλικ: {S.SHORT[outcome2]} {row['after_cov']}/{len(kws)}", flush=True)

    if frozen.errors:
        print(f"!! σφάλματα Gemini: {frozen.errors[:2]} — ΤΟ ΤΡΕΞΙΜΟ ΔΕΝ ΜΕΤΡΑΕΙ")
        return 1
    frozen.save()
    report(rows, frozen)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Σώθηκε: {OUT}")
    return 0


def report(rows: list[dict], frozen) -> None:
    n = len(rows)
    print("\n" + "#" * 96)
    oc = {o: sum(r["outcome"] == o for r in rows) for o in ("passed_gate", "passed_corrective", "cut")}
    print(f"ΠΟΥ ΚΑΤΑΛΗΓΟΥΝ ({n}): φύλακας {oc['passed_gate']} · corrective {oc['passed_corrective']} · σιωπή {oc['cut']}"
          f"   ·   σήμα ανιχνευτή {sum(r['flagged'] for r in rows)}/{n}")
    print(f"ΕΝΕΡΓΟΠΟΙΕΙΤΑΙ Η ΔΙΕΥΚΡΙΝΙΣΗ (σιωπή ΚΑΙ σήμα): {sum(r['clarify'] for r in rows)}/{n}")
    passed = [r for r in rows if r["outcome"] != "cut"]
    if passed:
        wrong = [r["id"] for r in passed if r["right_doc_pages"] == 0]
        print(f"όσες ΠΕΡΝΑΝΕ ({len(passed)}): κάλυψη {sum(r['cov'] for r in passed)} έναντι γονικής "
              f"{sum(r['cov_parent'] for r in passed)} /{sum(r['n_kw'] for r in passed)} · ΚΑΜΙΑ σελίδα του σωστού "
              f"εγγράφου: {len(wrong)} {wrong}")
    for name in ("rerank", "dense"):
        ranks = [r[f"{name}_rank"] for r in rows]
        print(f"προτάσεις «{'κριτής' if name == 'rerank' else 'dense'}»: σωστό στις 3 {sum(x is not None for x in ranks)}/{n}"
              f" · 1η θέση {sum(x == 1 for x in ranks)}/{n}   (τύχη στις 3: 43%)")
    ao = sum(r["after_outcome"] != "cut" for r in rows)
    print(f"ΜΕΤΑ ΤΟ ΚΛΙΚ: περνάει ο φύλακας {ao}/{n} · κάλυψη ≥ 1 σε {sum(r['after_cov'] >= 1 for r in rows)}/{n}"
          f" · ίση με της γονικής σε {sum(r['after_cov'] >= r['cov_parent'] for r in rows)}/{n} · σύνολο "
          f"{sum(r['after_cov'] for r in rows)} έναντι {sum(r['cov_parent'] for r in rows)}")
    best = max(sum(r["rerank_rank"] is not None for r in rows), sum(r["dense_rank"] is not None for r in rows))
    after_ok = sum(r["after_cov"] >= 1 for r in rows)
    ok = best >= MIN_HIT3 and after_ok >= MIN_AFTER
    print(f"\nΚΑΝΟΝΑΣ: σωστό στις 3 ≥ {MIN_HIT3} ({best}) · μετά το κλικ κάλυψη ≥ 1 σε ≥ {MIN_AFTER} ({after_ok}) -> "
          + ("ΥΛΟΠΟΙΕΙΤΑΙ" if ok else "ΜΕΝΕΙ Η ΓΕΝΙΚΗ ΕΡΩΤΗΣΗ"))
    print(f"Gemini: {frozen.misses} νέες κλήσεις · {frozen.hits} από το πάγωμα")
    print("#" * 96)


def main() -> int:
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
