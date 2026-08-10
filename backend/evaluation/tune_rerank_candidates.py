"""Πόσα υποψήφια χρειάζεται πραγματικά ο reranker; — σάρωση RERANK_CANDIDATES.

ΓΙΑΤΙ ΤΩΡΑ:
Το 15 βαθμονομήθηκε για το MiniLM-L-6. Το L-12 κατατάσσει καλύτερα (σωστό chunk
στη θέση 1: 40/56 αντί 36/56 · στις 3 πρώτες: 51/56 αντί 49). Αν το σωστό υλικό
είναι σχεδόν πάντα ψηλά, το να βαθμολογούμε 15 ζεύγη είναι σπατάλη. Το rerank
είναι ΓΡΑΜΜΙΚΟ στα candidates και τρώει το 97% του retrieval χρόνου -> κάθε
candidate που φεύγει είναι ~47 ms.

ΤΙ ΜΕΤΡΑΕΙ (τρία πράγματα, και το 3ο έχει δικαίωμα ΒΕΤΟ):
  1. coverage / kw_rank  — χάνεται σωστό υλικό;
  2. latency του rerank  — πόσο κερδίζουμε πραγματικά (μετρημένο, όχι εκτίμηση)
  3. ΤΟ ΚΕΝΟ ΤΟΥ GATE    — min(in-corpus) vs max(out_of_corpus).
     ΑΝ ΣΤΕΝΕΨΕΙ, ΣΤΑΜΑΤΑΜΕ. Η ταχύτητα ΔΕΝ αγοράζεται με την άμυνα κατά της
     ψευδαίσθησης. Λιγότερα candidates σημαίνει και λιγότερες ευκαιρίες για ένα
     out_of_corpus να βρει κάτι που «μοιάζει» σχετικό — μπορεί να πάει και προς
     τις δύο κατευθύνσεις, γι' αυτό μετριέται αντί να υποτίθεται.

ΠΡΟΣΟΧΗ: τα υποψήφια για N μικρότερο είναι ΥΠΟΣΥΝΟΛΟ αυτών του μεγαλύτερου
(ίδιο RRF, μικρότερο top_n) -> η σύγκριση είναι καθαρή, αλλάζει ΜΟΝΟ το πλήθος.

!!! ΤΟ «ΑΣΦΑΛΕΣ» ΕΔΩ ΔΕΝ ΑΡΚΕΙ — ΜΕΤΡΗΘΗΚΕ ΚΑΙ ΔΙΑΨΕΥΣΤΗΚΕ (10/8/2026) !!!
Αυτό το script σταματά στο rerank. Το pipeline συνεχίζει με
`_expand_to_pages(sorted_final[:EXPAND_INPUT])` -> ΟΙ ΣΕΛΙΔΕΣ είναι το prompt.
Το N=10 βγήκε «ΑΣΦΑΛΕΣ» εδώ (ταυτόσημο best-logit, κενό, kw@1, 56/56) και
παρ' όλα αυτά άλλαξε τις σελίδες σε 52/56 ερωτήσεις. Ο λόγος: ο reranker
ΑΝΑΔΙΑΤΑΣΣΕΙ — chunk στη θέση 14 του RRF μπορεί να ανέβει στη θέση 3 και να
μπει κανονικά στα 12 του expand. Το «candidates >= EXPAND_INPUT» ΔΕΝ εγγυάται
ίδιες σελίδες.
=> Μετά από ΚΑΘΕ «ΑΣΦΑΛΕΣ» εδώ, τρέξε ΥΠΟΧΡΕΩΤΙΚΑ:
     compare_pages_rerankers.py --candidates N
   Αν αλλάζουν σελίδες, χρειάζεται judge run — και τότε ξανακρίνε αν αξίζει.

ΚΟΣΤΟΣ: μηδέν API (μεταφράσεις από το μόνιμο cache).

    docker compose exec backend python evaluation/tune_rerank_candidates.py
    docker compose exec backend python evaluation/tune_rerank_candidates.py --values 8 10 12 15
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
import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

DEFAULT_SETS = [
    "/app/evaluation/golden_set_50.jsonl",
    "/app/evaluation/golden_multihop_new.jsonl",
]


def load_sets(paths):
    tests = []
    for p in paths:
        if not os.path.exists(p):
            print(f"ΠΑΡΑΛΕΙΨΗ (δεν υπάρχει): {p}")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tests.append(json.loads(line))
    return tests


def first_hit(ordered, kws):
    for r, (_s, text) in enumerate(ordered, 1):
        low = text.lower()
        if any(k.lower() in low for k in kws):
            return r
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", nargs="*", type=int, default=[6, 8, 10, 12, 15, 20])
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    tests = load_sets(args.sets)
    missing = [t["id"] for t in tests
               if ai_core._has_greek(t["question"])
               and t["question"] not in ai_core._translation_cache]
    if missing:
        print(f"ΠΡΟΣΟΧΗ: {len(missing)} ελληνικές εκτός cache -> κλήσεις: {missing}\n")
    else:
        print("Translation cache: πλήρες -> ΜΗΔΕΝ κλήσεις API.\n")

    where = ai_core._build_where(GOLDEN_CORPUS, None)
    allowed = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    top = max(args.values)
    print(f"{len(tests)} ερωτήσεις · μοντέλο {ai_core.RERANKER_MODEL}")
    print(f"gate {ai_core.MIN_RERANK_SCORE} · τρέχον RERANK_CANDIDATES="
          f"{ai_core.RERANK_CANDIDATES}\n")

    # --- ΕΝΑ RRF pass στο ΜΕΓΙΣΤΟ N· κάθε μικρότερο N είναι πρόθεμα αυτού ---
    print(f"Υποψήφια (ένα RRF pass στο top-{top})...", flush=True)
    shared = []
    for t in tests:
        q = await ai_core.optimize_query(t["question"])
        d = ai_core._dense_exact_ids(dm, q, allowed,
                                     min(ai_core.DENSE_CANDIDATES, len(allowed)))
        s = ai_core._bm25_sparse_ids(idx, q, allowed, ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(d, s, idx["ids"], idx["texts"], idx["metas"],
                                k=60, top_n=top, pos=idx["pos"])
        shared.append((t, q, rrf))
    print(f"  {len(shared)} σύνολα έτοιμα\n")

    print(f"{'N':>4} {'ms/ερώτ':>9} {'in-min':>8} {'ooc-max':>8} {'ΚΕΝΟ':>7} "
          f"{'kw@ διάμ':>9} {'kw@1':>6} {'βρέθηκε':>8} {'ΚΟΠΕΣ':>6}")
    print("-" * 76)

    rows, results = [], {}
    for n in sorted(args.values):
        t0 = time.time()
        recs = []
        for t, q, rrf in shared:
            cand = rrf[:n]
            pairs = [[q, it[1]] for it in cand]
            sc = [float(x) for x in ai_core.reranker.predict(
                pairs, batch_size=ai_core.RERANK_BATCH_SIZE)]
            ordered = sorted(zip(sc, [it[1] for it in cand]), key=lambda x: -x[0])
            ooc = t.get("category") == "out_of_corpus"
            recs.append(dict(
                n=n, id=t["id"], category=t.get("category", ""), best=ordered[0][0],
                kw_rank=(None if ooc else first_hit(ordered, t["keywords"]))))
        el = 1000 * (time.time() - t0) / len(shared)

        inc = [r for r in recs if r["category"] != "out_of_corpus"]
        ooc_r = [r for r in recs if r["category"] == "out_of_corpus"]
        lo = min(r["best"] for r in inc)
        hi = max(r["best"] for r in ooc_r)
        ranks = [r["kw_rank"] for r in inc if r["kw_rank"]]
        cuts = [r["id"] for r in inc if r["best"] < ai_core.MIN_RERANK_SCORE]
        leaks = [r["id"] for r in ooc_r if r["best"] >= ai_core.MIN_RERANK_SCORE]

        results[n] = dict(gap=lo - hi, ms=el, found=len(ranks), n_inc=len(inc),
                          cuts=cuts, leaks=leaks)
        rows.extend(recs)
        print(f"{n:>4} {el:>9.0f} {lo:>8.2f} {hi:>8.2f} {lo-hi:>7.2f} "
              f"{statistics.median(ranks):>9.1f} {sum(1 for x in ranks if x == 1):>6} "
              f"{f'{len(ranks)}/{len(inc)}':>8} {len(cuts):>6}", flush=True)

    base = ai_core.RERANK_CANDIDATES
    print("\n" + "=" * 76)
    print("ΕΤΥΜΗΓΟΡΙΑ")
    print("=" * 76)
    if base not in results:
        print(f"  (το τρέχον N={base} δεν ήταν στη σάρωση — δεν γίνεται σύγκριση)")
        return 0

    b = results[base]
    print(f"  Βάση: N={base} · {b['ms']:.0f} ms · κενό {b['gap']:.2f} · "
          f"βρέθηκε {b['found']}/{b['n_inc']}")
    for n in sorted(results):
        if n >= base:
            continue
        r = results[n]
        ok = (not r["leaks"] and not r["cuts"] and r["gap"] >= b["gap"] - 0.05
              and r["found"] >= b["found"])
        verdict = "ΑΣΦΑΛΕΣ" if ok else "ΟΧΙ"
        why = []
        if r["leaks"]:
            why.append(f"ΔΙΑΡΡΟΗ {r['leaks']}")
        if r["cuts"]:
            why.append(f"ΚΟΒΕΙ {r['cuts']}")
        if r["gap"] < b["gap"] - 0.05:
            why.append(f"κενό {b['gap']:.2f}->{r['gap']:.2f}")
        if r["found"] < b["found"]:
            why.append(f"χάνει υλικό {b['found']}->{r['found']}")
        print(f"  N={n:<3} {verdict:<8} {b['ms']-r['ms']:>6.0f} ms κέρδος "
              f"({100*(b['ms']-r['ms'])/b['ms']:>4.0f}%)"
              + ("   " + " · ".join(why) if why else ""))
    print("\n  ΚΡΙΤΗΡΙΟ: καμία διαρροή, καμία νέα κοπή, κενό όχι μικρότερο κατά >0.05,")
    print("  και το ίδιο πλήθος ερωτήσεων με βρεθέν keyword. Αλλιώς ΟΧΙ.")
    print("\n  !!! ΤΟ «ΑΣΦΑΛΕΣ» ΕΙΝΑΙ ΜΟΝΟ ΜΕΧΡΙ ΤΟ RERANK. Το prompt είναι ΣΕΛΙΔΕΣ.")
    print("  Μετρήθηκε: N=10 «ΑΣΦΑΛΕΣ» εδώ -> άλλαξε σελίδες σε 52/56 ερωτήσεις,")
    print("  γιατί ο reranker ΑΝΑΔΙΑΤΑΣΣΕΙ (chunk #14 του RRF -> θέση 3 μετά).")
    print("  ΥΠΟΧΡΕΩΤΙΚΟ επόμενο βήμα:  compare_pages_rerankers.py --candidates N")

    if args.csv and rows:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
