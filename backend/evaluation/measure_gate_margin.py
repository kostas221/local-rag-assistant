"""Κατανομή του relevance gate: πόσο περιθώριο έχει το MIN_RERANK_SCORE;

ΓΙΑΤΙ ΥΠΑΡΧΕΙ (μέτρα πριν αλλάξεις):
Ο corrective retrieval agent ενεργοποιείται ΟΤΑΝ ΤΟ GATE ΚΟΒΕΙ. Πριν γραφτεί,
πρέπει να ξέρουμε αν το gate κόβει ποτέ ερώτηση που ΘΑ ΕΠΡΕΠΕ να απαντηθεί.
Αν δεν κόβει καμία, ο agent λύνει ανύπαρκτο πρόβλημα και ρισκάρει το 5/5 των
out_of_corpus — που είναι το ΜΟΝΟ σκληρό κριτήριο του.

ΤΙ ΜΕΤΡΑΕΙ (read-only, ΜΗΔΕΝ αλλαγή στον κώδικα παραγωγής):
  1. best rerank logit ανά ερώτηση — δηλαδή ακριβώς ό,τι συγκρίνει το gate.
  2. αν το τρέχον κατώφλι θα έκοβε την ερώτηση.
  3. θέση (1-based) του πρώτου υποψήφιου chunk που περιέχει keyword. Διακρίνει
     "το gate έκοψε άδικα" (υπήρχε σωστό chunk στα 15) από "το retrieval απέτυχε"
     (δεν υπήρχε καν) — η αναδιατύπωση λύνει ΜΟΝΟ το δεύτερο.
  4. το ΚΕΝΟ: min(in-corpus) vs max(out_of_corpus). Αυτό είναι το πραγματικό
     περιθώριο ασφαλείας του -2.0, και ο μόνος αριθμός που λέει πόσο μπορεί
     να κινηθεί το κατώφλι χωρίς να σπάσει το 5/5.

ΚΟΣΤΟΣ API: μηδέν, ΕΦΟΣΟΝ οι ελληνικές ερωτήσεις έχουν ήδη μεταφραστεί (μόνιμο
translation cache). Και τα δύο golden sets έχουν τρέξει -> cache hit. Το script
τυπώνει προειδοποίηση αν χρειαστεί έστω μία νέα κλήση.

ΠΡΟΣΟΧΗ: αναπαράγει το pipeline του search_documents (βήματα 2-5) καλώντας τις
ΙΔΙΕΣ συναρτήσεις του ai_core — δεν το αντιγράφει λογικά. Αν αλλάξει το pipeline,
αυτό το script πρέπει να ακολουθήσει.

    docker compose exec backend python evaluation/measure_gate_margin.py
    docker compose exec backend python evaluation/measure_gate_margin.py --csv evaluation/runs/gate_margin.csv
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import sys

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
                    t = json.loads(line)
                    t["_set"] = os.path.basename(p)
                    tests.append(t)
    return tests


def first_hit(ordered, kws):
    """Θέση (1-based) του πρώτου chunk με keyword στα reranked υποψήφια."""
    for r, (_s, text) in enumerate(ordered, 1):
        low = text.lower()
        if any(k.lower() in low for k in kws):
            return r
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="προαιρετική εξαγωγή ανά ερώτηση")
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS)
    args = ap.parse_args()

    tests = load_sets(args.sets)
    if not tests:
        print("Κανένα test.")
        return 1

    # Πόσες ελληνικές ερωτήσεις ΔΕΝ είναι στο translation cache -> κόστος API.
    missing = [t["id"] for t in tests
               if ai_core._has_greek(t["question"])
               and t["question"] not in ai_core._translation_cache]
    if missing:
        print(f"ΠΡΟΣΟΧΗ: {len(missing)} ελληνικές ερωτήσεις εκτός cache "
              f"-> τόσες κλήσεις Gemini: {', '.join(missing)}\n")
    else:
        print("Translation cache: πλήρες -> ΜΗΔΕΝ κλήσεις API.\n")

    where = ai_core._build_where(GOLDEN_CORPUS, None)
    allowed_ids = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()
    thr = ai_core.MIN_RERANK_SCORE
    print(f"corpus: {len(allowed_ids)} chunks · MIN_RERANK_SCORE = {thr}\n")

    print(f"{'id':<7} {'set':<24} {'category':<14} {'best':>8} {'gate':>8} "
          f"{'kw@':>5} {'2ο':>8} {'μέσος15':>8}")
    print("-" * 92)

    rows = []
    for t in tests:
        query = await ai_core.optimize_query(t["question"])
        d_ids = ai_core._dense_exact_ids(dm, query, allowed_ids,
                                         min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
        s_ids = ai_core._bm25_sparse_ids(idx, query, allowed_ids,
                                         ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"],
                                idx["metas"], k=60, top_n=ai_core.RERANK_CANDIDATES,
                                pos=idx["pos"])
        pairs = [[query, it[1]] for it in rrf]
        scores = [float(x) for x in
                  ai_core.reranker.predict(pairs, batch_size=ai_core.RERANK_BATCH_SIZE)]
        ordered = sorted(zip(scores, [it[1] for it in rrf]), key=lambda x: -x[0])

        best = ordered[0][0]
        second = ordered[1][0] if len(ordered) > 1 else float("nan")
        cat = t.get("category", "")
        ooc = cat == "out_of_corpus"
        cut = best < thr
        # Σωστό = out_of_corpus που κόπηκε, ή in-corpus που πέρασε.
        ok = cut if ooc else not cut
        verdict = ("ΚΟΒΕΙ" if cut else "περνάει") + ("" if ok else " <<")
        hit = None if ooc else first_hit(ordered, t["keywords"])

        rows.append(dict(id=t["id"], set=t["_set"], category=cat, best=best,
                         second=second, mean15=statistics.mean(scores),
                         cut=cut, ok=ok, kw_rank=hit))
        print(f"{t['id']:<7} {t['_set']:<24} {cat:<14} {best:>8.2f} {verdict:>8} "
              f"{(hit if hit else '-')!s:>5} {second:>8.2f} "
              f"{statistics.mean(scores):>8.2f}", flush=True)

    inc = [r for r in rows if r["category"] != "out_of_corpus"]
    ooc = [r for r in rows if r["category"] == "out_of_corpus"]

    print("\n" + "=" * 92)
    print("ΚΑΤΑΝΟΜΗ best-logit")
    print("=" * 92)
    for label, grp in (("in-corpus", inc), ("out_of_corpus", ooc)):
        if not grp:
            continue
        v = sorted(r["best"] for r in grp)
        print(f"  {label:<14} n={len(v):<3} min {v[0]:>7.2f} · p10 "
              f"{v[max(0, len(v)//10)]:>7.2f} · διάμεσος {statistics.median(v):>7.2f} "
              f"· max {v[-1]:>7.2f}")

    print("\n" + "=" * 92)
    print(f"ΤΟ GATE ΣΗΜΕΡΑ (κατώφλι {thr})")
    print("=" * 92)
    cut_inc = [r for r in inc if r["cut"]]
    pass_ooc = [r for r in ooc if not r["cut"]]
    print(f"  in-corpus που ΚΟΠΗΚΑΝ:      {len(cut_inc)}/{len(inc)}"
          + (f"  -> {', '.join(r['id'] for r in cut_inc)}" if cut_inc else "  (καμία)"))
    print(f"  out_of_corpus που ΠΕΡΑΣΑΝ:  {len(pass_ooc)}/{len(ooc)}"
          + (f"  -> {', '.join(r['id'] for r in pass_ooc)}" if pass_ooc else "  (καμία) — 5/5 OK"))

    if cut_inc:
        print("\n  Ήταν άδικη η κοπή; (θέση σωστού chunk στα 15 υποψήφια)")
        for r in cut_inc:
            if r["kw_rank"]:
                print(f"    {r['id']}: keyword chunk στη θέση {r['kw_rank']} "
                      f"-> ΤΟ ΥΛΙΚΟ ΥΠΗΡΧΕ, το gate το πέταξε")
            else:
                print(f"    {r['id']}: κανένα keyword chunk στα 15 "
                      f"-> απέτυχε το RETRIEVAL, όχι το gate")

    if inc and ooc:
        lo = min(r["best"] for r in inc)
        hi = max(r["best"] for r in ooc)
        print("\n" + "=" * 92)
        print("ΠΕΡΙΘΩΡΙΟ ΑΣΦΑΛΕΙΑΣ")
        print("=" * 92)
        print(f"  χειρότερο in-corpus:      {lo:>7.2f}")
        print(f"  καλύτερο out_of_corpus:   {hi:>7.2f}")
        if lo > hi:
            print(f"  ΔΙΑΧΩΡΙΣΙΜΑ: κενό {lo - hi:.2f} logits. Κάθε κατώφλι στο "
                  f"({hi:.2f}, {lo:.2f}) δίνει 100% και στα δύο.")
        else:
            print(f"  ΕΠΙΚΑΛΥΨΗ {hi - lo:.2f} logits: ΚΑΝΕΝΑ κατώφλι δεν τα "
                  f"χωρίζει τέλεια. Το gate είναι εξ ορισμού συμβιβασμός.")
        print(f"  απόσταση κατωφλίου από out_of_corpus: {thr - hi:>7.2f} "
              f"({'ασφαλές' if thr > hi else 'ΔΙΑΡΡΟΗ'})")
        print(f"  απόσταση κατωφλίου από in-corpus:     {lo - thr:>7.2f} "
              f"({'ασφαλές' if lo > thr else 'ΚΟΒΕΙ in-corpus'})")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
