"""11ο (19/11/2026): ελέγχουμε τα chunks της αναδιατύπωσης απέναντι στην ΑΡΧΙΚΗ ερώτηση. 0 κλήσεις API.

Διαβάζει τις αναδιατυπώσεις που ήδη πληρώθηκαν (runs/corrective_ab.csv, παλιό prompt, 3 προσπάθειες),
ξανατρέχει την ανάκτηση με καθεμία (ντετερμινιστική, CPU) και βαθμολογεί τα 5 πρώτα chunks με τον ίδιο
reranker ως προς την ΑΡΧΙΚΗ (μεταφρασμένη) ερώτηση. Κανόνας απόφασης: scoreboard.py, «11ο».

    docker compose ... run --rm --no-deps -e DATABASE_URL=sqlite:// -v eval_tmp:/tmp backend \\
        python -u evaluation/probe_corrective_recheck.py --run evaluation/runs/scoreboard/g38b.json
"""
import argparse
import asyncio
import csv
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import eval_near_ooc as E
from probe_corrective_ab import LOCK, load_tests
from probe_corrective_attempts import _retrieve

TOP = 5
GATE = -3.8          # CORRECTIVE_MIN_SCORE, ίδιο με την παραγωγή
THRESHOLDS = [-6.0, -5.0, -4.0, -3.5, -3.0, -2.6, -2.0, -1.5, -1.0, -0.5, 0.0]


async def run(args) -> int:
    ai_core, _near = E.open_store(False)
    if ai_core.collection.count() != 418:
        print("!! Το store δεν έχει 418 chunks — σταματάω")
        return 1
    queries = {(t["_set"], t["id"]): t["_q"] for t in load_tests(args.run)}
    with open(args.csv_in, encoding="utf-8") as f:
        src = [r for r in csv.DictReader(f) if r["prompt"] == "old" and r["best"] not in ("", None)]
    allowed = ai_core.collection.get(where=ai_core._build_where(None, None), include=[])["ids"]
    idx, dm = ai_core._get_bm25_index(), ai_core._get_dense_matrix()

    rows = []
    for r in src:
        q = queries[(r["set"], r["id"])]
        ranked = await _retrieve(r["rewrite"], allowed, idx, dm)
        if abs(float(ranked[0][0]) - float(r["best"])) > 0.05:      # ίδια ανάκτηση με το probe_ab;
            print(f"!! {r['id']} #{r['attempt']}: best {ranked[0][0]:+.2f} ≠ {float(r['best']):+.2f} (csv)")
        texts = [it[1] for it in ranked[:TOP]]
        scores = await asyncio.to_thread(
            lambda q=q, texts=texts: ai_core.reranker.predict([[q, t] for t in texts],
                                                             batch_size=ai_core.RERANK_BATCH_SIZE))
        row = {"set": r["set"], "id": r["id"], "kind": r["kind"], "attempt": int(r["attempt"]),
               "best": float(r["best"]), "cov": float(r["cov"]),
               "orig_top1": round(float(scores[0]), 3), "orig_max": round(float(max(scores)), 3)}
        rows.append(row)
        print(f"{r['set']:<9}{r['id']:<6}[{r['kind']:<10}] #{row['attempt']} best {row['best']:+6.2f} · "
              f"αρχική: top1 {row['orig_top1']:+6.2f} max{TOP} {row['orig_max']:+6.2f}")

    print("\n" + "=" * 80)
    print(f"ΠΕΡΝΑ = best ≥ {GATE} ΚΑΙ orig_max ≥ T  (μέσος ανά προσπάθεια· σήμερα χωρίς T: βλ. 1η γραμμή)")
    print(f"{'T':>6} {'διαρροές ooc (ανά προσπάθεια)':>32} {'κοντινές':>9} {'σωσμένες':>9}")
    for t in [None, *THRESHOLDS]:
        def ok(x, t=t):
            return x["best"] >= GATE and (t is None or x["orig_max"] >= t)
        leaks = [sum(ok(x) for x in rows if x["kind"] == "ooc" and x["attempt"] == a) for a in (1, 2, 3)]
        near = sum(ok(x) for x in rows if x["kind"] == "near") / 3
        saved = sum(ok(x) for x in rows if x["kind"] == "answerable" and x["cov"] > 0) / 3
        print(f"{'—' if t is None else f'{t:+.1f}':>6} {str(leaks):>32} {near:>9.2f} {saved:>9.2f}")
    print("=" * 80)
    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ γράφτηκε {args.csv_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Έλεγχος των chunks του corrective με την αρχική ερώτηση")
    ap.add_argument("--run", required=True, help="runs/scoreboard/g38b.json — οι μεταφράσεις")
    ap.add_argument("--csv-in", default="/app/evaluation/runs/corrective_ab.csv")
    ap.add_argument("--csv-out", default="/app/evaluation/runs/corrective_recheck.csv")
    args = ap.parse_args()
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"!! Τρέχει ήδη scoreboard/probe στο ίδιο store (κλειδαριά {LOCK}) — σταματάω")
        return 1
    try:
        return asyncio.run(run(args))
    finally:
        os.close(fd)
        os.remove(LOCK)


if __name__ == "__main__":
    sys.exit(main())
