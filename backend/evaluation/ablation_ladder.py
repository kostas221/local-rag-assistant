"""Η ΘΕΤΙΚΗ ΣΚΑΛΑ: τι προσθέτει ΚΑΘΕ στάδιο του pipeline, στις ΙΔΙΕΣ ερωτήσεις.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Το project έχει ~15 τεκμηριωμένες ΑΠΟΡΡΙΨΕΙΣ («δοκίμασα X, χειρότερο, να γιατί»)
αλλά ΚΑΜΙΑ θετική σκάλα. Είναι το ίδιο δεδομένο ανάποδα — και η θετική εκδοχή
είναι αυτή που διαβάζεται σε 30 δευτερόλεπτα από εξεταστή ή recruiter:

    dense only          -> ...
    + BM25 / RRF        -> ...
    + cross-encoder     -> ...
    + relevance gate    -> ...   <-- εδώ φαίνεται η άμυνα κατά της ψευδαίσθησης

ΤΟ ΚΡΙΣΙΜΟ ΝΟΥΜΕΡΟ ΔΕΝ ΕΙΝΑΙ ΤΟ MRR. Είναι η στήλη `ooc blocked`: πόσα από τα 5
out_of_corpus κόβονται. Χωρίς gate είναι 0/5 — το σύστημα απαντά ΠΑΝΤΑ, ακόμα κι
όταν δεν υπάρχει υλικό. Αυτό είναι το μοναδικό στάδιο που αλλάζει ΕΙΔΟΣ
συμπεριφοράς, όχι βαθμό.

ΜΕΘΟΔΟΛΟΓΙΑ: κάθε στάδιο τερματίζει στο ΙΔΙΟ σημείο — page-expansion με τα ίδια
EXPAND_INPUT/MAX_PAGES — ώστε να συγκρίνουμε ΤΟ ΙΔΙΟ ΠΡΑΓΜΑ (τι φτάνει στο
Gemini) και όχι «πόσα chunks γύρισε το κάθε σκέλος». Το optimize_query τρέχει
ΜΙΑ φορά ανά ερώτηση και μοιράζεται σε όλα τα στάδια.

ΜΗΔΕΝ ΚΟΣΤΟΣ API (οι μεταφράσεις είναι ήδη cached· ο corrective agent ΔΕΝ
τρέχει — το στάδιο «+gate» μετράει το gate αυτό καθαυτό).

    docker compose exec backend python evaluation/ablation_ladder.py
    docker compose exec backend python evaluation/ablation_ladder.py \\
        --dataset evaluation/golden_hard_paraphrase.jsonl --csv evaluation/runs/ladder_hard.csv
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
from evaluation.eval_engine import TestQuestion, evaluate_retrieval

STAGES = ["dense", "+bm25/rrf", "+reranker", "+gate"]


def load_tests(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [TestQuestion(**json.loads(line)) for line in f if line.strip()]


def _items(ids, idx):
    """ids -> [(score, text, meta)] στη μορφή που περιμένει το _expand_to_pages.
    Το score είναι φθίνον placeholder: η ΣΕΙΡΑ είναι που μετράει εδώ."""
    pos, texts, metas = idx["pos"], idx["texts"], idx["metas"]
    return [(float(-i), texts[pos[cid]], metas[pos[cid]])
            for i, cid in enumerate(ids)]


async def run_stages(test, idx, dm, allowed_ids):
    """Επιστρέφει {στάδιο: retrieved} — ΟΛΑ τερματίζουν σε page-expansion."""
    query = await ai_core.optimize_query(test.question)
    out = {}

    # --- 1. DENSE ONLY: ένα σκέλος, καμία σύντηξη, κανένα φίλτρο ---------
    dense_ids = ai_core._dense_exact_ids(
        dm, query, allowed_ids, min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
    out["dense"] = ai_core._expand_to_pages(
        _items(dense_ids, idx)[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)

    # --- 2. + BM25 μέσω RRF: υβριδική ανάκτηση, ακόμα χωρίς reranker ------
    sparse_ids = ai_core._bm25_sparse_ids(
        idx, query, allowed_ids, ai_core.DENSE_CANDIDATES)
    rrf = ai_core._rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                            idx["metas"], k=60,
                            top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
    out["+bm25/rrf"] = ai_core._expand_to_pages(
        rrf[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)

    # --- 3. + CROSS-ENCODER: αναδιάταξη των ίδιων υποψηφίων ---------------
    scores = ai_core.reranker.predict([[query, it[1]] for it in rrf],
                                      batch_size=ai_core.RERANK_BATCH_SIZE)
    ranked = sorted(zip(scores, [it[1] for it in rrf], [it[2] for it in rrf]),
                    key=lambda x: x[0], reverse=True)
    out["+reranker"] = ai_core._expand_to_pages(
        ranked[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)

    # --- 4. + GATE: το ΜΟΝΟ στάδιο που μπορεί να επιστρέψει ΤΙΠΟΤΑ --------
    # Ο corrective agent ΔΕΝ τρέχει εδώ: μετράμε το gate αυτό καθαυτό, και ο
    # agent είναι μη-ντετερμινιστικός (rewrite από Gemini χωρίς seed).
    out["+gate"] = ([] if ranked[0][0] < ai_core.MIN_RERANK_SCORE
                    else out["+reranker"])
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evaluation/golden_set_50.jsonl")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    tests = load_tests(args.dataset)
    inc = [t for t in tests if t.category != "out_of_corpus"]
    ooc = [t for t in tests if t.category == "out_of_corpus"]
    print(f"Dataset: {args.dataset} — {len(inc)} in-corpus + {len(ooc)} out_of_corpus\n")

    allowed_ids = ai_core.collection.get(
        where=ai_core._build_where(None, None), include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    acc = {s: {"mrr": [], "cov": [], "ooc_blocked": 0, "silent": 0}
           for s in STAGES}
    rows = []

    for n, t in enumerate(tests, start=1):
        stages = await run_stages(t, idx, dm, allowed_ids)
        row = {"id": t.id, "category": t.category}
        for s in STAGES:
            retrieved = stages[s]
            if t.category == "out_of_corpus":
                # Για τα out_of_corpus η ΜΟΝΗ σωστή απάντηση είναι η σιωπή.
                if not retrieved:
                    acc[s]["ooc_blocked"] += 1
                row[f"{s}_blocked"] = int(not retrieved)
                continue
            if not retrieved:
                acc[s]["silent"] += 1
                acc[s]["mrr"].append(0.0)
                acc[s]["cov"].append(0.0)
                row[f"{s}_mrr"] = 0.0
                continue
            r = await evaluate_retrieval(t, retrieved=retrieved)
            acc[s]["mrr"].append(r.mrr)
            acc[s]["cov"].append(r.keyword_coverage)
            row[f"{s}_mrr"] = round(r.mrr, 3)
        rows.append(row)
        print(f"  [{n}/{len(tests)}] {t.id}", end="\r", flush=True)

    print(" " * 40)
    print("=" * 78)
    print(f"{'στάδιο':<14}{'MRR':>9}{'coverage':>11}{'σιωπηλές':>11}"
          f"{'ooc blocked':>14}")
    print("-" * 78)
    prev = None
    for s in STAGES:
        a = acc[s]
        mrr = statistics.mean(a["mrr"]) if a["mrr"] else 0.0
        cov = statistics.mean(a["cov"]) if a["cov"] else 0.0
        d = "" if prev is None else f"  ({mrr - prev:+.3f})"
        print(f"{s:<14}{mrr:>9.3f}{cov:>10.1f}%{a['silent']:>11}"
              f"{a['ooc_blocked']:>8}/{len(ooc):<5}{d}")
        prev = mrr
    print("=" * 78)
    print("ΠΡΟΣΟΧΗ ΣΤΗΝ ΑΝΑΓΝΩΣΗ:")
    print("  · Το `+gate` ΔΕΝ βελτιώνει MRR/coverage — μπορεί να τα ρίξει, γιατί")
    print("    μια κομμένη ερώτηση μετράει 0. Αυτό ΔΕΝ είναι υποβάθμιση: το gate")
    print("    ανταλλάσσει ανάκληση με ΑΞΙΟΠΙΣΤΙΑ, και η ανταλλαγή φαίνεται")
    print("    ΜΟΝΟ στη στήλη `ooc blocked`.")
    print("  · Χωρίς gate, το σύστημα απαντά ΠΑΝΤΑ — και σε ερωτήσεις που το")
    print("    corpus δεν μπορεί να απαντήσει. Αυτή είναι ψευδαίσθηση εξ ορισμού.")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "category"]
                               + [k for k in keys if k not in ("id", "category")])
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
