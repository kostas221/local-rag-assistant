"""Τρέχει το eval πάνω σε ένα golden set (.jsonl).

ΔΥΟ ΤΡΟΠΟΙ ΛΕΙΤΟΥΡΓΙΑΣ:

  python run_eval.py evaluation/golden_set_50.jsonl --retrieval-only
      Μόνο μετρικές ανάκτησης (MRR / nDCG / keyword coverage). Καμία κλήση
      judge, καμία γέννηση απάντησης -> πρακτικά μηδενικό κόστος API (μόνο η
      cached μετάφραση των ελληνικών ερωτήσεων). Αυτός είναι ο τρόπος για
      καθημερινές δοκιμές ΚΑΙ για το CI regression gate (βλ. --min-mrr).

  python run_eval.py evaluation/golden_set_50.jsonl
      Πλήρες: ανάκτηση + γέννηση απάντησης + LLM-judge. ΚΑΤΑΝΑΛΩΝΕΙ quota —
      κράτησέ το για ορόσημα (baseline, πριν/μετά πειράματος).

Η ανάκτηση γίνεται ΜΙΑ φορά ανά ερώτηση και μοιράζεται σε όλα τα στάδια
(eval_engine.retrieve). Παλιότερα γινόταν τρεις φορές — ~13s CPU η καθεμία.
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import sys

from evaluation.eval_engine import TestQuestion, evaluate_answer, evaluate_retrieval, retrieve

RETRIEVAL_FIELDS = ["mrr", "ndcg", "keyword_coverage"]
ANSWER_FIELDS = ["accuracy", "completeness", "relevance", "faithfulness"]


def load_tests(path: str) -> list:
    tests = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(TestQuestion(**json.loads(line)))
    return tests


async def run(args) -> int:
    tests = load_tests(args.dataset)
    if not tests:
        print(f"Δεν βρέθηκαν ερωτήσεις στο {args.dataset}")
        return 1

    mode = "RETRIEVAL-ONLY (χωρίς judge)" if args.retrieval_only else "ΠΛΗΡΕΣ (με judge)"
    print(f"Φόρτωσα {len(tests)} ερωτήσεις από {args.dataset}")
    print(f"Τρόπος: {mode}\n")

    rows = []
    for i, test in enumerate(tests, 1):
        label = test.id or f"#{i}"
        print(f"[{i}/{len(tests)}] {label} {test.question[:55]}...", flush=True)
        try:
            # ΜΙΑ ανάκτηση — τροφοδοτεί και τις μετρικές και τη γέννηση.
            retrieved = await retrieve(test)
            ret = await evaluate_retrieval(test, retrieved)
            row = {
                "id": test.id,
                "category": test.category,
                "question": test.question,
                "mrr": round(ret.mrr, 3),
                "ndcg": round(ret.ndcg, 3),
                "keyword_coverage": round(ret.keyword_coverage, 1),
            }
            if not args.retrieval_only:
                ans, generated = await evaluate_answer(test, retrieved)
                row.update({
                    "accuracy": ans.accuracy,
                    "completeness": ans.completeness,
                    "relevance": ans.relevance,
                    "faithfulness": ans.faithfulness,
                    "generated_answer": generated,
                    "feedback": ans.feedback,
                })
            rows.append(row)
        except Exception as e:
            print(f"    ⚠️ Παράλειψη (σφάλμα: {e})", flush=True)

        # Η παύση αφορά ΜΟΝΟ το rate limit του judge/γέννησης.
        if args.delay and i < len(tests) and not args.retrieval_only:
            await asyncio.sleep(args.delay)

    if not rows:
        print("\nΚαμία ερώτηση δεν ολοκληρώθηκε (πιθανόν εξαντλημένο quota).")
        return 1

    # --- Συγκεντρωτικά ---
    def avg(key):
        vals = [r[key] for r in rows if key in r]
        return statistics.mean(vals) if vals else 0.0

    # Το MRR ορίζεται μόνο στις in-corpus· οι out_of_corpus σκοράρουν 0 by design.
    in_corpus = [r for r in rows if r["category"] != "out_of_corpus"]

    def avg_in(key):
        vals = [r[key] for r in in_corpus if key in r]
        return statistics.mean(vals) if vals else 0.0

    print("\n" + "=" * 60)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ")
    print("=" * 60)
    print(f"RETRIEVAL (όλες, {len(rows)}) | MRR: {avg('mrr'):.3f} "
          f"| nDCG: {avg('ndcg'):.3f} | Coverage: {avg('keyword_coverage'):.1f}%")
    if in_corpus and len(in_corpus) != len(rows):
        print(f"RETRIEVAL (in-corpus, {len(in_corpus)}) | MRR: {avg_in('mrr'):.3f} "
              f"| nDCG: {avg_in('ndcg'):.3f} | Coverage: {avg_in('keyword_coverage'):.1f}%")
    if not args.retrieval_only:
        print(f"ANSWER    | Accuracy: {avg('accuracy'):.2f}/5 "
              f"| Completeness: {avg('completeness'):.2f}/5")
        print(f"          | Relevance: {avg('relevance'):.2f}/5 "
              f"| Faithfulness: {avg('faithfulness'):.2f}/5")

    # Ανά κατηγορία — εδώ φαίνεται αν βελτιώθηκε το enumeration.
    cats = sorted({r["category"] for r in rows if r["category"]})
    if cats:
        print("-" * 60)
        print("MRR ανά κατηγορία:")
        for c in cats:
            sub = [r["mrr"] for r in rows if r["category"] == c]
            print(f"  {c:<16} {statistics.mean(sub):.3f}  (n={len(sub)})")
    print("=" * 60)

    # --- CSV (utf-8-sig για σωστά ελληνικά στο Excel) ---
    out_path = args.out or ("evaluation/results_retrieval.csv" if args.retrieval_only
                            else "evaluation/results.csv")
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ Αναλυτικά ανά ερώτηση: {out_path}")

    # --- CI regression gate ---
    if args.min_mrr is not None:
        score = avg_in("mrr") if in_corpus else avg("mrr")
        if score < args.min_mrr:
            print(f"\n❌ GATE: in-corpus MRR {score:.3f} < κατώφλι {args.min_mrr:.3f}")
            return 1
        print(f"\n✅ GATE: in-corpus MRR {score:.3f} >= κατώφλι {args.min_mrr:.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval runner για golden sets.")
    ap.add_argument("dataset", nargs="?", default="evaluation/golden_set_50.jsonl",
                    help="διαδρομή στο .jsonl golden set")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="μόνο μετρικές ανάκτησης — χωρίς judge, χωρίς κόστος")
    ap.add_argument("--out", default=None, help="διαδρομή CSV εξόδου")
    ap.add_argument("--min-mrr", type=float, default=None,
                    help="CI gate: exit 1 αν το in-corpus MRR πέσει κάτω από αυτό")
    ap.add_argument("--delay", type=int, default=int(os.getenv("EVAL_DELAY", "30")),
                    help="παύση (δευτ.) ανάμεσα στις ερωτήσεις· αφορά μόνο το πλήρες run")
    args = ap.parse_args()

    if not os.path.exists(args.dataset):
        print(f"ΣΦΑΛΜΑ: δεν βρέθηκε το dataset: {args.dataset}")
        return 1
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
