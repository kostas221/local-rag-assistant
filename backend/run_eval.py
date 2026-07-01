import asyncio
import csv
import json
import os
import statistics
import sys

from evaluation.eval_engine import evaluate_retrieval, evaluate_answer, TestQuestion


async def run_all_tests():
    # 1. Διάβασε ΟΛΕΣ τις ερωτήσεις από το tests.jsonl
    tests = []
    data_file = sys.argv[1] if len(sys.argv) > 1 else "evaluation/tests.jsonl"
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tests.append(TestQuestion(**json.loads(line)))

    if not tests:
        print("Δεν βρέθηκαν ερωτήσεις στο tests.jsonl")
        return

    print(f"Φόρτωσα {len(tests)} ερωτήσεις. Ξεκινάω αξιολόγηση...\n")

    # Παύση ανάμεσα στις ερωτήσεις για να μη χτυπάμε το rate limit του Gemini free tier
    DELAY_BETWEEN_QUESTIONS = int(os.getenv("EVAL_DELAY", "30"))

    rows = []
    for i, test in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {test.question[:60]}...", flush=True)
        try:
            ret = await evaluate_retrieval(test)
            ans, generated = await evaluate_answer(test)
            rows.append({
                "question": test.question,
                "mrr": round(ret.mrr, 3),
                "ndcg": round(ret.ndcg, 3),
                "keyword_coverage": round(ret.keyword_coverage, 1),
                "accuracy": ans.accuracy,
                "completeness": ans.completeness,
                "relevance": ans.relevance,
                "faithfulness": ans.faithfulness,
                "generated_answer": generated,
                "feedback": ans.feedback,
            })
        except Exception as e:
            print(f"    ⚠️ Παράλειψη (σφάλμα: {e})", flush=True)

        if i < len(tests):
            await asyncio.sleep(DELAY_BETWEEN_QUESTIONS)

    if not rows:
        print("\nΚαμία ερώτηση δεν ολοκληρώθηκε (πιθανόν εξαντλημένο Gemini quota). "
              "Δοκίμασε αργότερα.")
        return

    # 2. Συγκεντρωτικά (μέσοι όροι)
    def avg(key):
        return statistics.mean([r[key] for r in rows])

    print("\n" + "=" * 55)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ (μέσοι όροι)")
    print("=" * 55)
    print(f"RETRIEVAL | MRR: {avg('mrr'):.3f} | nDCG: {avg('ndcg'):.3f} "
          f"| Coverage: {avg('keyword_coverage'):.1f}%")
    print(f"ANSWER    | Accuracy: {avg('accuracy'):.2f}/5 "
          f"| Completeness: {avg('completeness'):.2f}/5")
    print(f"          | Relevance: {avg('relevance'):.2f}/5 "
          f"| Faithfulness: {avg('faithfulness'):.2f}/5")
    print("=" * 55)

    # 3. Αποθήκευση αναλυτικών αποτελεσμάτων σε CSV (utf-8-sig για σωστά ελληνικά στο Excel)
    out_path = "evaluation/results.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ Αναλυτικά αποτελέσματα ανά ερώτηση: {out_path}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
