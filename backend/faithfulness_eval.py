"""Answer-quality eval (LLM-judge) στο config bge-m3 / bge-reranker / chunk=1500,
πάνω στο tests_cloud.jsonl. Μετράει retrieval MRR + answer accuracy/completeness/
relevance/faithfulness (1-5). Τρέξε:  docker compose exec backend python faithfulness_eval.py
"""
import asyncio
import csv
import json
import statistics

from evaluation.eval_engine import evaluate_retrieval, evaluate_answer, TestQuestion
from ai_core import search_documents

TESTS = "evaluation/golden_set_20.jsonl"
DELAY = 20  # sec παύση μεταξύ ερωτήσεων (για το per-minute rate limit του Gemini)


async def main():
    tests = []
    with open(TESTS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tests.append(TestQuestion(**json.loads(line)))

    print(f"{len(tests)} ερωτήσεις | config: bge-m3 / bge-reranker-v2-m3 / chunk=1500")
    print(f"(παύση {DELAY}s/ερώτηση για το rate limit — θα πάρει λίγα λεπτά)\n")

    rows = []
    ragas_rows = []
    for i, t in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {t.question[:55]}...", flush=True)
        try:
            ret = await evaluate_retrieval(t)
            ans, generated = await evaluate_answer(t)
            if ans.feedback == "Σφάλμα API":
                # Ο judge απέτυχε (π.χ. quota) -> ΕΞΑΙΡΕΙΤΑΙ από τους μέσους
                # όρους αντί να μετρήσει 0 και να τους παραμορφώσει.
                print("    ⚠️ judge failed -> εξαιρείται από τους μέσους όρους", flush=True)
                continue
                        # Κρατάμε ΚΑΙ τα contexts (τι "είδε" το μοντέλο) για το RAGAS dataset.
            # Η μετάφραση είναι ήδη cached από το evaluate_retrieval -> 0 επιπλέον API calls.
            retrieved = await search_documents(t.question)
            ragas_rows.append({
                "question": t.question,
                "answer": generated,
                "contexts": [text for text, meta in retrieved],
                "ground_truth": t.reference_answer,
            })
            rows.append({
                "question": t.question,
                "mrr": round(ret.mrr, 3),
                "accuracy": ans.accuracy,
                "completeness": ans.completeness,
                "relevance": ans.relevance,
                "faithfulness": ans.faithfulness,
                "generated_answer": generated,
                "feedback": ans.feedback,
            })
            print(f"    MRR={ret.mrr:.2f} | acc={ans.accuracy} compl={ans.completeness} "
                  f"rel={ans.relevance} faith={ans.faithfulness}", flush=True)
        except Exception as e:
            print(f"    ⚠️ skip ({e})", flush=True)
        if i < len(tests):
            await asyncio.sleep(DELAY)

    if not rows:
        print("\nΚαμία ερώτηση δεν ολοκληρώθηκε (πιθανόν εξαντλημένο Gemini quota).")
        return

    def avg(k):
        return statistics.mean(r[k] for r in rows)

    print("\n" + "=" * 56)
    print("ΜΕΣΟΙ ΟΡΟΙ ΠΟΙΟΤΗΤΑΣ ΑΠΑΝΤΗΣΕΩΝ")
    print("=" * 56)
    print(f"Retrieval MRR : {avg('mrr'):.3f}")
    print(f"Accuracy      : {avg('accuracy'):.2f} / 5")
    print(f"Completeness  : {avg('completeness'):.2f} / 5")
    print(f"Relevance     : {avg('relevance'):.2f} / 5")
    print(f"Faithfulness  : {avg('faithfulness'):.2f} / 5   (πιστότητα στις πηγές)")
    print("=" * 56)

        # RAGAS-ready dump: θα αξιολογηθεί offline από ξεχωριστό περιβάλλον (το RAGAS
    # θέλει pydantic v2, ασύμβατο με το pinned stack του backend).
    with open("evaluation/ragas_dataset.jsonl", "w", encoding="utf-8") as f:
        for r in ragas_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = "evaluation/results_golden20.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"✅ Αναλυτικά ανά ερώτηση (+ αιτιολόγηση judge): {out}")


if __name__ == "__main__":
    asyncio.run(main())
