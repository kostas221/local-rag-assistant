"""Φτιάχνει ragas_dataset.jsonl από ΥΠΑΡΧΟΝ judge run + φρέσκο retrieval.

ΓΙΑΤΙ ΟΧΙ ΤΟ faithfulness_eval.py: εκείνο ξανακαλεί το Gemini για να παράγει
απαντήσεις (50 κλήσεις, quota, ~20 λεπτά) — αλλά τις έχουμε ΗΔΗ στο CSV του
judge run. Εδώ ξανακάνουμε μόνο το retrieval (πλέον ~1s/ερώτηση μετά την αλλαγή
reranker) για να μαζέψουμε τα contexts, και δανειζόμαστε τις απαντήσεις.
Μηδέν κλήσεις generation, μηδέν κλήσεις judge.

ΓΙΑΤΙ ΜΑΣ ΝΟΙΑΖΕΙ ΤΟ RAGAS: ο δικός μας judge είναι Gemini 2.5 Flash που κρίνει
απαντήσεις που έγραψε το ΙΔΙΟ μοντέλο -> τεκμηριωμένο self-preference bias. Το
RAGAS χρησιμοποιεί άλλη μεθοδολογία (claim decomposition για faithfulness,
embeddings για answer_relevancy) και λειτουργεί ως ανεξάρτητη επικύρωση.

    docker compose exec backend python evaluation/build_ragas_dataset.py \
        --judge evaluation/judge_minilm.csv
"""
import argparse
import asyncio
import csv
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="evaluation/judge_minilm.csv",
                    help="CSV από πλήρες run_eval (με generated_answer)")
    ap.add_argument("--dataset", default="evaluation/golden_set_50.jsonl")
    ap.add_argument("--out", default="evaluation/ragas_dataset.jsonl")
    args = ap.parse_args()

    if not os.path.exists(args.judge):
        print(f"ΣΦΑΛΜΑ: δεν βρέθηκε {args.judge}")
        return 1

    with open(args.judge, encoding="utf-8-sig") as f:
        judged = {r["id"]: r for r in csv.DictReader(f)}
    if "generated_answer" not in next(iter(judged.values())):
        print("ΣΦΑΛΜΑ: το CSV δεν έχει generated_answer — τρέξε run_eval "
              "ΧΩΡΙΣ --retrieval-only.")
        return 1

    golden = {}
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                golden[t["id"]] = t

    rows, skipped_gate, missing = [], [], []
    for qid, g in golden.items():
        # Οι out-of-corpus ελέγχουν το GATE, όχι την ποιότητα RAG -> εξαιρούνται
        # (ίδια σύμβαση με τα in-corpus νούμερα του RESULTS.md).
        if g.get("category") == "out_of_corpus":
            continue
        j = judged.get(qid)
        if not j or not (j.get("generated_answer") or "").strip():
            missing.append(qid)
            continue

        retrieved = await ai_core.search_documents(
            g["question"], target_filenames=GOLDEN_CORPUS)
        if not retrieved:
            skipped_gate.append(qid)      # το gate το έκοψε -> κενά contexts
            continue

        rows.append({
            "question": g["question"],
            "answer": j["generated_answer"],
            "contexts": [text for text, _meta in retrieved],
            "ground_truth": g["reference_answer"],
        })
        print(f"  {qid}: {len(retrieved)} contexts", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nΓράφτηκαν {len(rows)} εγγραφές -> {args.out}")
    if skipped_gate:
        print(f"Εξαιρέθηκαν (τα έκοψε το gate): {skipped_gate}")
    if missing:
        print(f"*** Λείπει απάντηση από το judge CSV: {missing} ***")
    return 0


sys.exit(asyncio.run(main()))
