"""RAGAS-style retrieval metrics (context precision & recall) ΧΩΡΙΣ το RAGAS — οι
εξαρτήσεις του είναι σπασμένες στο langchain 1.x. Υπολογίζονται με τον ίδιο Gemini
judge που ήδη δουλεύει στο backend, πάνω στο ragas_dataset.jsonl.

- Context Precision: από τα ανακτημένα chunks, πόσα είναι ΣΧΕΤΙΚΑ — rank-weighted
  (ακριβώς ο ορισμός LLMContextPrecisionWithReference του RAGAS).
- Context Recall: τι ποσοστό των ισχυρισμών της σωστής απάντησης ΚΑΛΥΠΤΕΤΑΙ από τα chunks.

Τρέξε:  docker compose exec backend python context_eval.py
"""
import asyncio
import csv
import json
import os
import statistics

from genai_compat import genai  # deprecated SDK, τεκμηριωμένο: βλ. genai_compat.py

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
judge = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

DATA = "evaluation/ragas_dataset.jsonl"
DELAY = 4  # sec μεταξύ ερωτήσεων


async def _json(prompt):
    resp = await judge.generate_content_async(
        prompt,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", temperature=0.0))
    return json.loads(resp.text)


async def context_precision(question, contexts, reference):
    """Για κάθε chunk: σχετικό; -> rank-weighted average precision."""
    ctx_block = "\n".join(f"[{i+1}] {c[:800]}" for i, c in enumerate(contexts))
    prompt = (
        "You judge retrieval quality. For EACH numbered context decide if it is "
        "RELEVANT (contains information useful to answer the question / support the "
        "reference answer).\n\n"
        f"QUESTION: {question}\n\nREFERENCE ANSWER: {reference}\n\n"
        f"CONTEXTS:\n{ctx_block}\n\n"
        'Return ONLY JSON: {"relevant": [list of 0 or 1, one per context, in order]}')
    rel = (await _json(prompt)).get("relevant", [])
    rel = [int(bool(x)) for x in rel][:len(contexts)]
    if not any(rel):
        return 0.0
    hits, score = 0, 0.0
    for k, r in enumerate(rel, 1):
        if r:
            hits += 1
            score += hits / k
    return score / hits


async def context_recall(contexts, reference):
    """Σπάει τη σωστή απάντηση σε ισχυρισμούς -> πόσοι καλύπτονται από τα chunks."""
    ctx_block = "\n---\n".join(c[:800] for c in contexts)
    prompt = (
        "Break the REFERENCE ANSWER into individual factual claims. For each claim "
        "decide if it is supported by the CONTEXTS below.\n\n"
        f"REFERENCE ANSWER: {reference}\n\nCONTEXTS:\n{ctx_block}\n\n"
        'Return ONLY JSON: {"total_claims": int, "supported_claims": int}')
    d = await _json(prompt)
    tot = max(1, int(d.get("total_claims", 1)))
    sup = int(d.get("supported_claims", 0))
    return min(1.0, sup / tot)


async def main():
    with open(DATA, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    samples = [r for r in rows if r.get("contexts")]
    print(f"{len(samples)} / {len(rows)} samples με μη-κενό context\n")

    out = []
    for i, r in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] {r['question'][:50]}...", flush=True)
        try:
            cp = await context_precision(r["question"], r["contexts"], r["ground_truth"])
            cr = await context_recall(r["contexts"], r["ground_truth"])
            out.append({"question": r["question"],
                        "context_precision": round(cp, 3),
                        "context_recall": round(cr, 3)})
            print(f"    precision={cp:.3f}  recall={cr:.3f}", flush=True)
        except Exception as e:
            print(f"    ⚠️ skip ({e})", flush=True)
        if i < len(samples):
            await asyncio.sleep(DELAY)

    if not out:
        print("Καμία μέτρηση ολοκληρώθηκε.")
        return

    print("\n" + "=" * 46)
    print("RAGAS-style retrieval metrics (Gemini judge)")
    print("=" * 46)
    print(f"Context Precision : {statistics.mean(r['context_precision'] for r in out):.3f}")
    print(f"Context Recall    : {statistics.mean(r['context_recall'] for r in out):.3f}")
    print("=" * 46)

    with open("evaluation/context_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("✅ evaluation/context_results.csv")


asyncio.run(main())
