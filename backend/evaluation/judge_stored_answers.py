# backend/evaluation/judge_stored_answers.py
"""ΚΡΙΤΗΣ ΠΑΝΩ ΣΕ ΗΔΗ ΑΠΟΘΗΚΕΥΜΕΝΕΣ ΑΠΑΝΤΗΣΕΙΣ — ΜΗΔΕΝ ΝΕΑ ΓΕΝΝΗΣΗ.

ΤΟ ΕΡΩΤΗΜΑ: ο κανόνας παραπομπής έκοψε περιεχόμενο; Το ντετερμινιστικό proxy
(keywords του golden set μέσα στην απάντηση) έδειξε -0.030, δηλαδή ΜΙΑ λέξη σε
δύο ερωτήσεις = θόρυβος. Αλλά το q013 συρρικνώθηκε 1995 -> 808 χαρακτήρες
κρατώντας και τα 10 στοιχεία και χάνοντας τις ΕΠΕΞΗΓΗΣΕΙΣ τους. Τα keywords
δεν μπορούν να το δουν: μετράνε παρουσία όρου, όχι ανάπτυξη. Το completeness
του κριτή μπορεί.

ΓΙΑΤΙ ΕΝΑΣ JUDGE RUN ΕΙΝΑΙ ΕΔΩ ΣΩΣΤΟ ΕΡΓΑΛΕΙΟ: 48/50 ερωτήσεις είναι ήδη
5/5/5/5, άρα ένα τρέξιμο ΔΕΝ μπορεί να δείξει βελτίωση — μόνο υποβάθμιση.
Υποβάθμιση ψάχνουμε.

ΤΡΙΑ ΣΚΕΛΗ, ΟΧΙ ΔΥΟ:
  base          η απάντηση χωρίς κανόνα παραπομπής
  cite          η απάντηση ΟΠΩΣ ΤΗΝ ΒΛΕΠΕΙ Ο ΧΡΗΣΤΗΣ, με τις αγκύλες μέσα
  cite_stripped η ΙΔΙΑ απάντηση με τις αγκύλες αφαιρεμένες
Το τρίτο σκέλος διαχωρίζει δύο εντελώς διαφορετικές αιτίες: «το περιεχόμενο
χειροτέρεψε» (τότε πέφτει και το stripped) από «ο κριτής δεν συμπαθεί τα
[S3]» (τότε πέφτει μόνο το cite). Χωρίς αυτό, μια πτώση relevance δεν
ερμηνεύεται.

ΚΟΣΤΟΣ: 3 κλήσεις κριτή ανά ερώτηση, ΚΑΜΙΑ γέννηση απάντησης. Η ανάκτηση
ξανατρέχει (ντετερμινιστική, δωρεάν) μόνο για να ξαναχτιστεί το context που
χρειάζεται το faithfulness.

⚠️ ΕΞΑΙΡΕΣΗ h016: εκεί τρέχει ο corrective agent, που ΔΕΝ είναι επαναλήψιμος.
Το context του μπορεί να διαφέρει από εκείνο πάνω στο οποίο γράφτηκε η
απάντηση -> το faithfulness του h016 σημειώνεται με (*) και δεν διαβάζεται.

Τρέξε:
  docker compose exec backend python evaluation/judge_stored_answers.py \
      --csv evaluation/runs/inline_citations_bare.csv \
      --out evaluation/runs/judge_citations_bare.csv
"""
import argparse
import asyncio
import csv
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_inline_citations import HARD_SET, MAIN_SET, strip_citations

import ai_core
from evaluation import eval_engine

FIELDS = ["accuracy", "completeness", "relevance", "faithfulness"]

# Αντίγραφο του prompt του eval_engine.evaluate_answer — ΛΕΞΗ ΠΡΟΣ ΛΕΞΗ, ώστε
# τα νούμερα να συγκρίνονται με κάθε άλλο judge run του project. Ο έλεγχος
# από κάτω σκάει αν αποκλίνει.
JUDGE_PROMPT = """You are a strict expert evaluator for a RAG system. Only give 5/5 for perfect answers.

QUESTION: {question}

RETRIEVED CONTEXT (the ONLY source the system was allowed to use):
{context}

GENERATED ANSWER: {answer}

REFERENCE ANSWER (gold): {reference}

Evaluate (each 1-5):
1. accuracy: factual correctness vs the reference answer.
2. completeness: covers the key information of the reference.
3. relevance: directly answers the question, no filler.
4. faithfulness: EVERY claim in the generated answer is supported by the RETRIEVED CONTEXT. If the answer adds facts NOT in the context (hallucination), score this LOW even if they happen to be correct.

Return ONLY raw JSON (no markdown):
{{
    "feedback": "your detailed reasoning",
    "accuracy": 5,
    "completeness": 5,
    "relevance": 5,
    "faithfulness": 5
}}"""

_SRC = inspect.getsource(eval_engine.evaluate_answer)
for _probe in ("You are a strict expert evaluator for a RAG system.",
               "2. completeness: covers the key information of the reference.",
               "4. faithfulness: EVERY claim in the generated answer is supported",
               "temperature=0.0"):
    assert _probe in _SRC, f"ο κριτής του eval_engine ΑΛΛΑΞΕ — λείπει: {_probe[:45]}"


def load_golden() -> dict:
    out = {}
    for path in (MAIN_SET, HARD_SET):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r
    return out


async def judge(question: str, context: str, answer: str, reference: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, context=context,
                                 answer=answer, reference=reference)
    resp = await eval_engine.judge_model.generate_content_async(
        prompt,
        generation_config=eval_engine.genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.0),   # ίδιος ντετερμινιστικός κριτής με το run_eval
    )
    data = json.loads(resp.text)
    return {f: int(data[f]) for f in FIELDS} | {"feedback": data.get("feedback", "")}


def avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="evaluation/runs/inline_citations_bare.csv")
    ap.add_argument("--out", default="evaluation/runs/judge_citations_bare.csv")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], {})[r["arm"]] = r
    golden = load_golden()

    fmt = next((r["cite"].get("fmt", "full") for r in by_id.values()
                if "cite" in r), "full")
    ids = [q for q, a in by_id.items() if "base" in a and "cite" in a]
    print(f"Ερωτήσεις: {len(ids)} x 3 σκέλη = {3 * len(ids)} κλήσεις κριτή "
          f"(ΜΗΔΕΝ γέννηση) · μορφή: {fmt}\n")

    out, tally = [], {a: {f: [] for f in FIELDS} for a in
                      ("base", "cite", "cite_stripped")}
    print(f"{'id':6s} {'σκέλος':14s} {'acc':>4s} {'comp':>5s} {'rel':>4s} "
          f"{'faith':>6s}")
    print("-" * 46)

    for qid in ids:
        g = golden.get(qid)
        if not g:
            print(f"{qid:6s} ΠΑΡΑΛΕΙΨΗ — δεν βρέθηκε στο golden set")
            continue
        # Ίδια κλήση με το probe (target_filenames=None): το corpus έχει μόνο
        # τα 7 papers, αλλά η ταύτιση της κλήσης είναι αυτή που εγγυάται ίδιο
        # context με εκείνο πάνω στο οποίο γράφτηκαν οι απαντήσεις.
        retrieved = await ai_core.search_documents(g["question"], None, user_id=None)
        context = "\n---\n".join(text for text, _ in retrieved)
        mark = " (*)" if qid == "h016" else ""

        variants = {
            "base": by_id[qid]["base"]["answer"],
            "cite": by_id[qid]["cite"]["answer"],
            "cite_stripped": strip_citations(by_id[qid]["cite"]["answer"], fmt),
        }
        row = {"id": qid, "category": g.get("category", ""), "fmt": fmt}
        for arm, ans in variants.items():
            try:
                sc = await judge(g["question"], context, ans,
                                 g.get("reference_answer", ""))
            except Exception as e:
                print(f"{qid:6s} {arm:14s} ΣΦΑΛΜΑ: {e}")
                continue
            for f in FIELDS:
                tally[arm][f].append(sc[f])
                row[f"{arm}_{f}"] = sc[f]
            row[f"{arm}_feedback"] = sc["feedback"]
            print(f"{qid:6s} {arm + mark:14s} {sc['accuracy']:4d} "
                  f"{sc['completeness']:5d} {sc['relevance']:4d} "
                  f"{sc['faithfulness']:6d}")
            if args.delay:
                await asyncio.sleep(args.delay)
        out.append(row)
        print("-" * 46)

    print("\n--- ΣΥΓΚΕΝΤΡΩΤΙΚΑ ---")
    print(f"{'σκέλος':14s} {'acc':>5s} {'comp':>5s} {'rel':>5s} {'faith':>6s} "
          f"{'τέλειες':>8s}")
    for arm in ("base", "cite", "cite_stripped"):
        perfect = sum(1 for r in out
                      if all(r.get(f"{arm}_{f}") == 5 for f in FIELDS))
        print(f"{arm:14s} " + " ".join(f"{avg(tally[arm][f]):5.2f}"
                                       for f in FIELDS)
              + f" {perfect:5d}/{len(out)}")

    print("\n--- ΖΕΥΓΑΡΩΤΑ, ΑΝΑ ΕΡΩΤΗΣΗ (μόνο όπου ΑΛΛΑΞΕ κάτι) ---")
    for arm in ("cite", "cite_stripped"):
        worse, better = [], []
        for r in out:
            for f in FIELDS:
                b, c = r.get(f"base_{f}"), r.get(f"{arm}_{f}")
                if b is None or c is None or b == c:
                    continue
                (worse if c < b else better).append(f"{r['id']} {f} {b}->{c}")
        print(f"  base -> {arm}: ΧΕΙΡΟΤΕΡΑ {len(worse)} · ΚΑΛΥΤΕΡΑ {len(better)}")
        for x in worse:
            print(f"      ↓ {x}")
        for x in better:
            print(f"      ↑ {x}")

    if out:
        cols = ["id", "category", "fmt"] + [f"{a}_{f}" for a in
                                            ("base", "cite", "cite_stripped")
                                            for f in FIELDS] + \
               [f"{a}_feedback" for a in ("base", "cite", "cite_stripped")]
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(out)
        print(f"\nCSV -> {args.out}")
    print("\n(*) h016: ο corrective agent δεν είναι επαναλήψιμος — το context "
          "μπορεί να διαφέρει από εκείνο της γέννησης.")


if __name__ == "__main__":
    asyncio.run(main())
