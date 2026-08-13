"""Απαντάει σωστά το σύστημα σε ερωτήσεις ΠΙΝΑΚΩΝ;

ΤΟ ΕΡΩΤΗΜΑ
----------
Η `q027` έδειξε ότι η εξαγωγή ισοπεδώνει τους πίνακες κατά στήλη και το μοντέλο
μέτρησε 5 γραμμές αντί για 6. Όμως το σώμα έχει **20 πίνακες** ενώ όλα τα golden
sets μαζί είχαν **μία** ερώτηση πίνακα σε 77. Δηλαδή δεν ξέραμε αν η q027 είναι
εξαίρεση ή κανόνας — και το «δεν ξέρουμε» δεν είναι εύρημα.

ΓΙΑΤΙ ΔΕΝ ΧΡΕΙΑΖΕΤΑΙ ΚΡΙΤΗΣ
---------------------------
Στους πίνακες η σωστή απάντηση είναι **ακριβής**: συγκεκριμένοι αριθμοί και
ονόματα. Άρα ο έλεγχος γίνεται με αντιπαραβολή συμβολοσειρών και είναι
ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟΣ, χωρίς τον θόρυβο του κριτή (μετρημένος στο 2/9). Κάθε
αναμενόμενη τιμή δηλώνεται με τις αποδεκτές παραλλαγές της.

ΓΙΑΤΙ ΜΕΤΡΙΕΤΑΙ ΚΑΙ Η ΚΑΛΥΨΗ
----------------------------
Χωρίς αυτήν δεν ξεχωρίζεις τις δύο αιτίες αποτυχίας:
  · κάλυψη 0   -> η σελίδα ΔΕΝ ήρθε. Σφάλμα ΑΝΑΚΤΗΣΗΣ.
  · κάλυψη > 0 και τιμές λείπουν -> η σελίδα ήρθε και το μοντέλο δεν τη
    διάβασε σωστά. Σφάλμα ΣΥΝΘΕΣΗΣ, δηλαδή το πρόβλημα του ισοπεδωμένου πίνακα.
Μόνο το δεύτερο απαντάει στο ερώτημα αυτού του probe.

ΚΟΣΤΟΣ: μία γέννηση ανά ερώτηση, 12 συνολικά, γύρω στα 0.07 $.

    docker compose exec backend python evaluation/eval_tables.py \\
        --csv evaluation/runs/tables.csv
"""
import argparse
import asyncio
import csv
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

from ai_core import ask_ai, search_documents
from evaluation.eval_engine import GOLDEN_CORPUS

DEFAULT_SET = "/app/evaluation/golden_tables.jsonl"


def norm(s: str) -> str:
    """Πεζά + ενιαία κενά + πέταμα διαχωριστικού χιλιάδων ΜΟΝΟ ανάμεσα σε ψηφία.

    Το «29,423» και το «29423» πρέπει να ταιριάζουν. Δεν πετάμε κάθε κόμμα
    γιατί θα κολλούσαν άσχετες λέξεις μεταξύ τους.
    """
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"(?<=\d)[, ](?=\d)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def check(answer: str, expected) -> list:
    """Για κάθε αναμενόμενη τιμή: βρέθηκε έστω μία αποδεκτή παραλλαγή;"""
    a = norm(answer)
    out = []
    for variants in expected:
        vs = variants if isinstance(variants, list) else [variants]
        out.append(any(norm(v) in a for v in vs))
    return out


def coverage(texts, keywords) -> float:
    if not keywords:
        return float("nan")
    blob = norm("\n".join(texts))
    hit = sum(1 for kw in keywords if norm(str(kw)) in blob)
    return 100.0 * hit / len(keywords)


async def answer_of(question: str, retrieved) -> str:
    out = ""
    async for chunk in ask_ai(question, target_filenames=GOLDEN_CORPUS,
                              precomputed=retrieved):
        if chunk.get("type") == "text":
            out += chunk.get("data", "")
    return out


async def main(args) -> int:
    with open(args.dataset, encoding="utf-8") as fh:
        tests = [json.loads(line) for line in fh if line.strip()]
    print(f"{len(tests)} ερωτήσεις πινάκων από {os.path.basename(args.dataset)}\n")

    rows = []
    for i, t in enumerate(tests, 1):
        print(f"[{i}/{len(tests)}] {t['id']} {t['question'][:58]}...", flush=True)
        try:
            retrieved = await search_documents(t["question"],
                                               target_filenames=GOLDEN_CORPUS)
        except Exception as e:
            print(f"    σφάλμα ανάκτησης: {type(e).__name__}")
            continue
        texts = [text for text, _m in retrieved]
        cov = coverage(texts, t.get("keywords") or [])
        cov = 0.0 if cov != cov else cov

        if not texts:
            ans, found = "", [False] * len(t["expected"])
        else:
            try:
                ans = await answer_of(t["question"], retrieved)
            except Exception as e:
                print(f"    σφάλμα γέννησης: {type(e).__name__}")
                continue
            found = check(ans, t["expected"])

        n_ok, n_all = sum(found), len(found)
        if n_ok == n_all:
            verdict = "ΣΩΣΤΗ"
        elif cov == 0.0:
            verdict = "ΑΝΑΚΤΗΣΗ"
        else:
            verdict = "ΣΥΝΘΕΣΗ"
        rows.append({"id": t["id"], "shape": t.get("shape", ""),
                     "doc": t.get("doc", ""), "coverage": round(cov, 1),
                     "values_ok": n_ok, "values_total": n_all,
                     "verdict": verdict,
                     "missing": "|".join(
                         (v[0] if isinstance(v, list) else str(v))
                         for v, ok in zip(t["expected"], found) if not ok),
                     "answer": ans.replace("\n", " ")[:600]})
        print(f"    κάλυψη {cov:5.1f}  τιμές {n_ok}/{n_all}  {verdict}"
              + (f"  λείπει: {rows[-1]['missing']}" if n_ok < n_all else ""))
        if args.delay and i < len(tests):
            await asyncio.sleep(args.delay)

    if not rows:
        print("Καμία ερώτηση δεν ολοκληρώθηκε.")
        return 1

    ok = [r for r in rows if r["verdict"] == "ΣΩΣΤΗ"]
    synth = [r for r in rows if r["verdict"] == "ΣΥΝΘΕΣΗ"]
    retr = [r for r in rows if r["verdict"] == "ΑΝΑΚΤΗΣΗ"]
    vals_ok = sum(r["values_ok"] for r in rows)
    vals_all = sum(r["values_total"] for r in rows)

    print("\n" + "=" * 74)
    print(f"πλήρως σωστές          {len(ok):>3} / {len(rows)}")
    print(f"σφάλμα ΣΥΝΘΕΣΗΣ        {len(synth):>3}   (η σελίδα ήρθε, "
          f"η τιμή δεν βγήκε)")
    print(f"σφάλμα ΑΝΑΚΤΗΣΗΣ       {len(retr):>3}   (η σελίδα δεν ήρθε)")
    print(f"μεμονωμένες τιμές      {vals_ok:>3} / {vals_all}  "
          f"({100.0 * vals_ok / vals_all:.1f}%)")
    cov_all = sum(r["coverage"] for r in rows) / len(rows)
    print(f"μέση κάλυψη            {cov_all:.1f}%")
    print("-" * 74)
    print("ανά μορφή πίνακα:")
    shapes = {}
    for r in rows:
        s = shapes.setdefault(r["shape"], [0, 0])
        s[0] += r["verdict"] == "ΣΩΣΤΗ"
        s[1] += 1
    for s, (a, b) in sorted(shapes.items()):
        print(f"  {s:<20} {a}/{b}")
    print("=" * 74)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ γράφτηκε {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ερωτήσεις πινάκων: ανάκτηση ή σύνθεση φταίει;")
    ap.add_argument("dataset", nargs="?", default=DEFAULT_SET)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--csv", default=None)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
