"""Στέλνει το νέο reranker ΤΙΣ ΙΔΙΕΣ ΣΕΛΙΔΕΣ στο Gemini; — έλεγχος πριν το judge.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ (και γιατί τρέχει ΠΡΙΝ ξοδέψουμε quota):
Το judge run κοστίζει ~50 κλήσεις γέννησης + 50 κρίσης. Αλλά η ποιότητα της
απάντησης εξαρτάται ΜΟΝΟ από το κείμενο που φτάνει στο μοντέλο — δηλαδή από
τις σελίδες που επιστρέφει το `_expand_to_pages`. Αν οι σελίδες είναι
ΤΑΥΤΟΣΗΜΕΣ, το Gemini βλέπει κυριολεκτικά το ίδιο prompt και το judge δεν έχει
τίποτα να μετρήσει: κάθε διαφορά που θα έβγαζε θα ήταν θόρυβος δειγματοληψίας.

Αυτός ο έλεγχος στοιχίζει ΜΗΔΕΝ (οι μεταφράσεις είναι στο μόνιμο cache) και
απαντά ακριβώς στο ερώτημα «χρειάζεται καν judge run;». Αν οι σελίδες
διαφέρουν σε λίγες ερωτήσεις, τυπώνει ΠΟΙΕΣ -> judge μόνο σε αυτές.

ΠΩΣ ΣΥΓΚΡΙΝΕΙ ΔΥΟ ΜΟΝΤΕΛΑ ΣΕ ΜΙΑ ΔΙΕΡΓΑΣΙΑ:
Το RERANKER_MODEL διαβάζεται στο import, οπότε δεν γίνεται με env var. Αντ'
αυτού κάνει monkeypatch τα module-globals του ai_core (reranker, κατώφλια) —
ίδιο μοτίβο με το compare_rerankers_hard.py. Το pipeline μένει το ΠΡΑΓΜΑΤΙΚΟ
`search_documents`, με gate και corrective agent ενεργά.

ΠΡΟΣΟΧΗ: μόνο in-corpus. Τα out_of_corpus κόβονται (επαληθευμένο 5/5 και στα
δύο μοντέλα) και θα πυροδοτούσαν άσκοπα rewrites.

    docker compose exec backend python evaluation/compare_pages_rerankers.py
    docker compose exec backend python evaluation/compare_pages_rerankers.py \
        --model cross-encoder/ms-marco-MiniLM-L-12-v2 --gate -2.6 --corrective 0.4
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
                    tests.append(json.loads(line))
    return [t for t in tests if t.get("category") != "out_of_corpus"]


def page_keys(pages):
    """Ταυτότητα σελίδας: (αρχείο, σελίδα). Το κείμενο είναι ντετερμινιστικό
    δεδομένου του κλειδιού, οπότε δεν χρειάζεται να συγκριθεί."""
    out = []
    for _text, meta in pages:
        out.append((meta.get("file_name"), meta.get("page")))
    return out


async def collect(tests, label):
    res = {}
    for t in tests:
        pages = await ai_core.search_documents(t["question"], GOLDEN_CORPUS)
        res[t["id"]] = page_keys(pages)
    print(f"  {label}: {len(res)} ερωτήσεις, "
          f"{sum(len(v) for v in res.values())} σελίδες συνολικά")
    return res


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="άλλο reranker· κενό = κράτα το τρέχον")
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--corrective", type=float, default=None)
    ap.add_argument("--candidates", type=int, default=None,
                    help="άλλο RERANK_CANDIDATES. ΠΡΟΣΟΧΗ: αν πέσει κάτω από το "
                         "EXPAND_INPUT, το _expand_to_pages δουλεύει με λιγότερα "
                         "chunks -> ΑΛΛΕΣ σελίδες. Ακριβώς αυτό ελέγχεται εδώ.")
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if not any((args.model, args.gate, args.corrective, args.candidates)):
        print("ΣΦΑΛΜΑ: δώσε τουλάχιστον ένα από --model/--gate/--corrective/"
              "--candidates, αλλιώς οι δύο πλευρές είναι ταυτόσημες.")
        return 1

    tests = load_sets(args.sets)
    missing = [t["id"] for t in tests
               if ai_core._has_greek(t["question"])
               and t["question"] not in ai_core._translation_cache]
    if missing:
        print(f"ΠΡΟΣΟΧΗ: {len(missing)} ελληνικές εκτός cache -> τόσες κλήσεις: "
              f"{', '.join(missing)}\n")
    else:
        print("Translation cache: πλήρες -> ΜΗΔΕΝ κλήσεις API.\n")

    print(f"{len(tests)} in-corpus ερωτήσεις\n")

    def cfg():
        return (f"{ai_core.RERANKER_MODEL} @ gate {ai_core.MIN_RERANK_SCORE} · "
                f"corrective {ai_core.CORRECTIVE_MIN_SCORE} · "
                f"candidates {ai_core.RERANK_CANDIDATES} · "
                f"expand_input {ai_core.EXPAND_INPUT}")

    print(f"A. ΤΡΕΧΟΝ: {cfg()}")
    base = await collect(tests, "A")

    if args.model:
        from sentence_transformers import CrossEncoder
        print(f"\nΦόρτωση {args.model} ...", flush=True)
        ai_core.reranker = CrossEncoder(args.model, device=ai_core.DEVICE)
        ai_core.RERANKER_MODEL = args.model
    if args.gate is not None:
        ai_core.MIN_RERANK_SCORE = args.gate
    if args.corrective is not None:
        ai_core.CORRECTIVE_MIN_SCORE = args.corrective
    if args.candidates is not None:
        ai_core.RERANK_CANDIDATES = args.candidates
        if args.candidates < ai_core.EXPAND_INPUT:
            print(f"\n  ΠΡΟΣΟΧΗ: candidates {args.candidates} < EXPAND_INPUT "
                  f"{ai_core.EXPAND_INPUT} -> το expand θα δει "
                  f"{args.candidates} chunks αντί {ai_core.EXPAND_INPUT}.")
    print(f"\nB. ΥΠΟΨΗΦΙΟ: {cfg()}")
    cand = await collect(tests, "B")

    rows, identical, diff_ids = [], 0, []
    jac_sum = 0.0
    for t in tests:
        a, b = base[t["id"]], cand[t["id"]]
        sa, sb = set(a), set(b)
        inter, union = sa & sb, sa | sb
        jac = len(inter) / len(union) if union else 1.0
        jac_sum += jac
        same_set = sa == sb
        same_order = a == b
        if same_set:
            identical += 1
        else:
            diff_ids.append(t["id"])
        rows.append(dict(id=t["id"], category=t.get("category", ""),
                         n_a=len(a), n_b=len(b), common=len(inter),
                         jaccard=round(jac, 3), same_set=same_set,
                         same_order=same_order,
                         only_a=" | ".join(f"{f}:{p}" for f, p in sorted(sa - sb)),
                         only_b=" | ".join(f"{f}:{p}" for f, p in sorted(sb - sa))))

    n = len(tests)
    print("\n" + "=" * 78)
    print("ΤΑΥΤΙΣΗ ΣΕΛΙΔΩΝ — αυτό ακριβώς βλέπει το Gemini")
    print("=" * 78)
    print(f"  ίδιο ΣΥΝΟΛΟ σελίδων:       {identical}/{n}  "
          f"({100*identical/n:.1f}%)")
    print(f"  ίδια ΚΑΙ η σειρά:          "
          f"{sum(1 for r in rows if r['same_order'])}/{n}")
    print(f"  μέσος Jaccard:             {jac_sum/n:.3f}")

    if diff_ids:
        print(f"\n  ΔΙΑΦΟΡΕΣ σε {len(diff_ids)} ερωτήσεις:")
        for r in rows:
            if r["same_set"]:
                continue
            print(f"    {r['id']:<7} {r['category']:<14} κοινές "
                  f"{r['common']}/{max(r['n_a'], r['n_b'])} · Jaccard {r['jaccard']}")
            if r["only_a"]:
                print(f"        μόνο ΤΡΕΧΟΝ: {r['only_a']}")
            if r["only_b"]:
                print(f"        μόνο ΝΕΟ:    {r['only_b']}")
        print(f"\n  -> JUDGE ΜΟΝΟ ΣΕ ΑΥΤΕΣ: {' '.join(diff_ids)}")
    else:
        print("\n  ΚΑΜΙΑ ΔΙΑΦΟΡΑ. Το Gemini λαμβάνει ταυτόσημο prompt σε "
              "κάθε ερώτηση\n  -> το judge run ΔΕΝ μπορεί να δείξει τίποτα "
              "πέρα από θόρυβο δειγματοληψίας.")

    empty_a = [r["id"] for r in rows if r["n_a"] == 0]
    empty_b = [r["id"] for r in rows if r["n_b"] == 0]
    if empty_a or empty_b:
        print(f"\n  ΣΙΩΠΗΛΕΣ — ΤΡΕΧΟΝ: {empty_a or 'καμία'} · "
              f"ΝΕΟ: {empty_b or 'καμία'}")

    if args.csv and rows:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
