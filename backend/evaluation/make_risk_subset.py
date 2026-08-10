"""Judge subset οδηγούμενο από ΡΙΣΚΟ, όχι από ποσοστώσεις κατηγορίας.

ΓΙΑΤΙ ΟΧΙ ΤΟ make_judge_subset.py:
Εκείνο μοιράζει θέσεις ανά κατηγορία (5 enumeration, 5 reasoning...) — σωστό
όταν ελέγχεις μια αλλαγή που επηρεάζει ΟΛΕΣ τις ερωτήσεις το ίδιο (π.χ.
thinking budget). Η αλλαγή reranker ΔΕΝ είναι τέτοια: το `compare_pages_
rerankers.py` έδειξε ότι σε 21/56 ερωτήσεις το prompt μένει ΤΑΥΤΟΣΗΜΟ. Να
πληρώσεις judge γι' αυτές είναι να αγοράσεις θόρυβο δειγματοληψίας.

ΤΟ ΚΡΙΤΗΡΙΟ: συμμετρική διαφορά σελίδων = πόσες σελίδες μπήκαν ή βγήκαν.
Όσο περισσότερες, τόσο πιο διαφορετικό το κείμενο που βλέπει το Gemini, τόσο
μεγαλύτερο το ρίσκο υποβάθμισης. Ταξινόμηση φθίνουσα, top-N.

ΓΙΑΤΙ ΔΕΝ ΞΑΝΑΤΡΕΧΟΥΜΕ ΤΟ BASELINE:
Στο judge_minilm.csv οι 48/50 ερωτήσεις είναι ήδη 5/5/5/5 — και ΟΛΕΣ οι
top-ρίσκου είναι μέσα σε αυτές. Όταν το baseline είναι στο ταβάνι, ένα νέο
5/5/5/5 ΔΕΝ μπορεί να κρύβει υποβάθμιση: το 5 είναι το μέγιστο. Αρκεί λοιπόν
να τρέξει ΜΟΝΟ το νέο μοντέλο -> μισό κόστος. (Αν κάποια βγει <5, τότε ΚΑΙ
μόνο τότε χρειάζεται baseline run της ίδιας ερώτησης, γιατί το judge_minilm
είναι από 30/7 — πριν το domain-aware translation prompt.)

    docker compose exec backend python evaluation/make_risk_subset.py --n 12
    docker compose exec backend python run_eval.py evaluation/golden_risk_subset.jsonl \
        --out evaluation/runs/judge_l12.csv
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGES_CSV = os.path.join(HERE, "runs", "pages_l12.csv")
OUT = os.path.join(HERE, "golden_risk_subset.jsonl")
SOURCES = [
    os.path.join(HERE, "golden_set_50.jsonl"),
    os.path.join(HERE, "golden_multihop_new.jsonl"),
]
# Ερωτήσεις που μπαίνουν ΠΑΝΤΑ, ανεξάρτητα από το ρίσκο σελίδων.
# q036: η ΜΟΝΗ in-corpus με judge < 5 στο baseline (completeness 4) που άλλαξε
# κι από πάνω context -> η μόνη με πραγματικό χώρο να κινηθεί προς τα πάνω.
ALWAYS = ["q036"]


def load_sources():
    out = {}
    for p in SOURCES:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    out.setdefault(t["id"], t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="πόσες ερωτήσεις")
    ap.add_argument("--pages-csv", default=PAGES_CSV)
    args = ap.parse_args()

    if not os.path.exists(args.pages_csv):
        print(f"ΣΦΑΛΜΑ: δεν βρέθηκε {args.pages_csv}\n"
              f"Τρέξε πρώτα: compare_pages_rerankers.py --csv ...")
        return 1

    with open(args.pages_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        n_a, n_b, common = int(r["n_a"]), int(r["n_b"]), int(r["common"])
        r["_risk"] = (n_a - common) + (n_b - common)

    changed = [r for r in rows if r["_risk"] > 0]
    print(f"{len(rows)} ερωτήσεις · {len(changed)} με αλλαγμένο context · "
          f"{len(rows)-len(changed)} ΤΑΥΤΟΣΗΜΕΣ (παραλείπονται)\n")

    picked, seen = [], set()
    for qid in ALWAYS:
        row = next((r for r in rows if r["id"] == qid), None)
        if row and qid not in seen:
            picked.append(row)
            seen.add(qid)

    # Το ρίσκο έχει ΠΡΟΤΕΡΑΙΟΤΗΤΑ, αλλά οι ισόπαλες σπάνε με ΚΑΤΗΓΟΡΙΑ, όχι με
    # αλφαβητική σειρά id. Χωρίς αυτό, οι 8 τελευταίες θέσεις γέμιζαν από τις
    # ισόπαλες στο risk=2 κατά id -> ΜΗΔΕΝ reasoning στο δείγμα, ενώ τέσσερις
    # reasoning είχαν αλλάξει context. Ένα δείγμα που δεν μπορεί να δει μια
    # ολόκληρη κατηγορία δεν ελέγχει μη-υποβάθμιση.
    counts = {}
    for r in picked:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    pool = [r for r in changed if r["id"] not in seen]
    while pool and len(picked) < args.n:
        r = min(pool, key=lambda x: (-x["_risk"], counts.get(x["category"], 0),
                                     float(x["jaccard"]), x["id"]))
        pool.remove(r)
        picked.append(r)
        seen.add(r["id"])
        counts[r["category"]] = counts.get(r["category"], 0) + 1

    src = load_sources()
    missing = [r["id"] for r in picked if r["id"] not in src]
    if missing:
        print(f"ΠΡΟΣΟΧΗ: δεν βρέθηκαν στα golden sets: {missing}")

    print(f"{'id':<7} {'κατηγορία':<14} {'σελ άλλαξαν':>12} {'jaccard':>8}  λόγος")
    print("-" * 64)
    for r in picked:
        why = "πάντα (judge<5)" if r["id"] in ALWAYS else "ρίσκο + κατηγορία"
        print(f"  {r['id']:<5} {r['category']:<14} {r['_risk']:>12} "
              f"{float(r['jaccard']):>8.3f}  {why}")

    by_cat = {}
    for r in picked:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    all_cats = {r["category"] for r in changed}
    print("\nΚάλυψη κατηγοριών: "
          + " · ".join(f"{c} {by_cat.get(c, 0)}" for c in sorted(all_cats)))
    gaps = [c for c in all_cats if not by_cat.get(c)]
    if gaps:
        print(f"ΠΡΟΣΟΧΗ: καμία ερώτηση από {gaps} — αύξησε το --n")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in picked:
            if r["id"] in src:
                f.write(json.dumps(src[r["id"]], ensure_ascii=False) + "\n")

    n = sum(1 for r in picked if r["id"] in src)
    print(f"\nΓράφτηκε {OUT}: {n} ερωτήσεις")
    print(f"Κόστος judge run: ~{2 * n} κλήσεις Gemini "
          f"(αντί ~100 για πλήρες run των 50).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
