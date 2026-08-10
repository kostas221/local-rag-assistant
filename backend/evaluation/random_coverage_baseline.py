"""Πόσο coverage βγάζει η ΣΚΕΤΗ ΤΥΧΗ; — το πάτωμα κάθε golden set.

ΤΟ ΚΕΝΟ ΠΟΥ ΚΛΕΙΝΕΙ:
Το `verify_keywords.py` υπολογίζει τυχαίο baseline για το **MRR**. Αλλά τρεις
από τις σημαντικότερες μετρήσεις του project αναφέρονται σε **coverage**:
  - probe_conversational.py   40.0% χωρίς rewrite -> 90.0% με rewrite
  - verify_corrective.py      «ΣΩΘΗΚΕ» = βρέθηκαν keywords στις σελίδες
  - run_eval.py               keyword_coverage 98.5%
Κανένα από αυτά δεν είχε πάτωμα τύχης. Ένα «90%» δεν σημαίνει τίποτα αν η τύχη
δίνει 85%, και σημαίνει πολλά αν δίνει 10%.

Ο ΥΠΟΛΟΓΙΣΜΟΣ (αναλυτικός, ΟΧΙ προσομοίωση -> ντετερμινιστικός):
Αν ένα keyword υπάρχει σε f από N σελίδες και το σύστημα επιστρέφει k σελίδες
στην τύχη, η πιθανότητα να βρεθεί είναι υπεργεωμετρική:
    P = 1 - C(N-f, k) / C(N, k)
Το αναμενόμενο coverage μιας ερώτησης = μέσος όρος των P των keywords της
(γραμμικότητα της αναμενόμενης τιμής — ισχύει ΚΑΙ με εξαρτημένα keywords).

ΜΟΝΑΔΑ = ΣΕΛΙΔΑ, όπως στο verify_keywords: το search_documents τελειώνει με
_expand_to_pages και το eval κάνει substring match πάνω σε ΟΛΟΚΛΗΡΕΣ σελίδες.

ΔΕΝ κάνει import το ai_core — μόνο ανάγνωση κειμένων από τη ChromaDB.
Μηδέν κόστος API, read-only, τρέχει σε δευτερόλεπτα.

    docker compose exec backend python evaluation/random_coverage_baseline.py
    docker compose exec backend python evaluation/random_coverage_baseline.py \
        evaluation/golden_conversations.jsonl --observed 40.0 90.0
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, "/app")

DEFAULT_SETS = [
    "/app/evaluation/golden_set_50.jsonl",
    "/app/evaluation/golden_multihop_new.jsonl",
    "/app/evaluation/golden_hard_paraphrase.jsonl",
    "/app/evaluation/golden_conversations.jsonl",
]
MAX_PAGES = int(os.getenv("MAX_PAGES", "8"))


def load_pages():
    """{(file_name, page, doc_id): κείμενο} — μία εγγραφή ανά σελίδα.

    ΙΔΙΟ μοτίβο με το verify_keywords.py (ίδιο collection, ίδιο κλειδί): το
    doc_id μπαίνει ΕΠΙΤΗΔΕΣ στο κλειδί, όπως στο _expand_to_pages. Χωρίς αυτό,
    δύο ingests του ίδιου αρχείου πέφτουν στα ίδια κλειδιά και τα διπλότυπα
    γίνονται αόρατα — ενώ το eval τα βλέπει ως ξεχωριστές σελίδες.
    Αν τα δύο scripts δεν συμφωνούν στο N, τα δύο baselines δεν συγκρίνονται.
    """
    import chromadb
    db_path = os.getenv("VECTOR_DB_PATH", "./vector_db")
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection(name="ai_research_docs")
    got = col.get(include=["documents", "metadatas"])
    pages = {}
    for doc, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
        meta = meta or {}
        key = (str(meta.get("file_name", "?")), meta.get("page"),
               meta.get("doc_id"))
        pages.setdefault(key, []).append(doc or "")
    return {k: "\n".join(v).lower() for k, v in pages.items()}


def load_set(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def p_found(f: int, n: int, k: int) -> float:
    """Πιθανότητα να πέσει τουλάχιστον μία από τις f «καλές» σελίδες σε k
    τυχαίες επιλογές χωρίς επανάθεση, από σύνολο n."""
    if f <= 0:
        return 0.0
    if f >= n or k >= n:
        return 1.0
    return 1.0 - math.comb(n - f, k) / math.comb(n, k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", default=DEFAULT_SETS)
    ap.add_argument("--pages", type=int, default=MAX_PAGES,
                    help="πόσες σελίδες επιστρέφει το σύστημα (MAX_PAGES)")
    ap.add_argument("--observed", nargs="*", type=float, default=None,
                    help="πραγματικά coverage %% για σύγκριση (π.χ. 40 90 100)")
    args = ap.parse_args()

    pages = load_pages()
    n = len(pages)
    print(f"Corpus: {n} σελίδες · το σύστημα επιστρέφει {args.pages}\n")

    for path in args.sets:
        if not os.path.exists(path):
            print(f"ΠΑΡΑΛΕΙΨΗ (δεν υπάρχει): {path}")
            continue
        tests = [t for t in load_set(path)
                 if t.get("category") != "out_of_corpus"]
        if not tests:
            continue

        per_q, weak = [], []
        for t in tests:
            ps = []
            for kw in t["keywords"]:
                low = kw.lower()
                f = sum(1 for txt in pages.values() if low in txt)
                p = p_found(f, n, args.pages)
                ps.append(p)
                if p >= 0.60:
                    weak.append((t["id"], kw, f, p))
            per_q.append((t["id"], sum(ps) / len(ps)))

        exp = 100 * sum(p for _i, p in per_q) / len(per_q)
        print("=" * 74)
        print(f"{os.path.basename(path)} — {len(tests)} in-corpus ερωτήσεις")
        print("=" * 74)
        print(f"  ΑΝΑΜΕΝΟΜΕΝΟ COVERAGE ΑΠΟ ΤΥΧΗ: {exp:>5.1f}%")
        hi = sorted(per_q, key=lambda x: -x[1])[:3]
        print("  χειρότερες (πιο εύκολο να τις πετύχει η τύχη): "
              + " · ".join(f"{i} {100*p:.0f}%" for i, p in hi))
        if weak:
            print(f"\n  Keywords με >=60% πιθανότητα ΤΥΧΑΙΑΣ εύρεσης ({len(weak)}):")
            for qid, kw, f, p in sorted(weak, key=lambda x: -x[3]):
                print(f"    {qid:<6} {kw:<18} {f:>3}/{n} σελ  ->  {100*p:>5.1f}% "
                      f"τυχαία")
        else:
            print("\n  Κανένα keyword δεν βρίσκεται εύκολα στην τύχη.")

        if args.observed:
            print()
            for obs in args.observed:
                lift = obs - exp
                head = (obs - exp) / (100 - exp) * 100 if exp < 100 else 0.0
                verdict = ("ΘΟΡΥΒΟΣ" if lift <= 2 else
                           "ΟΡΙΑΚΟ" if lift <= 10 else "ΠΡΑΓΜΑΤΙΚΟ")
                print(f"  παρατηρήθηκε {obs:>5.1f}%  ->  {lift:>+5.1f} πάνω από "
                      f"την τύχη · κλείνει {head:>4.0f}% του διαθέσιμου "
                      f"περιθωρίου · {verdict}")
        print()

    print("ΣΗΜΕΙΩΣΗ: το πάτωμα ΔΕΝ σημαίνει ότι το σύστημα «κλέβει» — σημαίνει")
    print("ότι κάθε ποσοστό πρέπει να διαβάζεται ΩΣ ΠΡΟΣ αυτό. Ένα 90% με πάτωμα")
    print("40% είναι πολύ διαφορετικό από ένα 90% με πάτωμα 10%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
