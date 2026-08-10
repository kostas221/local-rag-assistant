"""Διαστήματα εμπιστοσύνης (bootstrap) για κάθε μετρική — και για τη ΔΙΑΦΟΡΑ δύο runs.

ΤΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΛΥΝΕΙ:
Όλο το project αναφέρει νούμερα σαν να είναι ακριβή: «MRR 0.793», «coverage 98.5%».
Με n=45 δεν είναι. Ο εμπειρικός κανόνας «μη πιστεύεις διαφορά <0.01» προέκυψε από
παρατήρηση (ο ίδιος κώδικας έδωσε 0.764 και 0.755) — εδώ γίνεται ΜΕΤΡΗΜΕΝΟΣ.

ΤΟ ΚΡΙΣΙΜΟ ΕΙΝΑΙ ΤΟ --compare, ΟΧΙ ΤΟ ΑΠΛΟ CI:
Το ερώτημα του project δεν είναι «πόσο είναι το MRR» αλλά «είναι το 0.770 -> 0.793
πραγματική βελτίωση;». Αυτό απαντιέται με ΖΕΥΓΑΡΩΤΟ (paired) bootstrap: κάνουμε
resample ΕΡΩΤΗΣΕΙΣ (όχι ανεξάρτητα τα δύο runs), γιατί τα δύο runs είδαν ΤΙΣ ΙΔΙΕΣ
ερωτήσεις. Το ζευγάρωμα αφαιρεί τη διακύμανση «κάποιες ερωτήσεις είναι πιο εύκολες»
και δίνει πολύ στενότερο διάστημα από δύο ανεξάρτητα CI.
    Αν το 95% CI της διαφοράς ΠΕΡΙΕΧΕΙ το μηδέν -> δεν αποδεικνύεται βελτίωση.
    ΠΡΟΣΟΧΗ: «δεν αποδεικνύεται» ΔΕΝ σημαίνει «δεν υπάρχει» — με n=45 ένα
    πραγματικό +0.02 συχνά δεν ανιχνεύεται. Απουσία απόδειξης, όχι απόδειξη απουσίας.

ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟ: σταθερό seed, ίδια έξοδος σε κάθε εκτέλεση — όπως κάθε άλλη
μέτρηση του project. Μηδέν εξαρτήσεις πέρα από τη stdlib, μηδέν κόστος API.

    docker compose exec backend python evaluation/bootstrap_ci.py \\
        --csv evaluation/runs/retrieval_l12.csv
    docker compose exec backend python evaluation/bootstrap_ci.py \\
        --compare evaluation/runs/r_control.csv evaluation/runs/retrieval_l12.csv
"""
import argparse
import csv
import random
import statistics
import sys

# Στήλες που ΔΕΝ είναι μετρικές (κείμενο/ετικέτες) — αγνοούνται αυτόματα.
SKIP = {"id", "category", "question", "generated_answer", "feedback", "set",
        "cut", "ok"}


def load(path: str) -> list:
    # utf-8-SIG, ΟΧΙ utf-8: τα CSV που γράφονται σε Windows ξεκινούν με BOM,
    # οπότε η πρώτη στήλη ονομάζεται '﻿id' αντί για 'id'. Με σκέτο utf-8 το
    # .get("id") επέστρεφε None σε ΚΑΘΕ γραμμή -> όλες οι γραμμές «ταίριαζαν»
    # μεταξύ τους με κλειδί None και η σύγκριση έβγαζε σιωπηλά σκουπίδια.
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def numeric_columns(rows: list) -> list:
    """Στήλες που έχουν αριθμό σε ΟΛΕΣ τις γραμμές (αλλιώς δεν είναι μετρική)."""
    if not rows:
        return []
    out = []
    for col in rows[0]:
        if col in SKIP:
            continue
        try:
            for r in rows:
                float(r[col])
        except (ValueError, TypeError, KeyError):
            continue
        out.append(col)
    return out


def ci(values: list, n_boot: int, seed: int, alpha: float = 0.05) -> tuple:
    """(mean, low, high) με percentile bootstrap."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    k = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(values) for _ in range(k)) / k)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return statistics.mean(values), lo, hi


def paired_ci(pairs: list, n_boot: int, seed: int, alpha: float = 0.05) -> tuple:
    """ΖΕΥΓΑΡΩΤΟ bootstrap της διαφοράς (new - old).

    Κάνουμε resample τις ΕΡΩΤΗΣΕΙΣ και κρατάμε και τις δύο τιμές μαζί. Αν
    κάναμε resample ανεξάρτητα, θα προσθέταμε τεχνητό θόρυβο από το ότι κάποιες
    ερωτήσεις είναι εγγενώς δυσκολότερες — θόρυβο που ΔΕΝ υπάρχει, αφού και τα
    δύο runs είδαν ακριβώς τις ίδιες ερωτήσεις.
    """
    if not pairs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    k = len(pairs)
    diffs = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(k):
            old, new = rng.choice(pairs)
            s += new - old
        diffs.append(s / k)
    diffs.sort()
    lo = diffs[int(alpha / 2 * n_boot)]
    hi = diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return statistics.mean(n - o for o, n in pairs), lo, hi


def filtered(rows: list, include_ooc: bool) -> list:
    if include_ooc:
        return rows
    return [r for r in rows if r.get("category") != "out_of_corpus"] or rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="ένα run -> CI ανά μετρική")
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"),
                    help="δύο runs -> ΖΕΥΓΑΡΩΤΟ CI της διαφοράς")
    ap.add_argument("--metrics", default=None,
                    help="π.χ. mrr,ndcg,keyword_coverage (default: όλες)")
    ap.add_argument("--n", type=int, default=10000, help="επαναλήψεις bootstrap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-ooc", action="store_true",
                    help="μέτρα και τα out_of_corpus (MRR 0 από κατασκευή)")
    args = ap.parse_args()

    if not args.csv and not args.compare:
        print("ΣΦΑΛΜΑ: δώσε --csv ή --compare")
        return 1

    # ---------- ΕΝΑ RUN ----------
    if args.csv:
        rows = filtered(load(args.csv), args.include_ooc)
        cols = (args.metrics.split(",") if args.metrics
                else numeric_columns(rows))
        print(f"{args.csv}  —  n={len(rows)}  ·  {args.n} bootstrap · seed={args.seed}\n")
        print(f"{'μετρική':<20}{'μέση':>9}{'95% CI':>22}{'±':>9}")
        print("-" * 62)
        for c in cols:
            vals = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
            m, lo, hi = ci(vals, args.n, args.seed)
            print(f"{c:<20}{m:>9.3f}   [{lo:>7.3f}, {hi:>7.3f}]{(hi - lo) / 2:>9.3f}")
        print("\nΤο ± είναι το ΜΙΣΟ πλάτος: κάθε διαφορά μικρότερη από αυτό μεταξύ")
        print("δύο ΑΝΕΞΑΡΤΗΤΩΝ runs δεν ξεχωρίζει από θόρυβο. Για σύγκριση δύο")
        print("εκδόσεων χρησιμοποίησε --compare (ζευγαρωτό, πολύ στενότερο).")
        return 0

    # ---------- ΔΥΟ RUNS, ΖΕΥΓΑΡΩΤΑ ----------
    old_rows, new_rows = (filtered(load(p), args.include_ooc)
                          for p in args.compare)
    for path, rows in zip(args.compare, (old_rows, new_rows)):
        if rows and "id" not in rows[0]:
            print(f"ΣΦΑΛΜΑ: το {path} δεν έχει στήλη 'id' — το ζευγάρωμα είναι\n"
                  f"       αδύνατο. Στήλες: {list(rows[0])[:6]}")
            return 1
    old_by_id = {r["id"]: r for r in old_rows}
    common = [r for r in new_rows if r["id"] in old_by_id]
    if not common:
        print("ΣΦΑΛΜΑ: κανένα κοινό id — τα δύο CSV δεν είναι συγκρίσιμα.")
        return 1
    cols = args.metrics.split(",") if args.metrics else numeric_columns(common)

    print(f"OLD {args.compare[0]}")
    print(f"NEW {args.compare[1]}")
    print(f"κοινές ερωτήσεις: {len(common)}  ·  {args.n} bootstrap · seed={args.seed}\n")
    print(f"{'μετρική':<20}{'old':>8}{'new':>8}{'Δ':>9}{'95% CI της Δ':>22}  ετυμηγορία")
    print("-" * 84)
    for c in cols:
        pairs = []
        for r in common:
            o, n = old_by_id[r["id"]].get(c), r.get(c)
            if o in (None, "") or n in (None, ""):
                continue
            pairs.append((float(o), float(n)))
        if not pairs:
            continue
        d, lo, hi = paired_ci(pairs, args.n, args.seed)
        o_mean = statistics.mean(o for o, _ in pairs)
        n_mean = statistics.mean(n for _, n in pairs)
        # CI μηδενικού πλάτους ΔΕΝ είναι αβεβαιότητα — είναι ταυτότητα: κάθε
        # resample έδωσε 0, άρα η διαφορά είναι 0 σε ΚΑΘΕ ερώτηση. Το να το
        # λέγαμε «θόρυβος» θα έκρυβε το ισχυρότερο δυνατό αποτέλεσμα.
        if lo == hi == 0.0:
            verdict = "ΤΑΥΤΟΣΗΜΟ (0 σε κάθε ερώτηση)"
        elif lo <= 0 <= hi:
            verdict = "ΘΟΡΥΒΟΣ (το CI περιέχει το 0)"
        else:
            verdict = "ΒΕΛΤΙΩΣΗ" if lo > 0 else "ΥΠΟΒΑΘΜΙΣΗ"
        print(f"{c:<20}{o_mean:>8.3f}{n_mean:>8.3f}{d:>+9.3f}   "
              f"[{lo:>+7.3f}, {hi:>+7.3f}]  {verdict}")
    print("-" * 84)
    print("«ΘΟΡΥΒΟΣ» σημαίνει «δεν αποδεικνύεται διαφορά σε αυτό το n» — ΟΧΙ")
    print("«δεν υπάρχει διαφορά». Με n≈45 ένα πραγματικό +0.02 συχνά δεν ανιχνεύεται.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
