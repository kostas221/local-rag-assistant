# backend/evaluation/compare_citation_formats.py
"""ΤΡΕΙΣ ΜΟΡΦΕΣ ΠΑΡΑΠΟΜΠΗΣ, ΕΝΑΣ ΠΙΝΑΚΑΣ — ΜΗΔΕΝ κλήσεις API.

Διαβάζει τα CSV του probe_inline_citations.py και τα βάζει δίπλα-δίπλα. Κάθε
τρέξιμο έχει ΤΟ ΔΙΚΟ ΤΟΥ base σκέλος, οπότε όλες οι συγκρίσεις είναι
ΖΕΥΓΑΡΩΤΕΣ μέσα στο τρέξιμο — ΠΟΤΕ base του ενός με cite του άλλου. Η γέννηση
δεν είναι ντετερμινιστική (μετρήθηκε: το q001 base έδωσε 60 και 139 tokens σε
δύο τρεξίματα), άρα σύγκριση μεταξύ τρεξιμάτων θα μετρούσε θόρυβο.

ΤΟ ΚΡΙΣΙΜΟ ΠΟΥ ΠΡΟΣΘΕΤΕΙ: ΠΛΗΡΟΤΗΤΑ. Το «λιγότερα tokens» είναι κέρδος μόνο
αν το κείμενο λέει τα ίδια πράγματα. Μετριέται ντετερμινιστικά ως ποσοστό των
keywords του golden set που εμφανίζονται ΜΕΣΑ ΣΤΗΝ ΑΠΑΝΤΗΣΗ (όχι στη σελίδα).
ΠΡΟΣΟΧΗ: τα keywords γράφτηκαν για τις ΣΕΛΙΔΕΣ, όχι για τις απαντήσεις, και
είναι αγγλικά ενώ 5/11 απαντήσεις είναι ελληνικές -> το απόλυτο νούμερο ΔΕΝ
διαβάζεται. Διαβάζεται ΜΟΝΟ η ζευγαρωτή διαφορά base -> cite μέσα στο ίδιο
τρέξιμο, όπου η γλώσσα και η επιλογή keywords είναι σταθερές.

Τρέξε:
  docker compose exec backend python evaluation/compare_citation_formats.py \
      full=evaluation/runs/inline_citations.csv \
      label=evaluation/runs/inline_citations_label.csv \
      bare=evaluation/runs/inline_citations_bare.csv
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probe_inline_citations import HARD_SET, MAIN_SET, norm, strip_citations


def load_keywords() -> dict:
    out = {}
    for path in (MAIN_SET, HARD_SET):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r.get("keywords", [])
    return out


def kw_coverage(answer: str, keywords: list, fmt: str) -> float:
    """Ποσοστό keywords που εμφανίζονται στην απάντηση (χωρίς τις αγκύλες).

    Οι αγκύλες αφαιρούνται: αλλιώς ένα «[excamera-nsdi17.pdf, p.15]» θα
    μετρούσε το keyword «excamera» ως καλυμμένο — η μορφή full θα κέρδιζε
    πλήρως τεχνητά.
    """
    if not keywords:
        return float("nan")
    body = norm(strip_citations(answer, fmt))
    hit = sum(1 for k in keywords if norm(k) and norm(k) in body)
    return hit / len(keywords)


def avg(v):
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else float("nan")


def median(v):
    v = sorted(x for x in v if x == x)
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
    if not specs:
        print(__doc__)
        return
    kws = load_keywords()

    print(f"{'μορφή':7s} {'n':>3s} {'tok base':>9s} {'tok cite':>9s} "
          f"{'Δ%':>7s} {'διάμ Δ%':>8s} {'cite>base':>10s} "
          f"{'πληρ.base':>10s} {'πληρ.cite':>10s} {'Δ πληρ.':>8s}")
    print("-" * 92)
    detail = {}

    for name, path in specs:
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        by_id = {}
        for r in rows:
            by_id.setdefault(r["id"], {})[r["arm"]] = r

        d_pct, tb, tc, cb, cc, up, pairs = [], [], [], [], [], 0, []
        for qid, arms in by_id.items():
            if "base" not in arms or "cite" not in arms:
                continue
            b = float(arms["base"]["completion_tokens"] or "nan")
            c = float(arms["cite"]["completion_tokens"] or "nan")
            tb.append(b)
            tc.append(c)
            if b == b and c == c and b:
                d_pct.append(100 * (c - b) / b)
                up += c > b
            # base σκέλος: ποτέ αγκύλες -> fmt="full" είναι ουδέτερο εκεί
            cb.append(kw_coverage(arms["base"]["answer"], kws.get(qid, []), "full"))
            cc.append(kw_coverage(arms["cite"]["answer"], kws.get(qid, []),
                                  arms["cite"].get("fmt", "full")))
            pairs.append((qid, cb[-1], cc[-1]))

        n = len(d_pct)
        dcov = avg(cc) - avg(cb)
        print(f"{name:7s} {n:3d} {avg(tb):9.0f} {avg(tc):9.0f} "
              f"{100 * (avg(tc) - avg(tb)) / avg(tb):+7.1f} "
              f"{median(d_pct):+8.1f} {up:>7d}/{n:<2d} "
              f"{avg(cb):10.3f} {avg(cc):10.3f} {dcov:+8.3f}")

        cite = [r for r in rows if r["arm"] == "cite"]
        detail[name] = {
            "cites": sum(int(r["n_cites"]) for r in cite),
            "fab": sum(int(r["n_fabricated"]) for r in cite),
            "fabp": sum(int(r.get("n_fab_page") or 0) for r in cite),
            "bad": sum(int(r["n_bad_format"]) for r in cite),
            "gc": avg([float(r["ground_cited"]) for r in cite
                       if r["ground_cited"]]),
            "gf": avg([float(r["ground_floor"]) for r in cite
                       if r["ground_floor"]]),
            "chars_b": avg([float(r["chars"]) for r in rows
                            if r["arm"] == "base"]),
            "chars_c": avg([float(r["chars"]) for r in cite]),
            "ref": sum(int(r["refused"]) for r in cite),
            "cov_pairs": pairs,
        }

    print(f"\n{'μορφή':7s} {'παραπ.':>7s} {'λάθος πηγή':>11s} "
          f"{'λάθος σελ.':>11s} {'μορφή':>6s} {'θεμελ.':>7s} {'πάτωμα':>7s} "
          f"{'χαρ base':>9s} {'χαρ cite':>9s} {'αρνήσ.':>7s}")
    print("-" * 92)
    for name, d in detail.items():
        print(f"{name:7s} {d['cites']:7d} {d['fab'] - d['fabp']:11d} "
              f"{d['fabp']:11d} {d['bad']:6d} {d['gc']:7.3f} {d['gf']:7.3f} "
              f"{d['chars_b']:9.0f} {d['chars_c']:9.0f} {d['ref']:7d}")

    print("\n--- ΠΛΗΡΟΤΗΤΑ ΑΝΑ ΕΡΩΤΗΣΗ (keywords golden set μέσα στην απάντηση) ---")
    print("ΠΡΟΣΟΧΗ: ζευγαρωτό ΜΕΣΑ στο τρέξιμο· το απόλυτο νούμερο δεν διαβάζεται.")
    for name, d in detail.items():
        worse = [(q, b, c) for q, b, c in d["cov_pairs"]
                 if b == b and c == c and c < b - 1e-9]
        better = [(q, b, c) for q, b, c in d["cov_pairs"]
                  if b == b and c == c and c > b + 1e-9]
        print(f"  {name:7s} χειρότερα {len(worse)} · καλύτερα {len(better)}"
              + ("   " + ", ".join(f"{q} {b:.2f}->{c:.2f}" for q, b, c in worse)
                 if worse else ""))


if __name__ == "__main__":
    main()
