"""Προτείνει ΣΠΑΝΙΑ keywords για μια ερώτηση, από τη σωστή σελίδα.

ΤΟ ΠΡΟΒΛΗΜΑ (μετρημένο, random_coverage_baseline.py):
Keyword όπως «lambda» υπάρχει σε 41/122 σελίδες -> βρίσκεται σε 8 τυχαίες
σελίδες με πιθανότητα 96.7%. Ένα τεστ με τέτοιο keyword περνάει ΚΑΙ όταν το
σύστημα αποτυγχάνει. Χρειάζονται όροι που εμφανίζονται σε λίγες σελίδες.

ΠΩΣ ΒΡΙΣΚΕΙ ΤΗ ΣΩΣΤΗ ΣΕΛΙΔΑ (χωρίς να τρέξει retrieval):
Σελίδα-στόχος = αυτή με το μεγαλύτερο άθροισμα **1/συχνότητα** των υπαρχόντων
keywords. Τα keywords είναι ήδη επαληθευμένα (verify_keywords: 0 σφάλματα), άρα
δείχνουν σωστά — απλώς μερικά είναι πολύ κοινά. Κρατάμε το ΠΟΥ, αλλάζουμε το ΜΕ ΤΙ.

ΓΙΑΤΙ ΒΑΡΟΣ 1/df ΚΑΙ ΟΧΙ ΑΠΛΟ ΠΛΗΘΟΣ: με απλή μέτρηση «πόσα keywords έχει η
σελίδα», δεκάδες σελίδες βγαίνουν 3/3 ΤΥΧΑΙΑ — ακριβώς επειδή τα keywords είναι
κοινά. Μετρήθηκε: για το h003 (`transfer`/`disk`) η σελίδα-στόχος έβγαινε στο
semiconductor manufacturing. Το πρόβλημα που λύνουμε χαλούσε τη λύση.

ΤΙ ΠΡΟΤΕΙΝΕΙ: ΜΟΝΕΣ λέξεις από τη σελίδα-στόχο, κατά σπανιότητα. Η ΤΕΛΙΚΗ
ΕΠΙΛΟΓΗ ΕΙΝΑΙ ΑΝΘΡΩΠΙΝΗ: ο όρος πρέπει να απαντά ΟΝΤΩΣ στην ερώτηση, όχι απλώς
να είναι σπάνιος. Σπάνιο keyword άσχετο με την ερώτηση = χειρότερο από κοινό.

    docker compose exec backend python evaluation/suggest_keywords.py \
        evaluation/golden_hard_paraphrase.jsonl --ids h003 h005 h006 h007 h009
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, "/app")

_STOP_WORDS = """
the a an and or but if then than that this these those of in on at to for from
with without by as is are was were be been being it its it's they them their
we our you your he she his her not no nor so such can could may might must
will would shall should do does did done have has had having when where which
who whom whose what why how all any both each few more most other some only own
same too very s t just don now also into over under between during about above
below up down out off again further once here there both while because until
one two three first second new use used using make makes made way ways well
however thus therefore e.g i.e et al fig figure table section paper work
"""  # noqa: RUF100 — γραμμένο ως κείμενο επίτηδες: διαβάζεται/επεκτείνεται εύκολα
STOP = frozenset(_STOP_WORDS.split())


def load_pages():
    """{(file, page, doc_id): κείμενο} — ίδιο μοτίβο με verify_keywords.py."""
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
    return {k: "\n".join(v) for k, v in pages.items()}


def terms_of(text: str):
    """Μόνο ΜΟΝΕΣ λέξεις, πεζά, χωρίς stopwords/αριθμούς/κοντές λέξεις.

    ΓΙΑΤΙ ΟΧΙ BIGRAMS: δοκιμάστηκαν και ήταν θόρυβος. Τα chunks ενώνονται ανά
    σελίδα, οπότε τα bigrams πηδούν πάνω από όρια προτάσεων και παράγουν
    ψευδο-όρους («computing business-continuity», «encompass applicationspecific»)
    που είναι σπάνιοι επειδή είναι ΤΥΧΑΙΟΙ, όχι επειδή είναι διακριτικοί.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", text.lower())
    return {w for w in words if w not in STOP}


def p_found(f, n, k):
    if f <= 0:
        return 0.0
    if f >= n or k >= n:
        return 1.0
    return 1.0 - math.comb(n - f, k) / math.comb(n, k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--max-df", type=int, default=4,
                    help="μέγιστες σελίδες όπου επιτρέπεται να εμφανίζεται")
    ap.add_argument("--top", type=int, default=14)
    args = ap.parse_args()

    pages = load_pages()
    n = len(pages)
    page_terms = {k: terms_of(v) for k, v in pages.items()}
    df = Counter()
    for ts in page_terms.values():
        df.update(ts)

    tests = []
    with open(args.golden, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                if not args.ids or t["id"] in args.ids:
                    tests.append(t)

    print(f"Corpus: {n} σελίδες · σύστημα επιστρέφει {args.pages} · "
          f"προτείνονται όροι σε <= {args.max_df} σελίδες\n")

    for t in tests:
        kws = [k.lower() for k in t["keywords"]]
        # Συχνότητα σελίδων ανά keyword — ωμό count, πιάνει και φράσεις.
        kw_df = {k: max(1, sum(1 for txt in pages.values() if k in txt.lower()))
                 for k in kws}
        # ΒΑΡΟΣ 1/df: ένα keyword σε 5 σελίδες «δείχνει» 8x πιο δυνατά από ένα
        # σε 40. Χωρίς αυτό, δεκάδες σελίδες βγαίνουν 3/3 επειδή τα κοινά
        # keywords είναι παντού — και η σελίδα-στόχος βγαίνει ΛΑΘΟΣ (μετρήθηκε:
        # για το h003 πρότεινε σελίδες περί semiconductor manufacturing).
        scored = []
        for key, txt in pages.items():
            low = txt.lower()
            w = sum(1.0 / kw_df[k] for k in kws if k in low)
            hits = sum(1 for k in kws if k in low)
            if w > 0:
                scored.append((w, hits, key))
        scored.sort(key=lambda x: -x[0])
        best_hits = scored[0][1] if scored else 0
        targets = [k for _w, _h, k in scored[:3]]

        print("=" * 78)
        print(f"{t['id']} · {t.get('category', '')} · parent={t.get('parent', '?')}")
        print(f"  {t['question'][:74]}")
        print("  ΤΩΡΑ:")
        for k in t["keywords"]:
            f_ = df.get(k.lower(), 0)
            if not f_:  # φράση που δεν πιάνεται από terms_of -> μέτρα ωμά
                f_ = sum(1 for txt in pages.values() if k.lower() in txt.lower())
            p = p_found(f_, n, args.pages)
            flag = "  <-- ΑΝΤΙΚΑΤΕΣΤΗΣΕ" if p >= 0.60 else ""
            print(f"    {k:<16} {f_:>3}/{n} σελ  {100*p:>5.1f}% τυχαία{flag}")

        print(f"  ΣΕΛΙΔΑ-ΣΤΟΧΟΣ ({best_hits}/{len(kws)} keywords): "
              + " · ".join(f"{a}:{b}" for a, b, _c in targets))

        cands = set()
        for key in targets:
            cands |= page_terms[key]
        rare = sorted(((df[c], c) for c in cands if 0 < df[c] <= args.max_df),
                      key=lambda x: (x[0], -len(x[1])))
        print("  ΠΡΟΤΑΣΕΙΣ (σπάνιοι όροι από τη σελίδα-στόχο):")
        for f_, c in rare[:args.top]:
            print(f"    {c:<34} {f_:>3}/{n} σελ  {100*p_found(f_, n, args.pages):>5.1f}% τυχαία")
        print()

    print("ΔΙΑΛΕΞΕ ΜΕ ΤΟ ΧΕΡΙ: ο όρος πρέπει να ΑΠΑΝΤΑΕΙ στην ερώτηση.")
    print("Σπάνιο keyword άσχετο με την ερώτηση = χειρότερο από κοινό.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
