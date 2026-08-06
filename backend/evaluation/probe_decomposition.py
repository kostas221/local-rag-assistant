"""Βοηθά το query decomposition στα multi_hop; — ΠΡΙΝ γραφτεί ο agent.

ΤΟ ΜΕΤΡΗΜΕΝΟ ΠΡΟΒΛΗΜΑ:
multi_hop coverage 78.8% — η ΜΟΝΗ κατηγορία κάτω από 98%. Οι ερωτήσεις αυτές
χρειάζονται υλικό από ΔΥΟ papers, αλλά το σύστημα ψάχνει ΜΙΑ φορά με ΕΝΑ
ερώτημα: βρίσκει καλά το ένα έγγραφο και χάνει το άλλο.

Η ΥΠΟΘΕΣΗ: σπάσε την ερώτηση σε ανεξάρτητα υπο-ερωτήματα, ψάξε ΧΩΡΙΣΤΑ για το
καθένα, ένωσε τις σελίδες -> κάθε σκέλος εγγυημένα συνεισφέρει.

ΤΙ ΜΕΤΡΑΕΙ (read-only, ΜΗΔΕΝ αλλαγή στο ai_core.py):
  1. coverage baseline vs decomposition, ανά ερώτηση (page-level, όπως το
     eval_engine — γιατί το search_documents τελειώνει με _expand_to_pages)
  2. ROUTING: το μοντέλο καλείται να αποφασίσει ΜΟΝΟ ΤΟΥ αν χρειάζεται σπάσιμο.
     Έχουμε ground truth (το πεδίο category) -> μετριέται πόσο σωστά ξεχωρίζει
     multi_hop από τα υπόλοιπα. ΚΡΙΣΙΜΟ: αν το routing είναι κακό, το
     decomposition θα τρέχει σε ερωτήσεις που δεν το χρειάζονται και θα
     πληρώνουν όλες +0.5s χωρίς όφελος.
  3. Χρόνος ανά ερώτηση.

ΣΤΡΑΤΗΓΙΚΗ ΚΟΣΤΟΥΣ (το quota είναι περιορισμένο):
τρέξε ΠΡΩΤΑ `--categories multi_hop` (15 ερωτήσεις = 15 κλήσεις). Αν το coverage
δεν ανεβαίνει, σταμάτα εκεί — δεν χρειάζεται να μετρηθούν οι παρενέργειες μιας
αλλαγής που δεν ωφελεί.

    docker compose exec backend python evaluation/probe_decomposition.py --categories multi_hop
    docker compose exec backend python evaluation/probe_decomposition.py --limit 20 --csv evaluation/runs/decomp.csv
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import time

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core
import gemini_rest
from evaluation.eval_engine import GOLDEN_CORPUS

SETS = [
    "/app/evaluation/golden_multihop_new.jsonl",
    "/app/evaluation/golden_set_50.jsonl",
]

# Ο ΡΟΥΤΕΡ ΚΑΙ Ο ΔΙΑΣΠΑΣΤΗΣ ΣΕ ΕΝΑ PROMPT: μία κλήση αντί για δύο. Το μοντέλο
# επιστρέφει τη ΜΙΑ αρχική ερώτηση όταν δεν χρειάζεται σπάσιμο -> ο caller
# ξεχωρίζει τις περιπτώσεις από το πλήθος των γραμμών, χωρίς δεύτερο round-trip.
#
# «ανεξάρτητα» = καθένα να στέκει ΜΟΝΟ του σε αναζήτηση. Χωρίς αυτό το μοντέλο
# παράγει «and what about the second one?» — άχρηστο για retrieval.
DEFAULT_PROMPT = """\
You are preparing search queries for a corpus of computer-science papers on \
cloud computing, serverless computing and distributed systems.

Decide whether answering the question below requires looking up TWO DIFFERENT \
topics (typically described in two different papers), or just one.

- If ONE lookup is enough, output the question as a single search query, unchanged.
- If TWO are needed, output exactly two search queries, one per line. Each must \
stand ALONE as a search query: no pronouns, no "and what about...", no reference \
to the other line.

Use the standard technical terminology of the field. Do NOT add facts, system \
names or numbers that are not implied by the question.
Output ONLY the queries, one per line. No numbering, no quotes, no extra text.

Question: {query}
Queries:"""


def load(paths, categories=None):
    tests, seen = [], set()
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                t = json.loads(line)
                if t["id"] in seen or t.get("category") == "out_of_corpus":
                    continue
                if categories and t.get("category") not in categories:
                    continue
                seen.add(t["id"])
                tests.append(t)
    return tests


class Pipeline:
    """Το retrieval του search_documents, με το gate και το page expansion
    εκτεθειμένα ώστε να συνδυάζονται ελεύθερα."""

    def __init__(self):
        where = ai_core._build_where(GOLDEN_CORPUS, None)
        self.allowed = ai_core.collection.get(where=where, include=[])["ids"]
        self.idx = ai_core._get_bm25_index()
        self.dm = ai_core._get_dense_matrix()

    def rerank(self, query: str):
        """-> sorted_final [(score, text, meta)], χωρίς gate."""
        d_ids = ai_core._dense_exact_ids(
            self.dm, query, self.allowed,
            min(ai_core.DENSE_CANDIDATES, len(self.allowed)))
        s_ids = ai_core._bm25_sparse_ids(
            self.idx, query, self.allowed, ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(
            d_ids, s_ids, self.idx["ids"], self.idx["texts"], self.idx["metas"],
            k=60, top_n=ai_core.RERANK_CANDIDATES, pos=self.idx["pos"])
        pairs = [[query, it[1]] for it in rrf]
        scores = ai_core.reranker.predict(pairs, batch_size=ai_core.RERANK_BATCH_SIZE)
        return sorted(zip((float(x) for x in scores),
                          [it[1] for it in rrf], [it[2] for it in rrf]),
                      key=lambda x: x[0], reverse=True)

    def baseline(self, query: str):
        """Ακριβώς ό,τι κάνει σήμερα το search_documents."""
        sf = self.rerank(query)
        if sf[0][0] < ai_core.MIN_RERANK_SCORE:
            return []
        return ai_core._expand_to_pages(sf[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)

    def decomposed(self, subqueries: list):
        """Ξεχωριστό retrieval ανά υπο-ερώτημα, ΙΣΟΜΟΙΡΑΣΜΟΣ των σελίδων.

        ΓΙΑΤΙ ΙΣΟΜΟΙΡΑΣΜΟΣ και όχι ενιαία ταξινόμηση: το πρόβλημα των multi_hop
        είναι ΑΚΡΙΒΩΣ ότι το ένα έγγραφο κυριαρχεί. Αν ενώσουμε τα scores και
        πάρουμε τα top-8, το ισχυρό σκέλος θα ξαναπάρει και τις 8 θέσεις και δεν
        θα έχει αλλάξει τίποτα. Το budget ανά σκέλος είναι η ΟΥΣΙΑ της αλλαγής.

        Επιπλέον τα scores δύο ΔΙΑΦΟΡΕΤΙΚΩΝ ερωτημάτων δεν είναι στην ίδια
        κλίμακα — η σύγκρισή τους θα ήταν αυθαίρετη.
        """
        budget = max(1, ai_core.MAX_PAGES // len(subqueries))
        pages, seen = [], set()
        any_passed = False
        for q in subqueries:
            sf = self.rerank(q)
            # Gate ΑΝΑ ΣΚΕΛΟΣ: ένα σκέλος που δεν βρίσκει τίποτα δεν πρέπει να
            # μολύνει το αποτέλεσμα με άσχετες σελίδες.
            if sf[0][0] < ai_core.MIN_RERANK_SCORE:
                continue
            any_passed = True
            for text, meta in ai_core._expand_to_pages(
                    sf[:ai_core.EXPAND_INPUT], budget):
                key = (meta.get("file_name"), meta.get("page"))
                if key not in seen:
                    seen.add(key)
                    pages.append((text, meta))
        # Κανένα σκέλος δεν πέρασε -> σιωπή, όπως και σήμερα.
        return pages if any_passed else []


def covered(pages, kws):
    blob = "\n".join(t for t, _m in pages).lower()
    return sum(1 for k in kws if k.lower() in blob)


def docs_of(pages):
    return {m.get("file_name") for _t, m in pages}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    prompt_tpl = DEFAULT_PROMPT
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt_tpl = f.read()
    if "{query}" not in prompt_tpl:
        print("ΣΦΑΛΜΑ: το prompt δεν περιέχει {query}")
        return 1

    tests = load(SETS, args.categories)
    if args.limit:
        tests = tests[:args.limit]
    pipe = Pipeline()
    print(f"{len(tests)} ερωτήσεις · {len(tests)} κλήσεις Gemini "
          f"(1 ανά ερώτηση, decompose+route μαζί)\n")

    print(f"{'id':<6} {'category':<13} {'n':>2} {'base':>10} {'decomp':>10} "
          f"{'docs':>7} {'ms':>6}  {'έκβαση'}")
    print("-" * 82)

    rows = []
    for t in tests:
        q = await ai_core.optimize_query(t["question"])
        base = pipe.baseline(q)
        n_kw = len(t["keywords"])
        cov_b = covered(base, t["keywords"])

        t0 = time.perf_counter()
        try:
            raw = await gemini_rest.generate_once(
                prompt_tpl.format(query=q), model=ai_core.GEMINI_MODEL,
                api_key=ai_core.GEMINI_API_KEY)
        except Exception as e:
            print(f"{t['id']:<6} decompose ΑΠΕΤΥΧΕ: {e}")
            continue
        subs = [ln.strip(" -•\t") for ln in raw.strip().split("\n") if ln.strip()]
        subs = [s for s in subs if len(s) > 3][:3]
        if not subs:
            subs = [q]
        dec = pipe.decomposed(subs) if len(subs) > 1 else base
        ms = (time.perf_counter() - t0) * 1000
        cov_d = covered(dec, t["keywords"])

        is_mh = t.get("category") == "multi_hop"
        split = len(subs) > 1
        if cov_d > cov_b:
            outcome = "ΚΑΛΥΤΕΡΑ"
        elif cov_d < cov_b:
            outcome = "ΧΕΙΡΟΤΕΡΑ"
        elif split:
            outcome = "ίδιο (έσπασε)"
        else:
            outcome = "ίδιο"

        rows.append(dict(id=t["id"], category=t.get("category", ""),
                         n_sub=len(subs), split=split, is_multihop=is_mh,
                         cov_base=cov_b, cov_dec=cov_d, n_kw=n_kw,
                         docs_base=len(docs_of(base)), docs_dec=len(docs_of(dec)),
                         pages_base=len(base), pages_dec=len(dec),
                         ms=round(ms), subqueries=" || ".join(subs)))
        print(f"{t['id']:<6} {t.get('category',''):<13} {len(subs):>2} "
              f"{f'{cov_b}/{n_kw}':>10} {f'{cov_d}/{n_kw}':>10} "
              f"{f'{len(docs_of(base))}->{len(docs_of(dec))}':>7} {ms:>6.0f}  {outcome}",
              flush=True)

    if not rows:
        return 1

    mh = [r for r in rows if r["is_multihop"]]
    other = [r for r in rows if not r["is_multihop"]]

    def cov(group):
        tot = sum(r["n_kw"] for r in group)
        return (100 * sum(r["cov_base"] for r in group) / tot,
                100 * sum(r["cov_dec"] for r in group) / tot) if tot else (0, 0)

    print("\n" + "=" * 82)
    print("COVERAGE")
    print("=" * 82)
    for label, grp in (("multi_hop", mh), ("υπόλοιπες", other), ("ΣΥΝΟΛΟ", rows)):
        if not grp:
            continue
        b, d = cov(grp)
        arrow = "ΚΑΛΥΤΕΡΑ" if d > b + 0.05 else ("ΧΕΙΡΟΤΕΡΑ" if d < b - 0.05 else "ίδιο")
        print(f"  {label:<12} n={len(grp):<3} {b:>6.1f}% -> {d:>6.1f}%   ({arrow})")

    print("\n" + "=" * 82)
    print("ROUTING — αποφάσισε σωστά το μοντέλο πότε να σπάσει;")
    print("=" * 82)
    tp = sum(1 for r in mh if r["split"])
    fn = len(mh) - tp
    fp = sum(1 for r in other if r["split"])
    tn = len(other) - fp
    print(f"  multi_hop που ΕΣΠΑΣΕ:        {tp}/{len(mh)}"
          + (f"  (έχασε {fn})" if fn else ""))
    if other:
        print(f"  υπόλοιπες που ΕΣΠΑΣΕ:        {fp}/{len(other)}"
              f"  <- κάθε μία πληρώνει χρόνο χωρίς λόγο")
        print(f"  υπόλοιπες που ΔΕΝ έσπασε:    {tn}/{len(other)}")

    print("\n  ΤΙΜΗΜΑ ΤΟΥ ROUTING: η κλήση decompose γίνεται σε ΚΑΘΕ ερώτηση,")
    print("  ακόμα κι όταν η απόφαση είναι «μην σπάσεις». Διάμεσος χρόνος: "
          f"{sorted(r['ms'] for r in rows)[len(rows)//2]:.0f} ms")
    print("  (ο corrective agent, αντίθετα, κοστίζει ΜΟΝΟ όταν το gate έχει κόψει)")

    better = [r for r in rows if r["cov_dec"] > r["cov_base"]]
    worse = [r for r in rows if r["cov_dec"] < r["cov_base"]]
    print(f"\n  ΚΑΛΥΤΕΡΑ: {len(better)}" + (f" ({', '.join(r['id'] for r in better)})" if better else ""))
    print(f"  ΧΕΙΡΟΤΕΡΑ: {len(worse)}" + (f" ({', '.join(r['id'] for r in worse)})" if worse else ""))

    if better or worse:
        print("\n  ΤΙ ΠΑΡΗΓΑΓΕ ΤΟ ΣΠΑΣΙΜΟ:")
        for r in (better + worse):
            print(f"    {r['id']} [{r['cov_base']}/{r['n_kw']} -> {r['cov_dec']}/{r['n_kw']}] "
                  f"έγγραφα {r['docs_base']}->{r['docs_dec']}")
            for s in r["subqueries"].split(" || "):
                print(f"       · {s}")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
