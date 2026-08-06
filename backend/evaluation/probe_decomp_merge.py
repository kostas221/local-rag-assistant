"""Φταίει το decomposition ή ο τρόπος συγχώνευσης; — ΜΗΔΕΝ κόστος API.

Το probe_decomposition.py μέτρησε coverage 79.5% -> 68.2% με ισομοιρασμό
σελίδων (4+4). Πριν απορριφθεί η ιδέα, πρέπει να ξεχωρίσει η ΥΠΟΘΕΣΗ από την
ΥΛΟΠΟΙΗΣΗ: μήπως τα υπο-ερωτήματα ήταν καλά και απλώς το budget ανά σκέλος
πέταξε χρήσιμες σελίδες;

Τα υπο-ερωτήματα είναι ΗΔΗ στο CSV -> καμία νέα κλήση Gemini. Δοκιμάζονται
τρεις στρατηγικές συγχώνευσης πάνω στα ΙΔΙΑ ακριβώς queries:

  A  ισομοιρασμός        MAX_PAGES//n ανά σκέλος            (ό,τι μετρήθηκε)
  B  baseline + συμπλήρωμα  όλες οι 8 του αρχικού + έως 4 νέες από τα σκέλη
                            -> ΥΠΕΡΣΥΝΟΛΟ του σημερινού· το coverage ΔΕΝ μπορεί
                            να πέσει. Τίμημα: ~+50% prompt tokens
  C  ενιαία ταξινόμηση     ένωση όλων των chunks, rerank με το ΑΡΧΙΚΟ ερώτημα,
                            top-EXPAND_INPUT -> expand. Κοινή κλίμακα σκορ

Αν ΚΑΜΙΑ δεν ξεπερνά το baseline με αποδεκτό κόστος, η απόρριψη αφορά την ίδια
την ιδέα, όχι την υλοποίηση — και αυτό είναι που πρέπει να καταγραφεί.

    docker compose exec backend python evaluation/probe_decomp_merge.py
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")
import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

SETS = [
    "/app/evaluation/golden_multihop_new.jsonl",
    "/app/evaluation/golden_set_50.jsonl",
]


def load_tests():
    out = {}
    for p in SETS:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    out.setdefault(t["id"], t)
    return out


class Pipeline:
    def __init__(self):
        where = ai_core._build_where(GOLDEN_CORPUS, None)
        self.allowed = ai_core.collection.get(where=where, include=[])["ids"]
        self.idx = ai_core._get_bm25_index()
        self.dm = ai_core._get_dense_matrix()

    def candidates(self, query: str):
        """-> τα RRF υποψήφια ΠΡΙΝ το rerank (για τη στρατηγική C)."""
        d_ids = ai_core._dense_exact_ids(
            self.dm, query, self.allowed,
            min(ai_core.DENSE_CANDIDATES, len(self.allowed)))
        s_ids = ai_core._bm25_sparse_ids(
            self.idx, query, self.allowed, ai_core.DENSE_CANDIDATES)
        return ai_core._rrf_fuse(
            d_ids, s_ids, self.idx["ids"], self.idx["texts"], self.idx["metas"],
            k=60, top_n=ai_core.RERANK_CANDIDATES, pos=self.idx["pos"])

    def rerank_pairs(self, query: str, items):
        pairs = [[query, it[1]] for it in items]
        scores = ai_core.reranker.predict(pairs, batch_size=ai_core.RERANK_BATCH_SIZE)
        return sorted(zip((float(x) for x in scores),
                          [it[1] for it in items], [it[2] for it in items]),
                      key=lambda x: x[0], reverse=True)

    def rerank(self, query: str):
        return self.rerank_pairs(query, self.candidates(query))

    def baseline(self, query: str):
        sf = self.rerank(query)
        if sf[0][0] < ai_core.MIN_RERANK_SCORE:
            return []
        return ai_core._expand_to_pages(sf[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)

    def strategy_a(self, subs):
        budget = max(1, ai_core.MAX_PAGES // len(subs))
        pages, seen, ok = [], set(), False
        for q in subs:
            sf = self.rerank(q)
            if sf[0][0] < ai_core.MIN_RERANK_SCORE:
                continue
            ok = True
            for text, meta in ai_core._expand_to_pages(
                    sf[:ai_core.EXPAND_INPUT], budget):
                k = (meta.get("file_name"), meta.get("page"))
                if k not in seen:
                    seen.add(k)
                    pages.append((text, meta))
        return pages if ok else []

    def strategy_b(self, orig, subs, extra=4):
        """Κρατά ΟΛΟ το σημερινό αποτέλεσμα και προσθέτει ό,τι νέο φέρνουν τα
        σκέλη. Υπερσύνολο -> το coverage δεν μπορεί να πέσει, μόνο το
        context_precision (και τα tokens)."""
        base = self.baseline(orig)
        if not base:
            return []
        pages = list(base)
        seen = {(m.get("file_name"), m.get("page")) for _t, m in base}
        added = 0
        for q in subs:
            if added >= extra:
                break
            sf = self.rerank(q)
            if sf[0][0] < ai_core.MIN_RERANK_SCORE:
                continue
            for text, meta in ai_core._expand_to_pages(
                    sf[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES):
                k = (meta.get("file_name"), meta.get("page"))
                if k not in seen:
                    seen.add(k)
                    pages.append((text, meta))
                    added += 1
                    if added >= extra:
                        break
        return pages

    def strategy_c(self, orig, subs):
        """Ένωση υποψηφίων από όλα τα σκέλη, ΕΝΑ rerank με το ΑΡΧΙΚΟ ερώτημα ->
        κοινή κλίμακα σκορ, χωρίς αυθαίρετη σύγκριση μεταξύ queries."""
        pool, seen = [], set()
        for q in [orig, *subs]:
            for it in self.candidates(q):
                if it[1] not in seen:
                    seen.add(it[1])
                    pool.append(it)
        sf = self.rerank_pairs(orig, pool)
        if sf[0][0] < ai_core.MIN_RERANK_SCORE:
            return []
        return ai_core._expand_to_pages(sf[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)


def covered(pages, kws):
    blob = "\n".join(t for t, _m in pages).lower()
    return sum(1 for k in kws if k.lower() in blob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-in", default="/app/evaluation/runs/decomp_mh.csv")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.csv_in):
        print(f"Δεν βρέθηκε: {args.csv_in}")
        return 1
    with open(args.csv_in, encoding="utf-8") as f:
        prev = list(csv.DictReader(f))
    tests = load_tests()
    pipe = Pipeline()
    print(f"{len(prev)} ερωτήσεις · ΜΗΔΕΝ κλήσεις API "
          f"(τα υπο-ερωτήματα διαβάζονται από το CSV)\n")

    print(f"{'id':<6} {'kw':>3} {'base':>5} {'A':>5} {'B':>5} {'C':>5} "
          f"{'σελ base':>9} {'σελ B':>6}")
    print("-" * 60)

    rows = []
    for r in prev:
        t = tests.get(r["id"])
        if not t:
            continue
        subs = [s for s in r["subqueries"].split(" || ") if s.strip()]
        orig = subs[0] if len(subs) == 1 else None
        # Το αρχικό (μεταφρασμένο) ερώτημα: από το cache, μηδέν κλήση.
        q = ai_core._translation_cache.get(t["question"], t["question"])
        if orig:
            q = orig

        kws = t["keywords"]
        base = pipe.baseline(q)
        a = pipe.strategy_a(subs) if len(subs) > 1 else base
        b = pipe.strategy_b(q, subs) if len(subs) > 1 else base
        c = pipe.strategy_c(q, subs) if len(subs) > 1 else base

        row = dict(id=r["id"], n_kw=len(kws), base=covered(base, kws),
                   a=covered(a, kws), b=covered(b, kws), c=covered(c, kws),
                   pages_base=len(base), pages_a=len(a), pages_b=len(b),
                   pages_c=len(c))
        rows.append(row)
        print(f"{r['id']:<6} {len(kws):>3} {row['base']:>5} {row['a']:>5} "
              f"{row['b']:>5} {row['c']:>5} {len(base):>9} {len(b):>6}", flush=True)

    tot = sum(r["n_kw"] for r in rows)
    print("\n" + "=" * 60)
    print("COVERAGE (multi_hop)")
    print("=" * 60)
    base_pct = 100 * sum(r["base"] for r in rows) / tot
    for key, label, note in (
            ("a", "A ισομοιρασμός", "4+4 σελίδες"),
            ("b", "B baseline+συμπλ.", "έως 12 σελίδες"),
            ("c", "C ενιαίο rerank", "8 σελίδες")):
        pct = 100 * sum(r[key] for r in rows) / tot
        d = pct - base_pct
        verdict = "ΚΑΛΥΤΕΡΑ" if d > 0.05 else ("ΧΕΙΡΟΤΕΡΑ" if d < -0.05 else "ίδιο")
        print(f"  {label:<20} {pct:>6.1f}%  ({d:+.1f})  {verdict:<10} {note}")
    print(f"  {'baseline (σήμερα)':<20} {base_pct:>6.1f}%")

    avg_pages = {k: sum(r[f"pages_{k}"] for r in rows) / len(rows)
                 for k in ("base", "a", "b", "c")}
    print(f"\n  μέσες σελίδες: base {avg_pages['base']:.1f} · A {avg_pages['a']:.1f} "
          f"· B {avg_pages['b']:.1f} · C {avg_pages['c']:.1f}")
    print(f"  (κάθε σελίδα ≈ 1.200 tokens prompt -> η B κοστίζει "
          f"~{(avg_pages['b']-avg_pages['base'])*1200:.0f} επιπλέον tokens/ερώτηση)")

    if args.csv and rows:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
