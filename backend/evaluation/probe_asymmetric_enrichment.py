"""ΑΣΥΜΜΕΤΡΟΣ εμπλουτισμός: πλούσιο query στην ΑΝΑΖΗΤΗΣΗ, αρχικό στον ΚΡΙΤΗ.

ΤΟ ΕΥΡΗΜΑ ΠΟΥ ΤΟ ΓΕΝΝΗΣΕ (probe_query_enrichment v2, 10/8/2026):
Ο εμπλουτισμός του ερωτήματος έδωσε ταυτόχρονα το καλύτερο και το χειρότερο:
  ΚΕΡΔΟΣ  h002 coverage 0% -> 100% (πρόσθεσε «vpxenc» — ΑΚΡΙΒΩΣ τη λέξη που το
          trace_query.py είχε δείξει ότι λείπει), +4 ακόμα ερωτήσεις σε 100%
  ΖΗΜΙΑ   q048 «GDPR ... data privacy»: -3.12 -> -2.00, ΠΕΡΑΣΕ το gate

Τα δύο συμβαίνουν σε ΔΙΑΦΟΡΕΤΙΚΑ σημεία του pipeline:
  · Το BM25 ΧΡΕΙΑΖΕΤΑΙ τις έξτρα λέξεις — χωρίς το «vpxenc» η σελίδα δεν έρχεται.
  · Ο reranker ΠΑΡΑΠΛΑΝΑΤΑΙ από αυτές — και το gate διαβάζει τον reranker.

Άρα δεν είναι ανάγκη να δεχτούμε και τα δύο. Εδώ:
      dense + BM25   ->  ΕΜΠΛΟΥΤΙΣΜΕΝΟ query   (ψάξε ευρύτερα)
      reranker/gate  ->  ΑΡΧΙΚΟ query          (κρίνε ό,τι ΠΡΑΓΜΑΤΙΚΑ ρωτήθηκε)

Η αρχή είναι η ίδια με το CORRECTIVE_MIN_SCORE: όταν το ΣΥΣΤΗΜΑ γράφει το
ερώτημα (και όχι ο χρήστης), δεν του επιτρέπεις να πείσει και τον κριτή. Αλλιώς
το keyword stuffing παρακάμπτει την άμυνα — μετρημένο δύο φορές ήδη.

ΜΗΔΕΝ ΚΟΣΤΟΣ API: διαβάζει τα ΗΔΗ παραγμένα enriched queries από το CSV του
προηγούμενου probe. Ίδιες είσοδοι -> η σύγκριση είναι καθαρή.

    docker compose exec backend python evaluation/probe_asymmetric_enrichment.py \\
        --in evaluation/runs/enrich_v2.csv --csv evaluation/runs/asym.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core


def retrieve(search_q, judge_q, idx, dm, allowed_ids):
    """search_q -> dense/BM25/RRF.  judge_q -> reranker + gate."""
    dense_ids = ai_core._dense_exact_ids(
        dm, search_q, allowed_ids,
        min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
    sparse_ids = ai_core._bm25_sparse_ids(
        idx, search_q, allowed_ids, ai_core.DENSE_CANDIDATES)
    rrf = ai_core._rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                            idx["metas"], k=60,
                            top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
    # ΕΔΩ ΕΙΝΑΙ ΟΛΗ Η ΔΙΑΦΟΡΑ: ο cross-encoder βλέπει το judge_q.
    scores = ai_core.reranker.predict([[judge_q, it[1]] for it in rrf],
                                      batch_size=ai_core.RERANK_BATCH_SIZE)
    ranked = sorted(zip(scores, [it[1] for it in rrf], [it[2] for it in rrf]),
                    key=lambda x: x[0], reverse=True)
    best = float(ranked[0][0])
    if best < ai_core.MIN_RERANK_SCORE:
        return best, []
    return best, ai_core._expand_to_pages(ranked[:ai_core.EXPAND_INPUT],
                                          ai_core.MAX_PAGES)


def coverage(pages, keywords):
    if not pages or not keywords:
        return 0.0
    blob = "\n".join(t for t, _m in pages).lower()
    return 100.0 * sum(1 for k in keywords if k.lower() in blob) / len(keywords)


def load_keywords():
    """Τα keywords ζουν στα golden sets, όχι στο CSV του probe."""
    import json
    out = {}
    for name in ("golden_hard_paraphrase.jsonl", "golden_set_50.jsonl"):
        p = f"/app/evaluation/{name}"
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    t = json.loads(ln)
                    out[t["id"]] = t.get("keywords", [])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src",
                    default="/app/evaluation/runs/enrich_v2.csv")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kw = load_keywords()
    print(f"{len(rows)} ερωτήσεις από {args.src} (ΜΗΔΕΝ κλήσεις Gemini)\n")

    allowed_ids = ai_core.collection.get(
        where=ai_core._build_where(None, None), include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    out, tally = [], {"saved": [], "lost": [], "leak": [], "cov_up": []}
    print(f"{'id':<7}{'best':>8}{'cov':>7}   {'ετυμηγορία'}")
    print("-" * 62)
    for r in rows:
        q, eq = r["question"], r["enriched"]
        keys = kw.get(r["id"], [])
        best, pages = retrieve(eq, q, idx, dm, allowed_ids)   # <-- ασύμμετρο
        cov = coverage(pages, keys)
        b0, c0 = float(r["best_before"]), float(r["cov_before"])
        is_ooc = r["category"] == "out_of_corpus"

        v = "ίδιο"
        if is_ooc:
            if pages:
                v = "*** ΔΙΑΡΡΟΗ ***"
                tally["leak"].append(r["id"])
        elif c0 == 0 and cov > 0:
            v = "ΣΩΘΗΚΕ"
            tally["saved"].append(r["id"])
        elif c0 > 0 and cov == 0:
            v = "*** ΧΑΘΗΚΕ ***"
            tally["lost"].append(r["id"])
        elif cov > c0:
            v = f"coverage {c0:.0f}->{cov:.0f}"
            tally["cov_up"].append(r["id"])
        print(f"{r['id']:<7}{b0:>7.2f}->{best:<7.2f}{c0:>4.0f}->{cov:<4.0f} {v}")
        out.append(dict(id=r["id"], category=r["category"],
                        best_baseline=b0, best_asym=round(best, 3),
                        cov_baseline=c0, cov_asym=round(cov, 1),
                        best_symmetric=r["best_after"],
                        cov_symmetric=r["cov_after"], verdict=v))

    print("=" * 62)
    print(f"ΣΩΘΗΚΑΝ        {len(tally['saved']):>2}  {tally['saved']}")
    print(f"coverage πάνω  {len(tally['cov_up']):>2}  {tally['cov_up']}")
    print(f"ΧΑΘΗΚΑΝ        {len(tally['lost']):>2}  {tally['lost']}")
    print(f"ΔΙΑΡΡΟΕΣ ooc   {len(tally['leak']):>2}  {tally['leak']}")
    print("=" * 62)
    print("ΣΥΓΚΡΙΣΗ ΜΕ ΤΟΝ ΣΥΜΜΕΤΡΙΚΟ (enrich_v2): εκείνος έδωσε 1 ΔΙΑΡΡΟΗ (q048).")
    if tally["leak"]:
        print("-> Η ασυμμετρία ΔΕΝ έλυσε το πρόβλημα: ΑΠΟΡΡΙΠΤΕΤΑΙ οριστικά.")
    elif tally["saved"] or tally["cov_up"]:
        print("-> ΜΗΔΕΝ διαρροές ΚΑΙ κέρδος. ΕΠΟΜΕΝΟ, ΥΠΟΧΡΕΩΤΙΚΑ:")
        print("   1. πλήρες golden_set_50 (61/61 gate, MRR 0.793 να ΜΗΝ πέσει)")
        print("   2. κόστος: +1 κλήση Gemini ΣΕ ΚΑΘΕ ερώτηση, +0.5-0.9 s")
        print("      Αυτό ΜΟΝΟ του απέρριψε το query decomposition.")
    else:
        print("-> Καθαρό αλλά ΧΩΡΙΣ κέρδος: το πλούσιο query δεν περνά το RRF.")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
