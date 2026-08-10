"""Σε ΠΟΙΟ στάδιο χάνεται μια σελίδα; — βήμα-βήμα ίχνος μιας ερώτησης.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Το q028 ρωτά για τα vpxenc flags του ExCamera. Η σελίδα excamera-nsdi17.pdf:15
περιέχει ΑΥΤΟΥΣΙΑ και το `vpxenc` (μοναδικό σε όλο το corpus) και το
`--undershoot-pct=100` (keyword του golden set). Και όμως το MRR είναι 0.000:
η σελίδα ΔΕΝ φτάνει ποτέ στο τελικό αποτέλεσμα. Το judge έδωσε 5/5/5 — δηλαδή
το Gemini απάντησε από ΠΑΡΑΜΕΤΡΙΚΗ ΓΝΩΣΗ, χωρίς το υλικό.

Κανένα υπάρχον εργαλείο δεν δείχνει ΠΟΥ χάνεται. Το measure_gate_margin δίνει
μόνο το τελικό best-logit, το compare_pages μόνο τις τελικές σελίδες. Εδώ
βλέπουμε κάθε στάδιο ξεχωριστά και σε ποια θέση βρίσκεται η σελίδα-στόχος.

ΤΑ ΤΕΣΣΕΡΑ ΥΠΟΠΤΑ, με τη σειρά που τα ελέγχει το script:
  0. Η ΜΕΤΑΦΡΑΣΗ. Το optimize_query ξαναγράφει την ερώτηση ΠΡΙΝ την αναζήτηση.
     Αν το "vpxenc" γίνει "video encoder settings", ο μοναδικός σπάνιος όρος
     χάνεται και το BM25 τυφλώνεται. Το script τυπώνει ΚΑΙ τα δύο ερωτήματα.
  1. DENSE. Σελίδα με σκέτα CLI flags δεν έχει σημασιολογία φυσικής γλώσσας.
  2. BM25. Ο tokenizer είναι re.findall(r"\\w+") -> το `--undershoot-pct=100`
     γίνεται ['undershoot','pct','100']. Το `vpxenc` όμως επιβιώνει ακέραιο.
  3. RRF / RERANKER / GATE. Ακόμα κι αν έρθει υποψήφια, μπορεί να θαφτεί.

    docker compose exec backend python evaluation/trace_query.py --id q028
    docker compose exec backend python evaluation/trace_query.py \\
        --question "..." --target excamera-nsdi17.pdf:15
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core

HERE = "/app/evaluation"
SETS = ["golden_set_50.jsonl", "golden_multihop_new.jsonl",
        "golden_hard_paraphrase.jsonl"]


def find_question(qid: str):
    for name in SETS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                t = json.loads(line)
                if t.get("id") == qid:
                    return t, name
    return None, None


def _where(meta, target):
    """Είναι αυτό το chunk η σελίδα-στόχος;"""
    if not target:
        return False
    f, _, p = target.partition(":")
    return (str(meta.get("file_name")) == f
            and (not p or str(meta.get("page")) == p))


def show(title, ids, idx, target, limit=10):
    """Τυπώνει τα top-N ενός σκέλους και ΠΟΥ βρίσκεται η σελίδα-στόχος."""
    pos, metas = idx["pos"], idx["metas"]
    print(f"\n--- {title}  ({len(ids)} υποψήφιοι) ---")
    hit = None
    for rank, cid in enumerate(ids, start=1):
        m = metas[pos[cid]]
        if _where(m, target) and hit is None:
            hit = rank
        if rank <= limit:
            mark = "  <== ΣΤΟΧΟΣ" if _where(m, target) else ""
            print(f"  {rank:>3}. {m.get('file_name'):<26} σελ.{m.get('page'):<4}{mark}")
    if target:
        print(f"  ΣΤΟΧΟΣ {target}: "
              + (f"θέση {hit}" if hit else "*** ΔΕΝ ΕΙΝΑΙ ΜΕΣΑ ***"))
    return hit


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="id από τα golden sets, π.χ. q028")
    ap.add_argument("--question", help="ελεύθερη ερώτηση")
    ap.add_argument("--target", help="file.pdf:page που παρακολουθούμε")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    raw = args.question
    if args.id:
        t, src = find_question(args.id)
        if not t:
            print(f"*** Δεν βρέθηκε {args.id} στα {SETS}")
            return 1
        raw = t["question"]
        print(f"[{args.id}] από {src}")
        print(f"keywords: {t.get('keywords')}")
    if not raw:
        print("*** Δώσε --id ή --question")
        return 1

    print("=" * 74)
    print(f"ΕΡΩΤΗΣΗ (raw): {raw}")

    # --- 0. Μετάφραση/εμπλουτισμός -------------------------------------
    query = await ai_core.optimize_query(raw)
    print(f"ΜΕΤΑ optimize_query: {query}")
    if query.strip() != raw.strip():
        lost = [w for w in raw.split() if len(w) > 4 and w.lower() not in query.lower()]
        if lost:
            print(f"*** ΛΕΞΕΙΣ ΠΟΥ ΧΑΘΗΚΑΝ: {lost}")
            print("    (κάθε σπάνιος όρος που χάνεται εδώ ΤΥΦΛΩΝΕΙ το BM25)")
    print("=" * 74)

    where_filter = ai_core._build_where(None, None)
    allowed_ids = ai_core.collection.get(where=where_filter, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    # --- 1/2. Τα δύο σκέλη ----------------------------------------------
    dense_ids = ai_core._dense_exact_ids(
        dm, query, allowed_ids, min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
    sparse_ids = ai_core._bm25_sparse_ids(
        idx, query, allowed_ids, ai_core.DENSE_CANDIDATES)

    show(f"DENSE (bge-m3, top-{ai_core.DENSE_CANDIDATES})",
         dense_ids, idx, args.target, args.limit)
    show(f"BM25 (top-{ai_core.DENSE_CANDIDATES})",
         sparse_ids, idx, args.target, args.limit)

    # --- 3. RRF ----------------------------------------------------------
    rrf = ai_core._rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                            idx["metas"], k=60,
                            top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
    print(f"\n--- RRF -> top-{ai_core.RERANK_CANDIDATES} ---")
    hit = None
    for rank, item in enumerate(rrf, start=1):
        m = item[2]
        if _where(m, args.target):
            hit = rank
        print(f"  {rank:>3}. {m.get('file_name'):<26} σελ.{m.get('page'):<4}"
              + ("  <== ΣΤΟΧΟΣ" if _where(m, args.target) else ""))
    if args.target:
        print("  ΣΤΟΧΟΣ: " + (f"θέση {hit}" if hit else "*** ΚΟΠΗΚΕ ΕΔΩ ***"))

    # --- 4. Reranker + gate ---------------------------------------------
    scores = ai_core.reranker.predict([[query, it[1]] for it in rrf],
                                      batch_size=ai_core.RERANK_BATCH_SIZE)
    ranked = sorted(zip(scores, [it[1] for it in rrf], [it[2] for it in rrf]),
                    key=lambda x: x[0], reverse=True)
    print(f"\n--- RERANKER (ωμά logits· gate = {ai_core.MIN_RERANK_SCORE}) ---")
    for rank, (sc, _txt, m) in enumerate(ranked, start=1):
        flag = "  <== ΣΤΟΧΟΣ" if _where(m, args.target) else ""
        cut = "" if rank > 1 else ("   [GATE: ΠΕΡΝΑΕΙ]"
                                   if sc >= ai_core.MIN_RERANK_SCORE
                                   else "   [GATE: ΚΟΒΕΙ]")
        print(f"  {rank:>3}. {sc:>7.2f}  {m.get('file_name'):<26} "
              f"σελ.{m.get('page'):<4}{flag}{cut}")

    # --- 5. Τελικές σελίδες προς το Gemini ------------------------------
    pages = ai_core._expand_to_pages(ranked[:ai_core.EXPAND_INPUT],
                                     ai_core.MAX_PAGES)
    print(f"\n--- ΤΕΛΙΚΕΣ ΣΕΛΙΔΕΣ ΣΤΟ PROMPT ({len(pages)}) ---")
    found = False
    for _txt, m in pages:
        star = ""
        if _where(m, args.target):
            found = True
            star = "  <== ΣΤΟΧΟΣ"
        print(f"  {m.get('file_name'):<26} σελ.{m.get('page')}{star}")
    if args.target:
        print("\n" + ("*** Ο ΣΤΟΧΟΣ ΕΦΤΑΣΕ ΣΤΟ PROMPT ***" if found
                     else "*** Ο ΣΤΟΧΟΣ ΔΕΝ ΕΦΤΑΣΕ ΠΟΤΕ ***"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
