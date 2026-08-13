"""Ο εμπλουτισμός ΕΓΓΡΑΦΟΥ ανοίγει ή κλείνει το κενό του gate;

ΤΟ ΕΡΩΤΗΜΑ ΠΟΥ ΚΡΙΝΕΙ ΤΑ ΠΑΝΤΑ (12/8/2026)
------------------------------------------
Μετρήθηκε ότι ένα ΤΥΦΛΟ πρόθεμα (περιγραφή + doc2query, το prompt δεν βλέπει
ποτέ ερώτηση) ανεβάζει τον στόχο του `h002` από **−8.95 σε −3.97** και το dense
από τη θέση 15 στην 6. Χρειάζεται −2.60 για να περάσει, άρα λείπουν **1.37**.

Ο πειρασμός είναι να πει κανείς «ξαναβαθμονομούμε το gate». ΛΑΘΟΣ ΕΡΩΤΗΜΑ.
Αν ο εμπλουτισμός ανεβάζει ΚΑΙ τις out_of_corpus κατά +5, το κενό μένει ίδιο,
κάθε νέο κατώφλι είναι ισοδύναμο με το σημερινό και το h002 παραμένει κάτω.
Το μόνο που μετράει είναι η **ΔΙΑΧΩΡΙΣΙΜΟΤΗΤΑ**: ανεβαίνουν οι in-corpus
ΠΕΡΙΣΣΟΤΕΡΟ από τις out_of_corpus;

Ίδιο κριτήριο έκρινε το `bge-reranker-base`: βαθμολογούσε ψηλά τα out_of_corpus
(thr* = +0.963), άρα το κατώφλι που τα έκοβε σκότωνε και τα σωστά. Η απόλυτη
βαθμολογία δεν σημαίνει τίποτα — η ΑΠΟΣΤΑΣΗ σημαίνει.

ΤΙ ΚΑΝΕΙ
--------
Για κάθε ερώτηση τρέχει τα βήματα 2-5 του `search_documents` καλώντας τις ΙΔΙΕΣ
συναρτήσεις του ai_core (όπως το `measure_gate_margin.py`), βρίσκει το **top-1
chunk** που θα κρίνει το gate, παράγει γι' αυτό ΤΥΦΛΟ πρόθεμα και ξαναβαθμολογεί
το ίδιο ζεύγος. Το Δ είναι το κέρδος του εμπλουτισμού για ΑΥΤΗ την ερώτηση.

ΤΙ ΔΕΝ ΚΑΝΕΙ — ΓΡΑΨ' ΤΟ ΣΤΟ ΣΥΜΠΕΡΑΣΜΑ
--------------------------------------
Εμπλουτίζει ΜΟΝΟ το top-1 chunk, όχι όλο το corpus. Σε πραγματικό re-ingest
αλλάζουν και οι υποψήφιοι (το dense/BM25 φέρνουν άλλα chunks), άρα το top-1
μπορεί να είναι άλλο. Είναι **δείκτης κατεύθυνσης με n=1 chunk ανά ερώτηση**,
όχι πρόβλεψη του τελικού αποτελέσματος. Αν η κατεύθυνση βγει λάθος, δεν αξίζει
το πλήρες πείραμα· αν βγει σωστή, ΔΕΝ αποδεικνύει τίποτα μόνη της.

ΚΟΣΤΟΣ: 1-2 κλήσεις Gemini ανά **μοναδικό** chunk (cache σε JSON), δηλαδή
~0.005 $ για 20 ερωτήσεις. Οι μεταφράσεις είναι ήδη στο μόνιμο cache.

    docker compose exec backend python evaluation/probe_doc_enrichment_leak.py
    docker compose exec backend python evaluation/probe_doc_enrichment_leak.py \\
        --in-corpus 12 --csv evaluation/runs/enrich_leak.csv
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

from build_chunk_prefixes import describe

import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

SETS = [
    "/app/evaluation/golden_set_50.jsonl",
    "/app/evaluation/golden_hard_paraphrase.jsonl",
]
CACHE = "/app/evaluation/runs/chunk_prefixes_cache.json"


def load_sets(paths):
    tests = []
    for p in paths:
        if not os.path.exists(p):
            print(f"ΠΑΡΑΛΕΙΨΗ (δεν υπάρχει): {p}")
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    t["_set"] = os.path.basename(p).replace(".jsonl", "")
                    tests.append(t)
    return tests


def pick(tests, n_in: int):
    """ΟΛΕΣ οι out_of_corpus + ντετερμινιστικό δείγμα in-corpus + ΟΛΟ το hard.

    Οι out_of_corpus είναι το κρίσιμο σκέλος και είναι μόνο 5 — μπαίνουν όλες.
    Το in-corpus δείγμα είναι κάθε k-οστή ερώτηση (σταθερό, χωρίς seed).
    """
    ooc = [t for t in tests if t.get("category") == "out_of_corpus"]
    hard = [t for t in tests if t["_set"].startswith("golden_hard")]
    inc = [t for t in tests if t.get("category") != "out_of_corpus"
           and not t["_set"].startswith("golden_hard")]
    step = max(1, len(inc) // max(1, n_in))
    return ooc + inc[::step][:n_in] + hard


async def prefix_for(cid: str, text: str, cache: dict, styles) -> str:
    """Τυφλό πρόθεμα, με cache ανά (chunk_id, style). ΔΕΝ βλέπει την ερώτηση."""
    parts = []
    for st in styles:
        key = f"{cid}::{st}"
        if key not in cache:
            cache[key] = await describe(text, st)
        parts.append(cache[key])
    return "\n".join(parts)


async def main(args) -> int:
    tests = pick(load_sets(args.sets), args.in_corpus)
    print(f"{len(tests)} ερωτήσεις · στυλ προθέματος: {'+'.join(args.styles)}\n")

    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache, encoding="utf-8") as fh:
            cache = json.load(fh)

    where = ai_core._build_where(GOLDEN_CORPUS, None)
    allowed_ids = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()
    thr = ai_core.MIN_RERANK_SCORE
    # Το _rrf_fuse επιστρέφει (score, text, meta) χωρίς id -> αντιστοίχιση από
    # το κείμενο. Τα chunks είναι μοναδικά κείμενα στο corpus αυτού του μεγέθους.
    text_to_id = dict(zip(idx["texts"], idx["ids"]))

    print(f"{'id':<7}{'σετ':<22}{'κατηγορία':<15}{'πριν':>8}{'μετά':>8}"
          f"{'Δ':>8}   gate πριν -> μετά")
    print("-" * 88)

    rows = []
    for t in tests:
        query = await ai_core.optimize_query(t["question"])
        d_ids = ai_core._dense_exact_ids(
            dm, query, allowed_ids, min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
        s_ids = ai_core._bm25_sparse_ids(idx, query, allowed_ids,
                                         ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"],
                                idx["metas"], k=60,
                                top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
        pairs = [[query, it[1]] for it in rrf]
        scores = [float(x) for x in ai_core.reranker.predict(
            pairs, batch_size=ai_core.RERANK_BATCH_SIZE)]
        best_i = max(range(len(scores)), key=lambda i: scores[i])
        best, best_text = scores[best_i], rrf[best_i][1]
        cid = text_to_id.get(best_text, "?")

        pfx = await prefix_for(cid, best_text, cache, args.styles)
        after = float(ai_core.reranker.predict(
            [[query, pfx + "\n" + best_text]], batch_size=1)[0])

        cat = t.get("category") or ("hard" if t["_set"].startswith("golden_hard")
                                    else "-")
        rows.append(dict(id=t["id"], set=t["_set"], category=cat, chunk=cid,
                         before=round(best, 3), after=round(after, 3),
                         delta=round(after - best, 3),
                         pass_before=best >= thr, pass_after=after >= thr))
        print(f"{t['id']:<7}{t['_set'][:20]:<22}{cat:<15}{best:>8.2f}"
              f"{after:>8.2f}{after - best:>+8.2f}   "
              f"{'περνά' if best >= thr else 'ΚΟΒΕΙ':<6} -> "
              f"{'περνά' if after >= thr else 'ΚΟΒΕΙ'}", flush=True)

        with open(args.cache, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False, indent=1)

    # ---------------- ΤΟ ΣΥΜΠΕΡΑΣΜΑ ----------------
    ooc = [r for r in rows if r["category"] == "out_of_corpus"]
    inc = [r for r in rows if r["category"] not in ("out_of_corpus",)]
    hard = [r for r in rows if r["category"] == "hard"]
    main_in = [r for r in inc if r["category"] != "hard"]

    print("\n" + "=" * 88)
    print("ΜΕΣΟ ΚΕΡΔΟΣ ΑΝΑ ΟΜΑΔΑ  (αυτό κρίνει, όχι η απόλυτη βαθμολογία)")
    print("=" * 88)
    for label, grp in (("in-corpus (κύριο)", main_in), ("hard set", hard),
                       ("OUT_OF_CORPUS", ooc)):
        if not grp:
            continue
        d = [r["delta"] for r in grp]
        print(f"  {label:<20} n={len(d):<3} μέσο Δ {statistics.mean(d):>+7.2f}"
              f" · διάμεσος {statistics.median(d):>+7.2f}"
              f" · εύρος [{min(d):+.2f}, {max(d):+.2f}]")

    if inc and ooc:
        lo_b = min(r["before"] for r in inc)
        hi_b = max(r["before"] for r in ooc)
        lo_a = min(r["after"] for r in inc)
        hi_a = max(r["after"] for r in ooc)
        print("\n" + "=" * 88)
        print("ΤΟ ΚΕΝΟ (χειρότερο in-corpus μείον καλύτερο out_of_corpus)")
        print("=" * 88)
        print(f"  ΠΡΙΝ : in-min {lo_b:>7.2f} · ooc-max {hi_b:>7.2f} "
              f"-> κενό {lo_b - hi_b:>+7.2f}")
        print(f"  ΜΕΤΑ : in-min {lo_a:>7.2f} · ooc-max {hi_a:>7.2f} "
              f"-> κενό {lo_a - hi_a:>+7.2f}")
        d = (lo_a - hi_a) - (lo_b - hi_b)
        print(f"\n  ΜΕΤΑΒΟΛΗ ΚΕΝΟΥ: {d:+.2f} logits  ->  "
              + ("ΑΝΟΙΓΕΙ: η επαναβαθμονόμηση έχει νόημα"
                 if d > 0.5 else
                 "ΚΛΕΙΝΕΙ/ΙΔΙΟ: η επαναβαθμονόμηση ΔΕΝ σώζει τίποτα"))

    leaked = [r for r in ooc if r["pass_after"] and not r["pass_before"]]
    saved = [r for r in inc if r["pass_after"] and not r["pass_before"]]
    broken = [r for r in inc if r["pass_before"] and not r["pass_after"]]
    print(f"\n  ερωτήσεις που ΧΑΝΟΝΤΑΙ (περνούσαν, πλέον κόβονται): {len(broken)}"
          + (f" -> {', '.join(r['id'] for r in broken)}" if broken else ""))

    # ---------------- ΔΙΠΛΗ ΑΝΑΠΑΡΑΣΤΑΣΗ ----------------
    # Κρατάς ΚΑΙ το αρχικό chunk ΚΑΙ το εμπλουτισμένο ως ξεχωριστές εγγραφές:
    # η βαθμολογία της σελίδας γίνεται max(αρχικό, εμπλουτισμένο). Έτσι εξαφανίζεται
    # η αραίωση (τίποτα δεν χάνεται) — αλλά δίνεται δεύτερη ευκαιρία ΚΑΙ στις
    # out_of_corpus. Υπολογίζεται από τα ΙΔΙΑ δεδομένα, ΜΗΔΕΝ επιπλέον κόστος.
    for r in rows:
        r["dual"] = max(r["before"], r["after"])
    d_lo = min(r["dual"] for r in inc)
    d_hi = max(r["dual"] for r in ooc)
    d_saved = [r for r in inc if r["dual"] >= thr > r["before"]]
    d_leak = [r for r in ooc if r["dual"] >= thr > r["before"]]
    print("\n" + "=" * 88)
    print("ΔΙΠΛΗ ΑΝΑΠΑΡΑΣΤΑΣΗ  max(αρχικό, εμπλουτισμένο) — καμία αραίωση")
    print("=" * 88)
    print(f"  in-min {d_lo:>7.2f} · ooc-max {d_hi:>7.2f} -> κενό {d_lo - d_hi:>+7.2f}"
          f"  (ήταν {lo_b - hi_b:+.2f})")
    print(f"  ΣΩΖΟΝΤΑΙ {len(d_saved)}"
          + (f" ({', '.join(r['id'] for r in d_saved)})" if d_saved else "")
          + f"  ·  ΔΙΑΡΡΕΟΥΝ {len(d_leak)}"
          + (f" ({', '.join(r['id'] for r in d_leak)})" if d_leak else "")
          + "  ·  ΧΑΝΟΝΤΑΙ 0 εξ ορισμού")
    print(f"\n  out_of_corpus που ΘΑ ΠΕΡΝΑΓΑΝ με το σημερινό {thr}: "
          f"{len(leaked)}/{len(ooc)}"
          + (f" -> {', '.join(r['id'] for r in leaked)}" if leaked else ""))
    print(f"  ερωτήσεις που ΣΩΖΟΝΤΑΙ με το σημερινό {thr}: {len(saved)}"
          + (f" -> {', '.join(r['id'] for r in saved)}" if saved else ""))

    if args.csv and rows:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="διαχωρισιμότητα υπό εμπλουτισμό εγγράφου")
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--in-corpus", type=int, default=12,
                    help="πόσες in-corpus του κύριου σετ (δείγμα κάθε k-οστή)")
    ap.add_argument("--styles", nargs="*", default=["plain", "d2q"],
                    help="ποια τυφλά prompts μπαίνουν ως πρόθεμα")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--csv", default="/app/evaluation/runs/enrich_leak.csv")
    raise SystemExit(asyncio.run(main(ap.parse_args())))
