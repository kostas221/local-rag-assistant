"""Warm latency του retrieval ΑΝΑ ΣΤΑΔΙΟ, με προαιρετική σύγκριση thread counts.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ: τα νούμερα "cold 816ms / warm 611ms / rerank 589ms" ήταν ad-hoc
μετρήσεις που δεν επαναλαμβάνονταν. Κάθε φορά που αγγίζουμε το pipeline θέλουμε
την ΙΔΙΑ μέτρηση, αλλιώς συγκρίνουμε μήλα με πορτοκάλια.

ΤΙ ΜΕΤΡΑΕΙ: τα στάδια χωριστά, με τις ΙΔΙΕΣ συναρτήσεις που καλεί το
search_documents — dense (cached embedding + matmul), BM25, RRF, cross-encoder,
page expansion. Το άθροισμα είναι το warm retrieval· η γέννηση (Gemini) ΔΕΝ
μετριέται, είναι δικτυακή και εκτός ελέγχου μας.

WARM ΣΗΜΑΙΝΕΙ: BM25 index, dense matrix και query embeddings είναι ήδη στη
μνήμη. Γι' αυτό γίνεται ένα πλήρες πέρασμα warmup πριν τη μέτρηση — αλλιώς η
πρώτη ερώτηση πληρώνει ~1190ms embedding και μολύνει τη διάμεσο.

    docker compose exec backend python evaluation/measure_latency.py
    docker compose exec backend python evaluation/measure_latency.py --threads 4,8
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ai_core

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden_set_50.jsonl")
GOLDEN_CORPUS = None  # None -> όλο το corpus, όπως στην εφαρμογή


def load_questions(limit: int) -> list:
    """Μόνο in-corpus ερωτήσεις: οι out_of_corpus κόβονται από το gate ΠΡΙΝ το
    page expansion, οπότε θα μετρούσαν λιγότερα στάδια και θα κατέβαζαν ψευδώς
    τον μέσο όρο."""
    qs = []
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("category") == "out_of_corpus":
                continue
            qs.append(t["question"])
    return qs[:limit]


async def timed_stages(question: str) -> dict:
    """Ένα πλήρες retrieval, χρονομετρημένο ανά στάδιο (ms)."""
    t = {}
    t0 = time.perf_counter()
    query = await ai_core.optimize_query(question)   # cached -> ~0 για ελληνικά
    t["translate"] = (time.perf_counter() - t0) * 1000

    where = ai_core._build_where(GOLDEN_CORPUS, None)
    t0 = time.perf_counter()
    allowed = ai_core.collection.get(where=where, include=["metadatas"])["ids"]
    t["authz_get"] = (time.perf_counter() - t0) * 1000

    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    t0 = time.perf_counter()
    d_ids = ai_core._dense_exact_ids(dm, query, allowed, ai_core.DENSE_CANDIDATES)
    t["dense"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    s_ids = ai_core._bm25_sparse_ids(idx, query, allowed, ai_core.DENSE_CANDIDATES)
    t["bm25"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"], idx["metas"],
                            k=60, top_n=ai_core.RERANK_CANDIDATES)
    t["rrf"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    # ΙΔΙΑ κλήση με το search_documents, μαζί με το batch_size — αλλιώς το
    # script θα μετρούσε άλλη διαμόρφωση από αυτήν που τρέχει στην παραγωγή.
    scores = ai_core.reranker.predict([[query, item[1]] for item in rrf],
                                      batch_size=ai_core.RERANK_BATCH_SIZE)
    t["rerank"] = (time.perf_counter() - t0) * 1000

    final = sorted(zip(scores, [i[1] for i in rrf], [i[2] for i in rrf]),
                   key=lambda x: x[0], reverse=True)
    t0 = time.perf_counter()
    if float(final[0][0]) >= ai_core.MIN_RERANK_SCORE:
        ai_core._expand_to_pages(final[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)
    t["expand"] = (time.perf_counter() - t0) * 1000

    t["ΣΥΝΟΛΟ"] = sum(t.values())
    return t


async def run(questions: list, label: str) -> dict:
    # WARMUP: γεμίζει BM25/dense/embedding caches. Χωρίς αυτό η 1η ερώτηση
    # πληρώνει το embedding του μοντέλου (~1190ms) και σπάει τη διάμεσο.
    for q in questions:
        await timed_stages(q)

    runs = [await timed_stages(q) for q in questions]
    stages = list(runs[0].keys())
    med = {s: statistics.median(r[s] for r in runs) for s in stages}

    print(f"\n--- {label} · διάμεσος από {len(runs)} ερωτήσεις (ms) ---")
    total = med["ΣΥΝΟΛΟ"]
    for s in stages:
        if s == "ΣΥΝΟΛΟ":
            continue
        pct = 100 * med[s] / total if total else 0
        bar = "#" * max(1, round(pct / 2.5)) if pct >= 1 else ""
        print(f"  {s:<10} {med[s]:>8.1f}  {pct:>5.1f}%  {bar}")
    print(f"  {'ΣΥΝΟΛΟ':<10} {total:>8.1f}  100.0%")
    return med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="",
                    help="σύγκριση thread counts, π.χ. 4,8 (default: το τρέχον)")
    ap.add_argument("--n", type=int, default=12, help="πλήθος ερωτήσεων")
    args = ap.parse_args()

    questions = load_questions(args.n)
    print(f"Ερωτήσεις: {len(questions)} (in-corpus)")
    print(f"RERANK_CANDIDATES={ai_core.RERANK_CANDIDATES} "
          f"DENSE_CANDIDATES={ai_core.DENSE_CANDIDATES} "
          f"MAX_PAGES={ai_core.MAX_PAGES}")

    if not args.threads:
        asyncio.run(run(questions, f"torch threads = {torch.get_num_threads()}"))
        return 0

    counts = [int(x) for x in args.threads.split(",")]
    results = {}
    for n in counts:
        torch.set_num_threads(n)
        results[n] = asyncio.run(run(questions, f"torch threads = {n}"))

    base = results[counts[0]]
    print(f"\n--- Σύγκριση (βάση: {counts[0]} threads) ---")
    for n in counts:
        r = results[n]
        print(f"  {n:>2} threads: ΣΥΝΟΛΟ {r['ΣΥΝΟΛΟ']:>7.1f} ms "
              f"({base['ΣΥΝΟΛΟ'] / r['ΣΥΝΟΛΟ']:.2f}x)  "
              f"rerank {r['rerank']:>7.1f} ms "
              f"({base['rerank'] / r['rerank']:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
