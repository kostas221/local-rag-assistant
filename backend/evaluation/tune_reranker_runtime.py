"""Δύο ρυθμίσεις εκτέλεσης του ΙΔΙΟΥ reranker: batch_size και dynamic INT8.

ΓΙΑΤΙ: μετά το thread fix ο cross-encoder είναι ακόμα το 96.7% του warm
retrieval (479ms από 495ms). Δεν αλλάζουμε μοντέλο — ψάχνουμε αν ο ίδιος
υπολογισμός γίνεται φθηνότερα.

  1) batch_size: το CrossEncoder.predict() έχει default 32, οπότε και τα 15
     υποψήφια μπαίνουν σε ΕΝΑ batch και γίνονται όλα pad στο μήκος του
     μακρύτερου. Το sentence-transformers ταξινομεί κατά μήκος (smart batching)
     ΜΟΝΟ μεταξύ batches — με ένα batch η ταξινόμηση δεν κάνει τίποτα. Μικρότερα
     batches ομαδοποιούν κοντά μήκη -> λιγότερα padding tokens να υπολογιστούν.
  2) dynamic INT8 (torch.ao.quantization, fbgemm): κβαντίζει τα Linear layers σε
     int8 κατά την εκτέλεση. ΔΙΑΦΟΡΕΤΙΚΟ μονοπάτι από το ONNX INT8 που ήδη
     απορρίφθηκε (onnxruntime, μηδέν κέρδος) — αλλά με το ίδιο ιστορικό, γι'
     αυτό μετριέται εδώ αντί να θεωρηθεί δεδομένο.

ΠΟΙΟΤΗΤΑ, ΟΧΙ ΜΟΝΟ ΧΡΟΝΟΣ: κάθε παραλλαγή συγκρίνεται με το baseline σε
(α) συσχέτιση Pearson των ωμών σκορ και (β) πόσες φορές αλλάζει το TOP-1
chunk. Το (β) είναι που μετράει: το gate και το page expansion δουλεύουν πάνω
στην ΚΑΤΑΤΑΞΗ, όχι στις απόλυτες τιμές. Μια παραλλαγή με ρ=0.99 που όμως
αλλάζει το top-1 είναι ΑΠΟΡΡΙΠΤΕΑ.

    docker compose exec backend python evaluation/tune_reranker_runtime.py
"""
import asyncio
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ai_core

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden_set_50.jsonl")
BATCH_SIZES = [4, 8, 15, 32]      # 32 = default του sentence-transformers
REPEATS = 5


def load_tests(limit: int = 15) -> list:
    out = []
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("category") != "out_of_corpus":
                out.append(t)
    return out[:limit]


async def build_pairs(tests: list) -> list:
    """Τα ΠΡΑΓΜΑΤΙΚΑ ζεύγη (query, chunk) που θα έβλεπε ο reranker σε κάθε
    ερώτηση — ίδιο μονοπάτι με το search_documents μέχρι το RRF."""
    where = ai_core._build_where(None, None)
    allowed = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    batches = []
    for t in tests:
        query = await ai_core.optimize_query(t["question"])
        d_ids = ai_core._dense_exact_ids(dm, query, allowed, ai_core.DENSE_CANDIDATES)
        s_ids = ai_core._bm25_sparse_ids(idx, query, allowed, ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"],
                                idx["metas"], k=60, top_n=ai_core.RERANK_CANDIDATES)
        batches.append([[query, item[1]] for item in rrf])
    return batches


def score_all(model, batches: list, **kw) -> list:
    return [np.asarray(model.predict(p, **kw), dtype=np.float64) for p in batches]


def time_all(model, batches: list, **kw) -> float:
    """Διάμεσος χρόνου ΑΝΑ ΕΡΩΤΗΣΗ (ms) — ό,τι πληρώνει ο χρήστης."""
    model.predict(batches[0], **kw)          # warmup
    per_query = []
    for _ in range(REPEATS):
        for p in batches:
            t = time.perf_counter()
            model.predict(p, **kw)
            per_query.append((time.perf_counter() - t) * 1000)
    return statistics.median(per_query)


def compare(base: list, other: list) -> tuple:
    """(Pearson ρ σε όλα τα σκορ, πλήθος ερωτήσεων με αλλαγμένο top-1)."""
    flat_b = np.concatenate(base)
    flat_o = np.concatenate(other)
    rho = float(np.corrcoef(flat_b, flat_o)[0, 1])
    flips = sum(1 for b, o in zip(base, other)
                if int(np.argmax(b)) != int(np.argmax(o)))
    return rho, flips


def main() -> int:
    tests = load_tests()
    batches = asyncio.run(build_pairs(tests))
    n_pairs = sum(len(b) for b in batches)
    print(f"Ερωτήσεις: {len(batches)} · ζεύγη συνολικά: {n_pairs} "
          f"· threads: {torch.get_num_threads()}")
    print(f"Μοντέλο: {ai_core.RERANKER_MODEL}\n")

    model = ai_core.reranker
    base_scores = score_all(model, batches)
    base_ms = time_all(model, batches)
    print(f"{'παραλλαγή':<24} {'ms/ερώτηση':>11} {'σχέση':>7} "
          f"{'Pearson ρ':>10} {'top-1 flips':>12}")
    print("-" * 70)
    print(f"{'baseline (batch=32)':<24} {base_ms:>11.1f} {'1.00x':>7} "
          f"{'—':>10} {'—':>12}")

    # --- 1) batch_size ------------------------------------------------------
    for bs in BATCH_SIZES:
        if bs == 32:
            continue
        ms = time_all(model, batches, batch_size=bs)
        rho, flips = compare(base_scores, score_all(model, batches, batch_size=bs))
        print(f"{'batch_size=' + str(bs):<24} {ms:>11.1f} "
              f"{base_ms / ms:>6.2f}x {rho:>10.4f} {flips:>12}")

    # --- 2) dynamic INT8 ----------------------------------------------------
    try:
        qmodel = torch.ao.quantization.quantize_dynamic(
            model.model, {torch.nn.Linear}, dtype=torch.qint8)
        holder = ai_core.reranker
        original = holder.model
        holder.model = qmodel
        try:
            ms = time_all(holder, batches)
            rho, flips = compare(base_scores, score_all(holder, batches))
            print(f"{'dynamic INT8 (fbgemm)':<24} {ms:>11.1f} "
                  f"{base_ms / ms:>6.2f}x {rho:>10.4f} {flips:>12}")
        finally:
            holder.model = original      # ΠΑΝΤΑ πίσω: το reranker είναι singleton
    except Exception as e:
        print(f"{'dynamic INT8':<24} απέτυχε: {e}")

    print("\nΚΡΙΤΗΡΙΟ ΑΠΟΔΟΧΗΣ: top-1 flips == 0 ΚΑΙ κέρδος >= 1.10x.")
    print("Ένα flip σημαίνει άλλη σελίδα στο context -> άλλη απάντηση. "
          "Καμία επιτάχυνση δεν το αξίζει σε 15 ερωτήσεις δείγμα.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
