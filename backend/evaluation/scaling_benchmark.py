"""Πού σπάει το pipeline όταν μεγαλώσει το corpus;

ΤΟ ΕΡΩΤΗΜΑ ΠΟΥ ΑΠΑΝΤΑΕΙ: το σύστημα τρέχει σε 418 chunks και οι αρχιτεκτονικές
επιλογές είναι μετρημένα σωστές ΣΕ ΑΥΤΗ ΤΗΝ ΚΛΙΜΑΚΑ — exact brute-force αντί για
ANN, BM25 πάνω σε ΟΛΟ το corpus σε κάθε ερώτηση, caches μέσα στη διεργασία. Το
project όμως δεν είχε κανένα στοιχείο για το τι γίνεται στα 100.000 ή 1.000.000.
Αυτό ΔΕΝ είναι προσπάθεια να κλιμακώσει το σύστημα· είναι μέτρηση του ΠΟΥ ΣΠΑΕΙ
και με ποια σειρά, ώστε η απάντηση στο "τι θα άλλαζες" να έχει νούμερα.

ΤΙ ΜΕΤΡΙΕΤΑΙ ανά μέγεθος corpus:
  1. DENSE EXACT   — matmul N x 1024 + argsort top-30 (το σημερινό μονοπάτι)
  2. DENSE ANN     — hnswlib: κόστος χτισίματος, χρόνος ερωτήματος, ΚΑΙ recall@30
                     έναντι του exact. Δίνει το CROSSOVER POINT: πάνω από πόσα
                     vectors το ANN αρχίζει να αξίζει τη μη-ντετερμινιστικότητά του.
  3. BM25          — tokenization + χτίσιμο ευρετηρίου + ένα ερώτημα
  4. RRF           — η σύντηξη (καθαρή Python)
  5. RAM           — VmRSS από το /proc, πραγματική κατανάλωση

ΤΙ ΔΕΝ ΑΛΛΑΖΕΙ ΜΕ ΤΗΝ ΚΛΙΜΑΚΑ (και είναι το πιο χρήσιμο εύρημα): ο reranker
βλέπει ΠΑΝΤΑ RERANK_CANDIDATES=15 ζεύγη, ό,τι κι αν γίνει με το corpus. Το
βήμα που κυριαρχεί σήμερα στο latency είναι το ΜΟΝΟ που δεν κλιμακώνεται.

ΤΙΜΙΟΤΗΤΑ ΤΩΝ ΔΕΔΟΜΕΝΩΝ:
  • Τα διανύσματα είναι συνθετικά (τυχαία, κανονικοποιημένα). Για ΧΡΟΝΟ και ΜΝΗΜΗ
    αυτό είναι ισοδύναμο με πραγματικά — το matmul δεν ξέρει τι σημαίνουν. Για το
    recall του ANN είναι ΔΥΣΚΟΛΟΤΕΡΗ περίπτωση: τα τυχαία διανύσματα σε 1024
    διαστάσεις είναι σχεδόν ισαπέχοντα, άρα το πραγματικό recall σε δομημένα
    embeddings θα είναι καλύτερο. Το νούμερο είναι κάτω φράγμα, όχι πρόβλεψη.
  • Τα κείμενα του BM25 προκύπτουν από τα ΠΡΑΓΜΑΤΙΚΑ chunks του corpus, με
    ανακάτεμα προτάσεων και μοναδικά tokens ανά αντίγραφο ώστε το λεξιλόγιο να
    μεγαλώνει υπογραμμικά (νόμος του Heaps) αντί να μένει παγωμένο.

    docker compose exec backend python evaluation/scaling_benchmark.py
    docker compose exec backend python evaluation/scaling_benchmark.py --sizes 418,5000,50000
"""
import argparse
import gc
import os
import random
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Μετράμε τον ΠΡΑΓΜΑΤΙΚΟ κώδικα (el_tokenize, _rrf_fuse), όχι αντίγραφά τους —
# αλλιώς το benchmark θα μετρούσε κάτι που δεν τρέχει πουθενά. Το τίμημα είναι
# τα ~40-60s του import (φορτώνονται τα μοντέλα), αμελητέο για μέτρηση λεπτών.
import ai_core

DIM = 1024              # bge-m3
TOP_N = 30              # DENSE_CANDIDATES
RERANK_CANDIDATES = 15  # σταθερό, ανεξάρτητο από την κλίμακα
QUERY_REPEATS = 5

# Το κυρίαρχο σταθερό κόστος του pipeline (μετρημένο: measure_latency.py, 8
# threads, batch 4). Χρησιμεύει ως ΠΑΡΟΝΟΜΑΣΤΗΣ: μια βελτίωση 5ms σε ένα βήμα
# δεν σημαίνει τίποτα όταν ένα άλλο βήμα κοστίζει 434ms ό,τι κι αν γίνει.
RERANKER_MS = float(os.getenv("RERANKER_MS", "434"))
# Πόσο του κόστους rerank πρέπει να γλιτώνει μια αλλαγή για να τη λέμε ουσιαστική.
MATERIAL_PCT = 0.10

# Πάνω από αυτό το μέγεθος το BM25 παραλείπεται: το rank_bm25 είναι pure-Python
# και κρατάει ΚΑΘΕ tokenized document στη μνήμη ως list από strings. Το ίδιο το
# όριο είναι εύρημα, όχι παράλειψη — τυπώνεται στο πόρισμα.
BM25_MAX = int(os.getenv("BM25_MAX", "100000"))


def rss_mb() -> float:
    """Πραγματική κατανάλωση μνήμης της διεργασίας (MB), από το /proc.
    Χωρίς psutil — δεν είναι εγκατεστημένο και δεν αξίζει εξάρτηση γι' αυτό."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


# --- Παραγωγή δεδομένων ------------------------------------------------------

def make_vectors(n: int, seed: int = 0) -> np.ndarray:
    """N x 1024 κανονικοποιημένα float32 — ίδιο σχήμα και dtype με το παραγωγικό
    _get_dense_matrix, ώστε ο χρόνος και η μνήμη να είναι συγκρίσιμα."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, DIM), dtype=np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return M


def real_chunk_texts() -> list:
    """Τα ΠΡΑΓΜΑΤΙΚΑ κείμενα του corpus — ρεαλιστικό μήκος και λεξιλόγιο.
    Αν η ChromaDB δεν είναι προσβάσιμη, γυρνάμε σε συνθετικό κείμενο."""
    try:
        return ai_core.collection.get()["documents"]
    except Exception as e:
        print(f"  (χωρίς πρόσβαση στο corpus: {e} -> συνθετικά κείμενα)")
        words = ["cloud", "computing", "serverless", "datacenter", "elasticity", "utility", "virtualization", "latency", "throughput", "provisioning", "workload"]
        rng = random.Random(0)
        return [" ".join(rng.choices(words, k=220)) for _ in range(418)]


def grow_texts(base: list, n: int, seed: int = 0) -> list:
    """Μεγαλώνει το σύνολο κειμένων σε n, με ανακάτεμα προτάσεων και μοναδικά
    tokens ανά αντίγραφο. ΓΙΑΤΙ ΟΧΙ ΣΚΕΤΗ ΕΠΑΝΑΛΗΨΗ: με πανομοιότυπα κείμενα το
    λεξιλόγιο μένει σταθερό και το BM25 φαίνεται ψευδώς φθηνό — στην πραγματικότητα
    το κόστος του μεγαλώνει και με τους όρους, όχι μόνο με τα έγγραφα."""
    rng = random.Random(seed)
    out = list(base[:n])
    copy_idx = 0
    while len(out) < n:
        copy_idx += 1
        for text in base:
            if len(out) >= n:
                break
            sentences = text.split(". ")
            rng.shuffle(sentences)
            out.append(". ".join(sentences) + f" tok{copy_idx}x{len(out)}")
    return out


# --- Μετρήσεις ανά σκέλος ----------------------------------------------------

def bench_dense_exact(M: np.ndarray) -> dict:
    q = M[0].copy()
    M @ q  # warmup

    times = []
    for _ in range(QUERY_REPEATS):
        t0 = time.perf_counter()
        sims = M @ q
        np.argsort(-sims, kind="stable")[:TOP_N]
        times.append((time.perf_counter() - t0) * 1000)
    truth = set(np.argsort(-(M @ q), kind="stable")[:TOP_N].tolist())
    return {"ms": statistics.median(times), "truth": truth,
            "mb": M.nbytes / 1024 / 1024}


def bench_dense_ann(M: np.ndarray, truth: set) -> dict:
    """hnswlib με τις παραμέτρους που χρησιμοποιεί η ChromaDB by default."""
    import hnswlib
    n = M.shape[0]
    index = hnswlib.Index(space="cosine", dim=DIM)

    t0 = time.perf_counter()
    index.init_index(max_elements=n, ef_construction=100, M=16)
    index.add_items(M, np.arange(n))
    build_s = time.perf_counter() - t0

    index.set_ef(max(TOP_N + 20, 100))
    q = M[0].copy()
    index.knn_query(q, k=TOP_N)  # warmup
    times = []
    for _ in range(QUERY_REPEATS):
        t0 = time.perf_counter()
        index.knn_query(q, k=TOP_N)
        times.append((time.perf_counter() - t0) * 1000)

    labels, _ = index.knn_query(q, k=TOP_N)
    recall = len(truth & set(labels[0].tolist())) / len(truth)
    del index
    gc.collect()
    return {"ms": statistics.median(times), "build_s": build_s, "recall": recall}


def bench_bm25(texts: list) -> dict:
    from rank_bm25 import BM25Okapi

    t0 = time.perf_counter()
    tokenized = [ai_core.el_tokenize(t) for t in texts]
    tok_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    bm25 = BM25Okapi(tokenized)
    build_s = time.perf_counter() - t0

    query = ai_core.el_tokenize("cloud computing obstacles elasticity")
    bm25.get_scores(query)  # warmup
    times = []
    for _ in range(QUERY_REPEATS):
        t0 = time.perf_counter()
        scores = bm25.get_scores(query)
        np.argsort(-scores)[:TOP_N]
        times.append((time.perf_counter() - t0) * 1000)

    vocab = len({w for doc in tokenized for w in doc})
    del bm25, tokenized
    gc.collect()
    return {"tok_s": tok_s, "build_s": build_s,
            "ms": statistics.median(times), "vocab": vocab}


def bench_rrf(n: int) -> dict:
    """Η σύντηξη δουλεύει σε 2x30 υποψήφια αλλά χτίζει pos_by_id πάνω σε ΟΛΑ τα
    ids — εκεί κρύβεται η εξάρτηση από την κλίμακα."""
    ids = [f"id{i}" for i in range(n)]
    texts = ["t"] * n
    metas = [{"file_name": "f", "page": 1, "doc_id": 1}] * n
    dense_ids = ids[:TOP_N]
    sparse_ids = ids[TOP_N:2 * TOP_N]
    ai_core._rrf_fuse(dense_ids, sparse_ids, ids, texts, metas)  # warmup
    times = []
    for _ in range(QUERY_REPEATS):
        t0 = time.perf_counter()
        ai_core._rrf_fuse(dense_ids, sparse_ids, ids, texts, metas,
                          top_n=RERANK_CANDIDATES)
        times.append((time.perf_counter() - t0) * 1000)
    del ids, texts, metas
    gc.collect()
    return {"ms": statistics.median(times)}


# --- Κύριος βρόχος -----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="418,2000,10000,50000,200000",
                    help="μεγέθη corpus, χωρισμένα με κόμμα")
    ap.add_argument("--skip-ann", action="store_true")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    print("SCALING BENCHMARK — πού σπάει το pipeline καθώς μεγαλώνει το corpus")
    print(f"dim={DIM} · top_n={TOP_N} · rerank_candidates={RERANK_CANDIDATES} "
          f"(ΣΤΑΘΕΡΟ) · BM25 έως {BM25_MAX:,}\n")

    base_texts = real_chunk_texts()
    print(f"Βάση κειμένων: {len(base_texts)} πραγματικά chunks "
          f"(μέσο μήκος {statistics.mean(len(t) for t in base_texts):.0f} χαρ.)\n")

    rows = []
    for n in sizes:
        print(f"--- N = {n:,} " + "-" * 40)
        rss0 = rss_mb()
        row = {"n": n}

        M = make_vectors(n)
        d = bench_dense_exact(M)
        row["dense_ms"], row["mat_mb"] = d["ms"], d["mb"]
        print(f"  dense exact   {d['ms']:>9.2f} ms   (matrix {d['mb']:>7.0f} MB)")

        if not args.skip_ann:
            a = bench_dense_ann(M, d["truth"])
            row.update(ann_ms=a["ms"], ann_build=a["build_s"], recall=a["recall"])
            print(f"  dense ANN     {a['ms']:>9.2f} ms   "
                  f"(build {a['build_s']:>6.1f}s · recall@30 {a['recall']:.2f})")
        del M
        gc.collect()

        if n <= BM25_MAX:
            texts = grow_texts(base_texts, n)
            b = bench_bm25(texts)
            row.update(bm25_ms=b["ms"], bm25_build=b["tok_s"] + b["build_s"],
                       vocab=b["vocab"])
            print(f"  BM25 query    {b['ms']:>9.2f} ms   "
                  f"(build {b['tok_s'] + b['build_s']:>6.1f}s · "
                  f"λεξιλόγιο {b['vocab']:,})")
            del texts
            gc.collect()
        else:
            print(f"  BM25          ΠΑΡΑΛΕΙΠΕΤΑΙ (> {BM25_MAX:,}: in-memory)")

        r = bench_rrf(n)
        row["rrf_ms"] = r["ms"]
        print(f"  RRF           {r['ms']:>9.2f} ms")
        row["rss_mb"] = rss_mb()
        print(f"  RAM διεργασίας {row['rss_mb']:>8.0f} MB (από {rss0:.0f})\n")
        rows.append(row)

    # --- Πόρισμα -------------------------------------------------------------
    print("=" * 78)
    print(f"{'N':>10}{'dense':>10}{'ANN':>10}{'recall':>9}{'BM25 q':>10}"
          f"{'BM25 build':>12}{'RRF':>8}{'RAM MB':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['n']:>10,}{r['dense_ms']:>9.1f}m"
              f"{r.get('ann_ms', float('nan')):>9.2f}m"
              f"{r.get('recall', float('nan')):>9.2f}"
              f"{r.get('bm25_ms', float('nan')):>9.1f}m"
              f"{r.get('bm25_build', float('nan')):>11.1f}s"
              f"{r['rrf_ms']:>7.2f}m{r['rss_mb']:>9.0f}")

    print("\n--- ΣΥΜΠΕΡΑΣΜΑΤΑ ---")
    # ΓΙΑΤΙ ΟΧΙ ΣΚΕΤΟ "exact > ANN": στα 2.000 vectors το exact είναι 0.20ms και
    # το ANN 0.16ms — «το ANN κέρδισε», αλλά η διαφορά είναι 0.04ms σε pipeline
    # που τρέχει ~450ms. Θα ήταν αριθμητικά σωστό και πρακτικά ανόητο συμπέρασμα.
    # Το κατώφλι είναι πόσο ΤΟΥ ΣΥΝΟΛΟΥ κερδίζεις — γι' αυτό συγκρίνεται με τον
    # reranker, που είναι το κυρίαρχο σταθερό κόστος.
    print(f"  (κριτήριο: η διαφορά μετράει μόνο ως ποσοστό του συνολικού "
          f"latency· ο reranker κοστίζει ~{RERANKER_MS:.0f}ms ανεξάρτητα από N)")
    material = [r for r in rows if r.get("ann_ms")
                and (r["dense_ms"] - r["ann_ms"]) > MATERIAL_PCT * RERANKER_MS]
    if material:
        c = material[0]
        gain = c["dense_ms"] - c["ann_ms"]
        print(f"• ΟΥΣΙΑΣΤΙΚΟ CROSSOVER exact -> ANN: στα {c['n']:,} vectors "
              f"το ANN γλιτώνει {gain:.0f}ms "
              f"({100 * gain / RERANKER_MS:.0f}% του κόστους rerank).")
        print(f"  ΑΚΟΜΑ ΚΑΙ ΕΚΕΙ το τίμημα είναι {c.get('ann_build', 0):.0f}s "
              f"χτίσιμο ΣΕ ΚΑΘΕ ingest + απώλεια ντετερμινισμού.")
    else:
        big = rows[-1]
        gain = big["dense_ms"] - big.get("ann_ms", big["dense_ms"])
        print(f"• Το exact search παραμένει η ΣΩΣΤΗ επιλογή σε όλα τα μεγέθη που "
              f"δοκιμάστηκαν (έως {big['n']:,}): στο μεγαλύτερο, το ANN θα "
              f"γλίτωνε {gain:.0f}ms = {100 * gain / RERANKER_MS:.0f}% του "
              f"κόστους rerank, με τίμημα {big.get('ann_build', 0):.0f}s "
              f"χτίσιμο ανά ingest και μη-ντετερμινισμό.")

    bm = [r for r in rows if r.get("bm25_build")]
    if bm:
        worst = bm[-1]
        vs_dense = worst["bm25_ms"] / worst["dense_ms"] if worst["dense_ms"] else 0
        print(f"• ΤΟ BM25 ΣΠΑΕΙ ΠΡΩΤΟ, ΟΧΙ ΤΟ DENSE: στα {worst['n']:,} chunks το "
              f"ερώτημα θέλει {worst['bm25_ms']:.0f}ms — {vs_dense:.1f}x ΠΙΟ ΑΡΓΟ "
              f"από το exact dense ({worst['dense_ms']:.0f}ms) — και το χτίσιμο "
              f"{worst['bm25_build']:.0f}s επαναλαμβάνεται ΣΕ ΚΑΘΕ ingest.")
        print("  Είναι pure-Python (rank_bm25) και κρατιέται ολόκληρο στη μνήμη. "
              "Η αναβάθμιση που θα χρειαζόταν πρώτη είναι εδώ — Tantivy/Lucene ή "
              "Postgres full-text — ΟΧΙ το vector search.")

    rrf_rows = [r for r in rows if r.get("rrf_ms")]
    if len(rrf_rows) > 1 and rrf_rows[-1]["rrf_ms"] > 5:
        w = rrf_rows[-1]
        print(f"• ΤΟ RRF ΕΧΕΙ ΚΡΥΦΗ ΓΡΑΜΜΙΚΗ ΕΞΑΡΤΗΣΗ: {rrf_rows[0]['rrf_ms']:.2f}ms "
              f"-> {w['rrf_ms']:.0f}ms στα {w['n']:,}. Δουλεύει σε 2x30 υποψήφια, "
              f"αλλά χτίζει pos_by_id πάνω σε ΟΛΑ τα ids σε κάθε ερώτηση — ενώ το "
              f"ίδιο dict υπάρχει ΗΔΗ cached ως idx['pos'].")
    print(f"• Ο RERANKER ΔΕΝ ΚΛΙΜΑΚΩΝΕΤΑΙ: βλέπει πάντα {RERANK_CANDIDATES} "
          f"ζεύγη. Το βήμα που κυριαρχεί σήμερα (~96% του warm latency) είναι "
          f"το μόνο ανεξάρτητο από το μέγεθος του corpus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
