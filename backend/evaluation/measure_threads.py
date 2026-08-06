"""Πόσο κοστίζει ο reranker ανά αριθμό CPU threads;

ΓΙΑΤΙ: στο warm path ο cross-encoder είναι το 96% του retrieval latency
(589ms από 611ms). Ο container βλέπει 8 CPUs αλλά το PyTorch διαλέγει μόνο του
4 threads. Αν το 4 είναι συντηρητικό, υπάρχει κέρδος ΧΩΡΙΣ καμία αλλαγή στην
ποιότητα — δεν αλλάζει τίποτα στο pipeline, μόνο πόσοι πυρήνες δουλεύουν.
Αν το 4 είναι το βέλτιστο (π.χ. τα 8 είναι λογικοί πυρήνες πάνω σε 4 φυσικούς),
το μαθαίνουμε σε δύο λεπτά αντί να το υποθέσουμε.

ΔΕΝ κάνει import το ai_core: φορτώνει ΜΟΝΟ τον cross-encoder (22M), οπότε
τρέχει σε δευτερόλεπτα αντί για το 40-60s του πλήρους pipeline.

    docker compose exec backend python evaluation/measure_threads.py
"""
import os
import statistics
import sys
import time

import torch
from sentence_transformers import CrossEncoder

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "15"))
REPEATS = 7          # διάμεσος από 7 -> ανθεκτικό σε ένα τυχαίο spike
WARMUP = 2           # οι πρώτες κλήσεις πληρώνουν lazy init του backend

# Ρεαλιστικό φορτίο: RERANK_CANDIDATES ζεύγη (ερώτηση, chunk ~1500 χαρακτήρων),
# όσο ακριβώς δίνει το RRF στον reranker σε κάθε πραγματικό query.
QUERY = "what are the main obstacles to the adoption of cloud computing"
CHUNK = (
    "Cloud computing refers to both the applications delivered as services over "
    "the Internet and the hardware and systems software in the datacenters that "
    "provide those services. The services themselves have long been referred to "
    "as Software as a Service. The datacenter hardware and software is what we "
    "will call a Cloud. When a Cloud is made available in a pay-as-you-go manner "
    "to the general public, we call it a Public Cloud; the service being sold is "
    "Utility Computing. We use the term Private Cloud to refer to internal "
    "datacenters of a business or other organization, not made available to the "
    "general public. Thus, Cloud Computing is the sum of SaaS and Utility "
    "Computing, but does not include Private Clouds. Availability of service, "
    "data lock-in, data confidentiality and auditability, data transfer "
    "bottlenecks, performance unpredictability, scalable storage, bugs in large "
    "distributed systems, scaling quickly, reputation fate sharing, and software "
    "licensing are the top obstacles and opportunities we identified. "
)[:1500]

PAIRS = [[QUERY, CHUNK] for _ in range(RERANK_CANDIDATES)]


def bench(model) -> float:
    """Διάμεσος χρόνος ενός predict() σε ms."""
    for _ in range(WARMUP):
        model.predict(PAIRS)
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        model.predict(PAIRS)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main() -> None:
    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    default = torch.get_num_threads()
    print(f"CPUs ορατοί στον container : {n_cpu}")
    print(f"torch default threads      : {default}")
    print(f"Φορτίο                     : {RERANK_CANDIDATES} ζεύγη x {len(CHUNK)} χαρακτήρες")
    print(f"Μέθοδος                    : διάμεσος {REPEATS} μετρήσεων, {WARMUP} warmup\n")

    # Το μοντέλο φορτώνεται ΜΙΑ φορά· το set_num_threads δρα στο επόμενο forward.
    model = CrossEncoder(RERANKER_MODEL, device="cpu")

    candidates = sorted({1, 2, 4, 6, 8, n_cpu, default})
    candidates = [c for c in candidates if 1 <= c <= max(n_cpu, default)]

    print(f"{'threads':>8} | {'median ms':>10} | {'σε σχέση με default':>20}")
    print("-" * 46)
    results = {}
    for n in candidates:
        torch.set_num_threads(n)
        ms = bench(model)
        results[n] = ms
        print(f"{n:>8} | {ms:>10.1f} |", end=" ")
        print(f"{'(default)':>20}" if n == default else "")

    base = results[default]
    print("\nΕπιτάχυνση έναντι του default:")
    for n, ms in sorted(results.items()):
        mark = "  <-- default" if n == default else ""
        print(f"  {n} threads: {base / ms:.2f}x  ({ms:.1f} ms){mark}")

    best = min(results, key=results.get)
    gain = base / results[best]
    print()
    if best == default or gain < 1.10:
        print(f"ΣΥΜΠΕΡΑΣΜΑ: το default ({default}) είναι ήδη βέλτιστο "
              f"(καλύτερο: {best} threads, μόλις {gain:.2f}x). Καμία αλλαγή.")
    else:
        saved = base - results[best]
        print(f"ΣΥΜΠΕΡΑΣΜΑ: {best} threads -> {gain:.2f}x, "
              f"δηλαδή -{saved:.0f} ms ανά ερώτηση στο βήμα rerank.")
        print("Επόμενο: επιβεβαίωση end-to-end με run_eval --retrieval-only "
              "(τα σκορ ΠΡΕΠΕΙ να μείνουν ταυτόσημα — αλλάζει μόνο ο παραλληλισμός).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
