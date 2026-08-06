"""Πώς συμπεριφέρεται το retrieval όταν ρωτούν πολλοί ταυτόχρονα;

⚠️  ΤΟ THREAD COUNT ΔΕΝ ΑΛΛΑΖΕΙ ΜΕΣΑ ΣΤΗΝ ΙΔΙΑ ΔΙΕΡΓΑΣΙΑ — ΜΕΤΡΗΜΕΝΟ:
    το torch.set_num_threads() επηρεάζει ΜΟΝΟ τα threads που δημιουργούνται ΜΕΤΑ
    από αυτό. Τα worker threads του asyncio.to_thread υπάρχουν ήδη από το warmup,
    οπότε ένα set_num_threads(4) στη μέση του benchmark ΔΕΝ τα αγγίζει και οι δύο
    «συνθήκες» μετράνε το ίδιο πράγμα. Η πρώτη εκδοχή αυτού του script έκανε
    ακριβώς αυτό και έβγαλε 4 vs 8 threads «σχεδόν ίδια» — ήταν όντως ίδια.
    (Στην ΠΑΡΑΓΩΓΗ δεν υπάρχει θέμα: το ai_core το ρυθμίζει στο import, πριν
    δημιουργηθεί οποιοδήποτε worker thread. Επαληθεύτηκε ότι ο reranker τρέχει
    πράγματι με TORCH_THREADS threads μέσα στο to_thread.)

    ΓΙΑ ΣΥΓΚΡΙΣΗ THREAD COUNTS: ξεχωριστή διεργασία ανά τιμή, μέσω env —
      docker compose exec -e TORCH_THREADS=4 backend python evaluation/concurrency_benchmark.py
      docker compose exec -e TORCH_THREADS=8 backend python evaluation/concurrency_benchmark.py



ΤΟ ΧΡΕΟΣ ΠΟΥ ΞΟΦΛΑΕΙ: στο README και στο ARCHITECTURE γράφτηκε ότι «υπό
πραγματικό concurrency βάλε TORCH_THREADS=4». Αυτό ΔΕΝ είχε μετρηθεί — ήταν
λογικό συμπέρασμα από το ότι τα threads είναι ΑΝΑ predict() και τα requests
τρέχουν σε asyncio.to_thread, άρα N ταυτόχρονες ερωτήσεις ζητούν N x THREADS
πυρήνες. Λογικό, αλλά ισχυρισμός χωρίς νούμερα — ακριβώς αυτό που το project
δεν επιτρέπει να γράφεται.

ΓΙΑΤΙ ΟΧΙ ΤΟ loadtest.py: εκείνο χτυπάει το /chat, άρα κάθε αίτημα καίει μία
κλήση Gemini και ο θόρυβος του δικτύου/της γέννησης πνίγει το σήμα. Εδώ
καλείται ΑΠΕΥΘΕΙΑΣ το search_documents με ΑΓΓΛΙΚΕΣ ερωτήσεις (καμία μετάφραση,
άρα μηδέν κλήσεις API) — μετράμε καθαρά τον CPU ανταγωνισμό.

ΤΙ ΔΕΙΧΝΕΙ: για κάθε συνδυασμό (threads, ταυτόχρονοι χρήστες), το p50/p95 του
ΑΝΑ ΕΡΩΤΗΣΗ χρόνου και το throughput. Ο ένας χρήστης νοιάζεται για το p95· ο
διαχειριστής για το throughput. Τα δύο δεν βελτιστοποιούνται μαζί, και ο στόχος
είναι να φανεί ΠΟΥ ακριβώς αντιστρέφεται το ισοζύγιο.

    docker compose exec backend python evaluation/concurrency_benchmark.py
    docker compose exec backend python evaluation/concurrency_benchmark.py \
        --threads 2,4,8 --users 1,2,4,8
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


def pct(values: list, p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = max(0, min(len(s) - 1, round((p / 100) * (len(s) - 1))))
    return s[k]


def load_english_questions(n: int) -> list:
    """ΜΟΝΟ αγγλικές, in-corpus. Οι ελληνικές θα καλούσαν το optimize_query ->
    κλήση Gemini -> θόρυβος δικτύου μέσα στη μέτρηση CPU (και κάψιμο quota)."""
    out = []
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            q = t["question"]
            if t.get("category") != "out_of_corpus" and not ai_core._has_greek(q):
                out.append(q)
    return (out * ((n // max(1, len(out))) + 1))[:n]


async def timed_search(question: str) -> float:
    t0 = time.perf_counter()
    await ai_core.search_documents(question)
    return time.perf_counter() - t0


async def run_cell(questions: list, users: int, waves: int) -> dict:
    """`waves` διαδοχικά κύματα των `users` ταυτόχρονων ερωτήσεων.

    ΔΥΟ ΛΑΘΗ ΠΟΥ ΔΙΟΡΘΩΝΟΝΤΑΙ ΕΔΩ, και τα δύο έκαναν την πρώτη εκδοχή άχρηστη:

    1. ΕΝΑ ΔΕΙΓΜΑ ΑΝΑ ΚΕΛΙ. Με users=1 έτρεχε ΜΙΑ ερώτηση και το «p50» ήταν
       διάμεσος ενός δείγματος — καθαρός θόρυβος, που έβγαζε το 2-threads
       ταχύτερο από το 8 και αντέφασκε με το measure_latency.py.
    2. ΔΙΑΦΟΡΕΤΙΚΕΣ ΕΡΩΤΗΣΕΙΣ ΑΝΑ ΚΕΛΙ. Το users=1 έπαιρνε την ερώτηση #1, το
       users=8 τις #1-8. Άλλα chunks, άλλα μήκη, άλλος χρόνος rerank -> η
       σύγκριση μετρούσε τις ερωτήσεις, όχι τον ανταγωνισμό CPU.

    Τώρα ΚΑΘΕ κελί βλέπει τον ΙΔΙΟ κυκλικό κύκλο ερωτήσεων, και μαζεύονται
    users x waves δείγματα ώστε το p50/p95 να σημαίνει κάτι.
    """
    all_lat, wall_total = [], 0.0
    for w in range(waves):
        # Κυκλικά από την ΙΔΙΑ λίστα: ίδιο μείγμα φορτίου σε κάθε κελί.
        batch = [questions[(w * users + i) % len(questions)] for i in range(users)]
        wall0 = time.perf_counter()
        all_lat.extend(await asyncio.gather(*(timed_search(q) for q in batch)))
        wall_total += time.perf_counter() - wall0
    return {"p50": statistics.median(all_lat), "p95": pct(all_lat, 95),
            "max": max(all_lat), "n": len(all_lat),
            "throughput": len(all_lat) / wall_total}


async def main_async(user_counts: list, waves: int) -> int:
    # Αρκετές διαφορετικές ερωτήσεις ώστε ο κυκλικός κύκλος να μην επαναλαμβάνει
    # συνέχεια την ίδια (που θα ευνοούσε τεχνητά τα caches).
    questions = load_english_questions(max(16, max(user_counts) * 2))
    print(f"Ερωτήσεις: {len(questions)} αγγλικές in-corpus (καμία κλήση Gemini)")
    print(f"RERANK_CANDIDATES={ai_core.RERANK_CANDIDATES} "
          f"batch={ai_core.RERANK_BATCH_SIZE} · CPUs={os.cpu_count()} "
          f"· κύματα ανά κελί: {waves}\n")

    # Ζέσταμα ΕΞΩ από τη μέτρηση: BM25 index, dense matrix, query embeddings.
    print("Warmup...")
    for q in questions:
        await ai_core.search_documents(q)

    threads = torch.get_num_threads()
    # Επαλήθευση ΜΕΣΑ στο worker thread, όχι στο main: εκεί τρέχει ο reranker,
    # και εκεί μετράει ποια τιμή ισχύει πραγματικά.
    worker_threads = await asyncio.to_thread(torch.get_num_threads)
    print(f"\n=== TORCH_THREADS = {threads} "
          f"(worker threads: {worker_threads}) ===")
    if worker_threads != threads:
        print("  ⚠️  main και worker διαφέρουν — η μέτρηση ΔΕΝ είναι έγκυρη.")
    print(f"{'χρήστες':>8}{'p50':>9}{'p95':>9}{'max':>9}"
          f"{'throughput':>13}{'δείγμ.':>8}{'p50 vs 1':>10}")
    print("-" * 66)

    results, base_p50 = {}, None
    for users in user_counts:
        r = await run_cell(questions, users, waves)
        results[users] = r
        if base_p50 is None:
            base_p50 = r["p50"]
        print(f"{users:>8}{r['p50']:>8.2f}s{r['p95']:>8.2f}s{r['max']:>8.2f}s"
              f"{r['throughput']:>10.2f} req/s{r['n']:>8}"
              f"{r['p50'] / base_p50:>9.2f}x")

    print("\n--- ΠΟΡΙΣΜΑ ---")
    best_thr_users = max(results, key=lambda u: results[u]["throughput"])
    print(f"• Κορεσμός throughput στους {best_thr_users} ταυτόχρονους "
          f"({results[best_thr_users]['throughput']:.2f} req/s). Πέρα από εκεί, "
          f"περισσότεροι χρήστες ΔΕΝ δίνουν περισσότερες απαντήσεις — απλώς "
          f"περιμένουν όλοι περισσότερο.")
    for users in user_counts:
        r = results[users]
        # Ιδανική γραμμική κλιμάκωση θα κρατούσε το p50 σταθερό. Ο λόγος p50/base
        # δείχνει πόσο από τον ανταγωνισμό CPU πληρώνει ο κάθε χρήστης.
        print(f"  {users:>2} ταυτόχρονοι: p50 {r['p50']:.2f}s "
              f"({r['p50'] / base_p50:.2f}x του μόνου χρήστη) · "
              f"p95 {r['p95']:.2f}s · {r['throughput']:.2f} req/s")
    print("\nΓια σύγκριση thread counts τρέξε το ΞΑΝΑ σε νέα διεργασία με "
          "-e TORCH_THREADS=N (βλ. σχόλιο στην κορυφή του αρχείου).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="1,2,4,8")
    ap.add_argument("--waves", type=int, default=4,
                    help="κύματα ανά κελί — περισσότερα = λιγότερος θόρυβος")
    a = ap.parse_args()
    return asyncio.run(main_async(
        [int(x) for x in a.users.split(",")], a.waves))


if __name__ == "__main__":
    sys.exit(main())
