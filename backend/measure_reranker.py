"""Βαθμονόμηση relevance gate για το bge-reranker-v2-m3.
Τρέξε: docker compose exec backend python measure_reranker.py"""
import asyncio

from ai_core import collection, optimize_query, reranker

# Τα PDF σου είναι για cloud/serverless. RELEVANT = μέσα στα papers, IRRELEVANT = εκτός.
#
# ΤΡΕΙΣ ομάδες, όχι δύο. Το golden_set_50 αποκάλυψε ότι ο reranker συμπεριφέρεται
# πολύ διαφορετικά σε σελίδες ΠΡΟΖΑΣ και σε σελίδες ΔΟΜΗΜΕΝΕΣ (πίνακες αριθμών,
# μπλοκ εντολών). Η q027 — νόμιμη in-corpus ερώτηση προς πίνακα — κόπηκε από το
# gate με score 0.08, ενώ οι out-of-corpus σκοράρουν έως 0.02. Το περιθώριο είναι
# πολύ στενότερο απ' ό,τι δείχνει η μέτρηση με πρόζα μόνο.
QUERIES = [
    # Πρόζα — ο τύπος πάνω στον οποίο βαθμονομήθηκε το 0.15
    ("What is serverless computing and how does it simplify cloud programming?", "REL-proza"),
    ("What are the main obstacles to adopting cloud computing?",                 "REL-proza"),
    ("What is elasticity in cloud computing?",                                   "REL-proza"),
    # Δομημένες σελίδες — οι δύο πρώτες ΑΠΕΤΥΧΑΝ στο golden_set_50 (q027, q028)
    ("Which storage categories does the paper compare in its storage table, "
     "and what mean latency does each have?",                                    "REL-pinakas"),
    ("Which encoder and command-line settings did the ExCamera evaluation use?", "REL-pinakas"),
    # Πίνακας που ΔΟΥΛΕΨΕ (q025, MRR 1.0) — έχει επεξηγηματικό κείμενο στα κελιά,
    # σε αντίθεση με τους δύο παραπάνω που είναι σκέτοι αριθμοί/σημαίες.
    ("What electricity prices per kilowatt-hour does the paper list "
     "and why do they differ?",                                                  "REL-pinakas"),
    ("What is the capital of Australia?",                                        "IRRELEVANT"),
    ("How do I bake a chocolate cake?",                                          "IRRELEVANT"),
    ("What are the health benefits of running every day?",                       "IRRELEVANT"),
]

async def main():
    print(f"\n{'LABEL':<13}{'BEST':>9}   top-5 reranker scores")
    print("-" * 76)
    for q, label in QUERIES:
        query = await optimize_query(q)  # no-op στα αγγλικά (μεταφράζει μόνο τα ελληνικά)
        res = collection.query(query_texts=[query], n_results=15)
        docs = res["documents"][0]
        if not docs:
            print(f"{label:<13}{'—':>9}   (κανένα candidate)  | {q}")
            continue
        scores = sorted((float(s) for s in reranker.predict([[query, d] for d in docs])), reverse=True)
        top5 = " ".join(f"{s:+.3f}" for s in scores[:5])
        gate = "" if scores[0] >= 0.15 else "   ← ΚΟΒΕΤΑΙ από το gate (0.15)"
        print(f"{label:<13}{scores[0]:>+9.3f}   {top5}{gate}")
        print(f"{'':<13}{'':>9}   ↳ {q[:60]}")
    print("-" * 76)

asyncio.run(main())