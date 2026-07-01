"""Βαθμονόμηση relevance gate για το bge-reranker-v2-m3.
Τρέξε: docker compose exec backend python measure_reranker.py"""
import asyncio
from ai_core import collection, reranker, optimize_query

# Τα PDF σου είναι για cloud/serverless. RELEVANT = μέσα στα papers, IRRELEVANT = εκτός.
QUERIES = [
    ("What is serverless computing and how does it simplify cloud programming?", "RELEVANT"),
    ("What are the main obstacles to adopting cloud computing?",                 "RELEVANT"),
    ("What is elasticity in cloud computing?",                                   "RELEVANT"),
    ("What is the capital of Australia?",                                        "IRRELEVANT"),
    ("How do I bake a chocolate cake?",                                          "IRRELEVANT"),
    ("What are the health benefits of running every day?",                       "IRRELEVANT"),
]

async def main():
    print(f"\n{'LABEL':<11}{'BEST':>9}   top-5 reranker scores")
    print("-" * 70)
    for q, label in QUERIES:
        query = await optimize_query(q)  # no-op στα αγγλικά (μεταφράζει μόνο τα ελληνικά)
        res = collection.query(query_texts=[query], n_results=15)
        docs = res["documents"][0]
        if not docs:
            print(f"{label:<11}{'—':>9}   (κανένα candidate)  | {q}")
            continue
        scores = sorted((float(s) for s in reranker.predict([[query, d] for d in docs])), reverse=True)
        top5 = " ".join(f"{s:+.3f}" for s in scores[:5])
        print(f"{label:<11}{scores[0]:>+9.3f}   {top5}")
        print(f"{'':<11}{'':>9}   ↳ {q}")
    print("-" * 70)

asyncio.run(main())