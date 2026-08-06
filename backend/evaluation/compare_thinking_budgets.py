"""Χάνει περιεχόμενο η απάντηση όταν κόβουμε το thinking budget;

ΤΟ ΕΡΩΤΗΜΑ: το probe_thinking_budget.py έδειξε 3.76x ταχύτερο TTFT με
thinkingBudget=0, αλλά 34% ΜΙΚΡΟΤΕΡΗ απάντηση σε μία ερώτηση. Μικρότερη δεν
σημαίνει χειρότερη — μπορεί να είναι λιγότερη φλυαρία. Χρειάζεται αντικειμενικό
κριτήριο.

ΤΟ ΚΡΙΤΗΡΙΟ: το golden_set_50 έχει ήδη `keywords` ανά ερώτηση — τους όρους που
ΠΡΕΠΕΙ να περιέχει μια σωστή απάντηση. Μετράμε πόσα από αυτά εμφανίζονται στην
ΑΠΑΝΤΗΣΗ (όχι στο context, όπως κάνει το retrieval eval). Αυτό είναι φθηνό,
ντετερμινιστικό και δεν χρειάζεται LLM-judge -> μηδέν επιπλέον quota πέρα από
τις ίδιες τις κλήσεις γέννησης.

ΓΙΑΤΙ ENUMERATION: οι ερωτήσεις "λίστα όλων των X" είναι εκεί όπου η πληρότητα
πονάει πρώτη. Αν το thinking βοηθάει κάπου, είναι εδώ· αν το κόψιμό του χαλάει
κάτι, θα φανεί εδώ πριν φανεί οπουδήποτε αλλού.

ΚΟΣΤΟΣ: n ερωτήσεις x 3 συνθήκες κλήσεις Gemini (default 4 x 3 = 12).

    docker compose exec backend python evaluation/compare_thinking_budgets.py
    docker compose exec backend python evaluation/compare_thinking_budgets.py --n 6
"""
import argparse
import asyncio
import json
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probe_thinking_budget import call_rest

import ai_core

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden_set_50.jsonl")
BUDGETS = [("control", None), ("budget=0", 0), ("budget=512", 512)]


def norm(text: str) -> str:
    """lowercase + αφαίρεση τόνων — ίδια λογική με το el_tokenize, ώστε το
    matching να μη χάνεται σε τονισμό ή κεφαλαία."""
    text = text.lower()
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")


def keyword_hits(answer: str, keywords: list) -> float:
    """Ποσοστό των keywords που εμφανίζονται στην απάντηση."""
    if not keywords:
        return float("nan")
    hay = norm(answer)
    hits = sum(1 for k in keywords if norm(str(k)) in hay)
    return hits / len(keywords)


def load_tests(categories: set, n: int) -> list:
    out = []
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            if t.get("category") in categories and t.get("keywords"):
                out.append(t)
    return out[:n]


async def build_prompt(question: str) -> str:
    """ΙΔΙΟ prompt με την παραγωγή: ίδιο system prompt, ίδια persona, ίδιο
    context. Αλλιώς δεν συγκρίνουμε το thinking budget αλλά δύο άλλα prompts."""
    pages = await ai_core.search_documents(question)
    if not pages:
        return ""
    context = "".join(
        "\n[Source: {}, Page: {}]\n{}\n".format(
            m.get("file_name"), m.get("page"), t) for t, m in pages)
    persona_style = ai_core.PERSONA_STYLES["Researcher"]
    system_prompt = f"""You are an expert Research Assistant. Provide precise, evidence-based answers using ONLY the provided SOURCE TEXT.

    STRICT PROTOCOLS:
    1. LANGUAGE ENFORCEMENT: You MUST answer in the EXACT SAME LANGUAGE as the user's question.
    2. STYLE: {persona_style}
    3. NO HALLUCINATIONS: Base your answer EXCLUSIVELY on the SOURCE TEXT. If the answer is not in the text, clearly state that you cannot find the answer in the provided documents.
    4. FORMATTING: If you use a numbered list, number the items sequentially starting from 1.
    5. COMPLETENESS: List EVERY relevant item present in the sources.
    """
    return (f"{system_prompt}\n\n--- SOURCE TEXT ---\n{context}\n\n"
            f"CURRENT QUESTION: {question}\n\nANSWER:")


async def main_async(n: int, categories: set) -> int:
    tests = load_tests(categories, n)
    print(f"Ερωτήσεις: {len(tests)} ({'/'.join(sorted(categories))}) "
          f"· συνθήκες: {len(BUDGETS)} · κλήσεις Gemini: {len(tests) * len(BUDGETS)}\n")

    rows = {label: [] for label, _ in BUDGETS}
    for i, t in enumerate(tests, 1):
        prompt = await build_prompt(t["question"])
        if not prompt:
            print(f"{i}. (παραλείπεται — το gate έκοψε) {t['question'][:50]}")
            continue
        print(f"{i}. {t['question'][:66]}")
        for label, budget in BUDGETS:
            r = await call_rest(prompt, budget)
            if "error" in r:
                print(f"     {label:<11} ΣΦΑΛΜΑ: {r['error'][:90]}")
                continue
            cov = keyword_hits(r["text"], t["keywords"])
            rows[label].append({"cov": cov, "ttft": r["ttft"] or 0,
                                "total": r["total"], "chars": len(r["text"]),
                                "thinking": r["thinking"]})
            print(f"     {label:<11} keywords {100 * cov:>5.1f}% · "
                  f"TTFT {r['ttft'] or 0:>4.2f}s · {len(r['text']):>5} χαρ. · "
                  f"thinking {r['thinking']:>4}")

    print(f"\n{'συνθήκη':<12}{'keyword cov':>13}{'TTFT':>9}{'σύνολο':>9}"
          f"{'χαρακτ.':>10}{'thinking':>10}")
    print("-" * 63)
    summary = {}
    for label, _ in BUDGETS:
        rs = rows[label]
        if not rs:
            continue
        avg = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
        summary[label] = avg
        print(f"{label:<12}{100 * avg['cov']:>12.1f}%{avg['ttft']:>8.2f}s"
              f"{avg['total']:>8.2f}s{avg['chars']:>10.0f}{avg['thinking']:>10.0f}")

    if "control" in summary:
        base = summary["control"]
        print("\n--- Έναντι control ---")
        for label, avg in summary.items():
            if label == "control":
                continue
            dcov = 100 * (avg["cov"] - base["cov"])
            spd = base["ttft"] / avg["ttft"] if avg["ttft"] else 0
            print(f"  {label:<11} keyword coverage {dcov:+.1f}pp · "
                  f"TTFT {spd:.2f}x ταχύτερο")
        print("\nΚΡΙΤΗΡΙΟ: coverage να ΜΗΝ πέσει (>= -1pp). Η ταχύτητα δεν "
              "αγοράζεται με χαμένο περιεχόμενο.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="ερωτήσεις (x3 κλήσεις!)")
    ap.add_argument("--categories", default="enumeration",
                    help="κατηγορίες golden set, χωρισμένες με κόμμα")
    a = ap.parse_args()
    return asyncio.run(main_async(a.n, set(a.categories.split(","))))


if __name__ == "__main__":
    sys.exit(main())
