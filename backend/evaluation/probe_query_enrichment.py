"""Αξίζει να ΕΜΠΛΟΥΤΙΖΟΥΜΕ το ερώτημα με ορολογία; — offline probe, ΜΗΔΕΝ αλλαγή κώδικα.

ΤΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΨΑΧΝΕΙ ΝΑ ΛΥΣΕΙ (μετρημένο, trace_query.py 10/8/2026):
Το h002 («What exact settings did they use when they ran the encoder?») ζητά την
ΙΔΙΑ σελίδα με το q028 («Which specific vpxenc command-line flags...»). Το q028
τη βρίσκει στη θέση 1 του BM25 επειδή περιέχει το `vpxenc` — μοναδικό σε όλο το
corpus. Το h002 λέει σκέτο «encoder» και η σελίδα ΔΕΝ ΕΡΧΕΤΑΙ ΠΟΤΕ.
Υπόθεση: αν προσθέσουμε πιθανούς τεχνικούς όρους στο ερώτημα, το BM25 ξαναβλέπει.

ΓΙΑΤΙ ΔΕΝ ΤΟ ΚΑΛΥΠΤΕΙ ΤΟ ΥΠΑΡΧΟΝ optimize_query:
ai_core.py:422 -> `if not _has_greek(query): return query`. Οι ΑΓΓΛΙΚΕΣ ερωτήσεις
δεν περνούν καθόλου. Το hard set είναι αγγλικό. Για να δουλέψει enrichment θα
έπρεπε να καλείται το Gemini σε ΚΑΘΕ ερώτηση: +0.5-0.9 s παντού + 1 κλήση πάντα
— ΑΚΡΙΒΩΣ το κόστος για το οποίο απορρίφθηκε το query decomposition.
Άρα το μετράμε ΠΡΙΝ αγγίξουμε κώδικα.

ΤΟ ΚΡΙΣΙΜΟ ΔΕΝ ΕΙΝΑΙ ΤΟ HARD SET — ΕΙΝΑΙ ΤΑ out_of_corpus:
Ένα ερώτημα «τιμή Bitcoin» εμπλουτισμένο σε «Bitcoin price distributed ledger
consensus» μπορεί να ΜΟΙΑΣΕΙ σχετικό με cloud papers και να ΠΕΡΑΣΕΙ το gate.
Δηλαδή ένας εμπλουτισμός που κερδίζει 2 hard ερωτήσεις μπορεί να ΣΠΑΣΕΙ την
άμυνα κατά της ψευδαίσθησης. Αν έστω ΕΝΑ out_of_corpus περάσει, ΑΠΟΡΡΙΠΤΕΤΑΙ
ανεξάρτητα από το τι κέρδισε.

ΚΟΣΤΟΣ: 1 κλήση Gemini ανά ερώτηση (~21 για hard+ooc). Ο corrective ΔΕΝ τρέχει.

    docker compose exec backend python evaluation/probe_query_enrichment.py
    docker compose exec backend python evaluation/probe_query_enrichment.py \\
        --prompt-file evaluation/prompts/enrich_v2.txt --csv evaluation/runs/enrich.csv
"""
import argparse
import asyncio
import csv
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core
import gemini_rest

HERE = "/app/evaluation"

# Ουδέτερο επίτηδες: ΔΕΝ ζητάει από το μοντέλο να κρίνει αν η ερώτηση είναι
# εντός πεδίου — αυτή είναι δουλειά του gate. Αν του δίναμε δικαίωμα κρίσης,
# θα μετρούσαμε το φίλτρο του Gemini, όχι τον εμπλουτισμό.
DEFAULT_PROMPT = (
    "You expand search queries for an academic search engine. The corpus is "
    "computer-science papers on cloud computing, serverless computing and "
    "distributed systems.\n"
    "Rewrite the question as a search query, KEEPING every word of the original "
    "intent, and ADD 2-4 specific technical terms that such a paper would "
    "plausibly use for this topic (tool names, parameter names, standard "
    "jargon).\n"
    "Output ONLY the search query. No quotes, no explanation.\n\n"
    "Question: {q}"
)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


async def enrich(question: str, template: str, max_words: int = 0) -> str:
    """max_words > 0: ΜΗΧΑΝΙΚΟ όριο μήκους, ανεξάρτητο από το prompt.

    ΓΙΑΤΙ ΔΕΝ ΑΡΚΕΙ Ο ΚΑΝΟΝΑΣ ΣΤΟ PROMPT: στο v1 το μοντέλο παρήγαγε queries με
    100+ όρους («crypto price OR crypto value OR crypto cost OR ...») και ένα
    από αυτά ΠΕΡΑΣΕ το gate σε out_of_corpus ερώτηση. Ένα prompt είναι παράκληση·
    το κόψιμο εδώ είναι εγγύηση.
    """
    try:
        out = await gemini_rest.generate_once(
            template.format(q=question), model=ai_core.GEMINI_MODEL,
            api_key=ai_core.GEMINI_API_KEY)
        out = out.strip(' "\'\n') or question
    except Exception as e:
        print(f"  (enrich απέτυχε: {e})")
        return question
    if max_words and len(out.split()) > max_words:
        out = " ".join(out.split()[:max_words])
        print(f"  (κόπηκε στα {max_words} λέξεις)")
    return out


def pipeline(query, idx, dm, allowed_ids):
    """Επιστρέφει (best_logit, σελίδες) ΧΩΡΙΣ corrective — μετράμε το gate καθαρά."""
    dense_ids = ai_core._dense_exact_ids(
        dm, query, allowed_ids, min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
    sparse_ids = ai_core._bm25_sparse_ids(
        idx, query, allowed_ids, ai_core.DENSE_CANDIDATES)
    rrf = ai_core._rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                            idx["metas"], k=60,
                            top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
    scores = ai_core.reranker.predict([[query, it[1]] for it in rrf],
                                      batch_size=ai_core.RERANK_BATCH_SIZE)
    ranked = sorted(zip(scores, [it[1] for it in rrf], [it[2] for it in rrf]),
                    key=lambda x: x[0], reverse=True)
    best = float(ranked[0][0])
    if best < ai_core.MIN_RERANK_SCORE:
        return best, []
    return best, ai_core._expand_to_pages(ranked[:ai_core.EXPAND_INPUT],
                                          ai_core.MAX_PAGES)


def coverage(pages, keywords):
    if not pages:
        return 0.0
    blob = "\n".join(t for t, _m in pages).lower()
    return 100.0 * sum(1 for k in keywords if k.lower() in blob) / len(keywords)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="golden_hard_paraphrase.jsonl,golden_set_50.jsonl")
    ap.add_argument("--only-ooc-from", default="golden_set_50.jsonl",
                    help="από αυτό το σετ κρατάμε ΜΟΝΟ τα out_of_corpus")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-words", type=int, default=0,
                    help="μηχανικό όριο λέξεων στο εμπλουτισμένο query (0 = χωρίς)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    template = DEFAULT_PROMPT
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            template = f.read()
        print(f"prompt: {args.prompt_file}")

    tests = []
    for name in args.sets.split(","):
        path = os.path.join(HERE, name.strip())
        if not os.path.exists(path):
            print(f"(παραλείπω, δεν υπάρχει: {path})")
            continue
        for t in load(path):
            if name.strip() == args.only_ooc_from and t.get("category") != "out_of_corpus":
                continue
            tests.append(t)

    ooc = [t for t in tests if t.get("category") == "out_of_corpus"]
    print(f"{len(tests)} ερωτήσεις ({len(ooc)} out_of_corpus) — "
          f"{len(tests)} κλήσεις Gemini\n")

    allowed_ids = ai_core.collection.get(
        where=ai_core._build_where(None, None), include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    rows = []
    tally = {"saved": [], "lost": [], "leak": [], "same": 0}

    for n, t in enumerate(tests, start=1):
        kws = t.get("keywords", [])
        # ΤΟ BASELINE ΠΡΕΠΕΙ ΝΑ ΠΕΡΑΣΕΙ ΑΠΟ optimize_query, ΟΠΩΣ Η ΠΑΡΑΓΩΓΗ.
        # Χωρίς αυτό, οι ΕΛΛΗΝΙΚΕΣ ερωτήσεις συγκρίνονταν ωμές (αμετάφραστες)
        # με εμπλουτισμένες -> τεχνητά χαμηλό baseline και ψευδείς «σωτηρίες».
        # Βρέθηκε 10/8/2026: h012 έδινε -1.477 εδώ και -5.30 στο trace_query.
        q = await ai_core.optimize_query(t["question"])
        eq = await enrich(t["question"], template, args.max_words)
        b0, p0 = pipeline(q, idx, dm, allowed_ids)
        b1, p1 = pipeline(eq, idx, dm, allowed_ids)
        c0, c1 = coverage(p0, kws), coverage(p1, kws)
        is_ooc = t.get("category") == "out_of_corpus"

        verdict = "ίδιο"
        if is_ooc:
            if p1:
                verdict = "*** ΔΙΑΡΡΟΗ ***"
                tally["leak"].append(t["id"])
        elif not p0 and p1:
            verdict = "ΣΩΘΗΚΕ"
            tally["saved"].append(t["id"])
        elif p0 and not p1:
            verdict = "*** ΧΑΘΗΚΕ ***"
            tally["lost"].append(t["id"])
        else:
            tally["same"] += 1

        print(f"[{n}/{len(tests)}] {t['id']:<6} {b0:>7.2f} -> {b1:>7.2f}  "
              f"cov {c0:>5.1f} -> {c1:>5.1f}   {verdict}")
        print(f"         «{eq[:96]}»")
        rows.append(dict(id=t["id"], category=t.get("category", ""),
                         question=q, enriched=eq,
                         best_before=round(b0, 3), best_after=round(b1, 3),
                         cov_before=round(c0, 1), cov_after=round(c1, 1),
                         verdict=verdict))

    print("\n" + "=" * 74)
    print(f"ΣΩΘΗΚΑΝ      {len(tally['saved']):>2}  {tally['saved']}")
    print(f"ΧΑΘΗΚΑΝ      {len(tally['lost']):>2}  {tally['lost']}")
    print(f"αμετάβλητες  {tally['same']:>2}")
    print(f"ΔΙΑΡΡΟΕΣ ooc {len(tally['leak']):>2}  {tally['leak']}")
    print("=" * 74)
    if tally["leak"]:
        print("*** ΑΠΟΡΡΙΠΤΕΤΑΙ: έστω ΜΙΑ διαρροή out_of_corpus ακυρώνει ό,τι")
        print("    κέρδισε. Η άμυνα κατά της ψευδαίσθησης δεν είναι διαπραγματεύσιμη.")
    elif len(tally["saved"]) > len(tally["lost"]):
        print("ΥΠΟΨΗΦΙΟ. ΕΠΟΜΕΝΑ, ΥΠΟΧΡΕΩΤΙΚΑ ΠΡΙΝ ΜΠΕΙ ΣΤΟΝ ΚΩΔΙΚΑ:")
        print("  1. πλήρες golden_set_50 (61/61 gate; MRR 0.793 να ΜΗΝ πέσει)")
        print("  2. κόστος: +1 κλήση Gemini ΣΕ ΚΑΘΕ ερώτηση + 0.5-0.9 s")
        print("     (το query decomposition απορρίφθηκε ΓΙ' ΑΥΤΟΝ ΤΟΝ ΛΟΓΟ)")
    else:
        print("ΔΕΝ ΑΞΙΖΕΙ: δεν κερδίζει περισσότερα απ' όσα χάνει.")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
