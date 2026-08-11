"""Μπορεί το ΙΔΙΟ το μοντέλο γέννησης να πει «αυτές οι σελίδες δεν απαντούν»;

ΤΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΣΤΟΧΕΥΕΙ
-------------------------
Το gate πιάνει το «δεν ξέρω» και ΟΧΙ το «νομίζω ότι ξέρω». Το h002 περνάει με
best +1.27 ενώ η σελίδα-στόχος δεν έφτασε ποτέ, οπότε ο corrective agent δεν
τρέχει καν. Τρεις προσπάθειες να γίνει το gate πιο ευαίσθητο απέτυχαν
(άνοιγμα κατωφλίου, απόλυτη βαθμολογία, best−mean15): κανένα σήμα εκείνου του
σταδίου δεν ξεχωρίζει «δεν βρήκα» από «βρήκα λάθος», γιατί ο cross-encoder
μετράει ΛΕΞΙΛΟΓΙΚΗ επικάλυψη.

Ο generator όμως **διαβάζει τις σελίδες**. Είναι το μόνο σημείο του αγωγού που
ξέρει τι ΛΕΕΙ το κείμενο. Εδώ μετριέται αν αυτή η γνώση μπορεί να γίνει σήμα.

ΤΙ ΚΑΝΕΙ
--------
Τρέχει τον ΠΡΑΓΜΑΤΙΚΟ αγωγό ως τις σελίδες (translate -> dense -> BM25 -> RRF ->
rerank -> expand), και μετά ρωτάει το Gemini μία ερώτηση ΝΑΙ/ΟΧΙ πάνω στις ΙΔΙΕΣ
σελίδες. Δεν αλλάζει τίποτα στην παραγωγή.

Ο έλεγχος είναι ΑΣΥΜΜΕΤΡΟΣ και εκεί κρίνεται:
  · in-corpus (45)      -> πρέπει ΟΛΕΣ ΝΑΙ.  Έστω ΜΙΑ λανθασμένη άρνηση και η
                          ιδέα πέφτει: το gate σήμερα έχει ΜΗΔΕΝ εκεί.
  · out_of_corpus (5)   -> πρέπει ΟΛΕΣ ΟΧΙ (το gate τις κόβει ήδη· εδώ ελέγχεται
                          αν το σήμα στέκει ΜΟΝΟ του).
  · hard set (16)       -> το ενδιαφέρον· ειδικά το h002 πρέπει να βγει ΟΧΙ.

Το gate ΔΕΝ εφαρμόζεται: θέλουμε τη διακριτική ικανότητα του σήματος καθαρή.
Η στήλη `cut` λέει τι θα έκανε το gate, ώστε να φανεί ΤΙ ΠΡΟΣΘΕΤΕΙ η ετυμηγορία.

ΚΟΣΤΟΣ: 1 κλήση/ερώτηση, ~9.700 tokens εισόδου και ~2 εξόδου -> λίγα λεπτά του
δολαρίου συνολικά. Η γέννηση απάντησης ΔΕΝ τρέχει.

    docker compose exec backend python evaluation/probe_grounding_verdict.py
    docker compose exec backend python evaluation/probe_grounding_verdict.py \
        --csv evaluation/runs/grounding.csv
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
SETS = ["golden_set_50.jsonl", "golden_multihop_new.jsonl",
        "golden_hard_paraphrase.jsonl"]

# Σκόπιμα ΑΥΣΤΗΡΟ στο σχήμα (μία λέξη) και ΧΑΛΑΡΟ στο κριτήριο («έστω και
# μερικώς»): αν βγει ψευδώς αρνητικό ΕΔΩ, δεν σώζεται με πιο επιεική διατύπωση.
VERDICT_PROMPT = """You are verifying whether retrieved material is sufficient to answer a question.

Read the SOURCE TEXT and the QUESTION. Decide whether the SOURCE TEXT actually
contains the information needed to answer the QUESTION, even partially.

Answer with EXACTLY ONE WORD:
YES - the SOURCE TEXT contains information that answers the question
NO  - the SOURCE TEXT does not contain it; answering would need outside knowledge

Do not explain. Output only YES or NO.

--- SOURCE TEXT ---
{context}

QUESTION: {question}

VERDICT:"""


def load_sets(only=None):
    out = []
    for name in SETS:
        if only and name != only:
            continue
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    t["_set"] = name
                    out.append(t)
    return out


def coverage(text: str, keywords) -> float:
    if not keywords:
        return float("nan")
    low = text.lower()
    return 100.0 * sum(1 for kw in keywords if str(kw).lower() in low) / len(keywords)


async def verdict_for(context: str, question: str, retries: int = 4) -> str:
    """Επιστρέφει 'YES' / 'NO' / 'ERR'. Backoff στα rate limits, όπως ο agent."""
    prompt = VERDICT_PROMPT.format(context=context, question=question)
    for attempt in range(retries):
        try:
            raw = await gemini_rest.generate_once(
                prompt, model=ai_core.GEMINI_MODEL, api_key=ai_core.GEMINI_API_KEY)
            token = (raw or "").strip().strip('".\'').upper()
            if token.startswith("YES"):
                return "YES"
            if token.startswith("NO"):
                return "NO"
            return f"?{token[:12]}"
        except Exception as e:   # best effort: το probe δεν πρέπει να σκάει
            if attempt == retries - 1:
                print(f"    (σφάλμα κλήσης: {type(e).__name__})")
                return "ERR"
            await asyncio.sleep(2 ** (attempt + 1))
    return "ERR"


async def pages_for(question_raw, allowed_ids, idx, dm):
    """Ο πραγματικός αγωγός ως τις σελίδες. Χωρίς gate, χωρίς corrective."""
    q = await ai_core.optimize_query(question_raw)
    dense_ids = ai_core._dense_exact_ids(
        dm, q, allowed_ids, min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
    sparse_ids = ai_core._bm25_sparse_ids(
        idx, q, allowed_ids, ai_core.DENSE_CANDIDATES)
    rrf = ai_core._rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                            idx["metas"], k=60,
                            top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
    pairs = [[q, item[1]] for item in rrf]
    scores = await asyncio.to_thread(
        lambda: ai_core.reranker.predict(pairs, batch_size=ai_core.RERANK_BATCH_SIZE))
    ranked = sorted(zip([float(s) for s in scores], [it[1] for it in rrf],
                        [it[2] for it in rrf]), key=lambda x: x[0], reverse=True)
    pages = await asyncio.to_thread(
        ai_core._expand_to_pages, ranked[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)
    return ranked[0][0], pages, q


async def main(args) -> int:
    tests = load_sets(args.only)
    if args.limit:
        tests = tests[:args.limit]
    print(f"{len(tests)} ερωτήσεις · gate={ai_core.MIN_RERANK_SCORE} "
          f"(ΔΕΝ εφαρμόζεται, μόνο καταγράφεται)\n")

    where_filter = ai_core._build_where(None, None)
    allowed_ids = ai_core.collection.get(where=where_filter, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    rows = []
    for i, t in enumerate(tests, 1):
        best, pages, q_en = await pages_for(t["question"], allowed_ids, idx, dm)
        ctx = "\n\n".join(txt for txt, _m in pages)
        cov = coverage(ctx, t.get("keywords") or [])
        # Κρίνεται η ΜΕΤΑΦΡΑΣΜΕΝΗ ερώτηση, η ίδια που είδε η ανάκτηση. Με την ωμή
        # ελληνική πάνω σε αγγλικό κείμενο το σήμα καταρρέει: 14/20 λανθασμένες
        # αρνήσεις στα ελληνικά έναντι 2/36 στα αγγλικά (runs/grounding.csv).
        v = await verdict_for(ctx, t["question"] if args.raw_question else q_en)
        rows.append({
            "id": t["id"], "set": t["_set"], "category": t.get("category", ""),
            "best": round(best, 3), "cut": best < ai_core.MIN_RERANK_SCORE,
            "coverage": None if cov != cov else round(cov, 1),
            "verdict": v, "n_pages": len(pages),
        })
        print(f"  {i:>3}/{len(tests)}  {t['id']:<6} best {best:+7.2f}  "
              f"cov {'-' if cov != cov else f'{cov:5.1f}'}  -> {v}", flush=True)
        if args.sleep:
            await asyncio.sleep(args.sleep)

    # ------------------------------------------------------------ αναφορά
    print("\n" + "=" * 78)
    groups = [
        ("in-corpus κύριο (πρέπει ΟΛΕΣ ΝΑΙ)",
         [r for r in rows if r["set"] == "golden_set_50.jsonl"
          and r["category"] != "out_of_corpus"], "YES"),
        ("out_of_corpus (πρέπει ΟΛΕΣ ΟΧΙ)",
         [r for r in rows if r["category"] == "out_of_corpus"], "NO"),
        ("multi_hop σύνολο (πρέπει ΟΛΕΣ ΝΑΙ)",
         [r for r in rows if r["set"] == "golden_multihop_new.jsonl"], "YES"),
    ]
    for label, sub, want in groups:
        if not sub:
            continue
        ok = [r for r in sub if r["verdict"] == want]
        bad = [r for r in sub if r["verdict"] != want]
        print(f"{label}: {len(ok)}/{len(sub)} σωστά")
        for r in bad:
            print(f"    ΛΑΘΟΣ {r['id']:<6} verdict={r['verdict']:<4} "
                  f"cov={r['coverage']}  best={r['best']:+.2f}")

    hard = [r for r in rows if r["set"] == "golden_hard_paraphrase.jsonl"]
    if hard:
        print(f"\nδύσκολο σύνολο ({len(hard)}):")
        for r in hard:
            flag = ""
            if r["coverage"] == 0 and r["verdict"] == "NO":
                flag = "  <== ΣΩΣΤΑ ΤΟ ΕΠΙΑΣΕ"
            elif r["coverage"] == 0 and r["verdict"] == "YES":
                flag = "  <== ΤΟ ΕΧΑΣΕ"
            print(f"    {r['id']:<6} cov {r['coverage']!s:>5}  best {r['best']:+7.2f}  "
                  f"{'ΚΟΜΜΕΝΗ' if r['cut'] else 'περνάει'}  -> {r['verdict']}{flag}")

    # Το κρίσιμο: συμφωνεί η ετυμηγορία με το ΑΝ ΥΠΑΡΧΕΙ ΥΛΙΚΟ (coverage);
    known = [r for r in rows if r["coverage"] is not None]
    zero = [r for r in known if r["coverage"] == 0]
    some = [r for r in known if r["coverage"] > 0]
    print("\nΕΤΥΜΗΓΟΡΙΑ vs ΠΡΑΓΜΑΤΙΚΟ ΥΛΙΚΟ")
    print(f"  κάλυψη 0%  (n={len(zero):<3}): ΟΧΙ σε {sum(1 for r in zero if r['verdict'] == 'NO')}"
          f"  ΝΑΙ σε {sum(1 for r in zero if r['verdict'] == 'YES')}")
    print(f"  κάλυψη >0% (n={len(some):<3}): ΝΑΙ σε {sum(1 for r in some if r['verdict'] == 'YES')}"
          f"  ΟΧΙ σε {sum(1 for r in some if r['verdict'] == 'NO')}")
    print("=" * 78)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ γράφτηκε {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ετυμηγορία θεμελίωσης από τον generator.")
    ap.add_argument("--only", default=None, help="μόνο ένα σύνολο")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.4, help="παύση μεταξύ κλήσεων")
    ap.add_argument("--raw-question", action="store_true",
                    help="κρίνε με την ΩΜΗ ερώτηση (αναπαράγει το σφάλμα γλώσσας)")
    ap.add_argument("--csv", default=None)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
