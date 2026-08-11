"""Τρίτο στάδιο: κριτής θεμελίωσης ΜΟΝΟ πάνω στο 2ο πέρασμα του corrective agent.

Η ΠΑΡΑΤΗΡΗΣΗ ΠΟΥ ΤΟ ΓΕΝΝΗΣΕ
----------------------------
Το `probe_corrective_attempts.py` έδειξε κάτι που δεν είχε φανεί ποτέ: στις
ερωτήσεις που μένουν άλυτες, η αναδιατύπωση **βρίσκει** το σωστό υλικό και μετά
το σύστημα το **πετάει**.

    h008   best -3.16   coverage 100.0%
    h009   best -4.45   coverage 100.0%

Κάλυψη 100% σημαίνει ότι ΟΛΕΣ οι λέξεις-κλειδιά της ερώτησης βρίσκονται στις
σελίδες που ανακτήθηκαν. Απορρίπτονται επειδή το `CORRECTIVE_MIN_SCORE=+0.4`
βλέπει μόνο τη βαθμολογία του cross-encoder, δηλαδή ΛΕΞΙΛΟΓΙΚΗ ομοιότητα.

ΓΙΑΤΙ ΕΔΩ ΔΟΥΛΕΥΕΙ Ο ΚΡΙΤΗΣ ΕΝΩ ΩΣ ΓΕΝΙΚΟ GATE ΑΠΕΡΡΙΦΘΗ
----------------------------------------------------------
Το `probe_grounding_verdict.py` τον απέρριψε: 4 λανθασμένες αρνήσεις σε ερωτήσεις
που δούλευαν κανονικά. Στο 2ο πέρασμα το κόστος είναι **ΑΣΥΜΜΕΤΡΟ**:
  · λανθασμένο «ΟΧΙ»  -> το σύστημα θα σώπαινε ΟΥΤΩΣ Ή ΑΛΛΩΣ. Μηδέν ζημιά.
  · σωστό «ΝΑΙ»       -> κερδίζεται ερώτηση από το μηδέν.
Και το κόστος που τον σκότωσε (1 κλήση σε ΚΑΘΕ ερώτηση) εξαφανίζεται: τρέχει
μόνο στις ΚΟΜΜΕΝΕΣ, ~12% του συνόλου και λιγότερες στην πραγματική χρήση.

ΤΑ ΤΡΙΑ ΚΡΙΤΗΡΙΑ ΠΟΥ ΣΥΓΚΡΙΝΟΝΤΑΙ (ίδιο τρέξιμο, μηδέν επιπλέον κλήσεις)
  score     : σημερινή συμπεριφορά, best2 >= CORRECTIVE_MIN_SCORE
  verdict   : ο κριτής λέει ΝΑΙ πάνω στις σελίδες του 2ου περάσματος
  ή (or)    : περνάει ό,τι ικανοποιεί ΕΣΤΩ ΕΝΑ από τα δύο

ΤΟ ΑΠΑΓΟΡΕΥΤΙΚΟ: έστω ΜΙΑ out_of_corpus που περνάει ρίχνει την ιδέα. Το «5/5
σιωπηλά» είναι το ισχυρότερο νούμερο της εργασίας.

Ο κριτής παίρνει τη **μεταφρασμένη** ερώτηση, όχι την ωμή: μετρημένο ότι με
ελληνική ερώτηση πάνω σε αγγλικό κείμενο το σήμα καταρρέει (14/20 λάθος).
Και παίρνει την ΑΡΧΙΚΗ ερώτηση, όχι την αναδιατύπωση — κρίνουμε αν απαντιέται
αυτό που ρώτησε ο χρήστης, όχι αυτό που έψαξε ο agent.

    docker compose exec backend python evaluation/probe_corrective_verdict.py \
        --csv evaluation/runs/corrective_verdict.csv
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
SETS = ["golden_set_50.jsonl", "golden_hard_paraphrase.jsonl"]

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


def load_sets():
    out = []
    for name in SETS:
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


async def _retrieve(query, allowed_ids, idx, dm):
    dense_ids = await asyncio.to_thread(
        ai_core._dense_exact_ids, dm, query, allowed_ids,
        min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
    sparse_ids = await asyncio.to_thread(
        ai_core._bm25_sparse_ids, idx, query, allowed_ids, ai_core.DENSE_CANDIDATES)
    rrf = ai_core._rrf_fuse(dense_ids, sparse_ids, idx["ids"], idx["texts"],
                            idx["metas"], k=60,
                            top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
    pairs = [[query, it[1]] for it in rrf]
    scores = await asyncio.to_thread(
        lambda: ai_core.reranker.predict(pairs, batch_size=ai_core.RERANK_BATCH_SIZE))
    return sorted(zip([float(s) for s in scores], [it[1] for it in rrf],
                      [it[2] for it in rrf]), key=lambda x: x[0], reverse=True)


async def _ask(prompt: str, retries: int = 4) -> str:
    for attempt in range(retries):
        try:
            return await gemini_rest.generate_once(
                prompt, model=ai_core.GEMINI_MODEL, api_key=ai_core.GEMINI_API_KEY)
        except Exception as e:   # best effort: το probe δεν πρέπει να σκάει
            if attempt == retries - 1:
                print(f"      (σφάλμα κλήσης: {type(e).__name__})")
                return ""
            await asyncio.sleep(2 ** (attempt + 1))
    return ""


async def main(args) -> int:
    tests = load_sets()
    thr = ai_core.CORRECTIVE_MIN_SCORE
    print(f"{len(tests)} ερωτήσεις · gate={ai_core.MIN_RERANK_SCORE} · "
          f"corrective_min={thr}\n")

    where_filter = ai_core._build_where(None, None)
    allowed_ids = ai_core.collection.get(where=where_filter, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    rows = []
    for t in tests:
        q = await ai_core.optimize_query(t["question"])
        ranked = await _retrieve(q, allowed_ids, idx, dm)
        if ranked[0][0] >= ai_core.MIN_RERANK_SCORE:
            continue                      # το gate δεν κόβει -> δεν φτάνει εδώ
        nq = (await _ask(ai_core._CORRECTIVE_PROMPT.format(query=q))).strip(' "\'\n')
        if not nq or nq.strip().lower() == q.strip().lower():
            print(f"{t['id']:<6} rewrite ταυτόσημο -> μένει σιωπηλή")
            rows.append({"id": t["id"], "set": t["_set"],
                         "category": t.get("category", ""),
                         "best2": None, "coverage": 0.0, "verdict": "-",
                         "rewrite": nq})
            continue

        r2 = await _retrieve(nq, allowed_ids, idx, dm)
        best2 = r2[0][0]
        pages = await asyncio.to_thread(
            ai_core._expand_to_pages, r2[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)
        ctx = "\n\n".join(p for p, _m in pages)
        cov = coverage(ctx, t.get("keywords") or [])
        # ΑΡΧΙΚΗ (μεταφρασμένη) ερώτηση, ΟΧΙ η αναδιατύπωση.
        raw = (await _ask(VERDICT_PROMPT.format(context=ctx, question=q))).strip().upper()
        v = "YES" if raw.startswith("YES") else ("NO" if raw.startswith("NO") else "?")
        rows.append({"id": t["id"], "set": t["_set"],
                     "category": t.get("category", ""),
                     "best2": best2, "coverage": 0.0 if cov != cov else cov,
                     "verdict": v, "rewrite": nq})
        print(f"{t['id']:<6} [{t.get('category', ''):<14}] best2 {best2:+7.2f}  "
              f"cov {'-' if cov != cov else f'{cov:5.1f}'}  κριτής {v:<3}  «{nq[:48]}»")

    if not rows:
        print("Καμία κομμένη ερώτηση.")
        return 0

    # --------------------------------------------------------- τα 3 κριτήρια
    def passes(r, mode):
        by_score = r["best2"] is not None and r["best2"] >= thr
        by_verdict = r["verdict"] == "YES"
        return {"score": by_score, "verdict": by_verdict,
                "or": by_score or by_verdict}[mode]

    print("\n" + "=" * 78)
    print(f"{'κριτήριο':<12}{'σωσμένες':>11}{'ΧΩΡΙΣ ΥΛΙΚΟ':>14}{'ΔΙΑΡΡΟΕΣ ooc':>15}")
    print("-" * 78)
    for mode in ("score", "verdict", "or"):
        saved = nomat = leaks = 0
        for r in rows:
            if not passes(r, mode):
                continue
            if r["category"] == "out_of_corpus":
                leaks += 1
            elif r["coverage"] > 0:
                saved += 1
            else:
                nomat += 1
        flag = "  <== ΑΠΑΓΟΡΕΥΤΙΚΟ" if leaks else ""
        print(f"{mode:<12}{saved:>11}{nomat:>14}{leaks:>15}{flag}")
    print("-" * 78)

    print("\nΠΟΙΕΣ ΑΛΛΑΖΟΥΝ ΜΕ ΤΟΝ ΚΡΙΤΗ")
    for r in rows:
        s, v = passes(r, "score"), passes(r, "verdict")
        if s == v:
            continue
        what = "ΚΕΡΔΙΖΕΤΑΙ" if v else "χάνεται"
        good = "σωστό υλικό" if r["coverage"] > 0 else "ΧΩΡΙΣ ΥΛΙΚΟ"
        if r["category"] == "out_of_corpus":
            good = "OUT_OF_CORPUS"
        b2 = "  -  " if r["best2"] is None else f"{r['best2']:+.2f}"
        print(f"  {r['id']:<6} {what:<11} best2 {b2}"
              f"  κάλυψη {r['coverage']:5.1f}  ({good})")
    print("=" * 78)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"→ γράφτηκε {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Κριτής θεμελίωσης ΜΟΝΟ στο 2ο πέρασμα του corrective.")
    ap.add_argument("--csv", default=None)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
