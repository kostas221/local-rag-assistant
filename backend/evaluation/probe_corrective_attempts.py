"""Βοηθούν ΠΟΛΛΑΠΛΕΣ προσπάθειες του corrective agent, ή απλώς ανεβάζουν τη διαρροή;

ΤΟ ΕΡΩΤΗΜΑ
----------
Ο agent κάνει ΜΙΑ αναδιατύπωση. Ξέρουμε ήδη ότι είναι μη ντετερμινιστική: δύο
εκτελέσεις έδωσαν «ΣΩΘΗΚΑΝ 2» και «ΣΩΘΗΚΑΝ 1» με το h015 να γυρίζει χωρίς καμία
αλλαγή στα δεδομένα. Άρα υπάρχει ΔΙΑΣΠΟΡΑ — και τρεις προσπάθειες θα την
εκμεταλλεύονταν αντί να την υφίστανται.

Ο κίνδυνος είναι συμμετρικός και δεν είναι μικρός: κρατώντας τη ΜΕΓΙΣΤΗ από N
προσπάθειες ανεβάζεις και την πιθανότητα μια ΛΑΘΟΣ αναδιατύπωση να ξεπεράσει
τυχαία το `CORRECTIVE_MIN_SCORE`. Κλασικό πρόβλημα πολλαπλών συγκρίσεων. Οι 5
ερωτήσεις out_of_corpus περνούν ΟΛΕΣ από τον agent, άρα από 5 ευκαιρίες διαρροής
πάμε σε 15. Το «5/5 σιωπηλά» είναι το ισχυρότερο νούμερο της εργασίας.

ΓΙΑΤΙ ΕΝΑ ΤΡΕΞΙΜΟ ΚΑΙ ΟΧΙ ΤΡΙΑ
-------------------------------
Παράγονται 3 ΑΝΕΞΑΡΤΗΤΕΣ αναδιατυπώσεις ανά κομμένη ερώτηση και καταγράφεται η
βαθμολογία ΚΑΘΕΜΙΑΣ ξεχωριστά. Το «τι θα έδινε N=1/2/3» υπολογίζεται μετά,
εκτός σύνδεσης, χωρίς άλλη κλήση. Μπόνους: βγαίνει επιτέλους ΝΟΥΜΕΡΟ για τη
διασπορά της αναδιατύπωσης, που σήμερα είναι καταγεγραμμένη ως προειδοποίηση.

ΤΙ ΜΕΤΡΙΕΤΑΙ ΑΝΑ N
------------------
  σωσμένες      : πέρασε το κατώφλι ΚΑΙ έφερε σωστό υλικό (coverage > 0)
  ψευδαισθήσεις : πέρασε το κατώφλι ΜΕ coverage == 0
  διαρροές      : out_of_corpus που πέρασε το κατώφλι — ΑΠΑΓΟΡΕΥΤΙΚΟ

ΚΟΣΤΟΣ: το corrective prompt παίρνει ΜΟΝΟ το ερώτημα (όχι τις σελίδες), άρα οι
κλήσεις είναι μικρές. Τρέχει μόνο στις ΚΟΜΜΕΝΕΣ ερωτήσεις.

    docker compose exec backend python evaluation/probe_corrective_attempts.py
    docker compose exec backend python evaluation/probe_corrective_attempts.py \
        --attempts 3 --csv evaluation/runs/corrective_attempts.csv
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
    """Ακριβώς τα στάδια 2-5 του search_documents / _corrective_retry."""
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
    ranked = sorted(zip([float(s) for s in scores], [it[1] for it in rrf],
                        [it[2] for it in rrf]), key=lambda x: x[0], reverse=True)
    return ranked


async def _rewrite(query: str) -> str:
    """Ίδιο prompt με την παραγωγή (`ai_core._CORRECTIVE_PROMPT`)."""
    raw = await gemini_rest.generate_once(
        ai_core._CORRECTIVE_PROMPT.format(query=query),
        model=ai_core.GEMINI_MODEL, api_key=ai_core.GEMINI_API_KEY)
    return (raw or "").strip(' "\'\n')


async def main(args) -> int:
    tests = load_sets()
    thr = ai_core.CORRECTIVE_MIN_SCORE
    print(f"{len(tests)} ερωτήσεις · gate={ai_core.MIN_RERANK_SCORE} · "
          f"corrective_min={thr} · προσπάθειες={args.attempts}\n")

    where_filter = ai_core._build_where(None, None)
    allowed_ids = ai_core.collection.get(where=where_filter, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    rows = []
    for t in tests:
        q = await ai_core.optimize_query(t["question"])
        ranked = await _retrieve(q, allowed_ids, idx, dm)
        first_best = ranked[0][0]
        if first_best >= ai_core.MIN_RERANK_SCORE:
            continue  # το gate ΔΕΝ κόβει -> ο agent δεν τρέχει ποτέ
        kw = t.get("keywords") or []
        print(f"{t['id']:<6} [{t.get('category', ''):<14}] 1ο pass {first_best:+7.2f} "
              f"-> ΚΟΜΜΕΝΗ, {args.attempts} προσπάθειες")
        attempts = []
        for a in range(args.attempts):
            try:
                nq = await _rewrite(q)
            except Exception as e:   # best effort: μία αποτυχία δεν ρίχνει το probe
                print(f"    #{a + 1} σφάλμα rewrite: {type(e).__name__}")
                attempts.append({"query": "", "best": float("-inf"), "cov": 0.0})
                continue
            if not nq or nq.strip().lower() == q.strip().lower():
                # Η παραγωγή σταματά εδώ: ίδιο ερώτημα -> ίδιο αποτέλεσμα.
                print(f"    #{a + 1} rewrite ταυτόσημο -> παραλείπεται")
                attempts.append({"query": nq, "best": float("-inf"), "cov": 0.0})
                continue
            r2 = await _retrieve(nq, allowed_ids, idx, dm)
            best2 = r2[0][0]
            pages = await asyncio.to_thread(
                ai_core._expand_to_pages, r2[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)
            cov = coverage("\n\n".join(p for p, _m in pages), kw)
            attempts.append({"query": nq, "best": best2,
                             "cov": 0.0 if cov != cov else cov})
            print(f"    #{a + 1} best {best2:+7.2f}  cov "
                  f"{'-' if cov != cov else f'{cov:5.1f}'}  «{nq[:56]}»")
        rows.append({"id": t["id"], "set": t["_set"],
                     "category": t.get("category", ""),
                     "first_best": first_best, "attempts": attempts})

    if not rows:
        print("Καμία ερώτηση δεν κόπηκε — δεν υπάρχει τίποτα να μετρηθεί.")
        return 0

    # ------------------------------------------------- ανάλυση ανά N, offline
    print("\n" + "=" * 78)
    print(f"ΤΙ ΘΑ ΕΔΙΝΕ N ΠΡΟΣΠΑΘΕΙΕΣ (κατώφλι {thr:+.2f})")
    print("=" * 78)
    print(f"{'N':>3}{'σωσμένες':>12}{'ψευδαισθ.':>12}{'ΔΙΑΡΡΟΕΣ ooc':>15}")
    for n in range(1, args.attempts + 1):
        saved = halluc = leaks = 0
        for r in rows:
            sub = r["attempts"][:n]
            if not sub:
                continue
            best = max(sub, key=lambda a: a["best"])
            if best["best"] < thr:
                continue
            if r["category"] == "out_of_corpus":
                leaks += 1
            elif best["cov"] > 0:
                saved += 1
            else:
                halluc += 1
        print(f"{n:>3}{saved:>12}{halluc:>12}{leaks:>15}")

    # ------------------------------------------------- διασπορά αναδιατύπωσης
    print("\nΔΙΑΣΠΟΡΑ ΤΗΣ ΑΝΑΔΙΑΤΥΠΩΣΗΣ (ίδια ερώτηση, ανεξάρτητες προσπάθειες)")
    spreads = []
    for r in rows:
        vals = [a["best"] for a in r["attempts"] if a["best"] > float("-inf")]
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        spreads.append(spread)
        flips = len({v >= thr for v in vals}) > 1
        print(f"  {r['id']:<6} εύρος {spread:5.2f} logits  "
              f"({min(vals):+.2f} … {max(vals):+.2f})"
              + ("   <== Η ΑΠΟΦΑΣΗ ΓΥΡΙΖΕΙ" if flips else ""))
    if spreads:
        print(f"  ΜΕΣΟ ΕΥΡΟΣ: {sum(spreads) / len(spreads):.2f} logits σε "
              f"{len(spreads)} ερωτήσεις")
        print("  Για σύγκριση: το κενό του gate είναι 1.04 logits.")
    print("=" * 78)

    if args.csv:
        flat = []
        for r in rows:
            for i, a in enumerate(r["attempts"], 1):
                flat.append({"id": r["id"], "set": r["set"],
                             "category": r["category"],
                             "first_best": round(r["first_best"], 3),
                             "attempt": i,
                             "best": (None if a["best"] == float("-inf")
                                      else round(a["best"], 3)),
                             "coverage": round(a["cov"], 1),
                             "rewrite": a["query"]})
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader()
            w.writerows(flat)
        print(f"→ γράφτηκε {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Πολλαπλές προσπάθειες corrective: σώζουν ή διαρρέουν;")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--csv", default=None)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
