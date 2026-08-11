"""Quote-First: υποχρεωτικό αυτούσιο απόσπασμα αντί για κατώφλι βαθμολογίας.

ΤΟ ΠΡΟΒΛΗΜΑ ΠΟΥ ΠΑΕΙ ΝΑ ΛΥΣΕΙ
------------------------------
Στο 2ο πέρασμα του corrective agent η αναδιατύπωση ΒΡΙΣΚΕΙ το σωστό υλικό και το
`CORRECTIVE_MIN_SCORE=+0.4` το ΠΕΤΑΕΙ, επειδή βλέπει μόνο τη βαθμολογία του
cross-encoder — δηλαδή λεξιλογική ομοιότητα:

    h008   best2 -3.16   κάλυψη 100%   (hydroelectric / kilowatt-hours / idaho)
    h009   best2 -4.45   κάλυψη 100%   (compositing / inter-thread)

Ο κριτής ΝΑΙ/ΟΧΙ (`probe_corrective_verdict.py`) τις σώζει, αλλά λέει ΝΑΙ και στο
h012 που έχει κάλυψη **0%**. Κερδίζεις 2, πληρώνεις 1 ψευδαίσθηση.

Η ΥΠΟΘΕΣΗ
---------
Η δέσμευση σε ΣΥΓΚΕΚΡΙΜΕΝΟ απόσπασμα είναι πολύ σφιχτότερη από ένα ΝΑΙ. Το «ναι
απαντιέται» το λες φτηνά· το «να, εδώ ακριβώς» όχι. Ζητάμε από το μοντέλο
αυτούσιο substring και το επαληθεύουμε με ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟ έλεγχο σε Python.

ΠΡΟΣΟΧΗ — ΤΙ ΔΕΝ ΕΓΓΥΑΤΑΙ
-------------------------
Ο έλεγχος περιεκτικότητας επαληθεύει ότι το απόσπασμα ΥΠΑΡΧΕΙ, όχι ότι ΑΠΑΝΤΑΕΙ.
Οι σελίδες του h012 είναι αληθινές σελίδες του corpus — απλώς λάθος σελίδες, άρα
οποιαδήποτε πραγματική πρόταση περνάει τον έλεγχο. Η δομική εγγύηση πιάνει την
ΠΑΡΑΜΕΤΡΙΚΗ ψευδαίσθηση (επινοημένο γεγονός), όχι τη δική μας. Γι' αυτό εδώ
μετριέται, δεν προϋποτίθεται.

ΤΙ ΑΛΛΟ ΜΕΤΡΙΕΤΑΙ ΤΑΥΤΟΧΡΟΝΑ (ΔΩΡΕΑΝ)
--------------------------------------
RBO (Rank-Biased Overlap) ανάμεσα στους top-30 του dense και του BM25 για το
αναδιατυπωμένο ερώτημα: συμφωνούν τα δύο σκέλη; Είναι listwise σήμα, διαφορετικό
από το `best − mean15` που δοκιμάστηκε και έδωσε πλήρη επικάλυψη. Κοστίζει μηδέν.

ΝΤΕΤΕΡΜΙΝΙΣΜΟΣ
--------------
Οι αναδιατυπώσεις ΔΕΝ ξαναπαράγονται — διαβάζονται από το `corrective_verdict.csv`.
Άρα η είσοδος είναι καρφωμένη και το τρέξιμο συγκρίσιμο με το προηγούμενο.
Μοναδική πηγή μη ντετερμινισμού: η ίδια η κλήση quote-first.

ΤΟ ΑΠΑΓΟΡΕΥΤΙΚΟ: έστω ΜΙΑ out_of_corpus που περνάει ρίχνει την ιδέα.

    docker compose exec backend python evaluation/probe_quote_first.py \\
        --csv evaluation/runs/quote_first.csv
"""
import argparse
import asyncio
import csv
import json
import os
import re
import sys
import unicodedata

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core
import gemini_rest

HERE = "/app/evaluation"
SETS = ["golden_set_50.jsonl", "golden_hard_paraphrase.jsonl"]
DEFAULT_SOURCE = os.path.join(HERE, "runs", "corrective_verdict.csv")

# Αγγλικά ΕΠΙΤΗΔΕΣ: μετρημένο ότι ελληνική ερώτηση πάνω σε αγγλικό κείμενο
# καταρρέει (6 ΝΑΙ/14 ΟΧΙ έναντι 34/2). Κάθε έλεγχος LLM τρέχει στη γλώσσα
# του ΣΩΜΑΤΟΣ.
QUOTE_PROMPT = """You are answering a question using ONLY the SOURCE TEXT below.

Your first task is EXTRACTION, not synthesis. Find the exact, unmodified substring
of the SOURCE TEXT that answers the QUESTION.

Rules:
- The quote MUST be copied character-for-character from the SOURCE TEXT.
- The quote MUST by itself contain the information the QUESTION asks for.
- A quote that is merely on the same topic is NOT acceptable.
- Do not merge distant sentences. One contiguous span only.
- If no such substring exists, set "verbatim_quote" to null.

Output ONLY a JSON object, nothing else:
{{"verbatim_quote": "<exact substring, or null>"}}

--- SOURCE TEXT ---
{context}

QUESTION: {question}

JSON:"""


# --------------------------------------------------------------------- βοηθοί
def load_sets():
    by_id = {}
    for name in SETS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    t["_set"] = name
                    by_id[t["id"]] = t
    return by_id


def coverage(text: str, keywords) -> float:
    if not keywords:
        return float("nan")
    low = text.lower()
    return 100.0 * sum(1 for kw in keywords if str(kw).lower() in low) / len(keywords)


def norm(s: str) -> str:
    """NFKD + πεζά + πέταμα διακριτικών + πέταμα ΚΑΘΕ μη αλφαριθμητικού.

    Χωρίς αυτό ο έλεγχος σπάει σε κάθε αλλαγή κενού, παύλας ή εισαγωγικού που
    κάνει το μοντέλο αντιγράφοντας — η κλασική ευθραυστότητα του quote-first.
    Τα διακριτικά φεύγουν ΕΠΙΤΗΔΕΣ: σε σώμα με ονόματα συγγραφέων το
    «Fernández» -> «Fernandez» θα ήταν λανθασμένη άρνηση, όχι ψευδαίσθηση.
    """
    s = unicodedata.normalize("NFKD", s or "").lower()
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-zα-ωΑ-Ω]+", "", s)


def rbo(list_a, list_b, p: float = 0.9, depth: int = 30) -> float:
    """Rank-Biased Overlap, περικομμένο στο `depth` (κάτω φράγμα).

    Βαραίνει τις κορυφές: διαφωνία στη θέση 1 μετράει πολύ περισσότερο από
    διαφωνία στη θέση 30.
    """
    a, b = list(list_a)[:depth], list(list_b)[:depth]
    k = min(len(a), len(b))
    if k == 0:
        return 0.0
    sa, sb, total = set(), set(), 0.0
    for d in range(1, k + 1):
        sa.add(a[d - 1])
        sb.add(b[d - 1])
        total += (p ** (d - 1)) * (len(sa & sb) / d)
    return (1.0 - p) * total


def parse_quote(raw: str):
    """Βγάζει το `verbatim_quote`. Ανέχεται ```json fences και σκέτο κείμενο."""
    t = (raw or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.MULTILINE).strip()
    obj = None
    try:
        obj = json.loads(t)
    except (ValueError, TypeError):
        m = re.search(r'"verbatim_quote"\s*:\s*(null|"(?:[^"\\]|\\.)*")', t, re.S)
        if not m:
            return None
        frag = m.group(1)
        if frag == "null":
            return None
        try:
            return json.loads(frag) or None
        except (ValueError, TypeError):
            return None
    if not isinstance(obj, dict):
        return None
    q = obj.get("verbatim_quote")
    return q.strip() if isinstance(q, str) and q.strip() else None


# ---------------------------------------------------------------- pipeline
async def retrieve(query, allowed_ids, idx, dm):
    """Στάδια 2-5 του `search_documents`, + οι ΩΜΕΣ λίστες για το RBO."""
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
    return ranked, list(dense_ids), list(sparse_ids)


async def ask(prompt: str, retries: int = 4) -> str:
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


# -------------------------------------------------------------------- main
async def main(args) -> int:
    if not os.path.exists(args.source):
        print(f"Λείπει το {args.source} — τρέξε πρώτα probe_corrective_verdict.py")
        return 1
    with open(args.source, encoding="utf-8") as fh:
        source = list(csv.DictReader(fh))
    by_id = load_sets()
    thr = ai_core.CORRECTIVE_MIN_SCORE
    print(f"{len(source)} κομμένες ερωτήσεις από {os.path.basename(args.source)} · "
          f"corrective_min={thr:+.2f} · RBO p={args.p}\n")

    where_filter = ai_core._build_where(None, None)
    allowed_ids = ai_core.collection.get(where=where_filter, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    rows = []
    for src in source:
        qid = src["id"]
        t = by_id.get(qid)
        if t is None:
            continue
        nq = (src.get("rewrite") or "").strip()
        cat = src.get("category", "")
        # Ταυτόσημη αναδιατύπωση -> η παραγωγή σταματά, δεν φτάνει ποτέ εδώ.
        if not src.get("best2"):
            print(f"{qid:<6} rewrite ταυτόσημο -> μένει σιωπηλή")
            rows.append({"id": qid, "set": src.get("set", ""), "category": cat,
                         "best2": None, "coverage": 0.0, "rbo": None,
                         "quote_ok": False, "quote_len": 0, "quote_kw": False,
                         "quote": "", "rewrite": nq})
            continue

        q = await ai_core.optimize_query(t["question"])
        ranked, dense_ids, sparse_ids = await retrieve(nq, allowed_ids, idx, dm)
        best2 = ranked[0][0]
        pages = await asyncio.to_thread(
            ai_core._expand_to_pages, ranked[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)
        ctx = "\n\n".join(p for p, _m in pages)
        kw = t.get("keywords") or []
        cov = coverage(ctx, kw)
        cov = 0.0 if cov != cov else cov
        r = rbo(dense_ids, sparse_ids, p=args.p, depth=ai_core.DENSE_CANDIDATES)

        # Η ΑΡΧΙΚΗ (μεταφρασμένη) ερώτηση, ΟΧΙ η αναδιατύπωση: κρίνουμε αν
        # απαντιέται αυτό που ρώτησε ο χρήστης, όχι αυτό που έψαξε ο agent.
        quote = parse_quote(await ask(QUOTE_PROMPT.format(context=ctx, question=q)))
        nctx = norm(ctx)
        nq_quote = norm(quote) if quote else ""
        contained = bool(nq_quote) and nq_quote in nctx
        quote_kw = bool(nq_quote) and any(norm(str(k)) in nq_quote for k in kw)

        rows.append({"id": qid, "set": src.get("set", ""), "category": cat,
                     "best2": best2, "coverage": cov, "rbo": r,
                     "quote_ok": contained, "quote_len": len(nq_quote),
                     "quote_kw": quote_kw, "quote": (quote or "")[:400],
                     "rewrite": nq})
        mark = "OK " if contained else ("ΕΚΤΟΣ" if quote else "null")
        print(f"{qid:<6} [{cat:<14}] best2 {best2:+7.2f}  cov {cov:5.1f}  "
              f"RBO {r:.3f}  απόσπασμα {mark:<5} ({len(nq_quote):>3} χαρ"
              f"{', kw' if quote_kw else ''})")
        if quote:
            print(f'        «{quote[:96]}»')

    if not rows:
        print("Καμία γραμμή.")
        return 0

    # ------------------------------------------------------------ κριτήρια
    def passes(r, mode):
        by_score = r["best2"] is not None and r["best2"] >= thr
        by_quote = bool(r["quote_ok"])
        by_q30 = by_quote and r["quote_len"] >= 30
        return {"score": by_score, "quote": by_quote, "quote30": by_q30,
                "or": by_score or by_q30}[mode]

    print("\n" + "=" * 78)
    print(f"{'κριτήριο':<12}{'σωσμένες':>11}{'ΧΩΡΙΣ ΥΛΙΚΟ':>14}{'ΔΙΑΡΡΟΕΣ ooc':>15}")
    print("-" * 78)
    for mode in ("score", "quote", "quote30", "or"):
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

    print("\nΠΟΙΕΣ ΑΛΛΑΖΟΥΝ ΜΕ ΤΟ ΑΠΟΣΠΑΣΜΑ (έναντι σημερινής βαθμολογίας)")
    changed = False
    for r in rows:
        s, v = passes(r, "score"), passes(r, "quote30")
        if s == v:
            continue
        changed = True
        what = "ΚΕΡΔΙΖΕΤΑΙ" if v else "χάνεται"
        good = "σωστό υλικό" if r["coverage"] > 0 else "ΧΩΡΙΣ ΥΛΙΚΟ"
        if r["category"] == "out_of_corpus":
            good = "OUT_OF_CORPUS"
        b2 = "  -  " if r["best2"] is None else f"{r['best2']:+.2f}"
        print(f"  {r['id']:<6} {what:<11} best2 {b2}  κάλυψη {r['coverage']:5.1f}"
              f"  ({good})")
    if not changed:
        print("  καμία")

    # ------------------------------------------------- διαχωρίζει το RBO;
    print("\nΔΙΑΧΩΡΙΖΕΙ ΤΟ RBO;  (δωρεάν σήμα, listwise συμφωνία dense/BM25)")
    with_mat = sorted(r["rbo"] for r in rows
                      if r["rbo"] is not None and r["coverage"] > 0)
    without = sorted(r["rbo"] for r in rows
                     if r["rbo"] is not None and r["coverage"] == 0)
    for label, vals in (("ΜΕ υλικό   ", with_mat), ("ΧΩΡΙΣ υλικό", without)):
        if vals:
            print(f"  {label} n={len(vals):<3} "
                  f"{min(vals):.3f} … {max(vals):.3f}   "
                  + " ".join(f"{v:.3f}" for v in vals))
    if with_mat and without:
        if min(with_mat) > max(without):
            print("  ΚΑΘΑΡΟΣ ΔΙΑΧΩΡΙΣΜΟΣ — υπάρχει κατώφλι RBO που τις χωρίζει")
        else:
            print("  ΕΠΙΚΑΛΥΨΗ — κανένα κατώφλι RBO δεν τις χωρίζει")
    print("=" * 78)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({**r,
                            "best2": None if r["best2"] is None else round(r["best2"], 3),
                            "coverage": round(r["coverage"], 1),
                            "rbo": None if r["rbo"] is None else round(r["rbo"], 4)})
        print(f"→ γράφτηκε {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Quote-first στο 2ο πέρασμα + RBO ως δωρεάν σήμα.")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="CSV με τις αποθηκευμένες αναδιατυπώσεις (ντετερμινισμός)")
    ap.add_argument("--p", type=float, default=0.9, help="βάρος κορυφής του RBO")
    ap.add_argument("--csv", default=None)
    raise SystemExit(asyncio.run(main(ap.parse_args())))
