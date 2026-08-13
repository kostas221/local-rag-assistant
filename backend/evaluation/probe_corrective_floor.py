"""Τι ΑΠΑΝΤΑΕΙ πραγματικά το σύστημα αν χαμηλώσει το CORRECTIVE_MIN_SCORE;

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Το h008 έχει στο 2ο πέρασμα ΣΩΣΤΟ υλικό (kw@4, top-5 όλα από το σωστό paper)
και το CORRECTIVE_MIN_SCORE=+0.4 το πετάει στο -3.16. Ένα κατώφλι στο -3.6 το
σώζει με περιθώριο 0.44 πάνω από τον μετρημένο θόρυβο αναδιατύπωσης (0.43) και
0.87 κάτω από το πρώτο out_of_corpus (q019, -4.47). Το κύριο σετ δεν αγγίζεται
ΑΠΟ ΚΑΤΑΣΚΕΥΗ: ο agent τρέχει μόνο όταν κόβει το gate, και εκεί κόβει μόνο τα 5
out_of_corpus.

Το τίμημα είναι ΕΝΑ: περνάει και το h016 (-1.15), που δεν έχει keyword σε
κανέναν από τους 15 υποψηφίους του. Κάθε άλλο σήμα δοκιμάστηκε και έδωσε πλήρη
επικάλυψη (best-mean15, RBO, κριτής ΝΑΙ/ΟΧΙ, quote-first, ασύμμετρη
βαθμολόγηση με το αρχικό ερώτημα). Άρα το ερώτημα έχει συρρικνωθεί σε ένα
πράγμα που ΔΕΝ ΕΧΕΙ ΜΕΤΡΗΘΕΙ ΠΟΤΕ:

    τι απαντάει το Gemini για το h016 με ΕΚΕΙΝΕΣ τις σελίδες;

  ΑΡΝΗΣΗ        -> το τίμημα είναι μηδέν, το σύστημα σωπαίνει από άλλο
                   μηχανισμό (κανόνας 3 του system prompt) και κερδίζουμε
                   καθαρά +1: hard set 12/16 -> 13/16.
  ΨΕΥΔΑΙΣΘΗΣΗ   -> 1 κερδισμένη έναντι 1 χαμένης, η ίδια αριθμητική που
                   απέρριψε το bge-v2-m3 και το doc2query. Κλείνει οριστικά.

ΤΙ ΚΑΝΕΙ (read-only — ΜΗΔΕΝ αλλαγή στο ai_core.py):
παίρνει τις ΑΠΟΘΗΚΕΥΜΕΝΕΣ αναδιατυπώσεις από το CSV, ξανατρέχει ντετερμινιστικά
το 2ο πέρασμα, φτιάχνει τις σελίδες όπως ακριβώς το search_documents και τις
δίνει στο ask_ai μέσω precomputed= (το όρισμα υπάρχει ήδη για το eval).

ΚΟΣΤΟΣ: μία κλήση γέννησης ανά ερώτηση (προεπιλογή: 2).

    docker compose exec backend python evaluation/probe_corrective_floor.py
    docker compose exec backend python evaluation/probe_corrective_floor.py --ids h016,h008,h009
"""
import argparse
import asyncio
import csv
import json
import os
import sys
import unicodedata

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core

SETS = ["/app/evaluation/golden_hard_paraphrase.jsonl",
        "/app/evaluation/golden_set_50.jsonl",
        "/app/evaluation/golden_multihop_new.jsonl"]

# Δείκτες άρνησης — ΒΟΗΘΗΜΑ, όχι κριτής. Η απάντηση τυπώνεται ολόκληρη ώστε η
# ετυμηγορία να είναι ανθρώπινη· ένα regex δεν αποφασίζει αν κάτι είναι
# ψευδαίσθηση.
REFUSAL = [
    "cannot find", "could not find", "can't find", "do not contain",
    "does not contain", "no information", "not mentioned", "not provided",
    "not discussed", "not present", "δεν βρέθηκε", "δεν αναφέρεται",
    "δεν περιέχ", "δεν υπάρχει",
]


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                   if not unicodedata.combining(c))


def load_golden():
    out = {}
    for p in SETS:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    out.setdefault(t["id"], t)
    return out


def second_pass_pages(q2: str):
    """Ακριβώς τα βήματα 2-7 του search_documents, με το αναδιατυπωμένο query."""
    where = ai_core._build_where(None, None)
    allowed = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    d_ids = ai_core._dense_exact_ids(
        dm, q2, allowed, min(ai_core.DENSE_CANDIDATES, len(allowed)))
    s_ids = ai_core._bm25_sparse_ids(
        idx, q2, allowed, ai_core.DENSE_CANDIDATES)
    rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"], idx["metas"],
                            k=60, top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
    scores = ai_core.reranker.predict(
        [[q2, it[1]] for it in rrf], batch_size=ai_core.RERANK_BATCH_SIZE)
    final = sorted(zip((float(x) for x in scores),
                       [it[1] for it in rrf], [it[2] for it in rrf]),
                   key=lambda x: x[0], reverse=True)
    pages = ai_core._expand_to_pages(final[:ai_core.EXPAND_INPUT], ai_core.MAX_PAGES)
    return final[0][0], pages


async def answer(question: str, pages) -> str:
    out = []
    async for ev in ai_core.ask_ai(question, None, precomputed=pages):
        if ev.get("type") == "text":
            out.append(ev.get("data") or "")
    return "".join(out).strip()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="h016,h008")
    ap.add_argument("--in", dest="src",
                    default="/app/evaluation/runs/corr_v1_control.csv")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--repeat", type=int, default=1,
                    help="κλήσεις γέννησης ανά ερώτηση — ο generator ΔΕΝ είναι "
                         "ντετερμινιστικός· με 1 δεν ξέρεις αν η άρνηση κρατάει")
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as fh:
        stored = {r["id"]: r for r in csv.DictReader(fh)}
    golden = load_golden()
    ids = [i.strip() for i in args.ids.split(",") if i.strip()]

    print("κατώφλι σήμερα %+.1f · υποψήφιο -3.6 · κλήσεις γέννησης: %d\n"
          % (ai_core.CORRECTIVE_MIN_SCORE, len(ids)))

    rows = []
    for qid in ids:
        r, t = stored.get(qid), golden.get(qid)
        if not r or not t:
            print("%-6s ΛΕΙΠΕΙ από CSV ή golden set — παραλείπεται" % qid)
            continue

        best2, pages = second_pass_pages(r["q2"])
        if abs(best2 - float(r["best2"])) > 0.005:
            print("*** %s CONTROL ΑΠΕΤΥΧΕ: CSV %.2f έναντι τώρα %.2f"
                  % (qid, float(r["best2"]), best2))

        kws = t.get("keywords", [])
        blob = fold(" ".join(txt for txt, _m in pages))
        found = [k for k in kws if fold(k) in blob]
        cov = 100.0 * len(found) / len(kws) if kws else 0.0

        print("=" * 78)
        print("%s  [%s]  best2 %+.2f  (CSV %+.2f)" % (qid, t.get("hard_type",
              t.get("category", "")), best2, float(r["best2"])))
        print("  ερώτηση : %s" % t["question"])
        print("  rewrite : %s" % r["q2"])
        print("  σελίδες : %s" % ", ".join(
            "%s:%s" % (m["file_name"], m["page"]) for _x, m in pages))
        print("  ΚΑΛΥΨΗ  : %.0f%%  (%d/%d)  βρέθηκαν: %s"
              % (cov, len(found), len(kws), ", ".join(found) or "-"))

        for attempt in range(1, args.repeat + 1):
            txt = await answer(t["question"], pages)
            low = fold(txt)
            refused = any(fold(p) in low for p in REFUSAL)
            print("  [%d/%d] ΑΡΝΗΘΗΚΕ; %s"
                  % (attempt, args.repeat,
                     "ΝΑΙ (δείκτης άρνησης στο κείμενο)" if refused
                     else "ΟΧΙ — απάντησε"))
            for line in txt.splitlines():
                print("  | %s" % line)
            print()
            rows.append(dict(id=qid, attempt=attempt, best2=round(best2, 4),
                             coverage=round(cov, 1),
                             keywords_found=";".join(found),
                             pages=";".join("%s:%s" % (m["file_name"], m["page"])
                                            for _x, m in pages),
                             refused=refused, answer=txt))

    print("=" * 78)
    print("ΣΥΝΟΨΗ")
    for qid in ids:
        xs = [x for x in rows if x["id"] == qid]
        if not xs:
            continue
        x, n_ref = xs[0], sum(1 for x in xs if x["refused"])
        verdict = ("ΑΡΝΗΣΗ -> μηδενικό τίμημα" if n_ref == len(xs) else
                   ("ΑΠΑΝΤΗΣΕ ΜΕ ΥΛΙΚΟ" if x["coverage"] > 0
                    else "ΑΠΑΝΤΗΣΕ ΧΩΡΙΣ ΥΛΙΚΟ -> ΨΕΥΔΑΙΣΘΗΣΗ"))
        print("  %-6s best2 %+6.2f  κάλυψη %5.1f%%  αρνήσεις %d/%d  ->  %s"
              % (qid, x["best2"], x["coverage"], n_ref, len(xs), verdict))

    if args.csv and rows:
        path = args.csv if os.path.isabs(args.csv) else "/app/" + args.csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n-> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
