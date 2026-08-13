"""ΑΣΥΜΜΕΤΡΟ 2ο ΠΕΡΑΣΜΑ: η αναδιατύπωση ΒΡΙΣΚΕΙ, η αρχική ερώτηση ΚΡΙΝΕΙ.

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Σήμερα η αναδιατύπωση του corrective agent κάνει ΔΥΟ δουλειές:
    q2 = rewrite(q1)
      -> dense(q2) + bm25(q2) -> RRF      ΑΝΑΖΗΤΗΣΗ
      -> rerank(q2) -> gate CORRECTIVE_MIN_SCORE = +0.4    ΚΡΙΣΗ
Από τη δεύτερη βγαίνουν και τα δύο ανοιχτά προβλήματα:

  1. Η ΚΛΙΜΑΚΑ ΜΕΤΑΤΟΠΙΖΕΤΑΙ. Το ms-marco-MiniLM είναι εκπαιδευμένο σε
     ΕΡΩΤΗΣΕΙΣ· ο agent παράγει ΟΝΟΜΑΤΙΚΕΣ ΦΡΑΣΕΙΣ. Γι' αυτό χρειάστηκε ποτέ
     δεύτερο κατώφλι -- και γι' αυτό το h008 κάθεται στο -3.16 με ΣΩΣΤΗ
     κατάταξη (θέσεις 1-5 όλες από το σωστό paper).
  2. ΚΡΙΝΕΙ Η ΕΡΩΤΗΣΗ ΠΟΥ ΕΦΗΥΡΕ ΤΟ GEMINI, ΟΧΙ Η ΕΡΩΤΗΣΗ ΤΟΥ ΧΡΗΣΤΗ. Έτσι
     περνάει το h016 στο -1.15 ΧΩΡΙΣ ΥΛΙΚΟ: η αναδιατύπωσή του ταιριάζει σε
     γενικό κείμενο αποτυχιών, ενώ η πραγματική ερώτηση δεν απαντιέται.

Η παραλλαγή: rerank ΚΑΙ gate με το ΑΡΧΙΚΟ (μεταφρασμένο) ερώτημα. Τότε οι
βαθμολογίες γυρίζουν στην κλίμακα του 1ου περάσματος και το CORRECTIVE_MIN_SCORE
-- το πιο εύθραυστο σημείο της αλυσίδας, βαθμονομημένο σε ΔΥΟ ερωτήσεις --
παύει να χρειάζεται.

ΤΙ ΑΠΟΦΑΣΙΖΕΙ:
  h008 πάνω από το -2.6  ΚΑΙ  h016 κάτω  ΚΑΙ  out_of_corpus 5/5 κομμένα
      -> λύνεται η ερώτηση ΚΑΙ διαγράφεται ένα κατώφλι.
  h005/h015 πέφτουν      -> ισοπαλία σε αριθμό, αλλά πάλι ένα κατώφλι λιγότερο.
  οποιαδήποτε διαρροή ooc -> η σχεδίαση είναι λάθος, κλείνει επιτόπου.

ΜΗΔΕΝ κόστος: οι αναδιατυπώσεις q2 διαβάζονται από το CSV μιας προηγούμενης
εκτέλεσης του probe_corrective_rewrite.py. Τα υπόλοιπα είναι ντετερμινιστικά.

CONTROL: η βαθμολόγηση με q2 ΠΡΕΠΕΙ να αναπαράγει το best2 του CSV. Αν δεν το
αναπαράγει, το probe δεν αναπαριστά την παραγωγή και τίποτα άλλο δεν μετράει.

    docker compose exec backend python evaluation/probe_corrective_asymmetric.py
    docker compose exec backend python evaluation/probe_corrective_asymmetric.py \
        --in evaluation/runs/corr_v1_control.csv --csv evaluation/runs/corr_asym.csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

SETS = ["/app/evaluation/golden_hard_paraphrase.jsonl",
        "/app/evaluation/golden_set_50.jsonl",
        "/app/evaluation/golden_multihop_new.jsonl"]


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


def first_hit(ordered, kws):
    """Θέση του πρώτου chunk που περιέχει keyword — ΘΕΣΗ, όχι κάλυψη."""
    for r, (_s, text) in enumerate(ordered, 1):
        low = text.lower()
        if any(k.lower() in low for k in kws):
            return r
    return None


class Pipeline:
    """Το retrieval του search_documents μέχρι το gate, χωρίς το gate.

    Το retrieve γίνεται με ΕΝΑ ερώτημα και η βαθμολόγηση με ΑΛΛΟ — αυτή είναι
    ολόκληρη η παραλλαγή που δοκιμάζεται εδώ.
    """

    def __init__(self):
        where = ai_core._build_where(GOLDEN_CORPUS, None)
        self.allowed = ai_core.collection.get(where=where, include=[])["ids"]
        self.idx = ai_core._get_bm25_index()
        self.dm = ai_core._get_dense_matrix()

    def candidates(self, query):
        d_ids = ai_core._dense_exact_ids(
            self.dm, query, self.allowed,
            min(ai_core.DENSE_CANDIDATES, len(self.allowed)))
        s_ids = ai_core._bm25_sparse_ids(
            self.idx, query, self.allowed, ai_core.DENSE_CANDIDATES)
        return ai_core._rrf_fuse(
            d_ids, s_ids, self.idx["ids"], self.idx["texts"], self.idx["metas"],
            k=60, top_n=ai_core.RERANK_CANDIDATES, pos=self.idx["pos"])

    def score(self, query, rrf):
        pairs = [[query, it[1]] for it in rrf]
        scores = ai_core.reranker.predict(
            pairs, batch_size=ai_core.RERANK_BATCH_SIZE)
        return sorted(zip((float(x) for x in scores), [it[1] for it in rrf]),
                      key=lambda x: -x[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src",
                    default="/app/evaluation/runs/corr_v1_control.csv")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as fh:
        src_rows = list(csv.DictReader(fh))
    golden = load_golden()

    gate = ai_core.MIN_RERANK_SCORE
    corr = ai_core.CORRECTIVE_MIN_SCORE
    print("είσοδος: %s (%d αναδιατυπώσεις, ΜΗΔΕΝ κλήσεις Gemini)"
          % (args.src, len(src_rows)))
    print("σήμερα: rerank(q2) έναντι %+.1f   ·   παραλλαγή: rerank(q1) έναντι %+.1f\n"
          % (corr, gate))

    pipe = Pipeline()
    rows, drift = [], []

    hdr = ("%-6s %-14s %8s %8s %8s   %-13s %-13s %5s %5s"
           % ("id", "cat", "best2(q2)", "CSV", "best2(q1)",
              "σήμερα", "παραλλαγή", "kw@2", "kw@1"))
    print(hdr)
    print("-" * len(hdr))

    for r in src_rows:
        qid = r["id"]
        q1, q2 = r["q1"], r["q2"]
        t = golden.get(qid, {})
        kws = t.get("keywords", [])
        cat = r.get("category", "")
        ooc = cat == "out_of_corpus"

        rrf = pipe.candidates(q2)          # ΑΝΑΖΗΤΗΣΗ με την αναδιατύπωση
        ord2 = pipe.score(q2, rrf)         # control: όπως σήμερα
        ord1 = pipe.score(q1, rrf)         # παραλλαγή: κρίνει η αρχική

        b2, b1 = ord2[0][0], ord1[0][0]
        csv_b2 = float(r["best2"])
        if abs(b2 - csv_b2) > 0.005:
            drift.append((qid, csv_b2, b2))

        h2 = first_hit(ord2, kws) if kws else None
        h1 = first_hit(ord1, kws) if kws else None

        now = ("ΔΙΑΡΡΟΗ" if b2 >= corr else "κομμένο") if ooc else \
              ("ΠΕΡΝΑΕΙ" if b2 >= corr else "κομμένο")
        new = ("ΔΙΑΡΡΟΗ" if b1 >= gate else "κομμένο") if ooc else \
              ("ΠΕΡΝΑΕΙ" if b1 >= gate else "κομμένο")

        flag = ""
        if now != new:
            flag = "  <== ΑΛΛΑΖΕΙ"
        print("%-6s %-14s %8.2f %8.2f %8.2f   %-13s %-13s %5s %5s%s"
              % (qid, cat, b2, csv_b2, b1, now, new,
                 h2 or "-", h1 or "-", flag), flush=True)

        rows.append(dict(id=qid, category=cat, q1=q1, q2=q2,
                         best2_q2=round(b2, 4), best2_csv=csv_b2,
                         best2_q1=round(b1, 4), kw_q2=h2, kw_q1=h1,
                         now=now, variant=new))

    print("\n" + "=" * 78)
    if drift:
        print("*** CONTROL ΑΠΕΤΥΧΕ — το probe ΔΕΝ αναπαράγει το CSV:")
        for qid, a, b in drift:
            print("      %-6s CSV %.2f  έναντι  τώρα %.2f" % (qid, a, b))
        print("    Τίποτα παρακάτω δεν μετράει.")
    else:
        print("CONTROL: %d/%d ταυτόσημα με το CSV (|Δ| < 0.005) — το probe "
              "αναπαριστά την παραγωγή" % (len(src_rows), len(src_rows)))

    inc = [x for x in rows if x["category"] != "out_of_corpus"]
    ooc_rows = [x for x in rows if x["category"] == "out_of_corpus"]
    pass_now = [x["id"] for x in inc if x["now"] == "ΠΕΡΝΑΕΙ"]
    pass_new = [x["id"] for x in inc if x["variant"] == "ΠΕΡΝΑΕΙ"]
    leaks = [x["id"] for x in ooc_rows if x["variant"] == "ΔΙΑΡΡΟΗ"]

    print("\nin-corpus που περνούν το 2ο πέρασμα")
    print("  σήμερα    (%d): %s" % (len(pass_now), ", ".join(pass_now) or "-"))
    print("  παραλλαγή (%d): %s" % (len(pass_new), ", ".join(pass_new) or "-"))
    won = [i for i in pass_new if i not in pass_now]
    lost = [i for i in pass_now if i not in pass_new]
    print("  ΚΕΡΔΙΣΜΕΝΕΣ: %s" % (", ".join(won) or "καμία"))
    print("  ΧΑΜΕΝΕΣ:     %s" % (", ".join(lost) or "καμία"))

    print("\nout_of_corpus: %d/%d κομμένα %s"
          % (len(ooc_rows) - len(leaks), len(ooc_rows),
             "— ΤΟ ΚΡΙΤΗΡΙΟ ΚΡΑΤΗΣΕ" if not leaks
             else "— ΔΙΑΡΡΟΗ: " + ", ".join(leaks)))

    if won or lost:
        print("\nΤΙ ΑΛΛΑΞΕ ΚΑΙ ΓΙΑΤΙ")
        for x in rows:
            if x["id"] in won or x["id"] in lost:
                print("  %-6s %+.2f (q2) -> %+.2f (q1)   kw θέση %s -> %s"
                      % (x["id"], x["best2_q2"], x["best2_q1"],
                         x["kw_q2"] or "-", x["kw_q1"] or "-"))
                print("     q1: %s" % x["q1"])
                print("     q2: %s" % x["q2"])

    if args.csv:
        path = args.csv if os.path.isabs(args.csv) else "/app/" + args.csv
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n-> %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
