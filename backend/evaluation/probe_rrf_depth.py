"""Σε ποιο ΒΑΘΟΣ του RRF βρίσκεται η σελίδα-στόχος όταν κόβεται;

ΓΙΑΤΙ ΥΠΑΡΧΕΙ:
Το trace_query.py σταματάει στους RERANK_CANDIDATES=15 και τυπώνει σκέτο
"ΚΟΠΗΚΕ ΕΔΩ". Δεν ξέρουμε αν ο στόχος ήταν 16ος ή 27ος -- και η διαφορά
αποφασίζει αν το RERANK_CANDIDATES προς τα ΠΑΝΩ είναι μοχλός ή νεκρός δρόμος.

Η ΑΡΙΘΜΗΤΙΚΗ ΠΟΥ ΚΑΝΕΙ ΤΗ ΜΕΤΡΗΣΗ ΑΝΑΓΚΑΙΑ:
με k=60 και βάθος 30 ανά σκέλος, το ΧΕΙΡΟΤΕΡΟ διπλό (1/90+1/90 = 0.02222)
νικάει το ΚΑΛΥΤΕΡΟ μονό (1/61 = 0.01639). Άρα το RRF είναι λεξικογραφικό:
πρώτα ΟΛΗ η τομή dense∩bm25, μετά τα μοναχικά. Η θέση ενός μοναχικού είναι
    |τομή| + (πόσα μοναχικά έχουν καλύτερη θέση από αυτό)
Το script το επαληθεύει εμπειρικά αντί να το υποθέσει.

ΜΗΔΕΝ κόστος γέννησης. Μία κλήση optimize_query ανά ερώτηση (συνήθως cached).

    docker compose exec backend python evaluation/probe_rrf_depth.py --id h002
    docker compose exec backend python evaluation/probe_rrf_depth.py --all
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

import ai_core

HERE = "/app/evaluation"
SETS = ["golden_set_50.jsonl", "golden_multihop_new.jsonl",
        "golden_hard_paraphrase.jsonl"]


def load_all():
    out = []
    for name in SETS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    t["_src"] = name
                    out.append(t)
    return out


def target_of(t):
    """Ποια σελίδα είναι η σωστή; Από source_pages ή από τον γονέα."""
    sp = t.get("source_pages")
    if sp:
        s = sp[0]
        if isinstance(s, dict):
            return "%s:%s" % (s.get("file_name"), s.get("page"))
        return str(s)
    return None


def is_target(meta, target):
    f, _, p = target.partition(":")
    return (str(meta.get("file_name")) == f
            and (not p or str(meta.get("page")) == p))


async def analyse(raw, target, label="", score_target=False):
    query = await ai_core.optimize_query(raw)
    where_filter = ai_core._build_where(None, None)
    allowed_ids = ai_core.collection.get(where=where_filter, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()
    pos, metas = idx["pos"], idx["metas"]

    d_ids = ai_core._dense_exact_ids(
        dm, query, allowed_ids, min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
    s_ids = ai_core._bm25_sparse_ids(
        idx, query, allowed_ids, ai_core.DENSE_CANDIDATES)

    d_rank = {cid: i + 1 for i, cid in enumerate(d_ids)}
    s_rank = {cid: i + 1 for i, cid in enumerate(s_ids)}
    both = set(d_ids) & set(s_ids)

    # Πλήρης RRF ΧΩΡΙΣ κόψιμο -- ίδιος τύπος με το _rrf_fuse (rank 1000 όταν λείπει)
    universe = list(dict.fromkeys(list(d_ids) + list(s_ids)))
    scored = []
    for cid in universe:
        sc = 1.0 / (60 + d_rank.get(cid, 1000)) + 1.0 / (60 + s_rank.get(cid, 1000))
        scored.append((sc, cid))
    scored.sort(key=lambda x: (-x[0], x[1]))

    hits = [(r, cid, sc) for r, (sc, cid) in enumerate(scored, start=1)
            if is_target(metas[pos[cid]], target)]

    print("=" * 74)
    print("%s%s" % (label, raw))
    print("  query      : %s" % query)
    print("  στόχος     : %s" % target)
    print("  |dense|=%d  |bm25|=%d  |ΤΟΜΗ|=%d  |ένωση|=%d"
          % (len(d_ids), len(s_ids), len(both), len(universe)))
    if not hits:
        print("  *** Ο ΣΤΟΧΟΣ ΔΕΝ ΕΙΝΑΙ ΣΕ ΚΑΝΕΝΑ ΣΚΕΛΟΣ -- τέλος ***")
        return {"rrf": None, "inter": len(both), "dense": None, "bm25": None}

    best = hits[0]
    cid = best[1]
    print("  ΘΕΣΕΙΣ ΤΟΥ ΣΤΟΧΟΥ: dense=%s  bm25=%s  ->  RRF=%d  (score %.5f)"
          % (d_rank.get(cid, "-"), s_rank.get(cid, "-"), best[0], best[2]))
    if len(hits) > 1:
        print("  (άλλα chunks ίδιας σελίδας σε RRF: %s)"
              % ", ".join(str(h[0]) for h in hits[1:]))
    need = best[0]
    print("  --> RERANK_CANDIDATES που θα το περνούσε: >= %d  (τώρα %d)"
          % (need, ai_core.RERANK_CANDIDATES))

    # ΤΟ ΚΡΙΣΙΜΟ: ακόμα κι αν φτάσει στον reranker, τι logit παίρνει;
    # Αν είναι κάτω από το gate, ΚΑΘΕ αύξηση βάθους είναι νεκρός δρόμος.
    if score_target:
        rrf15 = [cid for _sc, cid in scored[:ai_core.RERANK_CANDIDATES]]
        # ΟΛΑ τα chunks της σελίδας-στόχου, ΟΧΙ μόνο όσα βρήκαν τα δύο σκέλη.
        tgt_ids = [c for c in idx["ids"] if is_target(metas[pos[c]], target)]
        print("  (η σελίδα-στόχος έχει %d chunks συνολικά· στα σκέλη βρέθηκαν %d)"
              % (len(tgt_ids), len(hits)))
        pool = list(dict.fromkeys(rrf15 + tgt_ids))
        pairs = [[query, idx["texts"][pos[c]]] for c in pool]
        sc = [float(x) for x in ai_core.reranker.predict(
            pairs, batch_size=ai_core.RERANK_BATCH_SIZE)]
        order = sorted(zip(sc, pool), key=lambda x: -x[0])
        print("  --- RERANKER με τον στόχο ΧΕΙΡΟΚΙΝΗΤΑ μέσα (gate %.1f) ---"
              % ai_core.MIN_RERANK_SCORE)
        for r, (s, c) in enumerate(order, start=1):
            if c in tgt_ids or r <= 3:
                m = metas[pos[c]]
                mark = "  <== ΣΤΟΧΟΣ" if c in tgt_ids else ""
                gate = "" if s >= ai_core.MIN_RERANK_SCORE else "  [ΚΑΤΩ ΑΠΟ ΤΟ GATE]"
                exp = "" if r <= ai_core.EXPAND_INPUT else "  [ΕΞΩ ΑΠΟ EXPAND_INPUT]"
                print("   %3d. %7.2f  %-26s σελ.%-4s%s%s%s"
                      % (r, s, m.get("file_name"), m.get("page"), mark, gate, exp))
        tpos = [r for r, (_s, c) in enumerate(order, start=1) if c in tgt_ids]
        tsc = max(s for s, c in order if c in tgt_ids)
        ok = tsc >= ai_core.MIN_RERANK_SCORE and min(tpos) <= ai_core.EXPAND_INPUT
        print("  ΕΤΥΜΗΓΟΡΙΑ: logit %.2f, θέση %d -> %s"
              % (tsc, min(tpos),
                 "ΘΑ ΕΦΤΑΝΕ ΣΤΟ PROMPT" if ok else "ΘΑ ΚΟΒΟΤΑΝ ΚΑΙ ΠΑΛΙ"))

    return {"rrf": best[0], "inter": len(both),
            "dense": d_rank.get(cid), "bm25": s_rank.get(cid)}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id")
    ap.add_argument("--question")
    ap.add_argument("--target")
    ap.add_argument("--all", action="store_true",
                    help="όλες οι ερωτήσεις με source_pages")
    args = ap.parse_args()

    if args.all:
        rows = []
        for t in load_all():
            tgt = target_of(t)
            if not tgt:
                continue
            r = await analyse(t["question"], tgt, label="[%s] " % t.get("id"))
            r["id"] = t.get("id")
            rows.append(r)
        miss = [r for r in rows if r["rrf"] is None or r["rrf"] > ai_core.RERANK_CANDIDATES]
        print("\n" + "=" * 74)
        print("ΣΥΝΟΨΗ: %d ερωτήσεις με στόχο· %d χάνουν τον στόχο στο RRF"
              % (len(rows), len(miss)))
        for r in sorted(miss, key=lambda x: (x["rrf"] is None, x["rrf"] or 0)):
            print("  %-6s RRF=%-5s dense=%-5s bm25=%-5s τομή=%d"
                  % (r["id"], r["rrf"], r["dense"], r["bm25"], r["inter"]))
        return 0

    raw, tgt = args.question, args.target
    if args.id:
        for t in load_all():
            if t.get("id") == args.id:
                raw = t["question"]
                tgt = tgt or target_of(t)
                break
    if not raw or not tgt:
        print("*** Δώσε --id (με source_pages) ή --question + --target")
        return 1
    await analyse(raw, tgt, score_target=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
