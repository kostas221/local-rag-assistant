"""Αφαίρεση βιβλιογραφίας από το ευρετήριο — πλήρες πείραμα σε shadow index.

ΓΙΑΤΙ ΑΥΤΟ ΚΑΙ ΟΧΙ Ο ΕΜΠΛΟΥΤΙΣΜΟΣ (12/8/2026)
---------------------------------------------
Ο εμπλουτισμός εγγράφου απορρίφθηκε (`probe_doc_enrichment_leak.py`): in-corpus
μέσο Δ **−0.61**, out_of_corpus **+0.24** — μειώνει τη διαχωρισιμότητα. Ο λόγος
βγήκε από τα δεδομένα: τα **δύο μεγαλύτερα κέρδη** ολόκληρου του τρεξίματος
(+2.72 στο `q050`, +2.99 στο `h009`) ήρθαν από **σελίδες βιβλιογραφίας**. Το
doc2query, μπροστά σε λίστα αναφορών, γράφει υποδειγματικές ερωτήσεις («Who
developed GFS?») και ο cross-encoder — εκπαιδευμένος σε ζεύγη (ερώτηση, passage)
— βλέπει μια λίστα αναφορών να **μοιάζει με απάντηση**.

Δηλαδή η βιβλιογραφία είναι **κανάλι θορύβου προς το gate**. Μετρημένο ήδη σε
ένα chunk: το `q028` πήγε **−1.97 -> −0.91** (+1.06) όταν αφαιρέθηκαν οι
αναφορές από τη σελίδα-στόχο του. Αυτό το script το κάνει σε ΟΛΟ το corpus.

ΤΙ ΑΛΛΑΖΕΙ
----------
Χτίζει προσωρινή συλλογή όπου κάθε chunk περνάει από τον `split_references` και
κρατάει **μόνο** το σώμα. Όσα μένουν κάτω από `--min-chars` (καθαρή λίστα
αναφορών) **φεύγουν εντελώς**. Τίποτα δεν προστίθεται — η αλλαγή είναι μόνο
αφαιρετική, άρα δεν μπορεί να «φυτέψει» ορολογία (ο λόγος που έπεσαν και οι
τρεις εκδοχές του query enrichment).

Η ΠΑΓΙΔΑ ΜΕΤΡΗΣΗΣ ΠΟΥ ΑΠΟΦΕΥΓΕΤΑΙ ΡΗΤΑ
--------------------------------------
Το coverage μετράει keywords μέσα στις σελίδες που φτάνουν στο Gemini. Αν ένα
keyword τυχαίνει να κάθεται μέσα σε **αναφορά**, η αφαίρεση θα το «χάσει» χωρίς
να έχει αλλάξει τίποτα στην ανάκτηση. Γι' αυτό υπολογίζονται **ΔΥΟ** coverage:
  · `cov`      πάνω στο ΚΑΘΑΡΟ κείμενο (τι θα δει όντως ο generator)
  · `cov_orig` πάνω στο ΑΡΧΙΚΟ κείμενο των ίδιων σελίδων (καθαρή ανάκτηση)
Διαφορά μεταξύ τους = artifact αφαίρεσης, ΟΧΙ αλλαγή ανάκτησης.

ΚΟΣΤΟΣ: μηδέν κλήσεις γέννησης. Οι μεταφράσεις είναι στο μόνιμο cache. Το ακριβό
μέρος είναι το embedding ~390 chunks με bge-m3 σε CPU (~2-4 λεπτά).

    docker compose exec backend python evaluation/probe_strip_references.py
    docker compose exec backend python evaluation/probe_strip_references.py \\
        --csv evaluation/runs/strip_refs.csv --show 3
"""
import argparse
import asyncio
import contextlib
import csv
import json
import os
import statistics
import sys

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/evaluation")

from probe_reference_contamination import split_references

import ai_core
from evaluation.eval_engine import GOLDEN_CORPUS

TMP_COLLECTION = "strip_refs_tmp"
SETS = [
    "/app/evaluation/golden_set_50.jsonl",
    "/app/evaluation/golden_hard_paraphrase.jsonl",
]


def load_sets(paths):
    tests = []
    for p in paths:
        if not os.path.exists(p):
            print(f"ΠΑΡΑΛΕΙΨΗ (δεν υπάρχει): {p}")
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    t["_set"] = os.path.basename(p).replace(".jsonl", "")
                    tests.append(t)
    return tests


async def run_all(tests, orig_pages, thr):
    """Τρέχει τα βήματα 2-7 όπως το search_documents, πάνω στην ΤΡΕΧΟΥΣΑ συλλογή."""
    where = ai_core._build_where(GOLDEN_CORPUS, None)
    allowed_ids = ai_core.collection.get(where=where, include=[])["ids"]
    idx = ai_core._get_bm25_index()
    dm = ai_core._get_dense_matrix()

    out = {}
    for t in tests:
        query = await ai_core.optimize_query(t["question"])
        d_ids = ai_core._dense_exact_ids(
            dm, query, allowed_ids, min(ai_core.DENSE_CANDIDATES, len(allowed_ids)))
        s_ids = ai_core._bm25_sparse_ids(idx, query, allowed_ids,
                                         ai_core.DENSE_CANDIDATES)
        rrf = ai_core._rrf_fuse(d_ids, s_ids, idx["ids"], idx["texts"],
                                idx["metas"], k=60,
                                top_n=ai_core.RERANK_CANDIDATES, pos=idx["pos"])
        scores = [float(x) for x in ai_core.reranker.predict(
            [[query, it[1]] for it in rrf], batch_size=ai_core.RERANK_BATCH_SIZE)]
        ranked = sorted(zip(scores, [it[1] for it in rrf], [it[2] for it in rrf]),
                        key=lambda x: -x[0])
        best = ranked[0][0] if ranked else -99.0

        # Το gate είναι φίλτρο ΑΝΑ chunk· ό,τι περνάει πάει στο page expansion.
        kept = [r for r in ranked if r[0] >= thr]
        pages = ai_core._expand_to_pages(kept[:ai_core.EXPAND_INPUT],
                                         ai_core.MAX_PAGES) if kept else []
        clean_txt = "\n".join(p[0] for p in pages).lower()
        orig_txt = "\n".join(
            orig_pages.get((m["file_name"], m["page"]), "") for _, m in pages).lower()

        kws = [k.lower() for k in t.get("keywords", [])]
        cov = 100.0 * sum(k in clean_txt for k in kws) / len(kws) if kws else 0.0
        cov_o = 100.0 * sum(k in orig_txt for k in kws) / len(kws) if kws else 0.0
        out[t["id"]] = dict(best=round(best, 3), cut=not kept, n_pages=len(pages),
                            cov=round(cov, 1), cov_orig=round(cov_o, 1))
    return out


async def main(args) -> int:
    tests = load_sets(args.sets)
    thr = ai_core.MIN_RERANK_SCORE
    data = ai_core.collection.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]

    # Αρχικό κείμενο ανά σελίδα — για το cov_orig (βλ. παγίδα μέτρησης).
    orig_pages = {}
    for _cid, txt, m in sorted(zip(ids, docs, metas),
                               key=lambda z: ai_core._chunk_idx_from_id(z[0])):
        key = (m["file_name"], m["page"])
        orig_pages[key] = orig_pages.get(key, "") + "\n" + txt

    # ---------- καθάρισμα ----------
    keep_ids, keep_docs, keep_metas = [], [], []
    dropped, trimmed, removed_chars = [], 0, 0
    for cid, txt, m in zip(ids, docs, metas):
        body, refs = split_references(txt)
        if refs:
            trimmed += 1
            removed_chars += len(txt) - len(body)
        if len(body.strip()) < args.min_chars:
            dropped.append((cid, len(txt)))
            continue
        keep_ids.append(cid)
        keep_docs.append(body.strip())
        keep_metas.append(m)

    print(f"corpus {len(ids)} chunks -> {len(keep_ids)} "
          f"(πετάχτηκαν {len(dropped)} καθαρές λίστες αναφορών)")
    print(f"chunks με αναφορές: {trimmed} · αφαιρέθηκαν {removed_chars:,} χαρακτήρες "
          f"({100.0 * removed_chars / sum(len(d) for d in docs):.1f}% του corpus)\n")
    for cid, n in dropped[:args.show]:
        print(f"  ΠΕΤΑΧΤΗΚΕ {cid}  ({n} χαρ)")
    if args.show and dropped:
        print()

    # ---------- control: η ΠΑΡΑΓΩΓΗ, ίδια διεργασία ----------
    print("--- control (παραγωγή) ---", flush=True)
    base = await run_all(tests, orig_pages, thr)

    # ---------- shadow index ----------
    saved = ai_core.collection
    # Δεν υπάρχει: μια χαρά — απλώς δεν έμεινε από προηγούμενο τρέξιμο.
    with contextlib.suppress(Exception):
        ai_core.chroma_client.delete_collection(TMP_COLLECTION)
    tmp = ai_core.chroma_client.get_or_create_collection(
        name=TMP_COLLECTION, embedding_function=ai_core.sentence_transformer_ef,
        metadata={"hnsw:space": "cosine"})
    try:
        print("--- χτίσιμο shadow index (embedding σε CPU) ---", flush=True)
        for i in range(0, len(keep_ids), 200):
            tmp.add(documents=keep_docs[i:i + 200], metadatas=keep_metas[i:i + 200],
                    ids=keep_ids[i:i + 200])
        ai_core.collection = tmp
        ai_core._bump_corpus_version()
        print("--- ΧΩΡΙΣ ΒΙΒΛΙΟΓΡΑΦΙΑ ---", flush=True)
        strip = await run_all(tests, orig_pages, thr)
    finally:
        ai_core.collection = saved
        ai_core._bump_corpus_version()
        with contextlib.suppress(Exception):
            ai_core.chroma_client.delete_collection(TMP_COLLECTION)

    # ---------- σύγκριση ----------
    rows = []
    for t in tests:
        b, s = base[t["id"]], strip[t["id"]]
        rows.append(dict(
            id=t["id"], set=t["_set"], category=t.get("category") or "-",
            best_before=b["best"], best_after=s["best"],
            d_best=round(s["best"] - b["best"], 3),
            cut_before=b["cut"], cut_after=s["cut"],
            cov_before=b["cov"], cov_after=s["cov"], cov_orig_after=s["cov_orig"],
            pages_before=b["n_pages"], pages_after=s["n_pages"]))

    print(f"\n{'id':<7}{'κατηγορία':<15}{'πριν':>8}{'μετά':>8}{'Δ':>8}"
          f"{'cov':>8}{'->':>4}{'cov':>7}{'  (αρχ.)':>9}  gate")
    print("-" * 88)
    for r in rows:
        flag = ""
        if r["cut_before"] != r["cut_after"]:
            flag = "  <<< ΚΟΒΕΙ ΤΩΡΑ" if r["cut_after"] else "  <<< ΠΕΡΝΑΕΙ ΤΩΡΑ"
        elif abs(r["cov_after"] - r["cov_before"]) >= 0.1:
            flag = "  <<< coverage"
        print(f"{r['id']:<7}{r['category'][:14]:<15}{r['best_before']:>8.2f}"
              f"{r['best_after']:>8.2f}{r['d_best']:>+8.2f}"
              f"{r['cov_before']:>8.1f}{'->':>4}{r['cov_after']:>7.1f}"
              f"{r['cov_orig_after']:>9.1f}"
              f"  {'ΚΟΒΕΙ' if r['cut_after'] else 'περνά'}{flag}")

    ooc = [r for r in rows if r["category"] == "out_of_corpus"]
    hard = [r for r in rows if r["set"].startswith("golden_hard")]
    main_in = [r for r in rows if r["category"] != "out_of_corpus"
               and not r["set"].startswith("golden_hard")]

    print("\n" + "=" * 88)
    print("ΣΥΓΚΕΝΤΡΩΤΙΚΑ")
    print("=" * 88)
    for label, grp in (("κύριο in-corpus", main_in), ("hard set", hard),
                       ("OUT_OF_CORPUS", ooc)):
        if not grp:
            continue
        d = [r["d_best"] for r in grp]
        cb = statistics.mean(r["cov_before"] for r in grp)
        ca = statistics.mean(r["cov_after"] for r in grp)
        co = statistics.mean(r["cov_orig_after"] for r in grp)
        print(f"  {label:<17} n={len(grp):<3} Δbest μέσο {statistics.mean(d):>+6.2f}"
              f" · διάμ {statistics.median(d):>+6.2f}"
              f"  ·  coverage {cb:>5.1f} -> {ca:>5.1f} (αρχικό κείμενο {co:>5.1f})")

    lo_b = min(r["best_before"] for r in rows if r["category"] != "out_of_corpus")
    hi_b = max(r["best_before"] for r in ooc) if ooc else 0
    lo_a = min(r["best_after"] for r in rows if r["category"] != "out_of_corpus")
    hi_a = max(r["best_after"] for r in ooc) if ooc else 0
    print(f"\n  ΚΕΝΟ GATE : πριν {lo_b - hi_b:+.2f} (in-min {lo_b:.2f} / ooc-max "
          f"{hi_b:.2f})  ->  μετά {lo_a - hi_a:+.2f} (in-min {lo_a:.2f} / ooc-max "
          f"{hi_a:.2f})")

    now_cut = [r for r in rows if r["cut_after"] and not r["cut_before"]]
    now_pass = [r for r in rows if r["cut_before"] and not r["cut_after"]]
    leak = [r for r in ooc if not r["cut_after"]]
    print(f"  ΑΛΛΑΓΕΣ GATE: περνούσαν->κόβονται {len(now_cut)}"
          + (f" ({', '.join(r['id'] for r in now_cut)})" if now_cut else "")
          + f"  ·  κόβονταν->περνούν {len(now_pass)}"
          + (f" ({', '.join(r['id'] for r in now_pass)})" if now_pass else ""))
    print(f"  out_of_corpus που ΔΙΑΡΡΕΟΥΝ: {len(leak)}/{len(ooc)}"
          + (f" ({', '.join(r['id'] for r in leak)})" if leak else " — ΟΛΑ ΣΙΩΠΗΛΑ"))

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.csv}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="αφαίρεση βιβλιογραφίας, shadow index")
    ap.add_argument("--sets", nargs="*", default=SETS)
    ap.add_argument("--min-chars", type=int, default=60,
                    help="κάτω από αυτό το chunk πετιέται ολόκληρο")
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--csv", default="/app/evaluation/runs/strip_refs.csv")
    raise SystemExit(asyncio.run(main(ap.parse_args())))
